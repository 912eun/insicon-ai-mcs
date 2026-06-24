"""
FAB AMR Simulator — SMAT2022 grid 기반 다중 AMR 반송 시뮬레이터.

수정 이력:
- AMR.step() request 즉시 초기화 버그 수정 (throughput=0 버그)
- Reservation Table 절대 시간 기반으로 수정 (타이밍 버그)
- 런타임 충돌 감지 및 해소 추가
- 데드락 감지 및 재경로 탐색 추가
- LOADING / UNLOADING 상태 추가 (픽업·드롭오프 소요 시간 반영)
- ReservationTable.release() 추가 (완료된 AMR 예약 해제 → 메모리 누수 방지)
- KPI 큐 대기시간 / 이동 시간 분리
"""

import heapq
import random
from enum import Enum
from typing import Optional

import numpy as np

from simulator.amr.pathfinding import astar_with_reservation, compute_step_costs

# SMAT2022 VehicleType (fab_layout.json 기준)
CELL_MM          = 2000.0   # mm/cell
V_MAX_MMS        = 1500.0   # mm/s  → 1step ≈ 1.33s
ACC_MMS2         = 2000.0   # mm/s² (가속도)
DEC_MMS2         = 3000.0   # mm/s² (감속도, 가속보다 빠름)

# 픽업/드롭오프 작업 시간 기본값: 각 10초 ≈ 8스텝
DEFAULT_LOAD_STEPS   = 8
DEFAULT_UNLOAD_STEPS = 8
DEADLOCK_THRESHOLD   = 30   # 이만큼 제자리 = "진짜 교착" (KPI 카운트)
REPLAN_STUCK         = 5    # 이만큼 막히면 낡은 예약 풀고 조기 재경로 (cascade 차단)
HOT_SLA_STEPS        = 30   # HOT 요청 큐 대기 ≤ 이 스텝이면 SLA 충족 (transport-layer 제약)

# ── 배터리/충전 (A2 phase-1, 기본 off) — Bischoff 2505.11136 포팅, docs/battery §2.5 ──
STEP_SEC            = CELL_MM / V_MAX_MMS   # 1 step의 실시간(초) ≈ 1.33s
BATTERY_CAP_AH     = 52.0                   # 용량 (Ah)
DRAIN_LOADED_AH_H  = 15.0                   # 적재 이동 소모 (Ah/h)
DRAIN_UNLOAD_AH_H  = 10.0                   # 공차 이동 소모 (Ah/h)
CHARGE_AH_H        = 104.0                  # 충전 (Ah/h = 52Ah/0.5h, 30분 만충)
SOC_LOW            = 0.20                   # 강제충전 하한 / threshold 트리거 / dispatch 제외선
N_CHARGERS         = 3                      # 충전소 개수 (구역 균형)
CHARGER_SLOTS      = 1                      # 충전소당 동시충전 슬롯 (제한 → contention=헤드룸)


class AMRStatus(Enum):
    IDLE      = "idle"
    MOVING    = "moving"
    LOADING   = "loading"    # 픽업 포트 도착, 화물 적재 중
    UNLOADING = "unloading"  # 드롭오프 포트 도착, 화물 하역 중
    DONE      = "done"
    CHARGING  = "charging"   # 충전소 도착, 충전 중 (배터리 on)
    DEPLETED  = "depleted"   # SoC 0 방전 = 그 자리 immobilized (deadlock급 실패)


class Priority(Enum):
    HOT    = 0
    NORMAL = 1


class _Request:
    """내부 반송 요청 (heapq 관리용)."""

    _counter = 0

    def __init__(self, from_port: str, to_port: str, priority: Priority):
        _Request._counter += 1
        self.id           = _Request._counter
        self.from_port    = from_port
        self.to_port      = to_port
        self.priority     = priority
        self.created_at:  Optional[int] = None
        self.assigned_at: Optional[int] = None
        self.done_at:     Optional[int] = None
        self.amr_id:      Optional[int] = None
        # 반송 완료 시 호출되는 콜백 (closed-loop 반송용 — PySCFabSim lot release).
        self.on_complete = None

    def waiting_time(self, current_step: int) -> int:
        return current_step - self.created_at if self.created_at is not None else 0

    @property
    def queue_time(self) -> Optional[int]:
        """큐 대기 시간: 요청 생성 → AMR 배정."""
        if self.created_at is None or self.assigned_at is None:
            return None
        return self.assigned_at - self.created_at

    @property
    def travel_time(self) -> Optional[int]:
        """이동 시간: AMR 배정 → 반송 완료."""
        if self.assigned_at is None or self.done_at is None:
            return None
        return self.done_at - self.assigned_at

    def __lt__(self, other):
        return self.priority.value < other.priority.value

    def __repr__(self):
        return (f"Request#{self.id}[{self.priority.name}] "
                f"{self.from_port[:6]}→{self.to_port[:6]}")


class AMR:
    def __init__(
        self,
        amr_id: int,
        x: int,
        y: int,
        load_steps: int = DEFAULT_LOAD_STEPS,
        unload_steps: int = DEFAULT_UNLOAD_STEPS,
    ):
        self.id      = amr_id
        self.x       = x
        self.y       = y
        self.status  = AMRStatus.IDLE
        self.path: list = []
        self.path_idx   = 0
        self.request: Optional[_Request] = None

        # 픽업 포트가 full_path에서 몇 번째 인덱스인지
        self.pickup_idx = 0

        # 작업 시간 (스텝 단위)
        self._load_steps   = load_steps
        self._unload_steps = unload_steps
        self._action_steps = 0   # 현재 LOADING/UNLOADING 남은 스텝

        # 데드락 감지용
        self._stuck_steps = 0
        self._prev_pos    = (x, y)

        # 속도 프로파일 (가속·감속 반영)
        self._step_costs: list = []   # 세그먼트별 스텝 비용 (float)
        self._step_frac:  float = 0.0 # 현재 세그먼트 누적 스텝

        # ── 배터리 (battery_enabled=True일 때만 활성) ──
        self.battery_enabled = False
        self.soc           = 1.0      # State of Charge (0~1)
        self.drain_loaded  = 0.0      # 적재 이동 step당 SoC 감소
        self.drain_unload  = 0.0      # 공차 이동 step당 SoC 감소
        self.charge_rate   = 0.0      # 충전 step당 SoC 증가
        self.going_to_charge = False  # 충전소로 이동 중 (도착 시 CHARGING)
        self._charger_idx  = -1       # 점유 중인 충전소 인덱스 (슬롯 반납용)
        self._depletion_counted = False

    @property
    def position(self) -> tuple:
        return (self.x, self.y)

    @property
    def is_idle(self) -> bool:
        return self.status == AMRStatus.IDLE

    def assign(self, request: _Request, path: list, pickup_idx: int,
               step_costs: Optional[list] = None):
        self.request    = request
        self.path       = path
        self.pickup_idx = pickup_idx
        self.path_idx   = 0
        self.status     = AMRStatus.MOVING
        self._step_costs = step_costs if step_costs else [1.0] * max(len(path) - 1, 1)
        self._step_frac  = 0.0

    def step(self):
        # 충전/방전은 sim 레벨에서 처리 (슬롯 관리) → AMR.step()은 무동작
        if self.status in (AMRStatus.CHARGING, AMRStatus.DEPLETED):
            return

        if self.status == AMRStatus.LOADING:
            self._action_steps -= 1
            if self._action_steps <= 0:
                self.status = AMRStatus.MOVING
                self._step_frac = 0.0

        elif self.status == AMRStatus.UNLOADING:
            self._action_steps -= 1
            if self._action_steps <= 0:
                self.status = AMRStatus.DONE

        elif self.status == AMRStatus.MOVING:
            # 배터리: 이동(MOVING) 스텝마다 SoC 소모. 적재 여부로 소모율 구분.
            if self.battery_enabled:
                carrying = (self.request is not None and self.pickup_idx >= 0
                            and self.path_idx >= self.pickup_idx)
                self.soc -= self.drain_loaded if carrying else self.drain_unload
                if self.soc <= 0.0:
                    self.soc = 0.0
                    self.status = AMRStatus.DEPLETED   # 그 자리 방전 = 실패
                    return
            self._step_frac += 1.0
            # 누적 스텝이 현재 세그먼트 비용을 넘으면 다음 셀로 이동
            while self.path_idx < len(self.path) - 1:
                cost = self._step_costs[min(self.path_idx, len(self._step_costs) - 1)]
                if self._step_frac < cost:
                    break
                self._step_frac -= cost
                self.path_idx += 1
                self.x = self.path[self.path_idx][0]
                self.y = self.path[self.path_idx][1]
                if self.path_idx == self.pickup_idx:
                    self.status        = AMRStatus.LOADING
                    self._action_steps = self._load_steps
                    self._step_frac    = 0.0
                    return
            if self.path_idx == len(self.path) - 1 and self.status == AMRStatus.MOVING:
                if self.going_to_charge:
                    # 충전소 도착 → 충전 시작 (sim이 soc 회복·슬롯 관리)
                    self.status     = AMRStatus.CHARGING
                    self._step_frac = 0.0
                else:
                    self.status        = AMRStatus.UNLOADING
                    self._action_steps = self._unload_steps
                    self._step_frac    = 0.0

    def reset_idle(self):
        self.status        = AMRStatus.IDLE
        self.request       = None
        self.path          = []
        self.path_idx      = 0
        self.pickup_idx    = 0
        self._action_steps = 0
        self._stuck_steps  = 0
        self._prev_pos     = (self.x, self.y)
        self._step_costs   = []
        self._step_frac    = 0.0
        self.going_to_charge = False   # soc는 유지 (배터리 상태는 idle 복귀로 안 리셋)


class ReservationTable:
    def __init__(self):
        self._vertex: dict = {}
        self._edge:   dict = {}
        # amr_id → 예약 키 목록 (메모리 누수 방지용)
        self._by_amr: dict[int, list] = {}

    def is_vertex_conflict(self, x, y, t) -> bool:
        return (x, y, t) in self._vertex

    def is_edge_conflict(self, x1, y1, x2, y2, t) -> bool:
        return (x2, y2, x1, y1, t) in self._edge

    def reserve(self, path: list, amr_id: int, time_offset: int = 0):
        """path를 time_offset 기준 절대 시간으로 예약."""
        keys = []
        for t, (x, y) in enumerate(path):
            abs_t = t + time_offset
            self._vertex[(x, y, abs_t)] = amr_id
            keys.append(('v', x, y, abs_t))
            if t > 0:
                px, py = path[t - 1]
                self._edge[(px, py, x, y, abs_t)] = amr_id
                keys.append(('e', px, py, x, y, abs_t))
        self._by_amr[amr_id] = keys

    def release(self, amr_id: int):
        """AMR 완료 시 해당 예약 항목 전부 삭제."""
        for key in self._by_amr.pop(amr_id, []):
            if key[0] == 'v':
                self._vertex.pop(key[1:], None)
            else:
                self._edge.pop(key[1:], None)

    def clear(self):
        self._vertex.clear()
        self._edge.clear()
        self._by_amr.clear()


class FABSimulator:
    """
    SMAT2022 grid_map + port_locations 기반 AMR 반송 시뮬레이터.

    RL 인터페이스:
        reset()            → initial state dict
        get_state()        → observation dict
        dispatch(amr_id)   → bool
        step()             → state dict (1스텝 진행)
        get_reward()       → float
        get_kpi()          → dict
        get_idle_amrs()    → List[AMR]
        get_next_request() → _Request | None
        add_request(from_port, to_port, priority)  (PySCFabSim 훅)
    """

    def __init__(
        self,
        grid_map: np.ndarray,
        port_locations: dict,
        n_amrs: int = 20,
        request_interval: int = 15,
        load_steps: int = DEFAULT_LOAD_STEPS,
        unload_steps: int = DEFAULT_UNLOAD_STEPS,
        seed: int = 42,
        auto_dispatch: bool = True,
        cell_mm: float = CELL_MM,
        v_max_mms: float = V_MAX_MMS,
        acc_mms2: float = ACC_MMS2,
        dec_mms2: float = DEC_MMS2,
        battery_enabled: bool = False,
        charge_policy: str = "threshold",   # threshold | opportunity | highlow (휴리스틱 baseline)
        battery_scale: float = 1.0,         # drain·charge 가속 배수 = 유효용량 1/scale (용량 sweep 축/dev 단축키)
        routing: str = "reservation",       # reservation(예약A*) | congestion(혼잡가중A*) — 라우팅 헤드룸 측정용
        w_cong_route: float = 2.0,          # 혼잡가중 셀 패널티 (routing=congestion일 때)
    ):
        random.seed(seed)
        self.grid_map         = grid_map
        self.port_locations   = port_locations
        self.port_names       = list(port_locations.keys())
        self.n_amrs           = n_amrs
        self.request_interval = request_interval
        self.load_steps       = load_steps
        self.unload_steps     = unload_steps
        self.auto_dispatch    = auto_dispatch
        self.cell_mm          = cell_mm
        self.v_max_mms        = v_max_mms
        self.acc_mms2         = acc_mms2
        self.dec_mms2         = dec_mms2

        # tool_family → [port_name, ...] 인덱스 (PySCFabSim 연동용)
        self.tool_to_ports: dict[str, list[str]] = {}
        for port_name, info in port_locations.items():
            tool = info.get("tool", "")
            if tool:
                self.tool_to_ports.setdefault(tool.upper(), []).append(port_name)

        grid_h, grid_w = grid_map.shape
        self._grid_w = grid_w
        self._grid_h = grid_h

        self.current_step     = 0
        self.total_completed  = 0
        self.total_queue_time  = 0   # 큐 대기 시간 합계
        self.total_travel_time = 0   # 이동 시간 합계
        self.deadlock_count   = 0
        self.blocking_count   = 0   # 충돌 롤백 누적 = blocking_time (deadlock 전단계 연속 신호)
        self.replan_count     = 0   # 조기 재경로 발동 횟수 (진단용)
        self.abort_count      = 0   # 재경로 실패 → 요청 반환 횟수 (진단용)
        # 조기 재경로(A안) 토글. False면 기존 동작(30스텝 교착에서만 replan).
        # ablation / 수정 전·후 비교용.
        self.early_replan     = True
        self.total_busy_steps  = 0   # 에피소드 전체 non-idle AMR 누적 (평균 utilization용)
        self.total_travel_dist = 0   # 누적 이동 셀 수 (fleet 효율/에너지 프록시)
        # per-priority 분리 집계 (transport-layer: HOT은 SLA 제약, NORMAL은 효율)
        self.completed_by_pri  = {Priority.HOT: 0, Priority.NORMAL: 0}
        self.queue_time_by_pri = {Priority.HOT: 0, Priority.NORMAL: 0}
        self.travel_time_by_pri = {Priority.HOT: 0, Priority.NORMAL: 0}
        self.hot_sla_met       = 0   # HOT 중 queue_time ≤ HOT_SLA_STEPS 건수
        self.reservation      = ReservationTable()

        # ── 배터리/충전 (A2 phase-1) ──
        self.battery_enabled  = battery_enabled
        self.charge_policy    = charge_policy
        self.battery_scale    = battery_scale
        self.routing          = routing
        self.w_cong_route     = w_cong_route
        self._density_cache   = None
        self._density_step    = -1
        # step당 SoC 환산 (Bischoff Ah/h → 1 step=STEP_SEC초) × battery_scale(유효용량 1/scale)
        self._drain_loaded = DRAIN_LOADED_AH_H * STEP_SEC / 3600 / BATTERY_CAP_AH * battery_scale
        self._drain_unload = DRAIN_UNLOAD_AH_H * STEP_SEC / 3600 / BATTERY_CAP_AH * battery_scale
        self._charge_rate  = CHARGE_AH_H       * STEP_SEC / 3600 / BATTERY_CAP_AH * battery_scale
        self.chargers: list = []                # 충전소 셀 좌표
        self.charger_occ: dict = {}             # 충전소 idx → 점유 amr_id set (슬롯 제한)
        self.depleted_count    = 0
        self.total_unavail_steps = 0            # 충전/방전으로 못 쓴 AMR·step 누적 (availability)
        self.total_drain_frac    = 0.0          # 누적 소모 SoC (energy 프록시)

        self.amrs: list[AMR]          = []
        self.request_queue: list      = []
        self.completed_requests: list = []

        self.reset()

    # ── 초기화 ────────────────────────────────────────────
    def reset(self) -> dict:
        self.current_step      = 0
        self.total_completed   = 0
        self.total_queue_time  = 0
        self.total_travel_time = 0
        self.deadlock_count    = 0
        self.blocking_count    = 0
        self.replan_count      = 0
        self.abort_count       = 0
        self.total_busy_steps  = 0
        self.total_travel_dist = 0
        self.completed_by_pri  = {Priority.HOT: 0, Priority.NORMAL: 0}
        self.queue_time_by_pri = {Priority.HOT: 0, Priority.NORMAL: 0}
        self.travel_time_by_pri = {Priority.HOT: 0, Priority.NORMAL: 0}
        self.hot_sla_met       = 0
        self.depleted_count    = 0
        self.total_unavail_steps = 0
        self.total_drain_frac    = 0.0
        self.reservation.clear()
        _Request._counter = 0

        start_ports = random.sample(self.port_names, self.n_amrs)
        self.amrs = [
            AMR(
                i + 1,
                self.port_locations[start_ports[i]]["grid_x"],
                self.port_locations[start_ports[i]]["grid_y"],
                load_steps=self.load_steps,
                unload_steps=self.unload_steps,
            )
            for i in range(self.n_amrs)
        ]
        if self.battery_enabled:
            for a in self.amrs:
                a.battery_enabled = True
                a.soc          = 1.0
                a.drain_loaded = self._drain_loaded
                a.drain_unload = self._drain_unload
                a.charge_rate  = self._charge_rate
            self._build_chargers()
        self.request_queue      = []
        self.completed_requests = []

        for _ in range(self.n_amrs):
            self._generate_request()

        return self.get_state()

    def _build_chargers(self):
        """충전소 N_CHARGERS개를 구역 균형 앵커에 가장 가까운 walkable 포트셀로 배치."""
        W, H = self._grid_w, self._grid_h
        anchors = [(W * 0.25, H * 0.30), (W * 0.75, H * 0.30), (W * 0.50, H * 0.75)]
        anchors = anchors[:N_CHARGERS]
        cells = []
        for ax, ay in anchors:
            best = min(self.port_names,
                       key=lambda p: (self.port_locations[p]["grid_x"] - ax) ** 2
                                   + (self.port_locations[p]["grid_y"] - ay) ** 2)
            info = self.port_locations[best]
            cells.append((info["grid_x"], info["grid_y"]))
        self.chargers = cells
        self.charger_occ = {i: set() for i in range(len(cells))}

    # ── 요청 생성 / 주입 ──────────────────────────────────
    def _generate_request(self, hot_prob: float = 0.1):
        from_port = random.choice(self.port_names)
        to_port   = random.choice(self.port_names)
        while to_port == from_port:
            to_port = random.choice(self.port_names)
        priority = Priority.HOT if random.random() < hot_prob else Priority.NORMAL
        self._enqueue(from_port, to_port, priority)

    def _enqueue(self, from_port: str, to_port: str, priority: Priority,
                 on_complete=None):
        req            = _Request(from_port, to_port, priority)
        req.created_at = self.current_step
        req.on_complete = on_complete
        heapq.heappush(self.request_queue, (priority.value, req.id, req))
        return req

    def add_request(self, from_port: str, to_port: str, priority: str = "NORMAL",
                    on_complete=None):
        """외부 반송 요청 주입 (PySCFabSim 이벤트 훅).

        on_complete: 배송 완료 시 호출될 콜백(req를 인자로). closed-loop에서
                     PySCFabSim의 lot을 다음 장비에 release하는 데 쓴다.
        반환: 생성된 _Request, 포트 미해결 시 None.
        """
        from_port = self._resolve_port(from_port)
        to_port   = self._resolve_port(to_port)
        if from_port is None or to_port is None or from_port == to_port:
            return None
        p = Priority.HOT if priority.upper() == "HOT" else Priority.NORMAL
        return self._enqueue(from_port, to_port, p, on_complete)

    def _resolve_port(self, name: str) -> Optional[str]:
        """
        1. 정확한 port_name 일치
        2. tool_family 일치 → 해당 포트 중 랜덤
        3. prefix 매칭 → 첫 번째 결과
        """
        if name in self.port_locations:
            return name
        name_up = name.upper()
        if name_up in self.tool_to_ports:
            return random.choice(self.tool_to_ports[name_up])
        for port_name in self.port_names:
            if port_name.upper().startswith(name_up):
                return port_name
        return None

    # ── RL 인터페이스 ──────────────────────────────────────
    def get_state(self) -> dict:
        amr_states = [
            {
                "id":      amr.id,
                "x":       amr.x,
                "y":       amr.y,
                "status":  amr.status.value,
                "is_idle": amr.is_idle,
            }
            for amr in self.amrs
        ]
        pending = [
            {
                "id":           req.id,
                "from_port":    req.from_port,
                "to_port":      req.to_port,
                "priority":     req.priority.name,
                "waiting_time": req.waiting_time(self.current_step),
            }
            for _, _, req in self.request_queue
        ]
        return {
            "step":          self.current_step,
            "amrs":          amr_states,
            "pending_queue": pending,
            "zone_density":  self._zone_density(),
            "n_idle_amrs":   sum(1 for a in self.amrs if a.is_idle),
            "n_pending":     len(self.request_queue),
            "throughput":    self.total_completed,
        }

    def get_idle_amrs(self) -> list:
        return [a for a in self.amrs if a.is_idle]

    def get_next_request(self) -> Optional[_Request]:
        return self.request_queue[0][2] if self.request_queue else None

    # ── 라우팅 (예약A* | 혼잡가중A*) ───────────────────────
    def _density_grid(self) -> np.ndarray:
        """현재 AMR 점유 밀도(5x5 box 스미어). step당 1회만 계산(캐시)."""
        if self._density_step == self.current_step and self._density_cache is not None:
            return self._density_cache
        d = np.zeros((self._grid_h, self._grid_w), dtype=np.float32)
        for a in self.amrs:
            if a.status != AMRStatus.DEPLETED:
                d[a.y, a.x] += 1.0
        blur = d.copy()
        for sy in range(-2, 3):
            for sx in range(-2, 3):
                if sx or sy:
                    blur += np.roll(np.roll(d, sy, axis=0), sx, axis=1)
        self._density_cache = blur
        self._density_step  = self.current_step
        return blur

    def _find_path(self, start, goal, t):
        """routing 모드에 따라 예약A* 또는 혼잡가중A*."""
        density = self._density_grid() if self.routing == "congestion" else None
        return astar_with_reservation(
            self.grid_map, start, goal, self.reservation, time_offset=t,
            density=density, w_cong=self.w_cong_route,
        )

    def dispatch(self, amr_id: int, request_id: Optional[int] = None) -> bool:
        amr = next((a for a in self.amrs if a.id == amr_id), None)
        if amr is None or not amr.is_idle:
            return False
        # 배터리: SoC 부족 차량은 배정 거부 (충전이 우선) — dispatch 후보 필터
        if self.battery_enabled and amr.soc < SOC_LOW:
            return False
        if not self.request_queue:
            return False

        if request_id is None:
            _, _, req = heapq.heappop(self.request_queue)
        else:
            req, temp = None, []
            while self.request_queue:
                item = heapq.heappop(self.request_queue)
                if item[2].id == request_id:
                    req = item[2]
                    break
                temp.append(item)
            for item in temp:
                heapq.heappush(self.request_queue, item)
            if req is None:
                return False

        from_p = self.port_locations[req.from_port]
        to_p   = self.port_locations[req.to_port]
        pickup = (from_p["grid_x"], from_p["grid_y"])
        goal   = (to_p["grid_x"],   to_p["grid_y"])

        t0 = self.current_step
        path_to_pickup = self._find_path(amr.position, pickup, t0)
        if path_to_pickup is None:
            heapq.heappush(self.request_queue, (req.priority.value, req.id, req))
            return False

        t1 = t0 + len(path_to_pickup) - 1
        path_to_goal = self._find_path(pickup, goal, t1)
        if path_to_goal is None:
            heapq.heappush(self.request_queue, (req.priority.value, req.id, req))
            return False

        full_path  = path_to_pickup + path_to_goal[1:]
        pickup_idx = len(path_to_pickup) - 1
        step_costs = compute_step_costs(
            full_path, self.cell_mm, self.v_max_mms, self.acc_mms2, self.dec_mms2
        )

        self.reservation.reserve(full_path, amr.id, time_offset=t0)
        req.assigned_at = self.current_step
        req.amr_id      = amr.id
        amr.assign(req, full_path, pickup_idx, step_costs)
        return True

    def step(self) -> dict:
        self.current_step += 1

        prev_pos = [(a.x, a.y) for a in self.amrs]
        for amr in self.amrs:
            amr.step()

        self._resolve_collisions()
        self._detect_deadlocks()

        # 누적 이동 셀 수 (충돌 롤백 후 실제 순이동 기준)
        self.total_travel_dist += sum(
            abs(a.x - px) + abs(a.y - py)
            for a, (px, py) in zip(self.amrs, prev_pos)
        )

        # 배터리: 소모 에너지 누적 + 신규 방전 카운트
        if self.battery_enabled:
            self.total_drain_frac += sum(
                self._drain_loaded if (a.request is not None and a.pickup_idx >= 0
                                       and a.path_idx >= a.pickup_idx)
                else self._drain_unload
                for a, (px, py) in zip(self.amrs, prev_pos)
                if (a.x, a.y) != (px, py)   # 실제 이동한 차만 (근사)
            )
            for amr in self.amrs:
                if amr.status == AMRStatus.DEPLETED and not amr._depletion_counted:
                    self.depleted_count += 1
                    amr._depletion_counted = True
                    if amr._charger_idx >= 0:
                        self.charger_occ[amr._charger_idx].discard(amr.id)
                        amr._charger_idx = -1
                    self.reservation.release(amr.id)
            self._charging_step()

        # 완료된 AMR 처리
        for amr in self.amrs:
            if amr.status == AMRStatus.DONE:
                if amr.request is not None:
                    req          = amr.request
                    req.done_at  = self.current_step
                    q  = req.queue_time or 0
                    tv = req.travel_time or 0
                    self.total_queue_time  += q
                    self.total_travel_time += tv
                    self.total_completed   += 1
                    pri = req.priority
                    self.completed_by_pri[pri]  += 1
                    self.queue_time_by_pri[pri] += q
                    self.travel_time_by_pri[pri] += tv
                    if pri == Priority.HOT and q <= HOT_SLA_STEPS:
                        self.hot_sla_met += 1
                    self.completed_requests.append(req)
                    # closed-loop: 배송 완료 → PySCFabSim lot release
                    if req.on_complete is not None:
                        req.on_complete(req)
                self.reservation.release(amr.id)
                amr.reset_idle()

        if self.current_step % self.request_interval == 0:
            for _ in range(random.randint(1, 3)):
                self._generate_request()

        # 배터리: 휴리스틱 충전 트리거 (dispatch보다 먼저 → 충전행 차량은 배정 제외)
        if self.battery_enabled:
            self._charging_decisions()
            # unavailable = 충전/방전/충전이동 + 저SoC로 묶인 idle(충전 못 잡고 dispatch도 거부)
            # → "available인데 실은 일 못 하는" 거짓 가용 제거 (effective availability)
            self.total_unavail_steps += sum(
                1 for a in self.amrs
                if a.status in (AMRStatus.CHARGING, AMRStatus.DEPLETED)
                or a.going_to_charge
                or (a.is_idle and a.soc < SOC_LOW)
            )

        if self.auto_dispatch:
            self._auto_dispatch()

        # 에피소드 전체 평균 utilization을 위한 누적
        self.total_busy_steps += sum(
            1 for a in self.amrs
            if a.status in (AMRStatus.MOVING, AMRStatus.LOADING, AMRStatus.UNLOADING)
        )

        return self.get_state()

    # ── 충돌 / 데드락 ──────────────────────────────────────
    def _resolve_collisions(self):
        """
        Vertex conflict: 같은 셀에 두 AMR → 높은 id를 이전 셀로 롤백.
        Edge conflict: 두 AMR이 위치 교환 → 둘 다 롤백.
        LOADING/UNLOADING 중인 AMR은 이동하지 않으므로 충돌 대상에서 제외.
        """
        moving = [a for a in self.amrs if a.status == AMRStatus.MOVING]

        # edge conflict
        for i in range(len(moving)):
            for j in range(i + 1, len(moving)):
                a, b = moving[i], moving[j]
                a_prev = a.path[a.path_idx - 1] if a.path_idx > 0 else a.position
                b_prev = b.path[b.path_idx - 1] if b.path_idx > 0 else b.position
                if (a.x, a.y) == b_prev and (b.x, b.y) == a_prev:
                    self._rollback(a)
                    self._rollback(b)

        # vertex conflict
        pos_map: dict = {}
        for amr in moving:
            pos_map.setdefault((amr.x, amr.y), []).append(amr)
        for amrs in pos_map.values():
            if len(amrs) > 1:
                amrs.sort(key=lambda a: a.id)
                for amr in amrs[1:]:
                    self._rollback(amr)

    def _rollback(self, amr: AMR):
        # 충돌로 뒤로 밀린 step = 다른 AMR에 막힌 시간 (blocking, Chung&Jang DPB 핵심 지표)
        self.blocking_count += 1
        if amr.path_idx > 0:
            amr.path_idx -= 1
        amr.x = amr.path[amr.path_idx][0]
        amr.y = amr.path[amr.path_idx][1]

    def _detect_deadlocks(self):
        for amr in self.amrs:
            if amr.status not in (AMRStatus.MOVING, AMRStatus.LOADING, AMRStatus.UNLOADING):
                amr._stuck_steps = 0
                amr._prev_pos    = amr.position
                continue

            # LOADING/UNLOADING은 정상 정지이므로 카운트 제외
            if amr.status == AMRStatus.MOVING:
                if amr.position == amr._prev_pos:
                    amr._stuck_steps += 1
                    if amr._stuck_steps >= DEADLOCK_THRESHOLD:
                        # 조기 재경로로도 못 풀린 진짜 교착
                        self.deadlock_count += 1
                        self._replan(amr)
                        amr._stuck_steps = 0
                    elif self.early_replan and amr._stuck_steps % REPLAN_STUCK == 0:
                        # 짧게 막힘 → 낡은(시간축 어긋난) 예약을 풀고 현재 위치 기준
                        # 재경로. 지연 AMR이 stale 예약으로 다른 AMR을 막는 cascade를 차단.
                        # stuck 카운터는 유지 → 재경로가 계속 실패하면 30에서 교착 감지.
                        self.replan_count += 1
                        self._replan(amr)
                else:
                    amr._stuck_steps = 0

            amr._prev_pos = amr.position

    def _replan(self, amr: AMR):
        if amr.request is None:
            self.reservation.release(amr.id)
            amr.reset_idle()
            return

        # 픽업 전이면 픽업으로, 픽업 후면 드롭오프로 재경로
        if amr.path_idx <= amr.pickup_idx:
            goal_name = amr.request.from_port
        else:
            goal_name = amr.request.to_port

        goal_info = self.port_locations.get(goal_name)
        if goal_info is None:
            self._abort_amr(amr)
            return

        goal     = (goal_info["grid_x"], goal_info["grid_y"])
        new_path = self._find_path(amr.position, goal, self.current_step)
        if new_path is None:
            self._abort_amr(amr)
        else:
            self.reservation.release(amr.id)
            self.reservation.reserve(new_path, amr.id, time_offset=self.current_step)
            going_to_pickup = amr.path_idx <= amr.pickup_idx
            step_costs = compute_step_costs(
                new_path, self.cell_mm, self.v_max_mms, self.acc_mms2, self.dec_mms2
            )
            amr.path        = new_path
            amr.path_idx    = 0
            amr.pickup_idx  = len(new_path) - 1 if going_to_pickup else -1
            amr._step_costs = step_costs
            amr._step_frac  = 0.0
            amr.status      = AMRStatus.MOVING

    def _abort_amr(self, amr: AMR):
        """재경로 실패 시 요청을 큐에 돌려보내고 AMR을 idle로."""
        self.abort_count += 1
        if amr.request is not None:
            req = amr.request
            heapq.heappush(self.request_queue, (req.priority.value, req.id, req))
        self.reservation.release(amr.id)
        amr.reset_idle()

    def _auto_dispatch(self):
        for amr in self.get_idle_amrs():
            if not self.request_queue:
                break
            self.dispatch(amr.id)

    # ── 배터리 / 충전 (A2 phase-1) ─────────────────────────
    def _send_to_charge(self, amr: AMR) -> bool:
        """여유 슬롯 있는 가장 가까운 충전소로 amr을 보냄. 슬롯은 출발 시 점유(여행+충전 동안 유지)."""
        free = [i for i in range(len(self.chargers))
                if len(self.charger_occ[i]) < CHARGER_SLOTS]
        if not free:
            return False   # 모든 충전소 만석 → 지금은 충전 불가 (contention)
        # 거리순으로 가까운 충전소부터 경로 시도
        free.sort(key=lambda i: abs(self.chargers[i][0] - amr.x)
                              + abs(self.chargers[i][1] - amr.y))
        for i in free:
            path = self._find_path(amr.position, self.chargers[i], self.current_step)
            if path is None:
                continue
            self.reservation.reserve(path, amr.id, time_offset=self.current_step)
            step_costs = compute_step_costs(
                path, self.cell_mm, self.v_max_mms, self.acc_mms2, self.dec_mms2
            )
            amr.path        = path
            amr.path_idx    = 0
            amr.pickup_idx  = -1            # 충전 트립엔 픽업 없음
            amr._step_costs = step_costs
            amr._step_frac  = 0.0
            amr.request     = None
            amr.going_to_charge = True
            amr._charger_idx = i
            amr.status      = AMRStatus.MOVING
            self.charger_occ[i].add(amr.id)
            return True
        return False

    def _charging_step(self):
        """CHARGING AMR의 SoC 회복 + 만충 시 슬롯 반납·IDLE 복귀."""
        for amr in self.amrs:
            if amr.status != AMRStatus.CHARGING:
                continue
            amr.soc = min(1.0, amr.soc + amr.charge_rate)
            if amr.soc >= 1.0:
                amr.soc = 1.0
                if amr._charger_idx >= 0:
                    self.charger_occ[amr._charger_idx].discard(amr.id)
                    amr._charger_idx = -1
                self.reservation.release(amr.id)
                amr.reset_idle()

    def _charging_decisions(self):
        """휴리스틱 충전 트리거 (baseline). IDLE AMR을 정책에 따라 충전소로.

        threshold   : soc < SOC_LOW 이면 충전.
        opportunity : threshold + (대기요청 없고 soc<0.9이면 기회충전).
        highlow     : 큐 길면 soc<SOC_LOW만, 큐 짧으면 soc<0.9까지 선제충전(적응 상한).
        """
        if not self.battery_enabled or not self.chargers:
            return
        q = len(self.request_queue)
        for amr in self.amrs:
            if not amr.is_idle:
                continue
            soc = amr.soc
            want = False
            if self.charge_policy == "threshold":
                want = soc < SOC_LOW
            elif self.charge_policy == "opportunity":
                want = soc < SOC_LOW or (q == 0 and soc < 0.9)
            elif self.charge_policy == "highlow":
                upper = 0.9 if q < self.n_amrs else SOC_LOW   # 큐 길면 선제충전 억제
                want = soc < upper
            if want:
                self._send_to_charge(amr)

    # ── KPI / 보상 ─────────────────────────────────────────
    def get_reward(self) -> float:
        avg_queue = (
            self.total_queue_time / self.total_completed
            if self.total_completed > 0 else 0
        )
        return (
            self.total_completed * 10
            - avg_queue * 0.5
            - self.deadlock_count * 100
        )

    def get_kpi(self) -> dict:
        n = self.total_completed
        avg_queue  = round(self.total_queue_time  / n, 2) if n > 0 else 0
        avg_travel = round(self.total_travel_time / n, 2) if n > 0 else 0
        # 에피소드 전체 평균 utilization (스냅샷 아닌 누적 평균)
        total_amr_steps = self.n_amrs * max(self.current_step, 1)
        utilization = round(self.total_busy_steps / total_amr_steps * 100, 2)
        step = max(self.current_step, 1)

        def _avg(d, pri):
            c = self.completed_by_pri[pri]
            return round(d[pri] / c, 2) if c > 0 else 0

        n_hot = self.completed_by_pri[Priority.HOT]
        return {
            "step":         self.current_step,
            "throughput":   self.total_completed,
            "avg_queue_time":  avg_queue,    # 큐 대기 시간 (배정까지)
            "avg_travel_time": avg_travel,   # 이동+작업 시간 (배정→완료)
            "avg_total_time":  round(avg_queue + avg_travel, 2),
            "utilization":  round(utilization, 2),
            "pending":      len(self.request_queue),
            "deadlocks":    self.deadlock_count,
            # ── transport-layer 1차 지표 (A1) ──
            # ① 강건성: 1k 스텝당 데드락 (에피소드 길이 무관 비교용)
            "deadlock_per_1k": round(self.deadlock_count / step * 1000, 3),
            # blocking = 충돌로 막혀 대기한 누적(연속, deadlock보다 민감) — Chung&Jang DPB 핵심
            "blocking":        self.blocking_count,   # raw (결합 스코어 per-delivery 환산용)
            "blocking_per_1k": round(self.blocking_count / step * 1000, 3),
            "replans":      self.replan_count,
            "aborts":       self.abort_count,
            # ② fleet 효율: 누적 이동 셀 수 + 완료당 이동거리(에너지 프록시)
            "total_travel_dist": self.total_travel_dist,
            "dist_per_done": round(self.total_travel_dist / n, 2) if n > 0 else 0,
            # per-priority 분리 (HOT=SLA 제약, NORMAL=효율)
            "hot_done":     n_hot,
            "normal_done":  self.completed_by_pri[Priority.NORMAL],
            "hot_queue_time":    _avg(self.queue_time_by_pri,  Priority.HOT),
            "normal_queue_time": _avg(self.queue_time_by_pri,  Priority.NORMAL),
            "hot_travel_time":   _avg(self.travel_time_by_pri, Priority.HOT),
            "normal_travel_time": _avg(self.travel_time_by_pri, Priority.NORMAL),
            # ③ HOT SLA 충족률 (queue_time ≤ HOT_SLA_STEPS)
            "hot_sla_rate": round(self.hot_sla_met / n_hot, 3) if n_hot > 0 else 1.0,
            # ── 배터리 (A2 phase-1, battery_enabled일 때만 유의미) ──
            # 가용성: 충전/방전으로 못 쓴 비율 1 - (충전·방전·충전이동 step / 전체 AMR·step)
            "availability": round(1 - self.total_unavail_steps / total_amr_steps, 4)
                            if self.battery_enabled else 1.0,
            # scale 불변(실제 Ah): drain은 scale배 빠르므로 /scale로 환산
            "energy_per_delivery": round(self.total_drain_frac * BATTERY_CAP_AH
                                         / self.battery_scale / n, 4)
                            if (self.battery_enabled and n > 0) else 0,
            "depleted":     self.depleted_count,
            "depleted_per_1k": round(self.depleted_count / step * 1000, 3)
                            if self.battery_enabled else 0,
            "reward":       round(self.get_reward(), 2),
        }

    def _zone_density(self) -> dict:
        zones = {"NW": 0, "NE": 0, "SW": 0, "SE": 0}
        for amr in self.amrs:
            col = "E" if amr.x >= self._grid_w // 2 else "W"
            row = "S" if amr.y >= self._grid_h // 2 else "N"
            zones[f"{row}{col}"] += 1
        return zones
