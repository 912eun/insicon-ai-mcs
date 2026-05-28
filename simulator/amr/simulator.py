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
DEADLOCK_THRESHOLD   = 30


class AMRStatus(Enum):
    IDLE      = "idle"
    MOVING    = "moving"
    LOADING   = "loading"    # 픽업 포트 도착, 화물 적재 중
    UNLOADING = "unloading"  # 드롭오프 포트 도착, 화물 하역 중
    DONE      = "done"


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
        self.reservation      = ReservationTable()

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
        self.request_queue      = []
        self.completed_requests = []

        for _ in range(self.n_amrs):
            self._generate_request()

        return self.get_state()

    # ── 요청 생성 / 주입 ──────────────────────────────────
    def _generate_request(self, hot_prob: float = 0.1):
        from_port = random.choice(self.port_names)
        to_port   = random.choice(self.port_names)
        while to_port == from_port:
            to_port = random.choice(self.port_names)
        priority = Priority.HOT if random.random() < hot_prob else Priority.NORMAL
        self._enqueue(from_port, to_port, priority)

    def _enqueue(self, from_port: str, to_port: str, priority: Priority):
        req            = _Request(from_port, to_port, priority)
        req.created_at = self.current_step
        heapq.heappush(self.request_queue, (priority.value, req.id, req))

    def add_request(self, from_port: str, to_port: str, priority: str = "NORMAL"):
        """외부 반송 요청 주입 (PySCFabSim 이벤트 훅)."""
        from_port = self._resolve_port(from_port)
        to_port   = self._resolve_port(to_port)
        if from_port is None or to_port is None or from_port == to_port:
            return
        p = Priority.HOT if priority.upper() == "HOT" else Priority.NORMAL
        self._enqueue(from_port, to_port, p)

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

    def dispatch(self, amr_id: int, request_id: Optional[int] = None) -> bool:
        amr = next((a for a in self.amrs if a.id == amr_id), None)
        if amr is None or not amr.is_idle:
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
        path_to_pickup = astar_with_reservation(
            self.grid_map, amr.position, pickup,
            self.reservation, time_offset=t0,
        )
        if path_to_pickup is None:
            heapq.heappush(self.request_queue, (req.priority.value, req.id, req))
            return False

        t1 = t0 + len(path_to_pickup) - 1
        path_to_goal = astar_with_reservation(
            self.grid_map, pickup, goal,
            self.reservation, time_offset=t1,
        )
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

        for amr in self.amrs:
            amr.step()

        self._resolve_collisions()
        self._detect_deadlocks()

        # 완료된 AMR 처리
        for amr in self.amrs:
            if amr.status == AMRStatus.DONE:
                if amr.request is not None:
                    req          = amr.request
                    req.done_at  = self.current_step
                    self.total_queue_time  += req.queue_time or 0
                    self.total_travel_time += req.travel_time or 0
                    self.total_completed   += 1
                    self.completed_requests.append(req)
                self.reservation.release(amr.id)
                amr.reset_idle()

        if self.current_step % self.request_interval == 0:
            for _ in range(random.randint(1, 3)):
                self._generate_request()

        if self.auto_dispatch:
            self._auto_dispatch()

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
                        self.deadlock_count += 1
                        self._replan(amr)
                        amr._stuck_steps = 0
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
        new_path = astar_with_reservation(
            self.grid_map, amr.position, goal,
            self.reservation, time_offset=self.current_step,
        )
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
        utilization = (
            sum(1 for a in self.amrs
                if a.status in (AMRStatus.MOVING, AMRStatus.LOADING, AMRStatus.UNLOADING))
            / self.n_amrs * 100
        )
        return {
            "step":         self.current_step,
            "throughput":   self.total_completed,
            "avg_queue_time":  avg_queue,    # 큐 대기 시간 (배정까지)
            "avg_travel_time": avg_travel,   # 이동+작업 시간 (배정→완료)
            "avg_total_time":  round(avg_queue + avg_travel, 2),
            "utilization":  round(utilization, 2),
            "pending":      len(self.request_queue),
            "deadlocks":    self.deadlock_count,
            "reward":       round(self.get_reward(), 2),
        }

    def _zone_density(self) -> dict:
        zones = {"NW": 0, "NE": 0, "SW": 0, "SE": 0}
        for amr in self.amrs:
            col = "E" if amr.x >= self._grid_w // 2 else "W"
            row = "S" if amr.y >= self._grid_h // 2 else "N"
            zones[f"{row}{col}"] += 1
        return zones
