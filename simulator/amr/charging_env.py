"""
ChargingFabEnv — 충전 결정 RL 환경 (A2 phase-2, docs/battery §2.5).

설계 (측정으로 디스패칭-RL 기각 후 → RL은 '충전'에만, jang_papers_review §5.4):
    dispatch  = 고정 휴리스틱 (idle AMR이 최우선 요청 take) — RL 아님
    charging  = RL ★ 유일 학습 레이어
    deadlock  = sim 내부 (구조적 처리 별도)

Action (event-triggered, A_binary):
    idle AMR 한 대마다 결정 → 0 = 서빙(최우선 요청 배정), 1 = 충전소로
    masking: soc<SOC_LOW & 충전소 여유 → 강제 충전 / soc≥0.99 or 충전소 만석 → 충전 불가

Observation (decision AMR 기준, ~13 feature, Bischoff 19 참고 축소):
    자기 soc·위치, 가까운 충전소 거리·여유, 대기요청(전체/HOT), fleet SoC 분포,
    충전 묶인 비율, idle 비율, 에피소드 진행도.

Reward (Light shaping — Bischoff: Basic 붕괴·Light 최고):
    +W_DONE·완료  −W_DEPLETE·방전  −W_ENERGY·에너지  ± 가벼운 충전맥락 shaping

호환: gym==0.19.0 / sb3-contrib MaskablePPO (action_masks)
"""

import json
import random
from pathlib import Path
from typing import Optional

import gym
import numpy as np
from gym import spaces

from simulator.amr.simulator import (
    FABSimulator, AMRStatus, Priority, SOC_LOW, CHARGER_SLOTS, BATTERY_CAP_AH,
    CELL_MM, V_MAX_MMS, ACC_MMS2, DEC_MMS2,
)

DEFAULT_GRID  = Path(__file__).parent.parent.parent / "rware_simulator" / "output" / "grid_map.npy"
DEFAULT_PORTS = Path(__file__).parent.parent.parent / "rware_simulator" / "output" / "port_locations.json"

OBS_DIM = 13


class ChargingFabEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    # ── 보상 가중치 (팀 합의 상수, 우선 초안 — docs/battery §3) ──
    W_DONE    = 1.0     # 서비스 베이스라인 (작게)
    W_DEPLETE = 50.0    # 방전 = deadlock급 실패 (무겁게)
    W_ENERGY  = 0.05    # fleet 에너지 (Ah당)
    SHAPE     = 0.2     # 충전맥락 light shaping 크기

    def __init__(
        self,
        grid_map: Optional[np.ndarray] = None,
        port_locations: Optional[dict] = None,
        n_amrs: int = 20,
        request_interval: int = 15,
        max_steps: int = 5000,
        seed: int = 42,
        battery_scale: float = 1.0,
        coupled: bool = False,
        dataset_dir: Optional[str] = None,
        sim_days: float = 30.0,
        w_block: float = 0.0,             # 혼잡(blocking) 패널티 — judge 정렬(충전이동 혼잡 억제)
        w_dl: float = 0.0,                # 데드락 패널티
        w_sla: float = 0.0,               # HOT SLA 위반 완료 패널티
        warmup_max: int = 0,              # >0 → 에피소드마다 랜덤 T(0~warmup_max)로 warm-start
        routing: str = "reservation",     # reservation | congestion | diversify(중앙쏠림 분산)
        w_cong_route: float = 2.0,
        n_chargers: int = 3,
        charger_slots: int = 1,           # 충전소당 동시충전 슬롯 — death-spiral 방지 핵심
        dispatch_policy: str = "fifo",    # fifo(최우선 요청) | nearest(최우선 tier 중 픽업 최단)
        random_start: bool = False,
    ):
        super().__init__()
        self.w_block, self.w_dl, self.w_sla = w_block, w_dl, w_sla
        self.warmup_max = warmup_max
        if grid_map is None:
            grid_map = np.load(DEFAULT_GRID)
        if port_locations is None:
            with open(DEFAULT_PORTS) as f:
                port_locations = json.load(f)

        # ── coupled(실공정) 셋업 ──
        self.coupled       = coupled
        self.sim_days      = sim_days
        self.random_start  = random_start
        self._start_rng    = np.random.RandomState(seed)
        self._pyscfab = self._hook = self._fab_files = self._ptuple_fcn = None
        self._fab_time = 0.0
        self._fab_step_sec = None
        if coupled:
            request_interval = 10**9      # 합성 랜덤요청 끄고 공정 이벤트만 주입
            self._setup_coupled(dataset_dir)

        self.sim = FABSimulator(
            grid_map=grid_map, port_locations=port_locations, n_amrs=n_amrs,
            request_interval=request_interval, seed=seed,
            auto_dispatch=False,          # 서빙도 env가 통제 (충전 결정과 묶음)
            battery_enabled=True,
            charge_policy="external",     # 내부 휴리스틱 충전 off → RL이 충전 통제
            battery_scale=battery_scale,
            routing=routing, w_cong_route=w_cong_route,
            n_chargers=n_chargers, charger_slots=charger_slots,
            dispatch_policy=dispatch_policy,
        )
        self.max_steps = max_steps
        self._n_amrs   = n_amrs

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)   # 0=서빙, 1=충전

        self._decision_amr = None
        self._decided: set = set()      # 이번 sim-step에 이미 결정한 AMR (재결정 방지)
        self._last_action  = 0
        self._last_ctx     = (0, 0.0)   # (n_pending, soc) — shaping 맥락
        self._prev_done = self._prev_dep = 0
        self._prev_drain = 0.0
        self._prev_block = self._prev_dl = self._prev_hot_done = self._prev_hot_sla = 0
        self._seed = seed              # 결정론 재seed 기준 (noise 제거)
        self._episode = 0

    # ── coupled(실공정) 헬퍼 ──────────────────────────────
    def _setup_coupled(self, dataset_dir):
        """SMT2020 공정 파일·dispatcher 로드 (PySCFabSim, lazy)."""
        import sys
        from pathlib import Path as _P
        psp = str(_P(__file__).parents[2] / "external" / "PySCFabSim")
        if psp not in sys.path:
            sys.path.insert(0, psp)
        from simulation.read import read_all
        from simulation.dispatching.dispatcher import dispatcher_map
        from simulator.fab_process.smat2022.coupled_env import AMR_STEP_SEC, DEFAULT_DATASET
        self.dataset_dir   = dataset_dir or DEFAULT_DATASET
        self._fab_files    = read_all(self.dataset_dir)
        self._ptuple_fcn   = dispatcher_map["fifo"]
        self._fab_step_sec = AMR_STEP_SEC
        self._run_to_sec   = self.sim_days * 24 * 3600

    def _advance_fab(self, until):
        """PySCFabSim을 until(초)까지 진행하며 결정포인트마다 FIFO 공정배정 → 반송요청 주입."""
        from simulation.greedy import get_lots_to_dispatch_by_machine
        while self._pyscfab.current_time < until:
            if self._pyscfab.next_decision_point() or self._pyscfab.current_time > self._run_to_sec:
                break
            while self._pyscfab.usable_machines:
                m, lots = get_lots_to_dispatch_by_machine(self._pyscfab, self._ptuple_fcn)
                if lots is None:
                    self._pyscfab.usable_machines.discard(m)
                else:
                    self._pyscfab.dispatch(m, lots)

    def _fast_forward_fab(self, until: float) -> None:
        """에피소드 시작 시각을 until(초)로 스킵.

        _advance_fab와 달리 AMR 반송 요청을 처리하지 않음.
        호출 후 sim.request_queue를 비워야 한다.
        """
        from simulation.greedy import get_lots_to_dispatch_by_machine
        while self._pyscfab.current_time < until:
            if self._pyscfab.next_decision_point():
                break
            if self._pyscfab.current_time >= self._run_to_sec:
                break
            while self._pyscfab.usable_machines:
                m, lots = get_lots_to_dispatch_by_machine(
                    self._pyscfab, self._ptuple_fcn
                )
                if lots is None:
                    self._pyscfab.usable_machines.discard(m)
                else:
                    self._pyscfab.dispatch(m, lots)

    # ── gym API ───────────────────────────────────────────
    def reset(self):
        # 에피소드별 결정론 재seed: 같은 (env seed, 에피소드)면 같은 수요 →
        # 정책 간 paired 비교 + run-to-run 재현. 학습은 에피소드마다 다른 수요(다양성).
        ep_seed = self._seed + self._episode
        random.seed(ep_seed)
        np.random.seed(ep_seed)
        self._episode += 1

        self.sim.reset()
        self._decided.clear()
        if self.coupled:
            if self._fab_files is None:        # 피클 후 lazy 로드
                from simulation.read import read_all
                self._fab_files = read_all(self.dataset_dir)
            from simulation.randomizer import Randomizer
            Randomizer().random.seed(ep_seed)  # PySCFabSim 공정 RNG 결정론화 (process time·샘플링)
            from simulation.file_instance import FileInstance
            from simulator.fab_process.smat2022.transport_hook import AMRTransportHook
            self.sim.request_queue = []        # 공정 주입 전 큐 비움
            self._hook = AMRTransportHook(self.sim)
            self._pyscfab = FileInstance(self._fab_files, run_to=self._run_to_sec,
                                         lot_for_machine=True, plugins=[self._hook])
            self._pyscfab.external_transport = False
            self._fab_time = 0.0
            # 랜덤 시작 시각: 7일치 다양한 공장 구간을 고르게 학습
            if self.random_start and self._run_to_sec > 0:
                episode_sec = self.max_steps * self._fab_step_sec
                max_start = max(0.0, self._run_to_sec - episode_sec)
                if max_start > 0:
                    offset = float(self._start_rng.uniform(0, max_start))
                    self._fast_forward_fab(offset)
                    self.sim.request_queue = []
                    self._fab_time = offset

        # warm-start: 매 에피소드 랜덤 T로 점프 → cold-start(t=0 빈 공장) 편향 제거.
        # ep_seed로 seed돼 있어 (seed,에피소드)당 결정론 → 정책 간 paired 비교 유지.
        if self.warmup_max > 0:
            self._warmup(random.randint(0, self.warmup_max))

        # reward 누적 카운터를 warm 상태 기준으로 초기화 (warmup 구간은 보상에서 제외).
        self._prev_done  = self.sim.total_completed
        self._prev_dep   = self.sim.depleted_count
        self._prev_drain = self.sim.total_drain_frac
        self._prev_block = self.sim.blocking_count
        self._prev_dl    = self.sim.deadlock_count
        self._prev_hot_done = self.sim.completed_by_pri[Priority.HOT]
        self._prev_hot_sla  = self.sim.hot_sla_met

        obs, _ = self._advance_to_decision()
        return obs

    def _warmup(self, target_step):
        """기본 휴리스틱(저soc&충전소여유→충전, 아니면 서빙)으로 target_step까지 진행.
        에피소드를 빈 공장이 아닌 '운영 중' 상태에서 시작하게 함 (랜덤 운영점)."""
        target = min(target_step, self.max_steps - 1)
        while self.sim.current_step < target:
            amr = self._pick_decision_amr()
            if amr is not None:
                self._decided.add(amr.id)
                m = self._mask_for(amr)
                self._apply(amr, 1 if (m[1] and amr.soc < SOC_LOW) else 0)
            else:
                if self.coupled:
                    self._advance_fab(self._fab_time + self._fab_step_sec)
                    self._fab_time += self._fab_step_sec
                self.sim.step()
                self._decided.clear()

    def __getstate__(self):
        s = self.__dict__.copy()
        # simulation 모듈 참조(무거움·경로의존) 직렬화 제외 → 워커/메인 unpickle 안전
        s["_fab_files"] = s["_pyscfab"] = s["_hook"] = s["_ptuple_fcn"] = None
        return s

    def __setstate__(self, s):
        self.__dict__.update(s)
        if self.coupled:
            import sys
            from pathlib import Path as _P
            psp = str(_P(__file__).parents[2] / "external" / "PySCFabSim")
            if psp not in sys.path:
                sys.path.insert(0, psp)
            if self._ptuple_fcn is None:          # dispatcher 복원 (가벼움)
                from simulation.dispatching.dispatcher import dispatcher_map
                self._ptuple_fcn = dispatcher_map["fifo"]
            # _fab_files(무거운 read_all)는 reset()에서 lazy 로드

    def step(self, action: int):
        a = self._decision_amr
        if a is not None:
            self._decided.add(a.id)
            self._last_action = int(action)
            self._last_ctx = (len(self.sim.request_queue), a.soc)
            self._apply(a, int(action))

        obs, done = self._advance_to_decision()
        reward = self._reward()
        info = self.sim.get_kpi()
        return obs, reward, done, info

    def _apply(self, a, action: int):
        """결정 실행. 1=충전소로(만석 실패시 idle 유지), 0=서빙(요청없음 실패시 idle 유지)."""
        if action == 1:
            self.sim._send_to_charge(a)
        else:
            self.sim.dispatch(a.id)

    # ── event-triggered 진행 ──────────────────────────────
    def _advance_to_decision(self):
        """다음 결정 필요한 idle AMR이 나올 때까지 sim 진행. (obs, done) 반환.
        강제스텝(유효행동 1개)도 RL에 노출되지만 MaskablePPO가 gradient=0으로 무해 처리 →
        ②(자동skip)는 결정 희소 레짐서 결정당 sim진행 폭증(fps 13)으로 비실용 → 미적용."""
        while True:
            amr = self._pick_decision_amr()
            if amr is not None:
                self._decision_amr = amr
                return self._encode(amr), False
            if self.sim.current_step >= self.max_steps:
                self._decision_amr = None
                return np.zeros(self.observation_space.shape[0], dtype=np.float32), True
            if self.coupled:           # 실공정을 AMR_STEP_SEC만큼 진행 → 실수요 주입
                self._advance_fab(self._fab_time + self._fab_step_sec)
                self._fab_time += self._fab_step_sec
            self.sim.step()
            self._decided.clear()      # 새 sim-step → 결정 재허용

    def _pick_decision_amr(self):
        """결정 대상 = idle & 아직 이번 step 미결정 AMR. soc 낮은 순 우선."""
        cands = [a for a in self.sim.amrs if a.is_idle and a.id not in self._decided]
        if not cands:
            return None
        return min(cands, key=lambda a: a.soc)

    # ── action masking (① 강제충전 floor 제거) ─────────────
    def _mask_for(self, a):
        """[serve_ok, charge_ok]. 서빙은 항상 가능(저soc 강제충전 floor 제거 →
        RL이 충전 타이밍을 직접 학습, 방전 위험은 W_DEPLETE 패널티로 교정).
        충전은 배터리 여유(<0.99) & 충전소 빈 슬롯 있을 때만(만석엔 못 감)."""
        if a is None:
            return np.array([True, False])
        charger_free = any(len(occ) < self.sim.charger_slots
                           for occ in self.sim.charger_occ.values())
        charge_ok = (a.soc < 0.99) and charger_free
        return np.array([True, charge_ok])

    def action_masks(self):
        return self._mask_for(self._decision_amr)

    # ── 보상 (Light shaping) ──────────────────────────────
    def _reward(self) -> float:
        d_done = self.sim.total_completed - self._prev_done
        d_dep  = self.sim.depleted_count  - self._prev_dep
        d_eng  = (self.sim.total_drain_frac - self._prev_drain) * BATTERY_CAP_AH \
                 / self.sim.battery_scale
        self._prev_done  = self.sim.total_completed
        self._prev_dep   = self.sim.depleted_count
        self._prev_drain = self.sim.total_drain_frac

        r = self.W_DONE * d_done - self.W_DEPLETE * d_dep - self.W_ENERGY * d_eng

        # judge 정렬 항 (혼잡·SLA) — 충전 이동이 유발하는 혼잡/SLA 손실 억제.
        # 가중치 0(기본)이면 비활성 = 기존 Light reward 그대로.
        d_block = self.sim.blocking_count - self._prev_block
        d_dl    = self.sim.deadlock_count - self._prev_dl
        hot_done = self.sim.completed_by_pri[Priority.HOT]
        d_hot_over = (hot_done - self._prev_hot_done) \
                     - (self.sim.hot_sla_met - self._prev_hot_sla)   # 이번 step HOT SLA 위반 완료
        self._prev_block, self._prev_dl = self.sim.blocking_count, self.sim.deadlock_count
        self._prev_hot_done, self._prev_hot_sla = hot_done, self.sim.hot_sla_met
        r -= self.w_block * d_block + self.w_dl * d_dl + self.w_sla * d_hot_over

        # light 충전맥락 shaping: 충전했을 때만
        if self._last_action == 1:
            n_pend, soc = self._last_ctx
            if n_pend == 0 and soc < 0.5:
                r += self.SHAPE          # 한가 + 저soc에 충전 = 좋은 타이밍
            elif n_pend >= self._n_amrs and soc > 0.6:
                r -= self.SHAPE          # 수요 폭주 + 충분soc에 충전 = 나쁜 타이밍
        return r

    # ── observation ───────────────────────────────────────
    def _encode(self, a) -> np.ndarray:
        W, H = self.sim._grid_w, self.sim._grid_h
        socs = [v.soc for v in self.sim.amrs]
        # 가까운 충전소 거리·여유
        if self.sim.chargers:
            dmin = min(abs(a.x - cx) + abs(a.y - cy) for cx, cy in self.sim.chargers)
            free = sum(max(self.sim.charger_slots - len(occ), 0)
                       for occ in self.sim.charger_occ.values())
            free_frac = free / (self.sim.charger_slots * len(self.sim.chargers))
        else:
            dmin, free_frac = W + H, 1.0
        n_pend = len(self.sim.request_queue)
        n_hot  = sum(1 for _, _, r in self.sim.request_queue if r.priority == Priority.HOT)
        n_charging = sum(1 for v in self.sim.amrs
                         if v.status in (AMRStatus.CHARGING, AMRStatus.DEPLETED)
                         or v.going_to_charge)
        n_idle = sum(1 for v in self.sim.amrs if v.is_idle)
        return np.array([
            a.soc,
            a.x / W, a.y / H,
            dmin / (W + H),
            free_frac,
            min(n_pend / 20.0, 5.0),
            min(n_hot / 10.0, 5.0),
            float(np.mean(socs)),
            float(np.min(socs)),
            np.mean([s < 0.3 for s in socs]),
            n_charging / self._n_amrs,
            n_idle / self._n_amrs,
            self.sim.current_step / self.max_steps,
        ], dtype=np.float32)

    def close(self):
        pass
