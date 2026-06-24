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
    ):
        super().__init__()
        if grid_map is None:
            grid_map = np.load(DEFAULT_GRID)
        if port_locations is None:
            with open(DEFAULT_PORTS) as f:
                port_locations = json.load(f)

        self.sim = FABSimulator(
            grid_map=grid_map, port_locations=port_locations, n_amrs=n_amrs,
            request_interval=request_interval, seed=seed,
            auto_dispatch=False,          # 서빙도 env가 통제 (충전 결정과 묶음)
            battery_enabled=True,
            charge_policy="external",     # 내부 휴리스틱 충전 off → RL이 충전 통제
            battery_scale=battery_scale,
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

    # ── gym API ───────────────────────────────────────────
    def reset(self):
        self.sim.reset()
        self._decided.clear()
        self._prev_done = self._prev_dep = 0
        self._prev_drain = 0.0
        obs, _ = self._advance_to_decision()
        return obs

    def step(self, action: int):
        a = self._decision_amr
        if a is not None:
            self._decided.add(a.id)
            self._last_action = int(action)
            self._last_ctx = (len(self.sim.request_queue), a.soc)
            if action == 1:
                self.sim._send_to_charge(a)        # 실패(만석) 시 idle 유지 → 이번 step 보류
            else:
                self.sim.dispatch(a.id)            # 실패(요청없음/저soc) 시 idle 유지 → 보류

        obs, done = self._advance_to_decision()
        reward = self._reward()
        info = self.sim.get_kpi()
        return obs, reward, done, info

    # ── event-triggered 진행 ──────────────────────────────
    def _advance_to_decision(self):
        """다음 결정 필요한 idle AMR이 나올 때까지 sim 진행. (obs, done) 반환."""
        while True:
            amr = self._pick_decision_amr()
            if amr is not None:
                self._decision_amr = amr
                return self._encode(amr), False
            if self.sim.current_step >= self.max_steps:
                self._decision_amr = None
                return np.zeros(OBS_DIM, dtype=np.float32), True
            self.sim.step()
            self._decided.clear()      # 새 sim-step → 결정 재허용

    def _pick_decision_amr(self):
        """결정 대상 = idle & 아직 이번 step 미결정 AMR. soc 낮은 순 우선."""
        cands = [a for a in self.sim.amrs if a.is_idle and a.id not in self._decided]
        if not cands:
            return None
        return min(cands, key=lambda a: a.soc)

    # ── action masking ────────────────────────────────────
    def action_masks(self):
        a = self._decision_amr
        if a is None:
            return np.array([True, False])
        charger_free = any(len(occ) < CHARGER_SLOTS
                           for occ in self.sim.charger_occ.values())
        charge_ok = (a.soc < 0.99) and charger_free
        serve_ok  = True
        if a.soc < SOC_LOW and charger_free:
            serve_ok = False               # 강제 충전
        if not (serve_ok or charge_ok):
            serve_ok = True                # 둘 다 불가(저soc·만석) → 서빙(보류)로 fallback
        return np.array([serve_ok, charge_ok])

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
            free = sum(max(CHARGER_SLOTS - len(occ), 0)
                       for occ in self.sim.charger_occ.values())
            free_frac = free / (CHARGER_SLOTS * len(self.sim.chargers))
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
