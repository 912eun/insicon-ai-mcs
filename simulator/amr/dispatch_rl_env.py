"""
dispatch_rl_env.py — 진짜 Dispatch RL 환경 (Phase 3).

[설계 배경]
DispatchChargingEnv(Phase 2)의 한계:
    - action 0 = EDD 하드코딩 (HOT 우선 + 대기시간 순) → RL이 배정 정책을 선택 못 함
    - 결과: 충전 타이밍만 RL, 배정은 휴리스틱 그대로 → FIFO와 차이 2.76점 (dep/1k만 차이)

[핵심 아이디어]
EDD는 "우선순위 + 대기시간"만 봄. 하지만 최적 배정은:
    ① 거리 (dist_per_done ↓ → transport_score ↑)
    ② HOT SLA 마감 긴박도
    ③ 배정 직전 AMR SoC (낮으면 먼 요청 피해야)
를 함께 고려해야 함. 이 세 가지를 jointly 학습하는 게 RL의 진짜 가치.

[Action 설계: Discrete(TOP_K + 1)]
    0 .. TOP_K-1  : EDD 정렬된 상위 K개 요청 중 index번 선택해서 배정
    TOP_K         : 충전소로
    TOP_K = 5 선택 이유:
        - K=1: EDD 고정과 동일 → 의미 없음
        - K=3: 거리 trade-off 학습 가능하나 다양성 부족
        - K=5: coverage 충분 + Discrete(6)으로 학습 부담 낮음
        - K=10+: action space 커서 수렴 느림, 문서(jang §5.1) "N³ 재배정 이득 없다" 방어

[Observation: 13(base) + TOP_K × 3 = 28-dim]
    Base 13-dim (ChargingFabEnv 그대로):
        soc, x/W, y/H, dist_charger, free_frac,
        n_pend/20, n_hot/10, fleet_soc_mean, fleet_soc_min,
        low_soc_frac, charging_frac, idle_frac, progress
    Per-request (TOP_K × 3):
        [priority_bit, norm_wait, norm_pickup_dist_from_amr]
        - priority_bit: HOT=1, NORMAL=0
        - norm_wait: waiting_time / 500 (클수록 오래 기다림)
        - norm_dist: Manhattan(AMR, pickup) / (W+H) (클수록 멀다)
        요청이 K개 미만이면 남은 슬롯은 [0, 0, 0] 패딩

[충전 로직]
    DispatchChargingEnv 완전 상속:
        - 강제충전 마스킹 (soc < SOC_LOW & 충전소 여유 → charge만 허용)
        - W_DONE=50, W_DEPLETE=50, SHAPE_HOT=1.5, SHAPE_DISPATCH_IMM=0.05
    추가:
        - SHAPE_NEAR=0.02: 가까운 요청 선택 시 소량 보너스 (dist_per_done 개선 유도)

[호환]
    gym==0.19.0 / sb3-contrib MaskablePPO
    ChargingFabEnv → DispatchChargingEnv → DispatchRLEnv (계층 상속)
    학습 스크립트: experiments/train_dispatch_rl.py
"""

import numpy as np
from gym import spaces

from simulator.amr.simulator import Priority, SOC_LOW
from simulator.amr.dispatch_charging_env import DispatchChargingEnv


class DispatchRLEnv(DispatchChargingEnv):
    """
    Top-K 요청 선택 Dispatch RL.

    Discrete(TOP_K + 1):
        0 .. TOP_K-1 : 상위 K개 EDD 요청 중 선택 배정
        TOP_K        : 충전소로
    """

    TOP_K = 5
    SHAPE_NEAR = 0.02   # 가까운 요청 선택 보너스 (dist_per_done 개선)
    SHAPE_HOT  = 1.5    # HOT 배정 보너스 (was 0.3 — hot_sla_rate 복구 핵심)

    def __init__(self, top_k: int = None, **kwargs):
        self._top_k = top_k if top_k is not None else self.TOP_K
        super().__init__(**kwargs)

        obs_dim = 13 + self._top_k * 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self._top_k + 1)

        self._chosen_dist = 0.0   # 선택한 요청까지의 거리 (SHAPE_NEAR 계산용)

    # ── reset ─────────────────────────────────────────────
    def reset(self):
        self._chosen_dist = 0.0
        return super().reset()

    # ── step: serve/charge 카운터 재정의 ──────────────────
    def step(self, action: int):
        # DispatchChargingEnv.step()은 action==0 → serve, else → charge로 카운트
        # 여기선 action < top_k → serve, action == top_k → charge
        if action < self._top_k:
            self._cnt_serve += 1
        else:
            self._cnt_charge += 1

        # 부모의 부모(ChargingFabEnv) step 로직을 직접 실행
        # (DispatchChargingEnv.step()은 카운터만 다르고 나머지 동일)
        a = self._decision_amr
        if a is not None:
            self._decided.add(a.id)
            self._last_action = int(action)
            self._last_ctx = (len(self.sim.request_queue), a.soc)
            self._apply(a, int(action))

        obs, done = self._advance_to_decision()
        reward = self._reward()
        info = self.sim.get_kpi()

        if done:
            total = max(self._cnt_serve + self._cnt_charge, 1)
            info["serve_rate"] = round(self._cnt_serve / total, 4)
            info["charge_rate"] = round(self._cnt_charge / total, 4)

        return obs, reward, done, info

    # ── observation ───────────────────────────────────────
    def _encode(self, a) -> np.ndarray:
        # ChargingFabEnv._encode() = 13-dim base
        from simulator.amr.charging_env import ChargingFabEnv
        base = ChargingFabEnv._encode(self, a)
        return np.concatenate([base, self._req_features(a)])

    def _req_features(self, a) -> np.ndarray:
        """상위 TOP_K 요청의 (priority, norm_wait, norm_dist) 벡터. 없으면 0 패딩."""
        W, H = self.sim._grid_w, self.sim._grid_h
        step = self.sim.current_step
        reqs = self._sorted_requests()[:self._top_k]
        feats = np.zeros(self._top_k * 3, dtype=np.float32)
        for i, r in enumerate(reqs):
            px = self.sim.port_locations[r.from_port]["grid_x"]
            py = self.sim.port_locations[r.from_port]["grid_y"]
            dist = (abs(a.x - px) + abs(a.y - py)) / (W + H)
            feats[i * 3]     = 1.0 if r.priority == Priority.HOT else 0.0
            feats[i * 3 + 1] = min(r.waiting_time(step) / 500.0, 1.0)
            feats[i * 3 + 2] = dist
        return feats

    # ── action 실행 ───────────────────────────────────────
    def _apply(self, a, action: int):
        """action < top_k → 해당 index 요청 배정, action == top_k → 충전."""
        self._last_dispatch_hot = False
        self._chosen_dist = 0.0

        if action == self._top_k:
            self.sim._send_to_charge(a)
            return

        reqs = self._sorted_requests()[:self._top_k]
        # 선택한 index의 요청 배정; 범위 초과 시 top-0 fallback
        req = reqs[action] if action < len(reqs) else (reqs[0] if reqs else None)
        if req is None:
            return  # 큐 비어있으면 implicit idle

        self._last_dispatch_hot = (req.priority == Priority.HOT)
        px = self.sim.port_locations[req.from_port]["grid_x"]
        py = self.sim.port_locations[req.from_port]["grid_y"]
        self._chosen_dist = (abs(a.x - px) + abs(a.y - py)) / (
            self.sim._grid_w + self.sim._grid_h
        )
        self.sim.dispatch(a.id, request_id=req.id)

    # ── action masking ────────────────────────────────────
    def action_masks(self) -> np.ndarray:
        a = self._decision_amr
        charger_free = any(
            len(occ) < self.sim.charger_slots
            for occ in self.sim.charger_occ.values()
        )

        # 강제 충전: soc < SOC_LOW & 충전소 여유 → charge만 허용
        if a is not None and a.soc < SOC_LOW and charger_free:
            return np.array([False] * self._top_k + [True])

        charge_ok = (a is not None and a.soc < 0.99 and charger_free)
        reqs = self._sorted_requests()[:self._top_k]
        n = len(reqs)

        if n == 0:
            req_mask = [True] * self._top_k
        else:
            # HOT 요청이 있으면 NORMAL 슬롯 마스킹 → hot_sla_rate 보장
            hot_set = {i for i, r in enumerate(reqs) if r.priority == Priority.HOT}
            if hot_set:
                req_mask = [(i in hot_set) for i in range(self._top_k)]
            else:
                req_mask = [(i < n) for i in range(self._top_k)]

        return np.array(req_mask + [charge_ok])

    # ── 보상 ──────────────────────────────────────────────
    def _reward(self) -> float:
        r = super(DispatchChargingEnv, self)._reward()   # ChargingFabEnv 보상
        n_pend, _ = self._last_ctx

        # serve action (0..top_k-1) & 배정한 요청이 있을 때
        if self._last_action < self._top_k and n_pend > 0:
            r += self.SHAPE_DISPATCH_IMM            # +0.05 즉시 선불
            if self._last_dispatch_hot:
                r += self.SHAPE_HOT                 # +1.5 HOT 우선 보너스
            # 가까운 요청 선택 보너스: 거리가 0에 가까울수록 최대 +0.02
            if self._chosen_dist > 0:
                r += self.SHAPE_NEAR * (1.0 - self._chosen_dist)

        return r
