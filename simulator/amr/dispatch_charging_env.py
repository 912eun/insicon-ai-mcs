"""
DispatchChargingEnv — 배정(dispatch) + 충전(charging) 통합 RL 환경 (Phase 2).

Phase 1 (ChargingFabEnv): dispatch=휴리스틱, charging=RL
Phase 2 (이 파일)        : dispatch=RL + charging=RL (통합 결정)

설계 근거:
    dispatch-only RL은 배터리 없는 환경에서 headroom=0으로 기각됐으나,
    배터리 환경에서 "어느 요청을, SoC를 감안해 배정할지"는 nearest 휴리스틱이
    해결하지 못하는 joint 결정 → 탐색 가치 있음 (Phase 2 가설).

Action (event-triggered, 결정 대상 idle AMR 1대씩):
    0          = 대기 (이번 이벤트 pass, 다음 이벤트에서 재검토)
    1          = 충전소로
    2 .. k+1   = 상위 k번째 요청 배정 (EDD 정렬: HOT↑ 대기시간↓)
    total      : Discrete(2 + MAX_DISPATCH) = Discrete(7)

Observation (결정 AMR 기준, 33-dim):
    [ChargingFabEnv 기본 13] + [상위 요청 MAX_DISPATCH개 × 4 feature]
    요청 feature: [AMR→픽업 거리, priority, 대기시간, 픽업→드롭 거리]  (각 정규화)

Action masking:
    대기 : 항상 허용 (단, soc < SOC_LOW & 충전소 여유 있으면 금지 → 강제 충전)
    충전 : soc < 0.99 & 충전소 슬롯 여유 있을 때
    배정k: request_queue에 k번째 요청이 존재할 때

Reward: ChargingFabEnv Light shaping 계승 (완료·방전·에너지)
    + HOT 요청 즉시 배정 시 SHAPE_HOT 보너스 (dispatch 품질 light signal)
    + 요청 있는데 대기 선택 시 SHAPE 패널티 (policy collapse 방지)

Attribution (Phase 1 vs Phase 2 기여 분리):
    에피소드 종료 info에 wait_rate / charge_rate / dispatch_rate 기록.
    Phase 1(충전만) vs Phase 2 score 차이 = dispatch RL 순기여.

호환: gym==0.19.0 / sb3-contrib MaskablePPO
"""

import numpy as np
from gym import spaces

from simulator.amr.simulator import Priority, SOC_LOW
from simulator.amr.charging_env import ChargingFabEnv

MAX_DISPATCH = 5
SHAPE_HOT = 0.3   # HOT 요청 즉시 배정 light shaping 보너스


class DispatchChargingEnv(ChargingFabEnv):
    """배정 + 충전 통합 RL 환경. ChargingFabEnv 확장."""

    OBS_DIM = 13 + MAX_DISPATCH * 4   # 33

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # action / obs space 재정의
        self.action_space = spaces.Discrete(2 + MAX_DISPATCH)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self._last_dispatch_hot = False
        self._cnt_wait = self._cnt_charge = self._cnt_dispatch = 0

    # ── gym API ───────────────────────────────────────────

    def reset(self):
        self._last_dispatch_hot = False
        self._cnt_wait = self._cnt_charge = self._cnt_dispatch = 0
        return super().reset()

    def step(self, action: int):
        # attribution 카운트 (결정 유형 기록)
        if action == 0:
            self._cnt_wait += 1
        elif action == 1:
            self._cnt_charge += 1
        else:
            self._cnt_dispatch += 1

        obs, reward, done, info = super().step(action)

        if done:
            total = max(
                self._cnt_wait + self._cnt_charge + self._cnt_dispatch, 1
            )
            info["wait_count"] = self._cnt_wait
            info["charge_count"] = self._cnt_charge
            info["dispatch_count"] = self._cnt_dispatch
            info["wait_rate"] = round(self._cnt_wait / total, 4)
            info["charge_rate"] = round(self._cnt_charge / total, 4)
            info["dispatch_rate"] = round(self._cnt_dispatch / total, 4)

        return obs, reward, done, info

    # ── action 실행 ───────────────────────────────────────

    def _apply(self, a, action: int):
        """0=대기, 1=충전, 2..=k번째 요청 배정."""
        self._last_dispatch_hot = False
        if action == 0:
            pass   # 대기: 이 AMR 건너뜀, _advance_to_decision이 다음 이벤트로 진행
        elif action == 1:
            self.sim._send_to_charge(a)
        else:
            req_idx = action - 2
            reqs = self._sorted_requests()
            if req_idx < len(reqs):
                req = reqs[req_idx]
                self._last_dispatch_hot = (req.priority == Priority.HOT)
                self.sim.dispatch(a.id, request_id=req.id)

    # ── action masking ────────────────────────────────────

    def _mask_for(self, a):
        """[대기, 충전, 요청0 .. 요청{MAX_DISPATCH-1}] 마스크."""
        mask = np.zeros(2 + MAX_DISPATCH, dtype=bool)
        if a is None:
            mask[0] = True
            return mask

        charger_free = any(
            len(occ) < self.sim.charger_slots
            for occ in self.sim.charger_occ.values()
        )

        # 극저 SoC + 충전소 여유 → 강제 충전 (대기·배정 금지)
        if a.soc < SOC_LOW and charger_free:
            mask[1] = True
            return mask

        mask[0] = True                             # 대기: 항상 허용
        mask[1] = (a.soc < 0.99) and charger_free  # 충전 조건

        reqs = self._sorted_requests()
        for k in range(min(len(reqs), MAX_DISPATCH)):
            mask[2 + k] = True

        return mask

    def action_masks(self):
        return self._mask_for(self._decision_amr)

    # ── 보상 ──────────────────────────────────────────────

    def _reward(self) -> float:
        r = super()._reward()
        # HOT 요청 즉시 배정 light shaping (dispatch 품질 신호, 작게)
        if self._last_dispatch_hot:
            r += SHAPE_HOT
        # _last_ctx: 결정 직전 상태 (_advance_to_decision 이후 큐 상태가 아닌 결정 시점 기준)
        n_pend, last_soc = self._last_ctx
        # 대기 패널티: 요청 있는데 대기 선택 → policy collapse(항상 대기) 방지
        if self._last_action == 0 and n_pend > 0:
            r -= self.SHAPE
        # per-AMR SoC 신호 (결정 AMR 기준, fleet-wide 패널티 ×):
        #   fleet-wide 패널티는 "dispatch → 배터리 소모 → 패널티"로 dispatch 자체를
        #   억제해 always-wait collapse를 유발했음 (5.0 계수로 확인).
        #   대신 결정 시점 해당 AMR SoC(_last_ctx[1])만 참조:
        #     soc < 0.30 & 배정 → 작은 패널티 (충전 고려 유도)
        #     soc < 0.30 & 충전 → 보너스 (수요 있어도 충전할 유인)
        #   → dispatch 지배하되 양방향 collapse 없음.
        if last_soc < 0.30:
            if self._last_action == 1:      # 충전 선택: 보너스
                r += 0.3
            elif self._last_action >= 2:    # dispatch 선택: 약한 패널티
                r -= 0.2
        return r

    # ── observation ───────────────────────────────────────

    def _encode(self, a) -> np.ndarray:
        base = super()._encode(a)   # (13,)
        W, H = self.sim._grid_w, self.sim._grid_h

        req_feats = []
        for req in self._sorted_requests()[:MAX_DISPATCH]:
            fp = self.sim.port_locations[req.from_port]
            tp = self.sim.port_locations[req.to_port]
            dist_amr = (
                abs(a.x - fp["grid_x"]) + abs(a.y - fp["grid_y"])
            ) / (W + H)
            route_len = (
                abs(fp["grid_x"] - tp["grid_x"])
                + abs(fp["grid_y"] - tp["grid_y"])
            ) / (W + H)
            req_feats += [
                dist_amr,
                0.0 if req.priority == Priority.HOT else 1.0,
                min(req.waiting_time(self.sim.current_step) / 50.0, 1.0),
                route_len,
            ]
        # 요청 수 부족 시 0으로 패딩
        req_feats += [0.0] * (MAX_DISPATCH * 4 - len(req_feats))

        return np.concatenate([base, np.array(req_feats, dtype=np.float32)])

    # ── 내부 유틸 ─────────────────────────────────────────

    def _sorted_requests(self) -> list:
        """요청 정렬: HOT 우선, 동일 priority 내 대기시간 긴 순 (EDD)."""
        step = self.sim.current_step
        return sorted(
            [item[2] for item in self.sim.request_queue],
            key=lambda r: (r.priority.value, -r.waiting_time(step)),
        )
