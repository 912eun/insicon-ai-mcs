"""
FabAMREnv — SMAT2022 FAB 레이아웃 기반 다중 AMR 반송 gym 환경.

호환: gym==0.19.0 (stable-baselines3==1.3.0)

Action:
    Discrete(n_amrs + 1)
    0 → 아무것도 하지 않음
    k → AMR k번에 우선순위 큐 맨 앞 요청 배정

Observation:
    flat float32 벡터
    [AMR × n_amrs: x, y, is_idle]          n_amrs * 3
    [pending × MAX_PENDING: fx, fy, tx, ty, priority, wait]  MAX_PENDING * 6
    [zone_density: NW, NE, SW, SE]          4
    [summary: n_idle, n_pending, throughput] 3
    총 = n_amrs*3 + MAX_PENDING*6 + 7
"""

import json
from pathlib import Path
from typing import Optional

import gym
import numpy as np
from gym import spaces

from simulator.amr.simulator import FABSimulator, CELL_MM, V_MAX_MMS, ACC_MMS2, DEC_MMS2

MAX_PENDING = 10
DEFAULT_GRID  = Path(__file__).parent.parent.parent / "rware_simulator" / "output" / "grid_map.npy"
DEFAULT_PORTS = Path(__file__).parent.parent.parent / "rware_simulator" / "output" / "port_locations.json"


class FabAMREnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        grid_map: Optional[np.ndarray] = None,
        port_locations: Optional[dict] = None,
        n_amrs: int = 20,
        request_interval: int = 15,
        max_steps: int = 1000,
        seed: int = 42,
        cell_mm: float = CELL_MM,
        v_max_mms: float = V_MAX_MMS,
        acc_mms2: float = ACC_MMS2,
        dec_mms2: float = DEC_MMS2,
    ):
        super().__init__()

        if grid_map is None:
            grid_map = np.load(DEFAULT_GRID)
        if port_locations is None:
            with open(DEFAULT_PORTS) as f:
                port_locations = json.load(f)

        self.sim = FABSimulator(
            grid_map=grid_map,
            port_locations=port_locations,
            n_amrs=n_amrs,
            request_interval=request_interval,
            seed=seed,
            auto_dispatch=False,   # RL 에이전트가 직접 dispatch
            cell_mm=cell_mm,
            v_max_mms=v_max_mms,
            acc_mms2=acc_mms2,
            dec_mms2=dec_mms2,
        )
        self.max_steps = max_steps
        self._n_amrs   = n_amrs

        n_obs = n_amrs * 3 + MAX_PENDING * 6 + 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        # 0 = no-op, 1..n_amrs = dispatch to that AMR
        self.action_space = spaces.Discrete(n_amrs + 1)

        self._prev_throughput = 0
        self._prev_waiting    = 0
        self._prev_deadlocks  = 0

    # ── gym API ───────────────────────────────────────────
    def reset(self):
        state = self.sim.reset()
        self._prev_throughput = 0
        self._prev_waiting    = 0
        self._prev_deadlocks  = 0
        return self._encode(state)

    def step(self, action: int):
        """
        action: 0 (no-op) 또는 1..n_amrs (AMR id)
        내부 _auto_dispatch()는 비활성화하고 RL이 직접 배정.
        """
        if action > 0:
            self.sim.dispatch(amr_id=int(action))

        state  = self.sim.step()
        reward = self._delta_reward()
        done   = self.sim.current_step >= self.max_steps
        info   = self.sim.get_kpi()
        return self._encode(state), reward, done, info

    def render(self, mode="human"):
        kpi = self.sim.get_kpi()
        print(
            f"Step {kpi['step']:4d} | throughput={kpi['throughput']} "
            f"avg_total={kpi['avg_total_time']} util={kpi['utilization']}% "
            f"pending={kpi['pending']}"
        )

    def close(self):
        pass

    # ── 내부 유틸 ─────────────────────────────────────────
    def _delta_reward(self) -> float:
        """이전 스텝 대비 처리량 증가분 - 큐 대기시간 증가분으로 dense reward 계산."""
        kpi = self.sim.get_kpi()
        delta_tp       = self.sim.total_completed - self._prev_throughput
        n              = max(self.sim.total_completed, 1)
        avg_queue      = self.sim.total_queue_time / n
        delta_wait     = avg_queue - self._prev_waiting
        delta_deadlock = kpi["deadlocks"] - self._prev_deadlocks
        self._prev_throughput = self.sim.total_completed
        self._prev_waiting    = avg_queue
        self._prev_deadlocks  = kpi["deadlocks"]
        return delta_tp * 10.0 - delta_wait * 0.5 - delta_deadlock * 100.0

    def _encode(self, state: dict) -> np.ndarray:
        grid_w = self.sim._grid_w
        grid_h = self.sim._grid_h

        # AMR features (normalised)
        amr_vec = []
        for a in state["amrs"]:
            amr_vec += [a["x"] / grid_w, a["y"] / grid_h, float(a["is_idle"])]

        # Pending queue (top MAX_PENDING)
        ports = self.sim.port_locations
        pq_vec = []
        for req in state["pending_queue"][:MAX_PENDING]:
            fp = ports[req["from_port"]]
            tp = ports[req["to_port"]]
            pq_vec += [
                fp["grid_x"] / grid_w,
                fp["grid_y"] / grid_h,
                tp["grid_x"] / grid_w,
                tp["grid_y"] / grid_h,
                1.0 if req["priority"] == "HOT" else 0.0,
                min(req["waiting_time"] / 100.0, 10.0),
            ]
        # 패딩
        pq_vec += [0.0] * (MAX_PENDING * 6 - len(pq_vec))

        # Zone density + summary
        zd = state["zone_density"]
        summary = [
            state["n_idle_amrs"] / self._n_amrs,
            min(state["n_pending"] / 20.0, 5.0),
            min(state["throughput"] / 100.0, 10.0),
            zd["NW"] / self._n_amrs,
            zd["NE"] / self._n_amrs,
            zd["SW"] / self._n_amrs,
            zd["SE"] / self._n_amrs,
        ]

        return np.array(amr_vec + pq_vec + summary, dtype=np.float32)
