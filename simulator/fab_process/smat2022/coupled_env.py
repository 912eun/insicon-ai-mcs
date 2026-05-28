"""
CoupledFabEnv — PySCFabSim(공정) + FABSimulator(AMR) 동기화 gym 환경.

시간 동기화:
    AMR 1스텝 = cell_size / max_speed = 2000mm / 1500mm/s ≈ 1.33초
    매 AMR 스텝마다 PySCFabSim을 AMR_STEP_SEC만큼 진행시켜 공정 이벤트를 뽑는다.

    PySCFabSim ──── next_decision_point() × N ──→ lot 이동 이벤트
                                                       │
                                        AMRTransportHook.on_step_done()
                                                       │
    FABSimulator ←─────────── add_request() ──────────┘
         │
      step() → AMR 이동 + dispatch

Action / Observation:
    FabAMREnv(env.py)와 동일한 인터페이스를 사용.
    (env.py의 FabAMREnv를 그대로 쓰되, FABSimulator.request_interval=∞ 로
     랜덤 생성을 끄고 이 클래스에서 공정 이벤트만 주입)

사용법:
    env = CoupledFabEnv(dataset_dir='external/PySCFabSim/datasets/SMT2020_LVHM')
    obs = env.reset()
    obs, reward, done, info = env.step(action)
"""

import sys
import os
from pathlib import Path

import gym
import numpy as np
from gym import spaces

_pyscfabsim_path = str(Path(__file__).parent.parent.parent.parent / "external" / "PySCFabSim")
if _pyscfabsim_path not in sys.path:
    sys.path.insert(0, _pyscfabsim_path)

from simulation.read import read_all
from simulation.file_instance import FileInstance
from simulation.greedy import get_lots_to_dispatch_by_machine
from simulation.dispatching.dispatcher import dispatcher_map

from simulator.amr.simulator import FABSimulator, CELL_MM, V_MAX_MMS, ACC_MMS2, DEC_MMS2
from simulator.amr.env import FabAMREnv, MAX_PENDING
from simulator.fab_process.smat2022.transport_hook import AMRTransportHook

_FAB_LAYOUT = (
    Path(__file__).parent.parent.parent.parent
    / "rware_simulator" / "output" / "fab_layout.json"
)

def _load_vehicle_params() -> dict:
    """fab_layout.json의 vehicle_type에서 실측 파라미터를 읽는다."""
    if not _FAB_LAYOUT.exists():
        return {}
    import json
    with open(_FAB_LAYOUT) as f:
        layout = json.load(f)
    vt = layout.get("vehicle_type", {})
    return {
        "cell_mm":   layout.get("meta", {}).get("cell_size", CELL_MM),
        "v_max_mms": vt.get("max_speed",    V_MAX_MMS),
        "acc_mms2":  vt.get("acceleration", ACC_MMS2),
        "dec_mms2":  vt.get("deceleration", DEC_MMS2),
    }

_VEHICLE = _load_vehicle_params()
# 1 AMR step = cell_size / v_max (seconds)
AMR_STEP_SEC = _VEHICLE.get("cell_mm", CELL_MM) / _VEHICLE.get("v_max_mms", V_MAX_MMS)

DEFAULT_DATASET = str(
    Path(__file__).parent.parent.parent.parent
    / "external" / "PySCFabSim" / "datasets" / "SMT2020_LVHM"
)


class CoupledFabEnv(gym.Env):
    """
    PySCFabSim 공정 이벤트 기반 반송 요청 + FABSimulator AMR 배정을
    하나의 gym 환경으로 묶은 클래스.

    FabAMREnv와 동일한 action/observation 공간을 사용하므로
    학습 코드 변경 없이 교체 가능.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        grid_map: np.ndarray = None,
        port_locations: dict = None,
        dataset_dir: str = DEFAULT_DATASET,
        n_amrs: int = 20,
        max_steps: int = 1000,
        sim_days: float = 30.0,
        seed: int = 42,
    ):
        super().__init__()

        # FabAMREnv를 내부 래퍼로 사용 (observation/action 공간 재사용)
        self._inner = FabAMREnv(
            grid_map=grid_map,
            port_locations=port_locations,
            n_amrs=n_amrs,
            request_interval=10**9,   # 랜덤 생성 비활성화
            max_steps=max_steps,
            seed=seed,
            **_VEHICLE,               # acc/dec/v_max/cell_mm from fab_layout.json
        )
        self.observation_space = self._inner.observation_space
        self.action_space      = self._inner.action_space

        self.dataset_dir  = dataset_dir
        self.sim_days     = sim_days
        self._fab_files   = read_all(dataset_dir)
        self._ptuple_fcn  = dispatcher_map["fifo"]
        self._run_to_sec  = sim_days * 24 * 3600

        self._pyscfab     = None
        self._hook        = None
        self._fab_time    = 0.0       # PySCFabSim 현재 시각 (초)

    # ── gym API ───────────────────────────────────────────
    def reset(self):
        obs = self._inner.reset()
        # 요청 큐 비우기 (FabAMREnv.reset()이 랜덤 요청 넣으므로 제거)
        self._inner.sim.request_queue = []

        self._hook    = AMRTransportHook(self._inner.sim)
        self._pyscfab = FileInstance(
            self._fab_files,
            run_to=self._run_to_sec,
            lot_for_machine=True,
            plugins=[self._hook],
        )
        self._fab_time = 0.0
        return obs

    def step(self, action: int):
        # 1. PySCFabSim을 AMR_STEP_SEC만큼 진행 → 반송 요청 주입
        self._advance_fab(until=self._fab_time + AMR_STEP_SEC)
        self._fab_time += AMR_STEP_SEC

        # 2. RL 액션 적용 + FABSimulator 1스텝
        obs, reward, done, info = self._inner.step(action)

        info["fab_time_sec"]    = self._fab_time
        info["fab_time_hours"]  = self._fab_time / 3600
        return obs, reward, done, info

    def render(self, mode="human"):
        self._inner.render(mode)
        print(f"  PySCFabSim 시각: {self._fab_time/3600:.2f}h")

    def close(self):
        self._inner.close()

    # ── 내부 유틸 ─────────────────────────────────────────
    def _advance_fab(self, until: float):
        """PySCFabSim을 until(초)까지 진행하면서 결정 포인트마다 FIFO 배정."""
        if self._pyscfab is None:
            return
        while self._pyscfab.current_time < until:
            done = self._pyscfab.next_decision_point()
            if done or self._pyscfab.current_time > self._run_to_sec:
                break
            # 공정 내부 배정 (FIFO) — AMR 배정과 무관
            # PySCFabSim greedy runner 패턴 그대로:
            # usable_machines 소진될 때까지 배정 시도
            while self._pyscfab.usable_machines:
                m, lots = get_lots_to_dispatch_by_machine(
                    self._pyscfab, self._ptuple_fcn
                )
                if lots is None:
                    self._pyscfab.usable_machines.discard(m)
                else:
                    self._pyscfab.dispatch(m, lots)
