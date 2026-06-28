"""
sim_source — 디지털 트윈/대시보드용 스냅샷 소스 (단일 데이터 계약).

핵심 아이디어: 시뮬레이터는 이미 매 스텝 모든 걸 내보낸다.
    FABSimulator.get_state() → AMR 위치/상태, pending 큐, zone 밀도  (트윈 렌더링용)
    env.step() info(=get_kpi())  → deadlock/HOT SLA/배터리 등 KPI    (LLM 보고·차트용)

여기서는 그 둘을 합친 "스냅샷" 스트림을 만든다. recorder(JSONL 저장)와
server(라이브 push)가 *같은* 스냅샷 스키마를 공유하므로 프론트엔드는 소스(replay/live)를
구분할 필요가 없다.

스냅샷 스키마 (한 스텝):
    {
      "step":   int,
      "amrs":   [{"id","x","y","status","is_idle"}],
      "pending":[{"id","from_port","to_port","priority","waiting_time"}],
      "zone_density": {"NW","NE","SW","SE"},
      "kpi":    { ... get_kpi() ... },
    }

정책은 평가 baseline과 동일한 nearest(기하 최근접) 휴리스틱을 기본으로 쓴다 —
트윈은 "어떤 정책이 돌 때 공장이 어떻게 보이나"를 보여주는 게 목적이므로 학습 모델로
교체 가능하도록 정책을 콜백으로 분리했다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = str(ROOT / "external" / "PySCFabSim" / "datasets" / "SMT2020_LVHM")
RAIL_GRID = ROOT / "rware_simulator" / "output" / "rail_grid.npy"
GRID_MAP = ROOT / "rware_simulator" / "output" / "grid_map.npy"
PORTS = ROOT / "rware_simulator" / "output" / "port_locations.json"


# ── 정책 ──────────────────────────────────────────────────────────
def nearest_action(sim) -> int:
    """최우선 요청을 기하 최근접 IDLE AMR에 배정 (eval_baseline의 nearest와 동일)."""
    if not sim.request_queue or not sim.get_idle_amrs():
        return 0
    req = sim.get_next_request()
    loc = sim.port_locations[req.from_port]
    gx, gy = loc["grid_x"], loc["grid_y"]
    best = min(sim.get_idle_amrs(), key=lambda a: abs(a.x - gx) + abs(a.y - gy))
    return best.id


SOC_LOW = 0.30  # 충전 임계 (eval_charging_coupled의 p_threshold와 동일 컨셉)


def charge_action(env) -> int:
    """배터리(충전) env용 정책: 충전 가능하고 SoC가 낮으면 충전(1), 아니면 0.

    ChargingFabEnv의 action은 dispatch가 아니라 '충전 결정'(Discrete(2)) — 배차는 env 내부
    휴리스틱이 처리한다. obs[0]=현재 결정 AMR의 SoC, mask[1]=충전 가능 여부.
    """
    obs = env._last_obs
    mask = env.action_masks()
    return 1 if (mask[1] and obs is not None and obs[0] < SOC_LOW) else 0


def make_env(
    n_amrs: int = 76,
    max_steps: int = 4000,
    seed: int = 0,
    oht: bool = False,
    battery: bool = False,
    early_replan: bool = True,
    battery_scale: float = 1.0,
):
    """반송 트윈 env 구성.

    battery=False → CoupledFabEnv(실공정 주입, dispatch nearest). `env._inner.sim` 사용.
    battery=True  → ChargingFabEnv(coupled=True) 를 *직접* 만든다. 이건 base env를 감싸는
        inner가 아니라 배터리 동역학을 내장한 독립 coupled env다(생성자/역할이 다름). `env.sim` 사용.
        battery_scale>1 → 방전이 그만큼 빨라져 짧은 구간에도 충전/방전 장면이 보인다(데모용 가속).
    """
    # 무거운 import는 함수 안에서 (서버가 sim 없이도 import 가능하도록)
    if battery:
        # OHT 레일은 충전 모델과 별개 트랙이라 배터리 트윈에선 floor 그리드 사용.
        from simulator.amr.charging_env import ChargingFabEnv

        # 확정 운영점: lanes 차선제 + nearest 디스패치 + 충전소 3x3
        # (train/eval과 동일 — 데모가 '지금 상황'을 그대로 보여주도록)
        env = ChargingFabEnv(
            n_amrs=n_amrs, max_steps=max_steps, seed=seed, coupled=True,
            battery_scale=battery_scale,
            routing="lanes", w_cong_route=1.0,
            n_chargers=3, charger_slots=3,
            dispatch_policy="nearest",
        )
        env.sim.early_replan = early_replan
        return env

    from simulator.fab_process.smat2022.coupled_env import CoupledFabEnv

    grid = np.load(RAIL_GRID) if oht else None
    # 확정 운영점: floor 트윈도 lanes 차선제로 (nearest 디스패치는 snapshots의 action 루프가 주입).
    # OHT(레일)는 단방향 레일 자체가 라우팅이라 lanes 미적용.
    inner_kwargs = None if oht else {"routing": "lanes", "w_cong_route": 1.0}
    env = CoupledFabEnv(
        grid_map=grid,
        dataset_dir=DATASET_DIR,
        n_amrs=n_amrs,
        max_steps=max_steps,
        sim_days=7.0,
        seed=seed,
        inner_env_cls=None,
        inner_env_kwargs=inner_kwargs,
    )
    env._inner.sim.early_replan = early_replan
    return env


def _sim_of(env):
    """env에서 FABSimulator를 꺼낸다. CoupledFabEnv는 ._inner.sim, ChargingFabEnv는 .sim."""
    inner = getattr(env, "_inner", None)
    return inner.sim if inner is not None else env.sim


MAX_PENDING_SHOWN = 30  # 큐는 수천 건까지 쌓이므로 트윈엔 상위 N건 + 총계만 보낸다 (페이로드 경감)


def _amr_detail(sim, amr) -> dict:
    """클릭-인스펙터용 AMR별 상세. RL이 공유하는 get_state()는 안 건드리고 여기서 보강한다.

    필드: 배터리(soc), 적재 여부/내용, 목적지(충전소·픽업·드롭), 경로 진행률.
    sim.amrs의 실제 객체에서 직접 읽는다(필드가 작아 40대라도 가벼움 → 전 AMR 클릭 가능).
    """
    req = getattr(amr, "request", None)
    # 적재 여부: 픽업 지점을 지났는가 (simulator AMR.step과 동일 판정)
    carrying = bool(req is not None and getattr(amr, "pickup_idx", -1) >= 0
                    and amr.path_idx >= amr.pickup_idx)
    # 목적지: 충전 이동 중>충전소 / 적재 중>드롭(to) / 미적재 배정>픽업(from) / 그 외 없음
    if getattr(amr, "going_to_charge", False):
        dest_kind, dest_port = "charger", None
    elif req is not None and carrying:
        dest_kind, dest_port = "dropoff", req.to_port
    elif req is not None:
        dest_kind, dest_port = "pickup", req.from_port
    else:
        dest_kind, dest_port = None, None
    path = getattr(amr, "path", []) or []
    progress = round(amr.path_idx / (len(path) - 1), 2) if len(path) > 1 else None
    d = {
        "carrying": carrying,
        "dest_kind": dest_kind,        # charger | pickup | dropoff | None
        "dest_port": dest_port,
        "from_port": req.from_port if req else None,
        "to_port": req.to_port if req else None,
        "priority": req.priority.name if req else None,
        "progress": progress,          # 0~1 경로 진행률
    }
    if getattr(amr, "battery_enabled", False):
        d["soc"] = round(float(amr.soc), 3)   # 0~1
    return d


def _snapshot(sim, kpi: dict) -> dict:
    state = sim.get_state()
    pending = state["pending_queue"]
    # HOT 먼저, 그다음 오래 기다린 순 — 보고/하이라이트에 의미 있는 상위 건만
    pending_top = sorted(
        pending,
        key=lambda r: (r["priority"] != "HOT", -r["waiting_time"]),
    )[:MAX_PENDING_SHOWN]
    # AMR 기본 상태(get_state) + 클릭 인스펙터용 상세를 id로 병합
    detail = {a.id: _amr_detail(sim, a) for a in sim.amrs}
    amrs = [{**a, **detail.get(a["id"], {})} for a in state["amrs"]]
    # 충전소 좌표(배터리 env일 때만 채워짐; 정적이라 매 프레임 동일) — 트윈에 표시용
    chargers = [{"x": cx, "y": cy} for cx, cy in getattr(sim, "chargers", [])]
    return {
        "step": state["step"],
        "amrs": amrs,
        "pending": pending_top,
        "n_pending": len(pending),
        "zone_density": state["zone_density"],
        "chargers": chargers,
        "kpi": kpi,
    }


def snapshots(
    n_amrs: int = 76,
    max_steps: int = 4000,
    seed: int = 0,
    oht: bool = False,
    battery: bool = False,
    battery_scale: float = 1.0,
    policy: Optional[Callable] = None,
    env=None,
) -> Iterator[dict]:
    """한 에피소드를 스텝하며 매 스텝 스냅샷을 yield 한다.

    정책: battery면 charge_action(env), 아니면 nearest_action(sim). policy로 직접 주입도 가능
    (학습 모델이면 obs 기반 predict 를 감싸 넘기면 된다).
    """
    if env is None:
        env = make_env(n_amrs, max_steps, seed, oht, battery, battery_scale=battery_scale)
    obs = env.reset()
    env._last_obs = obs  # charge_action이 obs[0]=SoC 참조
    sim = _sim_of(env)
    if policy is None:
        policy = charge_action if battery else nearest_action

    if battery:
        # 충전 env는 event-triggered라 env.step 한 번이 sim을 1~수십 스텝 한꺼번에 진행한다
        # (결정 필요한 idle AMR이 나올 때까지). 그 묶음을 한 프레임으로 yield하면 프레임당
        # 이동량이 들쭉날쭉해 재생 속도가 제멋대로가 된다. → env의 _on_sim_step 훅으로 sim-step
        # *한 칸마다* 스냅샷을 모아 yield → 일반 demo와 동일한 1스텝=1프레임 균일 재생.
        buf: list = []
        env._on_sim_step = lambda: buf.append(_snapshot(sim, sim.get_kpi()))
        done = False
        while not done:
            action = policy(env)
            obs, _, done, _info = env.step(action)
            env._last_obs = obs
            while buf:
                yield buf.pop(0)
        return

    done = False
    while not done:
        action = policy(sim)
        obs, _, done, info = env.step(action)
        env._last_obs = obs
        yield _snapshot(sim, info)


def export_meta(out_path: Path, oht: bool = False) -> dict:
    """프론트엔드 배경(공장 평면도) 메타를 한 번 내보낸다: grid + ports.

    grid: 86×159 int8 (0=통로, 1=벽/장비, 2=포트셀 등) → 캔버스 배경.
    ports: 작은 점으로 표시. 전부 보내도 1116개라 가볍다.
    """
    import json

    grid = np.load(RAIL_GRID if oht else GRID_MAP)
    with open(PORTS) as f:
        ports_raw = json.load(f)
    ports = [
        {"name": k, "x": v["grid_x"], "y": v["grid_y"], "tool": v.get("tool", "")}
        for k, v in ports_raw.items()
    ]
    meta = {
        "grid_h": int(grid.shape[0]),
        "grid_w": int(grid.shape[1]),
        "grid": grid.astype(int).tolist(),
        "ports": ports,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(meta, f)
    return {"grid_h": meta["grid_h"], "grid_w": meta["grid_w"], "n_ports": len(ports)}
