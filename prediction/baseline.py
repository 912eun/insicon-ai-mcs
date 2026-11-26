"""
FAB AMR Transport Baseline Evaluation
======================================
3가지 휴리스틱 전략을 5개 seed 평균으로 평가한다.

전략:
  FIFO-Nearest: 가장 오래된 요청(req.id, priority 무시) -> 맨해튼 최근접 idle AMR
                1스텝 1배정 (매 스텝 최대 1쌍)
  EDD-Nearest:  HOT 우선, 같은 priority면 대기시간 긴 요청 -> 가장 가까운 idle AMR
                매 스텝 idle AMR 전부 배정 시도
  Random:       랜덤 idle AMR -> 우선순위 큐 맨 앞 요청
                매 스텝 idle AMR 전부 배정 시도

한계:
  맨해튼 거리 abs(dx)+abs(dy)는 장애물/벽을 무시한 근사값.
  실제 A* 경로 길이와 다를 수 있음.

실행:
  cd insicon-ai-mcs
  python -m prediction.baseline
"""

import json
import random as _random
from pathlib import Path

import numpy as np

from simulator.amr.simulator import FABSimulator, Priority
# 팀 단일 Judge: eval_baseline.transport_score 를 공유 (가중치 중복 정의 금지)
from experiments.eval_baseline import (
    transport_score, W_DL, W_BLOCK, W_TRAV, W_DEPLETE, W_ENERGY, W_SLA, HOT_SLA_MIN,
)

# ── 경로 ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
GRID_PATH = _ROOT / "rware_simulator" / "output" / "grid_map.npy"
PORTS_PATH = _ROOT / "rware_simulator" / "output" / "port_locations.json"
OUT_DIR = Path(__file__).parent / "data"
OUT_PATH = OUT_DIR / "baseline_results.json"

# ── 실험 설정 ──────────────────────────────────────────────────────
N_AMRS = 20
MAX_STEPS = 500
REQUEST_INTERVAL = 15
SEEDS = [42, 99, 123, 256, 512]
HOT_SLA_THRESH = 30   # simulator.py의 HOT_SLA_STEPS와 동일


# ── KPI 수집 ──────────────────────────────────────────────────────

def _collect_kpi(sim: FABSimulator) -> dict:
    """
    sim.get_kpi() 기반에 추가 지표를 덧붙인다.

    추가/수정:
      replans_aborts   : replan_count + abort_count (신규 합산 키)
      hot_sla_rate     : completed_requests 직접 계산 (검증용)
      hot/normal_queue_time, travel_time : completed_requests 직접 계산
      availability / energy_per_delivery / depleted_per_1k
        : battery=False 이므로 None (get_kpi()의 1.0/0 대신)
    """
    kpi = sim.get_kpi()

    # 강건성: replan + abort 합산
    kpi["replans_aborts"] = sim.replan_count + sim.abort_count

    # per-priority 완료 목록
    hot_reqs = [
        r for r in sim.completed_requests if r.priority == Priority.HOT
    ]
    normal_reqs = [
        r for r in sim.completed_requests if r.priority == Priority.NORMAL
    ]

    # HOT SLA: queue_time <= HOT_SLA_THRESH 비율
    if hot_reqs:
        met = sum(
            1 for r in hot_reqs
            if r.queue_time is not None and r.queue_time <= HOT_SLA_THRESH
        )
        kpi["hot_sla_rate"] = round(met / len(hot_reqs), 3)
    else:
        kpi["hot_sla_rate"] = None

    # per-priority 평균 (completed_requests 직접 계산)
    def _avg(vals):
        vs = [v for v in vals if v is not None]
        return round(sum(vs) / len(vs), 2) if vs else None

    kpi["hot_queue_time"] = _avg(r.queue_time for r in hot_reqs)
    kpi["normal_queue_time"] = _avg(r.queue_time for r in normal_reqs)
    kpi["hot_travel_time"] = _avg(r.travel_time for r in hot_reqs)
    kpi["normal_travel_time"] = _avg(r.travel_time for r in normal_reqs)

    # 배터리 미구현 -> None
    kpi["availability"] = None
    kpi["energy_per_delivery"] = None
    kpi["depleted_per_1k"] = None

    return kpi


# ── 맨해튼 거리 ───────────────────────────────────────────────────

def _manhattan(amr, port_info: dict) -> int:
    # 근사: 장애물/벽 미반영 (실제 A* 경로 길이와 다를 수 있음)
    return (
        abs(amr.x - port_info["grid_x"])
        + abs(amr.y - port_info["grid_y"])
    )


# ── 에피소드 실행 ─────────────────────────────────────────────────

def _run_episode_fifo(
    grid_map: np.ndarray, port_locations: dict, seed: int
) -> dict:
    """
    FIFO-Nearest: 1스텝 1배정.
    - req.id 최소(가장 오래된) 요청 선택, priority 무시 = 순수 FIFO
    - from_port에 맨해튼 최근접 idle AMR 1대만 배정
    - 배정 직후 sim.step() -> 1스텝 1배정 보장
    """
    sim = FABSimulator(
        grid_map=grid_map,
        port_locations=port_locations,
        n_amrs=N_AMRS,
        request_interval=REQUEST_INTERVAL,
        seed=seed,
        auto_dispatch=False,
    )

    for _ in range(MAX_STEPS):
        idle = sim.get_idle_amrs()
        if idle and sim.request_queue:
            # FIFO: priority 무관, 가장 먼저 생성된 요청
            req = min(
                (item[2] for item in sim.request_queue),
                key=lambda r: r.id,
            )
            port_inf = sim.port_locations[req.from_port]
            nearest = min(idle, key=lambda a: _manhattan(a, port_inf))
            sim.dispatch(nearest.id, request_id=req.id)  # 1대만
        sim.step()

    return _collect_kpi(sim)


def _dispatch_edd_nearest(sim: FABSimulator) -> bool:
    """
    EDD-Nearest: HOT 우선 -> 같은 priority면 대기시간 긴 것 -> 최근접 AMR.
    """
    idle = sim.get_idle_amrs()
    if not idle or not sim.request_queue:
        return False
    req = min(
        (item[2] for item in sim.request_queue),
        key=lambda r: (r.priority.value, -r.waiting_time(sim.current_step)),
    )
    port_inf = sim.port_locations[req.from_port]
    nearest = min(idle, key=lambda a: _manhattan(a, port_inf))
    return sim.dispatch(nearest.id, request_id=req.id)


def _dispatch_random(sim: FABSimulator) -> bool:
    """Random: idle AMR 랜덤 선택 -> 우선순위 큐 맨 앞 요청."""
    idle = sim.get_idle_amrs()
    if not idle or not sim.request_queue:
        return False
    return sim.dispatch(_random.choice(idle).id)


def _run_episode_generic(
    dispatch_fn, grid_map: np.ndarray, port_locations: dict, seed: int
) -> dict:
    """EDD-Nearest / Random: 매 스텝 idle AMR 전부 배정."""
    sim = FABSimulator(
        grid_map=grid_map,
        port_locations=port_locations,
        n_amrs=N_AMRS,
        request_interval=REQUEST_INTERVAL,
        seed=seed,
        auto_dispatch=False,
    )
    for _ in range(MAX_STEPS):
        while dispatch_fn(sim):
            pass
        sim.step()
    return _collect_kpi(sim)


# ── 평균 KPI ──────────────────────────────────────────────────────

def _avg_kpis(kpi_list: list) -> dict:
    avg: dict = {}
    for key in kpi_list[0].keys():
        vals = [k[key] for k in kpi_list if k.get(key) is not None]
        if vals and isinstance(vals[0], (int, float)):
            avg[key] = round(sum(vals) / len(vals), 4)
        else:
            avg[key] = None
    return avg


# ── 전략 실행 ─────────────────────────────────────────────────────

def _run_seeds(fn, *args) -> dict:
    return _avg_kpis([fn(*args, s) for s in SEEDS])


def run_all(grid_map: np.ndarray, port_locations: dict) -> dict:
    results: dict = {}

    print(
        f"  [FIFO-Nearest] 평가 중 (1스텝 1배정, {len(SEEDS)} seeds)...",
        end="", flush=True,
    )
    results["FIFO-Nearest"] = _run_seeds(
        _run_episode_fifo, grid_map, port_locations
    )
    _print_progress(results["FIFO-Nearest"])

    print(
        f"  [EDD-Nearest]  평가 중 ({len(SEEDS)} seeds)...",
        end="", flush=True,
    )
    results["EDD-Nearest"] = _run_seeds(
        _run_episode_generic, _dispatch_edd_nearest, grid_map, port_locations
    )
    _print_progress(results["EDD-Nearest"])

    print(
        f"  [Random]       평가 중 ({len(SEEDS)} seeds)...",
        end="", flush=True,
    )
    results["Random"] = _run_seeds(
        _run_episode_generic, _dispatch_random, grid_map, port_locations
    )
    _print_progress(results["Random"])

    return results


def _print_progress(kpi: dict) -> None:
    tp = kpi.get("throughput") or 0
    dl = kpi.get("deadlock_per_1k") or 0
    sla = kpi.get("hot_sla_rate")
    sla_s = f"{sla:.3f}" if sla is not None else "—"
    print(f" throughput={tp:.1f}  deadlock/1k={dl:.3f}  hot_sla={sla_s}")


# ── transport_score ───────────────────────────────────────────────

def compute_transport_scores(
    results: dict, baseline: str = "FIFO-Nearest"
) -> dict:
    """공유 Judge(eval_baseline.transport_score) 기반 — FIFO 대비 비율 (FIFO=1.0, 높을수록 좋음).

    transport_score는 '배달당 비용'(낮을수록 좋음, HOT SLA 게이트 포함) → 비율 = fifo_cost/method_cost.
    팀 단일 judge 사용(가중치 중복 제거). 배터리 off면 energy/deplete 항 자동 0.
    """
    costs = {name: -transport_score(kpi)[0] for name, kpi in results.items()}
    fifo_cost = costs.get(baseline) or 1.0
    return {
        name: round(fifo_cost / c, 4) if c > 0 else 9.99
        for name, c in costs.items()
    }


# ── 출력 ─────────────────────────────────────────────────────────

_DISPLAY = [
    # (key, unit, 설명)
    ("throughput", "건", "완료 반송"),
    ("avg_queue_time", "스텝", "평균 큐 대기"),
    ("avg_travel_time", "스텝", "평균 이동"),
    ("avg_total_time", "스텝", "평균 총 시간"),
    ("utilization", "%", "AMR 가동률"),
    ("pending", "건", "미처리 잔량"),
    ("deadlocks", "회", "교착 누적"),
    ("deadlock_per_1k", "", "교착/1k스텝"),
    ("blocking_per_1k", "", "충돌롤백/1k"),
    ("replans_aborts", "회", "재경로+포기"),
    ("hot_sla_rate", "", "HOT SLA (<=30스텝)"),
    ("hot_done", "건", "HOT 완료"),
    ("normal_done", "건", "NORMAL 완료"),
    ("hot_queue_time", "스텝", "HOT 큐 대기"),
    ("normal_queue_time", "스텝", "NORMAL 큐 대기"),
    ("hot_travel_time", "스텝", "HOT 이동"),
    ("normal_travel_time", "스텝", "NORMAL 이동"),
    ("dist_per_done", "셀", "완료당 이동거리"),
    ("total_travel_dist", "셀", "누적 이동거리"),
    ("availability", "", "가용률(배터리)"),
    ("energy_per_delivery", "", "에너지/건(배터리)"),
    ("depleted_per_1k", "", "방전/1k(배터리)"),
    ("reward", "", "내장 보상"),
]


def print_per_strategy(results: dict, scores: dict) -> None:
    """전략별 상세 출력."""
    for name, kpi in results.items():
        sc = scores.get(name)
        sc_s = f"{sc:.4f}  (FIFO 기준=1.0)" if sc is not None else "—"
        print(f"\n{'='*50}")
        print(f"  {name}  (5 seed 평균)")
        print(f"{'='*50}")
        for key, unit, desc in _DISPLAY:
            v = kpi.get(key)
            if v is None:
                val_s = "—"
            elif isinstance(v, float):
                val_s = f"{v:.3f}{unit}"
            else:
                val_s = f"{v}{unit}"
            print(f"  {key:<22}: {val_s:<14}  ({desc})")
        print(f"  {'transport_score':<22}: {sc_s}")


def print_table(results: dict, scores: dict) -> None:
    """전략 간 비교 테이블."""
    strategies = list(results.keys())
    lw, cw = 24, 16
    sep = "=" * (lw + cw * len(strategies))

    print(f"\n{sep}")
    print("  BASELINE KPI COMPARISON  (5-seed average)")
    print(sep)
    print(f"{'Metric':<{lw}}" + "".join(f"{s:>{cw}}" for s in strategies))
    print("-" * (lw + cw * len(strategies)))

    for key, unit, _ in _DISPLAY:
        row = f"{key:<{lw}}"
        for s in strategies:
            v = results[s].get(key)
            if v is None:
                row += f"{'—':>{cw}}"
            elif isinstance(v, float):
                row += f"{v:>{cw - len(unit)}.3f}{unit}"
            else:
                row += f"{str(v) + unit:>{cw}}"
        print(row)

    print("-" * (lw + cw * len(strategies)))
    row = f"{'transport_score':<{lw}}"
    for s in strategies:
        sc = scores.get(s)
        row += f"{sc:>{cw}.4f}" if sc is not None else f"{'—':>{cw}}"
    print(row)
    print(sep)
    print(
        f"\n  * transport_score: FIFO-Nearest=1.0 기준 (높을수록 좋음, 공유 Judge=배달당 비용 비율)"
        f"\n    weights(step/event): deadlock(x{W_DL:.0f}) blocking(x{W_BLOCK:.0f})"
        f" travel(x{W_TRAV:.0f}) deplete(x{W_DEPLETE:.0f}) energy(x{W_ENERGY:.0f})"
        f"\n    HOT SLA 게이트: <{HOT_SLA_MIN} 미달분당 x{W_SLA:.0f}"
    )


# ── 저장 ─────────────────────────────────────────────────────────

def save_results(results: dict, scores: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "n_amrs": N_AMRS,
            "max_steps": MAX_STEPS,
            "request_interval": REQUEST_INTERVAL,
            "seeds": SEEDS,
            "baseline_ref": "FIFO-Nearest",
            "transport_score_weights": {
                "deadlock": W_DL, "blocking": W_BLOCK, "travel": W_TRAV,
                "deplete": W_DEPLETE, "energy": W_ENERGY, "hot_sla": W_SLA,
                "hot_sla_min": HOT_SLA_MIN,
            },
        },
        "results": results,
        "transport_score": scores,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장 완료: {OUT_PATH}")


# ── 진입점 ───────────────────────────────────────────────────────

def main() -> None:
    for p in (GRID_PATH, PORTS_PATH):
        if not p.exists():
            raise FileNotFoundError(f"파일 없음: {p}")

    print("데이터 로딩...")
    grid_map = np.load(GRID_PATH)
    with open(PORTS_PATH, encoding="utf-8") as f:
        port_locations = json.load(f)

    print(f"  그리드: {grid_map.shape}  포트: {len(port_locations)}개")
    print(
        f"\n실험 설정: n_amrs={N_AMRS}  max_steps={MAX_STEPS}"
        f"  request_interval={REQUEST_INTERVAL}  seeds={SEEDS}\n"
    )

    results = run_all(grid_map, port_locations)
    scores = compute_transport_scores(results)

    print_per_strategy(results, scores)
    print_table(results, scores)
    save_results(results, scores)


if __name__ == "__main__":
    main()
