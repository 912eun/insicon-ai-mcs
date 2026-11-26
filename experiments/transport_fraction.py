"""
반송이 lot 사이클타임의 몇 %인지 측정 — "반송 최적화의 천장" 진단.

논리:
    lot 사이클타임 = 처리시간(장비) + 큐대기 + 반송시간.
    공정 라우트가 242~521스텝으로 길어 처리시간이 지배적이면, 반송은 작은 조각이라
    아무리 최적화해도 사이클타임/납기를 거의 못 움직인다(closed-loop CR 불변 현상의 원인).

측정:
    각 라우트의 처리시간 합(full_time)과 반송 move 수(연속 스텝의 family 전환 횟수)를 구해
    반송시간 = moves × per_move_delay(우리 sim 실측 ~25분/move)와 비교.
    lot 모집단(WIP+orders)으로 가중 평균.

실행:
    python experiments/transport_fraction.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "PySCFabSim"))

import numpy as np
from simulation.read import read_all
from simulation.file_instance import FileInstance

DATASET = str(ROOT / "external" / "PySCFabSim" / "datasets" / "SMT2020_LVHM")

# 우리 sim 실측 per-move 반송지연 (closed-loop AMR-heur 76대): queue~1021 + travel~113 스텝
# 1 step ≈ 1.33s → (1021+113)*1.33 ≈ 1508s ≈ 25.1분/move
PER_MOVE_MIN_AMR = 25.1
# 원본 PySCFabSim 가정 (fromto.txt): uniform 7.5분/move
PER_MOVE_MIN_STATIC = 7.5


def n_moves(steps):
    """라우트에서 연속 스텝의 family가 바뀌는 횟수 = 반송 요청 수."""
    fams = [s.family for s in steps]
    return sum(1 for i in range(1, len(fams)) if fams[i] != fams[i - 1])


def main():
    files = read_all(DATASET)
    inst = FileInstance(files, run_to=10, lot_for_machine=True, plugins=[])
    lots = inst.active_lots + inst.dispatchable_lots

    proc_h, mv = [], []
    for lot in lots:
        steps = lot.processed_steps + ([lot.actual_step] if lot.actual_step else []) + lot.remaining_steps
        if not steps:
            continue
        proc_h.append(lot.full_time / 3600.0)   # 처리시간 합 (시간)
        mv.append(n_moves(steps))

    proc_h = np.array(proc_h); mv = np.array(mv)
    for label, per_move in [("AMR 실측 ~25분/move", PER_MOVE_MIN_AMR),
                            ("원본 정적 7.5분/move", PER_MOVE_MIN_STATIC)]:
        tr_h = mv * per_move / 60.0   # 반송시간 합 (시간)
        # 사이클타임 ≈ 처리 + 반송 + 큐대기. 큐대기는 flow factor로 흡수(CR≈2.78 관측).
        cycle_proc_only = proc_h + tr_h                  # 큐 제외(하한)
        cycle_flow      = proc_h * 2.78 + tr_h           # flow factor 2.78 반영(현실적)
        print(f"\n=== {label} ===")
        print(f"lot 수: {len(proc_h)} | 평균 처리시간 {proc_h.mean():.0f}h | 평균 move {mv.mean():.0f}회 "
              f"| 평균 반송시간 {tr_h.mean():.0f}h")
        print(f"반송 / (처리+반송)          = {(tr_h.sum()/cycle_proc_only.sum())*100:.1f}%   (큐 제외, 상한)")
        print(f"반송 / (처리×2.78 + 반송)   = {(tr_h.sum()/cycle_flow.sum())*100:.1f}%   (flow factor 반영, 현실)")


if __name__ == "__main__":
    main()
