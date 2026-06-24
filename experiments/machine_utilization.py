"""
장비(tool) 가동률 측정 — "fab이 장비 병목이냐"를 직접 검증.

CR 불변(반송 20~76대 어디든 fab 미동)의 내 설명은 "장비가 병목 + WIP버퍼"였다.
그게 맞으려면 병목 tool family가 ~포화(가동률 높음)여야 한다. 직접 측정해 확인한다.

가동률 = (처리 + 셋업) 시간 / 경과시간.  고장/PM은 down time으로 분리.

실행:
    python experiments/machine_utilization.py --n-amrs 76 --max-steps 10000
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from simulator.fab_process.smat2022.coupled_env import CoupledFabEnv

DATASET = str(ROOT / "external" / "PySCFabSim" / "datasets" / "SMT2020_LVHM")


def run_heuristic(env, max_steps):
    env.reset()
    sim = env._inner.sim
    for _ in range(max_steps):
        if not sim.request_queue or not sim.get_idle_amrs():
            a = 0
        else:
            req = sim.get_next_request()
            loc = sim.port_locations[req.from_port]
            best = min(sim.get_idle_amrs(),
                       key=lambda m: abs(m.x - loc["grid_x"]) + abs(m.y - loc["grid_y"]))
            a = best.id
        _, _, done, _ = env.step(a)
        if done:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-amrs", type=int, default=76)
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = CoupledFabEnv(dataset_dir=DATASET, n_amrs=args.n_amrs, max_steps=args.max_steps,
                        sim_days=7.0, seed=args.seed, closed_loop=True)
    run_heuristic(env, args.max_steps)

    inst = env._pyscfab
    elapsed = inst.current_time
    if elapsed <= 0:
        print("경과시간 0"); return

    # 머신별 가동률
    busy_frac, down_frac = [], []
    fam_busy = defaultdict(list)
    for m in inst.machines:
        busy = (m.utilized_time + m.setuped_time) / elapsed
        down = (getattr(m, "bred_time", 0) + m.pmed_time) / elapsed
        busy_frac.append(busy)
        down_frac.append(down)
        fam_busy[m.family].append(busy)

    busy_frac = np.array(busy_frac) * 100
    down_frac = np.array(down_frac) * 100

    print(f"=== 장비 가동률 (n_amrs={args.n_amrs}, {elapsed/3600:.1f}h fab, 장비 {len(inst.machines)}대) ===")
    print(f"가동률(처리+셋업): 평균 {busy_frac.mean():.1f}% | 중앙 {np.median(busy_frac):.1f}% "
          f"| p90 {np.percentile(busy_frac,90):.1f}% | max {busy_frac.max():.1f}%")
    print(f"다운(고장+PM):     평균 {down_frac.mean():.1f}%")
    print(f"가동률 분포: ≥90% {np.mean(busy_frac>=90)*100:.0f}% | ≥80% {np.mean(busy_frac>=80)*100:.0f}% "
          f"| ≥50% {np.mean(busy_frac>=50)*100:.0f}% | <20% {np.mean(busy_frac<20)*100:.0f}% 의 장비")

    # 병목 family Top 10 (family 평균 가동률)
    fam_avg = sorted(((np.mean(v), f, len(v)) for f, v in fam_busy.items()), reverse=True)
    print("\n=== 병목 tool family Top 12 (family 평균 가동률) ===")
    for util, fam, n in fam_avg[:12]:
        print(f"  {fam:<22} {util*100:5.1f}%  (장비 {n}대)")
    print(f"\n전체 {len(fam_busy)} family 중 평균가동률 ≥80%: "
          f"{sum(1 for u,_,_ in fam_avg if u>=0.8)}개 | ≥90%: {sum(1 for u,_,_ in fam_avg if u>=0.9)}개")


if __name__ == "__main__":
    main()
