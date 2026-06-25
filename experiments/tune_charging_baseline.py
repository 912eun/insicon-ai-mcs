"""
충전 휴리스틱 baseline 튜닝 — coupled(실수요)에서 최강 튜닝 휴리스틱 찾기.

RL verdict를 정직하게 하려면 "튜닝 안 된 기본 휴리스틱"이 아니라 "coupled에서
최적 튜닝된 최강 휴리스틱"과 비교해야 한다(strawman 방지). 충전 임계값을 sweep해
transport_score 최고값을 baseline으로 확정한다.

정책(모두 charger 여유 있을 때만 충전):
    thr{X}            : soc < X 충전 (단순 임계)
    highlow(L,H,C)    : 대기요청 ≥ C면 soc<L만, 아니면 soc<H까지 (적응)

실행:
    python experiments/tune_charging_baseline.py --n-amrs 20 --battery-scale 3
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
from experiments.eval_charging_coupled import run_scripted, summarize


def make_thr(thr):
    def pol(obs, mask):
        return 1 if (mask[1] and obs[0] < thr) else 0
    return pol


def make_highlow(low, high, cut):
    def pol(obs, mask):
        soc, n_pend = obs[0], obs[5] * 20
        t = low if n_pend >= cut else high
        return 1 if (mask[1] and soc < t) else 0
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-amrs", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--battery-scale", type=float, default=3.0)
    args = ap.parse_args()

    configs = []
    for x in (0.2, 0.3, 0.4, 0.5, 0.6):
        configs.append((f"thr{x:g}", make_thr(x)))
    for low, high, cut in [(0.2, 0.6, 3), (0.2, 0.8, 5), (0.3, 0.7, 5), (0.3, 0.9, 8)]:
        configs.append((f"hl({low:g},{high:g},{cut})", make_highlow(low, high, cut)))

    print(f"=== 휴리스틱 baseline 튜닝 (coupled) | n={args.n_amrs} bs={args.battery_scale} "
          f"{args.episodes}ep×{args.max_steps} ===")
    rows = []
    for name, pol in configs:
        infos = [run_scripted(pol, args.n_amrs, args.max_steps, args.seed + e, args.battery_scale)
                 for e in range(args.episodes)]
        rows.append(summarize(name, infos))

    rows.sort(key=lambda r: r["score"], reverse=True)
    h = f"{'정책':<16}{'score':>10}{'hot_sla':>9}{'tput':>7}{'dep/1k':>8}{'util%':>7}"
    print(h); print("-" * len(h))
    for r in rows:
        print(f"{r['name']:<16}{r['score']:>10.2f}{r['hot_sla']:>9.3f}{r['tput']:>7.0f}"
              f"{r['dep1k']:>8.2f}{r['util']:>7.1f}")
    print(f"\n★ 최강 튜닝 휴리스틱 = {rows[0]['name']} (score {rows[0]['score']:.2f}) "
          f"→ 이걸 RL verdict의 baseline으로 사용")


if __name__ == "__main__":
    main()
