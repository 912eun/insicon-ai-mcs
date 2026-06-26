"""
Fleet-sizing 프로브 (coupled 실수요) — 이 fab 수요의 적정 AMR 대수 찾기.

핵심 질문: 큐가 안정화(feasible)되는 최소 fleet은? 폭증하면 과부하(demand>capacity).
대수별로 휴리스틱(threshold) 운영 시 큐 궤적·util·HOT SLA·deadlock 측정 → fleet-sizing 곡선.

실행:
    python experiments/fleet_probe_coupled.py --battery-scale 3 --max-steps 30000
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
from simulator.amr.charging_env import ChargingFabEnv

SOC_LOW = 0.20


def run(n_amrs, max_steps, seed, battery_scale):
    # feasible 운영점 고정: lanes 차선제 + 충전소 3x3 + nearest 디스패치 (train/eval과 동일)
    env = ChargingFabEnv(n_amrs=n_amrs, max_steps=max_steps, seed=seed,
                         battery_scale=battery_scale, coupled=True,
                         routing="lanes", w_cong_route=1.0,
                         n_chargers=3, charger_slots=3,
                         dispatch_policy="nearest")
    obs = env.reset()
    q_traj, marks = [], {int(max_steps * f) for f in (0.25, 0.5, 0.75)}
    q_at = {}
    done = False
    while not done:
        mask = env.action_masks()
        # threshold 휴리스틱: 충전 가능 & soc<0.2 면 충전, 아니면 서빙
        a = 1 if (mask[1] and obs[0] < SOC_LOW) else 0
        obs, _, done, info = env.step(a)
        step = env.sim.current_step
        if step in marks:
            q_at[step] = len(env.sim.request_queue)
    q_final = len(env.sim.request_queue)
    return info, q_at, q_final, max_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-scale", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fleets", type=int, nargs="+", default=[20, 30, 40, 50, 60])
    args = ap.parse_args()

    print(f"=== Fleet-sizing 프로브 (coupled 실수요) | bs={args.battery_scale} "
          f"max_steps={args.max_steps} (≈{args.max_steps/65000:.1f}일) ===")
    print("큐: 25%→50%→75%→끝. 계속 증가=과부하(demand>capacity), 평탄=feasible\n")
    hdr = (f"{'대수':>4}{'큐25%':>8}{'큐50%':>8}{'큐75%':>8}{'큐끝':>8}"
           f"{'추세':>6}{'util%':>7}{'hot_sla':>8}{'tput':>7}{'dl/1k':>7}{'q_time':>8}")
    print(hdr); print("-" * len(hdr))
    for n in args.fleets:
        info, q_at, q_final, ms = run(n, args.max_steps, args.seed, args.battery_scale)
        q25 = q_at.get(int(ms * 0.25), 0)
        q50 = q_at.get(int(ms * 0.5), 0)
        q75 = q_at.get(int(ms * 0.75), 0)
        trend = "↑과부하" if q_final > q75 * 1.15 else ("평탄" if q_final < q75 * 1.15 else "~")
        print(f"{n:>4}{q25:>8}{q50:>8}{q75:>8}{q_final:>8}{trend:>6}"
              f"{info.get('utilization', 0):>7.1f}{info.get('hot_sla_rate', 0):>8.3f}"
              f"{info.get('throughput', 0):>7.0f}{info.get('deadlock_per_1k', 0):>7.2f}"
              f"{info.get('avg_queue_time', 0):>8.1f}")
    print("\n→ 큐가 평탄해지는 최소 대수 = 이 fab 수요의 자연 운영점(적정 fleet).")


if __name__ == "__main__":
    main()
