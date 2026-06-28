"""
FIFO baseline for DispatchChargingEnv — AMR 수 / routing 조합별 throughput·transport_score 비교.

FIFO 정책:
  - 강제 충전(mask[0]=False): charge (action 1)
  - 요청 있으면: 항상 top-EDD 배정 (action 2)
  - 요청 없으면: wait (action 0)  — 선제 충전 없음, 순수 reactive

비교 조합:
  n_amrs=30  routing=diversify   ← 권장 최소 (기준)
  n_amrs=35  routing=diversify
  n_amrs=50  routing=reservation ← 이전 baseline
  n_amrs=50  routing=diversify   ← routing만 개선
"""

import sys
import os
os.chdir(r"C:\Users\안가은\insicon-ai-mcs")
sys.path.insert(0, r"C:\Users\안가은\insicon-ai-mcs")

import numpy as np
from simulator.amr.dispatch_charging_env import DispatchChargingEnv
from experiments.eval_baseline import transport_score

OUT = r"C:\Users\안가은\insicon-ai-mcs\experiments\dispatch_charging\fifo_dc_result.txt"

# stdout도 파일에 기록
_orig = sys.stdout
f = open(OUT, "w", encoding="utf-8")

class _Tee:
    def __init__(self, *s): self._s = s
    def write(self, x):
        for s in self._s: s.write(x)
    def flush(self):
        for s in self._s:
            try: s.flush()
            except: pass

sys.stdout = _Tee(_orig, f)


def fifo_action(mask):
    """순수 FIFO: 강제충전 > 배정 > 대기. 선제 충전 없음."""
    if not mask[0]:          # wait 금지 = 강제 충전 상황
        return 1
    if any(mask[2:]):        # 배정 가능한 요청 있음
        return 2             # 항상 top-EDD 요청 배정
    return 0                 # 요청 없음 → 대기


def run_episode(n_amrs, routing, w_cong, seed, max_steps=5000,
                battery_scale=8, sim_days=7.0):
    env = DispatchChargingEnv(
        n_amrs=n_amrs, max_steps=max_steps, seed=seed,
        battery_scale=battery_scale, coupled=True, sim_days=sim_days,
        random_start=False, routing=routing, w_cong_route=w_cong,
    )
    obs = env.reset()
    done = False
    info = {}
    cnt = [0, 0, 0]  # wait, charge, dispatch
    while not done:
        mask = env.action_masks()
        a = fifo_action(mask)
        cnt[a if a < 2 else 2] += 1
        obs, _, done, info = env.step(a)
    env.close()
    return info, cnt


CONFIGS = [
    (30, "diversify",   3.0),
    (35, "diversify",   3.0),
    (50, "reservation", 0.0),   # 이전 baseline
    (50, "diversify",   3.0),
]
SEEDS = [100, 101]

print("=" * 70)
print("FIFO baseline — n_amrs × routing 비교 (coupled SMAT2022, bs=8)")
print("=" * 70)
print(f"{'n':>4} {'routing':>11}  {'score':>9}  {'tp':>5}  "
      f"{'dep/1k':>7}  {'dl/1k':>6}  {'hot_sla':>8}  {'util%':>6}  "
      f"{'charge%':>8}  {'wait%':>7}")
print("-" * 70)

for (n, rt, w) in CONFIGS:
    ep_infos = []
    ep_cnts  = []
    for s in SEEDS:
        print(f"  running: n={n}, routing={rt}, seed={s} ...", end="", flush=True)
        info, cnt = run_episode(n, rt, w, s)
        ep_infos.append(info)
        ep_cnts.append(cnt)
        print(f" tp={info.get('throughput', 0)}", flush=True)

    scores = [transport_score(i)[0] for i in ep_infos]
    avg_score = float(np.mean(scores))

    def avg(k): return float(np.mean([i.get(k) or 0 for i in ep_infos]))

    total_actions = sum(sum(c) for c in ep_cnts)
    charge_rate = sum(c[1] for c in ep_cnts) / max(total_actions, 1)
    wait_rate   = sum(c[0] for c in ep_cnts) / max(total_actions, 1)

    print(f"{n:>4} {rt:>11}  {avg_score:>9.2f}  {avg('throughput'):>5.1f}  "
          f"{avg('depleted_per_1k'):>7.3f}  {avg('deadlock_per_1k'):>6.3f}  "
          f"{avg('hot_sla_rate'):>8.3f}  {avg('utilization'):>6.1f}  "
          f"{charge_rate*100:>7.1f}%  {wait_rate*100:>6.1f}%")

print("=" * 70)
print("\n컬럼 설명:")
print("  score      : transport_score (0에 가까울수록 좋음, 음수)")
print("  tp         : throughput (에피소드 완료 건수)")
print("  dep/1k     : 방전/1000step (낮을수록 좋음)")
print("  dl/1k      : 데드락/1000step (낮을수록 좋음)")
print("  hot_sla    : HOT 요청 SLA 충족률 (높을수록 좋음, 목표 ≥ 0.90)")
print("  util%      : AMR 평균 가동률")
print("  charge%    : 충전 결정 비율 (FIFO는 강제 충전만 → 낮게 나옴)")
print("  wait%      : 대기 결정 비율 (요청 없을 때만)")

sys.stdout = _orig
f.close()
print(f"\n결과 저장: {OUT}")
