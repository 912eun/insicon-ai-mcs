"""
충전 전략 비교 — coupled(SMT2020 실공정 수요) 위에서.

핵심: RL·휴리스틱 모두 **같은 ChargingFabEnv(coupled=True)**에서 정책만 바꿔 돌린다 →
동일 실수요·동일 커플링, 차이는 오직 '언제 충전하나'. 합성 균일요청(eval_charging.py)과
달리 실공정은 버스트성·실 HOT 타이밍 = RL이 exploit할 구조가 있는지 보는 공정한 시험.

정책(env obs로 스크립트): obs[0]=soc, obs[5]≈min(n_pend/20,5), mask[1]=충전가능
    threshold   : soc<0.2 충전
    opportunity : threshold + (대기≈0 & soc<0.9 선제충전)
    highlow     : 큐 길면 soc<0.2만, 짧으면 soc<0.9까지 (적응)
    rl          : 학습 모델 (--model)

실행:
    python experiments/eval_charging_coupled.py --n-amrs 20 --battery-scale 3
    python experiments/eval_charging_coupled.py --n-amrs 20 --battery-scale 3 --model <zip>
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
from experiments.eval_baseline import transport_score
from simulator.amr.charging_env import ChargingFabEnv

SOC_LOW = 0.20


def p_threshold(obs, mask):
    return 1 if (mask[1] and obs[0] < SOC_LOW) else 0


def p_opportunity(obs, mask):
    soc, n_pend = obs[0], obs[5] * 20
    return 1 if (mask[1] and (soc < SOC_LOW or (n_pend < 1 and soc < 0.9))) else 0


def p_highlow(obs, mask):
    soc, n_pend = obs[0], obs[5] * 20
    thr = SOC_LOW if n_pend >= 5 else 0.9      # 큐 길면 보수적, 짧으면 선제충전
    return 1 if (mask[1] and soc < thr) else 0


HEURISTICS = {"threshold": p_threshold, "opportunity": p_opportunity, "highlow": p_highlow}


def make_env(n_amrs, max_steps, seed, battery_scale):
    # feasible 운영점: lanes 차선제 + 충전소 3x3 + nearest 디스패치 (docs/feasibility_findings.md)
    return ChargingFabEnv(n_amrs=n_amrs, max_steps=max_steps, seed=seed,
                          battery_scale=battery_scale, coupled=True,
                          routing="lanes", w_cong_route=1.0,
                          n_chargers=3, charger_slots=3,
                          dispatch_policy="nearest")


def run_scripted(policy, n_amrs, max_steps, seed, battery_scale):
    env = make_env(n_amrs, max_steps, seed, battery_scale)
    obs = env.reset()
    info = {}
    while True:
        mask = env.action_masks()
        a = policy(obs, mask)
        obs, _, done, info = env.step(a)
        if done:
            return info


def run_rl(model_path, vecnorm_path, n_amrs, max_steps, seed, battery_scale):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    vec = DummyVecEnv([lambda: make_env(n_amrs, max_steps, seed, battery_scale)])
    vec = VecNormalize.load(vecnorm_path, vec)
    vec.training = False
    vec.norm_reward = False
    model = MaskablePPO.load(model_path)
    obs = vec.reset()
    while True:
        mask = vec.env_method("action_masks")[0]
        a, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, _, done, infos = vec.step(a)
        if done[0]:
            return infos[0]


def summarize(name, infos):
    scores = [transport_score(i)[0] for i in infos]
    avg = lambda k: float(np.mean([i.get(k) or 0 for i in infos]))
    return dict(name=name, score=float(np.mean(scores)),
                hot_sla=avg("hot_sla_rate"), tput=avg("throughput"),
                dep1k=avg("depleted_per_1k"), energy=avg("energy_per_delivery"),
                dist=avg("dist_per_done"), util=avg("utilization"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-amrs", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--battery-scale", type=float, default=3.0)
    ap.add_argument("--model", default=None, help="RL 모델 .zip (있으면 RL도 비교)")
    args = ap.parse_args()

    print(f"=== 충전 비교 (coupled 실수요) | n={args.n_amrs} bs={args.battery_scale} "
          f"{args.episodes}ep×{args.max_steps} ===")
    rows = []
    for name, pol in HEURISTICS.items():
        infos = [run_scripted(pol, args.n_amrs, args.max_steps, args.seed + e, args.battery_scale)
                 for e in range(args.episodes)]
        rows.append(summarize(name, infos))

    if args.model and Path(args.model).exists():
        vp = args.model.replace("_model.zip", "_vecnorm.pkl")
        infos = [run_rl(args.model, vp, args.n_amrs, args.max_steps, args.seed + e,
                        args.battery_scale) for e in range(args.episodes)]
        rows.append(summarize("rl ★", infos))

    rows.sort(key=lambda r: r["score"], reverse=True)
    h = f"{'정책':<13}{'score':>10}{'hot_sla':>9}{'tput':>7}{'dep/1k':>8}{'energy':>8}{'util%':>7}"
    print(h); print("-" * len(h))
    for r in rows:
        print(f"{r['name']:<13}{r['score']:>10.2f}{r['hot_sla']:>9.3f}{r['tput']:>7.0f}"
              f"{r['dep1k']:>8.2f}{r['energy']:>8.3f}{r['util']:>7.1f}")
    spread = rows[0]["score"] - rows[-1]["score"]
    print(f"\n최우수: {rows[0]['name']} | 최고-최저 score 차 = {spread:.1f}  "
          f"(클수록 '충전 정책이 실수요에서 실제로 갈린다'는 신호)")


if __name__ == "__main__":
    main()
