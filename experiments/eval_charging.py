"""
충전 전략 비교 (M3 판정) — RL 충전 vs 휴리스틱 충전을 단일 Judge로 채점.

문제마다 맞는 도구(roadmap): dispatch=휴리스틱 / routing=예약A* / deadlock=구조적 /
charging=RL ★. 이 스크립트는 **충전 결정만** 다른 4개 전략을 동일 조건에서 비교한다.

비교 대상 (모두 battery_enabled, 동일 n_amrs·seed·max_steps·battery_scale):
    threshold    : soc<SOC_LOW 이면 충전                         (휴리스틱 baseline)
    opportunity  : threshold + 한가할 때 기회충전                 (휴리스틱 baseline)
    highlow      : 큐 길이 적응 상한 선제충전                     (휴리스틱 baseline)
    rl           : ChargingFabEnv + 학습된 MaskablePPO            (★ 우리 정책)

공정성: 휴리스틱은 FABSimulator(auto_dispatch=True, charge_policy=X)로 자율 구동,
RL은 ChargingFabEnv(auto_dispatch=False, charge_policy="external")로 충전을 통제.
dispatch 규칙(idle이 최우선 요청 take)은 양쪽 동일 → 차이는 오직 '언제 충전하나'.
채점은 eval_baseline.transport_score(공유 Judge, 배달당 손실 step) 그대로 사용.

실행:
    conda activate ai-mcs
    python experiments/eval_charging.py --n-amrs 20               # 휴리스틱 3종 + (모델 있으면)RL
    python experiments/eval_charging.py --n-amrs 20 --model experiments/charging/charging_20_seed42_model.zip
    python experiments/eval_charging.py --n-amrs 50 --episodes 5
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import json

from simulator.amr.charging_env import ChargingFabEnv, DEFAULT_GRID, DEFAULT_PORTS
from simulator.amr.simulator import FABSimulator
from experiments.eval_baseline import transport_score   # 단일 공유 Judge (중복 금지)

HEURISTICS = ["threshold", "opportunity", "highlow"]


def _load_grid_ports():
    grid = np.load(DEFAULT_GRID)
    with open(DEFAULT_PORTS) as f:
        ports = json.load(f)
    return grid, ports


def run_heuristic(policy, n_amrs, max_steps, seed, battery_scale, request_interval=15):
    """FABSimulator를 charge_policy=휴리스틱으로 자율 구동 → KPI."""
    grid, ports = _load_grid_ports()
    sim = FABSimulator(
        grid_map=grid, port_locations=ports, n_amrs=n_amrs,
        request_interval=request_interval, seed=seed,
        auto_dispatch=True, battery_enabled=True,
        charge_policy=policy, battery_scale=battery_scale,
    )
    sim.reset()
    while sim.current_step < max_steps:
        sim.step()
    return sim.get_kpi()


def run_rl(model_path, vecnorm_path, n_amrs, max_steps, seed, battery_scale):
    """ChargingFabEnv + 학습 모델(obs 정규화 포함) → KPI."""
    def _init():
        return ChargingFabEnv(n_amrs=n_amrs, max_steps=max_steps, seed=seed,
                              battery_scale=battery_scale)
    vec = DummyVecEnv([_init])
    vec = VecNormalize.load(vecnorm_path, vec)
    vec.training = False
    vec.norm_reward = False
    model = MaskablePPO.load(model_path)

    obs = vec.reset()
    info = {}
    while True:
        masks = vec.env_method("action_masks")[0]
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, _, done, infos = vec.step(action)
        info = infos[0]
        if done[0]:
            break
    return info


def summarize(name, infos):
    """에피소드 KPI 리스트 → (평균 score, breakdown, 주요지표)."""
    scores, bds = zip(*[transport_score(i) for i in infos])
    bd = {k: float(np.mean([b[k] for b in bds])) for k in bds[0]}
    avg = lambda k: float(np.mean([i.get(k) or 0 for i in infos]))
    return {
        "name": name,
        "score": float(np.mean(scores)),
        "bd": bd,
        "throughput": avg("throughput"),
        "depleted_per_1k": avg("depleted_per_1k"),
        "hot_sla_rate": avg("hot_sla_rate"),
        "energy_per_delivery": avg("energy_per_delivery"),
        "dist_per_done": avg("dist_per_done"),
        "deadlock_per_1k": avg("deadlock_per_1k"),
        "utilization": avg("utilization"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-amrs", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=0, help="eval seed 베이스 (ep마다 +1)")
    ap.add_argument("--battery-scale", type=float, default=1.0)
    ap.add_argument("--model", default=None,
                    help="RL 모델 .zip (미지정 시 기본경로 자동탐색; 없으면 휴리스틱만)")
    ap.add_argument("--policies", default=",".join(HEURISTICS),
                    help="비교할 휴리스틱 (쉼표구분)")
    args = ap.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    # RL 모델 경로 해결 (기본: charging_n{n}_bs{bs}_seed42_model.zip — train_charging 규칙)
    model_path = args.model
    if model_path is None:
        cand = (Path(__file__).parent / "charging"
                / f"charging_n{args.n_amrs}_bs{args.battery_scale:g}_seed42_model.zip")
        model_path = str(cand) if cand.exists() else None
    vecnorm_path = model_path.replace("_model.zip", "_vecnorm.pkl") if model_path else None
    have_rl = bool(model_path and Path(model_path).exists()
                   and vecnorm_path and Path(vecnorm_path).exists())

    print(f"=== 충전 전략 비교 | n_amrs={args.n_amrs} | {args.episodes}ep × {args.max_steps:,}step "
          f"| battery_scale={args.battery_scale} ===")
    print(f"RL 모델: {model_path if have_rl else '없음 (휴리스틱만 비교)'}\n")

    rows = []
    for pol in policies:
        infos = [run_heuristic(pol, args.n_amrs, args.max_steps, args.seed + ep,
                               args.battery_scale)
                 for ep in range(args.episodes)]
        rows.append(summarize(pol, infos))

    if have_rl:
        infos = [run_rl(model_path, vecnorm_path, args.n_amrs, args.max_steps,
                        args.seed + ep, args.battery_scale)
                 for ep in range(args.episodes)]
        rows.append(summarize("rl ★", infos))

    rows.sort(key=lambda r: r["score"], reverse=True)   # score 높을수록(0에 가까울수록) 우수

    h = (f"{'전략':<14}{'score':>10}{'deplete/1k':>12}{'hot_sla':>9}"
         f"{'energy/d':>10}{'dist/done':>11}{'dl/1k':>8}{'util%':>8}{'tput':>7}")
    print(h)
    print("-" * len(h))
    for r in rows:
        print(f"{r['name']:<14}{r['score']:>10.2f}{r['depleted_per_1k']:>12.3f}"
              f"{r['hot_sla_rate']:>9.3f}{r['energy_per_delivery']:>10.3f}"
              f"{r['dist_per_done']:>11.1f}{r['deadlock_per_1k']:>8.3f}"
              f"{r['utilization']:>8.1f}{r['throughput']:>7.0f}")

    best = rows[0]
    print(f"\n최우수: {best['name']}  (score {best['score']:.2f}, 0에 가까울수록 우수)")
    if have_rl:
        rl = next(r for r in rows if r["name"].startswith("rl"))
        heur_best = max((r for r in rows if not r["name"].startswith("rl")),
                        key=lambda r: r["score"])
        gap = rl["score"] - heur_best["score"]
        verdict = "RL 지배 ✅" if gap > 0 else "휴리스틱 우위 ❌ (학습 더 필요)"
        print(f"판정: RL({rl['score']:.2f}) vs 최고 휴리스틱 {heur_best['name']}"
              f"({heur_best['score']:.2f}) → Δ{gap:+.2f}  {verdict}")


if __name__ == "__main__":
    main()
