"""
CoupledFabEnv 정책/휴리스틱 평가 — deadlock 수정 전·후 비교 및 baseline 측정.

같은 eval 설정(n_amrs, max_steps, seed)을 수정 전·후에 동일하게 써서 KPI를 비교한다.
deadlock·replan·abort는 에피소드 길이에 비례하므로 1000스텝당 정규화 값도 출력.

모드:
    휴리스틱(nearest)  : 학습 모델 없이 sim만 검증 → 시뮬레이터 수정 효과를 정책과 분리
    모델(MaskablePPO)  : 학습된 정책 평가

실행:
    conda activate ai-mcs
    # 시뮬 수정 A/B (휴리스틱, 76대)
    python experiments/eval_baseline.py --heuristic --n-amrs 76 --early-replan off   # 수정 전
    python experiments/eval_baseline.py --heuristic --n-amrs 76 --early-replan on    # 수정 후
    # 학습 모델 평가
    python experiments/eval_baseline.py --model path/to/model.zip --n-amrs 20
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.save_util import load_from_zip_file

from simulator.fab_process.smat2022.coupled_env import CoupledFabEnv

DATASET_DIR = str(Path(__file__).parents[1] / "external" / "PySCFabSim" / "datasets" / "SMT2020_LVHM")
RAIL_GRID   = Path(__file__).parents[1] / "rware_simulator" / "output" / "rail_grid.npy"


def make_coupled(n_amrs, max_steps, seed, flow_aware, early_replan, oht=False,
                 closed_loop=False):
    grid_map = np.load(RAIL_GRID) if oht else None   # OHT=레일 그리드, AMR=기본 floor
    inner_env_cls = None
    if flow_aware:
        # 평가 스크립트가 합성 루트(composition root): 전략 env를 여기서 주입한다.
        from mcs.dispatcher.strategies.fa import FlowAwareFabEnv
        inner_env_cls = FlowAwareFabEnv
    env = CoupledFabEnv(
        grid_map=grid_map,
        dataset_dir=DATASET_DIR, n_amrs=n_amrs,
        max_steps=max_steps, sim_days=7.0, seed=seed, inner_env_cls=inner_env_cls,
        closed_loop=closed_loop,
    )
    env._inner.sim.early_replan = early_replan
    return env


def load_policy_only(model_path, env):
    """정책 가중치만 로드 (optimizer 상태 무시 — train_phase4가 optimizer를 재구성하므로)."""
    _, params, _ = load_from_zip_file(model_path, device="cpu")
    model = MaskablePPO("MlpPolicy", env, device="cpu")
    model.policy.load_state_dict(params["policy"])
    return model


def run_heuristic_episode(env):
    """nearest: 최우선 요청을 가장 가까운 IDLE AMR에 배정."""
    env.reset()
    sim = env._inner.sim
    info = {}
    while True:
        if not sim.request_queue or not sim.get_idle_amrs():
            action = 0
        else:
            req = sim.get_next_request()
            loc = sim.port_locations[req.from_port]
            best = min(sim.get_idle_amrs(),
                       key=lambda a: abs(a.x - loc["grid_x"]) + abs(a.y - loc["grid_y"]))
            action = best.id
        _, _, done, info = env.step(action)
        if done:
            return info


def run_model_episode(vec, env, model):
    # DummyVecEnv는 done 시 자동 reset → completed_requests가 비워진다.
    # 따라서 split은 step 직전(리셋 전) sim 상태에서 스냅샷해 둔다.
    obs = vec.reset()
    last_split = split_kpi(env._inner.sim)
    while True:
        masks = vec.env_method("action_masks")[0]
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, _, done, info = vec.step(action)
        if done[0]:
            return info[0], last_split
        last_split = split_kpi(env._inner.sim)


def split_kpi(sim):
    """완료 요청을 HOT/NORMAL로 분리해 건수·평균 큐대기(생성→배정)·평균 이동(배정→완료) 계산.

    집계 throughput은 76대 용량 포화로 비기지만, FA 설계(HOT×20, hot_urgency)의
    값어치는 HOT 큐 대기시간에서 드러난다. completed_requests에서 직접 뽑는다.
    """
    out = {}
    for prio in ("HOT", "NORMAL"):
        reqs = [r for r in sim.completed_requests if r.priority.name == prio]
        q = [r.queue_time for r in reqs if r.queue_time is not None]
        t = [r.travel_time for r in reqs if r.travel_time is not None]
        out[prio] = {
            "n":      len(reqs),
            "queue":  float(np.mean(q)) if q else 0.0,
            "travel": float(np.mean(t)) if t else 0.0,
            "p95":    float(np.percentile(q, 95)) if q else 0.0,
        }
    pend = {"HOT": 0, "NORMAL": 0}
    for _, _, req in sim.request_queue:
        pend[req.priority.name] += 1
    out["pending"] = pend
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="평가할 MaskablePPO .zip (미지정 시 --heuristic 필요)")
    ap.add_argument("--heuristic", action="store_true", help="nearest 휴리스틱 (모델 불필요)")
    ap.add_argument("--n-amrs", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--early-replan", choices=["on", "off"], default="on",
                    help="조기 재경로(A안) 토글 — 수정 전 비교는 off")
    ap.add_argument("--oht", action="store_true",
                    help="OHT 레일 그리드에서 평가 (기본=AMR floor 그리드)")
    ap.add_argument("--closed-loop", action="store_true",
                    help="반송지연을 lot 사이클타임·납기에 반영 (tardiness 측정)")
    args = ap.parse_args()

    early = args.early_replan == "on"
    if not args.heuristic and not args.model:
        print("❌ --model 또는 --heuristic 중 하나 필요"); return
    if args.model and not Path(args.model).exists():
        print(f"❌ 모델 없음: {args.model}"); return

    tag = "휴리스틱(nearest)" if args.heuristic else Path(args.model).name
    grid_tag = "OHT(rail)" if args.oht else "AMR(floor)"
    print(f"=== 평가: {tag} | {grid_tag} ===")
    print(f"n_amrs={args.n_amrs} | 에피소드={args.episodes} | max_steps={args.max_steps:,} "
          f"(≈{args.max_steps*1.33/3600:.1f}h fab) | early_replan={'ON' if early else 'OFF'}\n")

    infos, splits, model = [], [], None
    for ep in range(args.episodes):
        env = make_coupled(args.n_amrs, args.max_steps, args.seed + ep,
                           flow_aware=not args.heuristic, early_replan=early, oht=args.oht,
                           closed_loop=args.closed_loop)
        if args.heuristic:
            info = run_heuristic_episode(env)
            split = split_kpi(env._inner.sim)   # 휴리스틱은 auto-reset 없음 → 직접 읽어도 안전
        else:
            vec = DummyVecEnv([lambda e=env: Monitor(e)])
            if model is None:
                model = load_policy_only(args.model, vec)
            info, split = run_model_episode(vec, env, model)
        infos.append(info)
        splits.append(split)
        print(f"  ep{ep}: throughput={info['throughput']:>5} "
              f"deadlocks={info['deadlocks']:>5} replans={info.get('replans',0):>5} "
              f"aborts={info.get('aborts',0):>5} queue={info['avg_queue_time']:.0f}")

    def avg(k): return np.mean([i.get(k, 0) for i in infos])
    per1k = lambda k: avg(k) / args.max_steps * 1000
    def savg(prio, k): return np.mean([s[prio][k] for s in splits])

    print("\n=== 평균 (집계) ===")
    print(f"throughput:        {avg('throughput'):.1f}")
    print(f"deadlocks:         {avg('deadlocks'):.1f}   ({per1k('deadlocks'):.2f} / 1000스텝)")
    print(f"replans:           {avg('replans'):.1f}   ({per1k('replans'):.2f} / 1000스텝)")
    print(f"aborts:            {avg('aborts'):.1f}   ({per1k('aborts'):.2f} / 1000스텝)")
    print(f"utilization:       {avg('utilization'):.1f}%")
    print(f"avg_queue_time:    {avg('avg_queue_time'):.1f}")
    print(f"avg_travel_time:   {avg('avg_travel_time'):.1f}")

    print("\n=== HOT / NORMAL 분리 (FA 우선순위 설계 검증) ===")
    print(f"{'priority':<10}{'완료':>8}{'큐대기(평균)':>14}{'큐대기 p95':>12}{'이동시간':>12}{'미완료대기':>12}")
    pend_avg = {p: np.mean([s['pending'][p] for s in splits]) for p in ('HOT', 'NORMAL')}
    for prio in ("HOT", "NORMAL"):
        print(f"{prio:<10}{savg(prio,'n'):>8.0f}{savg(prio,'queue'):>14.1f}"
              f"{savg(prio,'p95'):>12.1f}{savg(prio,'travel'):>12.1f}{pend_avg[prio]:>12.1f}")

    if args.closed_loop:
        fk = lambda k: np.mean([i.get(k, 0) for i in infos])
        print("\n=== fab 일정 준수 (closed-loop, WIP 임계비 CR) ===")
        print(f"WIP lot:          {fk('fab_wip_lots'):.0f}   (완료 {fk('fab_done_lots'):.0f})")
        print(f"평균 CR:          {fk('fab_cr_mean'):.3f}   (HOT {fk('fab_hot_cr_mean'):.3f})   ※높을수록 일정 여유")
        print(f"납기 뒤처짐(CR<1): {fk('fab_behind_rate'):.1f}%   (HOT {fk('fab_hot_behind_rate'):.1f}%)")
        print(f"평균 진행 스텝:    {fk('fab_steps_done_mean'):.1f}")
        if fk('fab_done_lots') > 0:
            print(f"[완료분] 정시율 {fk('fab_ontime_rate'):.1f}% | 평균지각 {fk('fab_tardiness_mean')/3600:.2f}h "
                  f"| 사이클 {fk('fab_cycle_mean')/3600:.2f}h")


if __name__ == "__main__":
    main()
