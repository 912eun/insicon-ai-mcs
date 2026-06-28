"""
train_dispatch_rl.py — Phase 3 Dispatch RL 학습 (DispatchRLEnv).

Phase 2 (DispatchChargingEnv) 한계:
    action 0 = EDD 하드코딩 → RL이 배정 정책을 선택 못 함
    결과: 충전 타이밍만 RL, FIFO와 score 차이 2.76점

Phase 3 (DispatchRLEnv) 개선:
    action 0..4 = 상위 5개 요청 중 선택 배정 (거리+우선순위+SoC jointly 학습)
    action 5   = 충전소로
    obs: 13(base) + 5×3(요청별 priority/wait/dist) = 28-dim

실행:
    conda activate ai-mcs
    python experiments/train_dispatch_rl.py --smoke
    python experiments/train_dispatch_rl.py \\
        --n-amrs 35 --days 0.25 --battery-scale 8 --coupled --routing lanes \\
        --charger-slots 3 --seed 0

산출물:
    experiments/dispatch_rl/drl_n{N}_bs{S}_seed{K}_model.zip
    experiments/dispatch_rl/drl_n{N}_bs{S}_seed{K}_vecnorm.pkl
    experiments/dispatch_rl/checkpoints/
"""

import sys
import time
import argparse
import functools
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from simulator.amr.dispatch_rl_env import DispatchRLEnv
from experiments.eval_baseline import transport_score

HERE = Path(__file__).parent
OUT_DIR = HERE / "dispatch_rl"
CKPT_DIR = OUT_DIR / "checkpoints"

TRAIN_SEEDS = list(range(10))
TEST_SEEDS  = [100, 101, 102, 103]

MAX_STEPS    = 5_000
LR           = 3e-4
DAYS_DEFAULT = 7


class TimeLimitCallback(BaseCallback):
    def __init__(self, max_seconds: float, verbose: int = 1):
        super().__init__(verbose)
        self.max_seconds = max_seconds
        self._start: float = 0.0

    def _on_training_start(self):
        self._start = time.time()
        if self.verbose:
            h = self.max_seconds / 3600
            print(f"[TimeLimitCallback] 최대 {h:.1f}h 학습")

    def _on_step(self) -> bool:
        if time.time() - self._start >= self.max_seconds:
            if self.verbose:
                elapsed = (time.time() - self._start) / 3600
                print(f"[TimeLimitCallback] {elapsed:.2f}h 경과 → 종료")
            return False
        return True


class CheckpointEvalCallback(BaseCallback):
    """
    체크포인트마다 VecNorm 저장 + transport_score 즉시 평가.

    CheckpointCallback과 같은 save_freq로 동작:
        → 모델 저장 시점에 vecnorm_<step>.pkl 저장
        → n_ep=2 seed로 빠른 평가 (~10분)
        → FIFO baseline(-202.22)과 실시간 비교 출력

    training 흐름을 블로킹하므로 n_ep=2로 최소화.
    """

    FIFO_BASELINE = -202.22

    def __init__(self, save_freq: int, ckpt_dir: Path,
                 eval_kwargs: dict, n_ep: int = 2, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq  = save_freq
        self.ckpt_dir   = Path(ckpt_dir)
        self.eval_kwargs = eval_kwargs
        self.n_ep       = n_ep
        self._history: list = []   # [(step, score)]

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True

        step = self.num_timesteps

        # ── VecNorm 저장 ──────────────────────────────────
        vecnorm = self.model.get_vec_normalize_env()
        vn_path = str(self.ckpt_dir / f"vecnorm_{step}.pkl")
        vecnorm.save(vn_path)

        if self.verbose:
            print(f"\n[EvalCB] step={step:,} | VecNorm 저장 → {vn_path}")
            print(f"[EvalCB] transport_score 평가 중 (seed {TEST_SEEDS[:self.n_ep]})...")

        # ── 평가 (학습 일시 중단, ~10분) ─────────────────
        score = evaluate(
            self.model, vn_path,
            n_ep=self.n_ep,
            **self.eval_kwargs,
        )
        self._history.append((step, score))

        gap = score - self.FIFO_BASELINE
        sign = "▲ FIFO 이김!" if gap > 0 else f"▼ FIFO 대비 {gap:.2f}"
        print(f"[EvalCB] step={step:,} | score={score:.2f} | {sign}")

        return True

    def _on_training_end(self):
        if not self._history:
            return
        print("\n=== 체크포인트별 transport_score 추이 ===")
        print(f"{'step':>10}  {'score':>8}  {'vs FIFO':>10}")
        for s, sc in self._history:
            gap = sc - self.FIFO_BASELINE
            print(f"{s:>10,}  {sc:>8.2f}  {gap:>+10.2f}")


def _env_factory(n_amrs, max_steps, seed, battery_scale, coupled,
                 sim_days, random_start, routing, w_cong_route,
                 charger_slots, top_k, w_sla, w_dl):
    return Monitor(DispatchRLEnv(
        n_amrs=n_amrs, max_steps=max_steps, seed=seed,
        battery_scale=battery_scale, coupled=coupled,
        sim_days=sim_days, random_start=random_start,
        routing=routing, w_cong_route=w_cong_route,
        charger_slots=charger_slots, top_k=top_k,
        w_sla=w_sla, w_dl=w_dl,
    ))


def make_env(n_amrs, max_steps, seed, battery_scale=1.0, coupled=False,
             sim_days=7.0, random_start=False, routing="lanes",
             w_cong_route=1.0, charger_slots=3, top_k=5, w_sla=0.5, w_dl=1.0):
    return functools.partial(
        _env_factory, n_amrs, max_steps, seed, battery_scale, coupled,
        sim_days, random_start, routing, w_cong_route, charger_slots, top_k,
        w_sla, w_dl,
    )


def out_paths(n_amrs, seed, battery_scale, coupled=False, routing="lanes"):
    cp = "_coupled" if coupled else ""
    rt = "" if routing == "diversify" else f"_{routing}"
    tag = f"drl_n{n_amrs}_bs{battery_scale:g}{cp}{rt}_seed{seed}"
    return (
        str(OUT_DIR / f"{tag}_model"),
        str(OUT_DIR / f"{tag}_vecnorm.pkl"),
        tag,
    )


def _avg(infos, k):
    return float(np.mean([i.get(k) or 0 for i in infos]))


def evaluate(model, vecnorm_path, n_amrs, max_steps, battery_scale,
             n_ep=4, coupled=False, sim_days=7.0,
             routing="lanes", w_cong_route=1.0, charger_slots=3, top_k=5):
    seeds = TEST_SEEDS[:n_ep]
    infos = []

    for s in seeds:
        def _factory(seed=s):
            return DispatchRLEnv(
                n_amrs=n_amrs, max_steps=max_steps, seed=seed,
                battery_scale=battery_scale, coupled=coupled,
                sim_days=sim_days, random_start=False,
                routing=routing, w_cong_route=w_cong_route,
                charger_slots=charger_slots, top_k=top_k,
            )
        vec = DummyVecEnv([_factory])
        vec = VecNormalize.load(vecnorm_path, vec)
        vec.training = False
        vec.norm_reward = False

        obs = vec.reset()
        info = {}
        while True:
            masks = vec.env_method("action_masks")[0]
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, _, done, infos_ = vec.step(action)
            info = infos_[0]
            if done[0]:
                break
        infos.append(info)

    scores, _ = zip(*[transport_score(i) for i in infos])
    combined = float(np.mean(scores))

    print(f"\n=== Phase 3 평가 | n_amrs={n_amrs} | test_seeds={seeds} ===")
    print(f"transport_score : {combined:>8.2f}  (FIFO baseline: -202.22)")
    print(f"throughput      : {_avg(infos, 'throughput'):>8.1f}")
    print(f"depleted_per_1k : {_avg(infos, 'depleted_per_1k'):>8.3f}")
    print(f"hot_sla_rate    : {_avg(infos, 'hot_sla_rate'):>8.3f}")
    print(f"energy/delivery : {_avg(infos, 'energy_per_delivery'):>8.3f}")
    print(f"dist_per_done   : {_avg(infos, 'dist_per_done'):>8.1f}")
    print(f"deadlock_per_1k : {_avg(infos, 'deadlock_per_1k'):>8.3f}")
    print(f"utilization     : {_avg(infos, 'utilization'):>8.1f}%")
    print("--- Attribution ---")
    print(f"serve_rate      : {_avg(infos, 'serve_rate'):>8.3f}")
    print(f"charge_rate     : {_avg(infos, 'charge_rate'):>8.3f}")
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-amrs",       type=int,   default=20)
    ap.add_argument("--seed",         type=int,   default=0)
    ap.add_argument("--battery-scale",type=float, default=8.0)
    ap.add_argument("--days",         type=float, default=DAYS_DEFAULT)
    ap.add_argument("--max-steps",    type=int,   default=MAX_STEPS)
    ap.add_argument("--n-envs",       type=int,   default=0)
    ap.add_argument("--ent-coef",     type=float, default=0.05)
    ap.add_argument("--top-k",        type=int,   default=5,
                    help="상위 K개 요청 중 선택 (default 5)")
    ap.add_argument("--coupled",      action="store_true")
    ap.add_argument("--sim-days",     type=float, default=7.0)
    ap.add_argument("--routing",      default="lanes",
                    choices=["reservation", "congestion", "diversify", "lanes"])
    ap.add_argument("--w-cong-route", type=float, default=1.0)
    ap.add_argument("--charger-slots",type=int,   default=3)
    ap.add_argument("--w-sla",        type=float, default=0.5,
                    help="HOT SLA 위반 패널티 가중치 (default 0.5)")
    ap.add_argument("--w-dl",         type=float, default=1.0,
                    help="데드락 패널티 가중치 (default 1.0)")
    ap.add_argument("--smoke",        action="store_true", help="2분 빠른 검증")
    ap.add_argument("--resume-from",  type=int, default=None,
                    help="재개할 체크포인트 step 번호 (예: 602112)")
    args = ap.parse_args()

    if args.smoke:
        max_steps = 300
        n_steps   = 64
        max_sec   = 120.0
        n_envs    = 2
    else:
        max_steps = args.max_steps
        n_steps   = 2048
        max_sec   = args.days * 86400
        n_envs    = args.n_envs or max(1, os.cpu_count() - 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out_model, out_vecnorm, tag = out_paths(
        args.n_amrs, args.seed, args.battery_scale, args.coupled, args.routing
    )

    mode = "smoke 2분" if args.smoke else f"{args.days}일 학습"
    req  = "coupled(SMT2020)" if args.coupled else "합성요청"
    obs_dim = 13 + args.top_k * 3
    print("=" * 60)
    print("[Phase 3] DispatchRLEnv — 배정(top-K) + 충전 통합 RL")
    print(f"  n_amrs={args.n_amrs}  battery_scale={args.battery_scale:g}  n_envs={n_envs}")
    print(f"  {mode}  |  {req}")
    print(f"  action: Discrete({args.top_k + 1}) = top-{args.top_k}요청선택/충전")
    print(f"  obs: {obs_dim}-dim (13 base + {args.top_k}×3 request features)")
    print(f"  routing={args.routing}  charger_slots={args.charger_slots}")
    print("=" * 60)

    sim_days = args.sim_days if args.coupled else 30.0
    fns = [
        make_env(
            args.n_amrs, max_steps,
            TRAIN_SEEDS[i % len(TRAIN_SEEDS)],
            args.battery_scale, args.coupled,
            sim_days=sim_days,
            random_start=args.coupled,
            routing=args.routing,
            w_cong_route=args.w_cong_route,
            charger_slots=args.charger_slots,
            top_k=args.top_k,
            w_sla=args.w_sla,
            w_dl=args.w_dl,
        )
        for i in range(n_envs)
    ]
    vec = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)

    resume_step = args.resume_from
    if resume_step is not None:
        ckpt_model = str(CKPT_DIR / f"{tag}_ckpt_{resume_step}_steps.zip")
        ckpt_vn    = str(CKPT_DIR / f"vecnorm_{resume_step}.pkl")
        print(f"[Resume] 모델 로드: {ckpt_model}")
        print(f"[Resume] VecNorm 로드: {ckpt_vn}")
        vec = VecNormalize.load(ckpt_vn, vec)
        vec.training = True
        vec.norm_reward = True
        model = MaskablePPO.load(
            ckpt_model, env=vec,
            verbose=1,
            tensorboard_log=str(OUT_DIR / "logs"),
        )
        print(f"[Resume] step {resume_step:,}부터 재개")
    else:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_reward=10.0)
        model = MaskablePPO(
            "MlpPolicy", vec,
            learning_rate=LR,
            n_steps=n_steps,
            batch_size=256,
            gamma=0.99,
            ent_coef=args.ent_coef,
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(OUT_DIR / "logs"),
            policy_kwargs=dict(net_arch=[64, 64]),
        )

    # save_freq: n_steps 배수로 맞춰야 체크포인트가 정확히 찍힘
    # (n_calls는 VecEnv step 단위이므로 n_steps 배수인 값 중 ~2h 간격 선택)
    save_freq = n_steps * max(1, round(100_000 / (n_envs * n_steps)))
    eval_kwargs = dict(
        n_amrs=args.n_amrs, max_steps=max_steps,
        battery_scale=args.battery_scale, coupled=args.coupled,
        sim_days=sim_days, routing=args.routing,
        w_cong_route=args.w_cong_route,
        charger_slots=args.charger_slots, top_k=args.top_k,
    )
    callbacks = [
        TimeLimitCallback(max_sec),
        CheckpointCallback(
            save_freq=save_freq,
            save_path=str(CKPT_DIR),
            name_prefix=f"{tag}_ckpt",
            verbose=1,
        ),
        CheckpointEvalCallback(
            save_freq=save_freq,
            ckpt_dir=CKPT_DIR,
            eval_kwargs=eval_kwargs,
            n_ep=1 if args.smoke else 2,
            verbose=1,
        ),
    ]

    model.learn(
        total_timesteps=int(1e12),
        callback=callbacks,
        reset_num_timesteps=(resume_step is None),
    )
    model.save(out_model)
    vec.save(out_vecnorm)
    print(f"\n[저장 완료]\n  모델   : {out_model}.zip\n  VecNorm: {out_vecnorm}")

    n_ep = 2 if args.smoke else 4
    evaluate(
        model, out_vecnorm, args.n_amrs, max_steps,
        args.battery_scale, n_ep=n_ep, coupled=args.coupled,
        sim_days=sim_days, routing=args.routing,
        w_cong_route=args.w_cong_route,
        charger_slots=args.charger_slots, top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
