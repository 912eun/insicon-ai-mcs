"""
train_dispatch_charging.py — 배정(dispatch) + 충전(charging) 통합 RL 학습.

Phase 1 (train_charging.py) : dispatch=휴리스틱, charging=RL
Phase 2 (이 스크립트)       : dispatch=RL + charging=RL (통합)

설계 포인트:
    - Action: Discrete(7) = 대기 / 충전 / 요청x5
    - Obs: 33-dim (ChargingFabEnv 13 + 요청 5개 x 4 feature)
    - 시간 기반 학습: --days (TimeLimitCallback, 스텝 상한 없음)
      ※ --days = wall-clock 학습 시간(일), 시뮬레이션 기간 아님.
         max_steps=5000 sim-step = 약 1.85h 시뮬 시간/에피소드.
         Phase 1 policy collapse 대응: ent_coef 기본 0.05 (0.01에서 상향).
    - Train/Test 시드 분리: TRAIN_SEEDS(0~9) / TEST_SEEDS(100~103)
    - CPU 병렬: SubprocVecEnv(n_envs = cpu_count-1, auto)
    - 채점: eval_baseline.transport_score (단일 공유 Judge)
    - Attribution: info 에 wait/charge/dispatch_rate 기록
      → Phase 1 vs Phase 2 기여 분리 가능

실행:
    conda activate ai-mcs
    python experiments/train_dispatch_charging.py --smoke
    python experiments/train_dispatch_charging.py --n-amrs 20 --days 7
    python experiments/train_dispatch_charging.py \
        --n-amrs 20 --days 7 --battery-scale 8

산출물:
    experiments/dispatch_charging/dc_n{N}_bs{S}_seed{K}_model.zip
    experiments/dispatch_charging/dc_n{N}_bs{S}_seed{K}_vecnorm.pkl
    experiments/dispatch_charging/checkpoints/
"""

import sys
import time
import argparse
import functools
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    BaseCallback, CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv, SubprocVecEnv, VecNormalize,
)

from simulator.amr.dispatch_charging_env import (  # noqa: E402
    DispatchChargingEnv,
)
from experiments.eval_baseline import transport_score  # noqa: E402

# ── 경로 ──────────────────────────────────────────────────
HERE = Path(__file__).parent
OUT_DIR = HERE / "dispatch_charging"
CKPT_DIR = OUT_DIR / "checkpoints"

# ── 시드 분리 ──────────────────────────────────────────────
TRAIN_SEEDS = list(range(10))
TEST_SEEDS = [100, 101, 102, 103]

# ── 기본 하이퍼파라미터 ───────────────────────────────────
MAX_STEPS = 5_000
LR = 3e-4
DAYS_DEFAULT = 7


# ── 시간 기반 종료 콜백 ───────────────────────────────────
class TimeLimitCallback(BaseCallback):
    """학습 경과 시간이 max_seconds를 초과하면 종료."""

    def __init__(self, max_seconds: float, verbose: int = 1):
        super().__init__(verbose)
        self.max_seconds = max_seconds
        self._start: float = 0.0

    def _on_training_start(self):
        self._start = time.time()
        if self.verbose:
            h = self.max_seconds / 3600
            print(
                f"[TimeLimitCallback] 최대 {h:.1f}h "
                f"({self.max_seconds / 86400:.1f}일) 학습"
            )

    def _on_step(self) -> bool:
        if time.time() - self._start >= self.max_seconds:
            if self.verbose:
                elapsed = (time.time() - self._start) / 3600
                print(f"[TimeLimitCallback] {elapsed:.2f}h 경과 → 종료")
            return False
        return True


# ── 환경 팩토리 (SubprocVecEnv 피클 가능하도록 모듈 레벨) ──
def _env_factory(n_amrs, max_steps, seed, battery_scale, coupled,
                 sim_days, random_start, routing, w_cong_route):
    return Monitor(DispatchChargingEnv(
        n_amrs=n_amrs, max_steps=max_steps, seed=seed,
        battery_scale=battery_scale, coupled=coupled,
        sim_days=sim_days, random_start=random_start,
        routing=routing, w_cong_route=w_cong_route,
    ))


def make_env(n_amrs, max_steps, seed, battery_scale=1.0, coupled=False,
             sim_days=7.0, random_start=False,
             routing="diversify", w_cong_route=3.0):
    return functools.partial(
        _env_factory, n_amrs, max_steps, seed, battery_scale, coupled,
        sim_days, random_start, routing, w_cong_route,
    )


def out_paths(n_amrs, seed, battery_scale, coupled=False, routing="diversify"):
    cp = "_coupled" if coupled else ""
    rt = "" if routing == "diversify" else f"_{routing}"
    tag = f"dc_n{n_amrs}_bs{battery_scale:g}{cp}{rt}_seed{seed}"
    return (
        str(OUT_DIR / f"{tag}_model"),
        str(OUT_DIR / f"{tag}_vecnorm.pkl"),
        tag,
    )


# ── 평가 (TEST_SEEDS, transport_score 공유 Judge) ─────────
def _avg(infos, k):
    return float(np.mean([i.get(k) or 0 for i in infos]))


def evaluate(model, vecnorm_path, n_amrs, max_steps, battery_scale,
             n_ep=4, coupled=False, sim_days=7.0,
             routing="diversify", w_cong_route=3.0):
    """TEST_SEEDS로 평가 → transport_score 출력. random_start=False 고정."""
    seeds = TEST_SEEDS[:n_ep]
    infos = []

    for s in seeds:
        def _factory(seed=s):
            return DispatchChargingEnv(
                n_amrs=n_amrs, max_steps=max_steps, seed=seed,
                battery_scale=battery_scale, coupled=coupled,
                sim_days=sim_days, random_start=False,
                routing=routing, w_cong_route=w_cong_route,
            )
        vec = DummyVecEnv([_factory])
        vec = VecNormalize.load(vecnorm_path, vec)
        vec.training = False
        vec.norm_reward = False

        obs = vec.reset()
        info = {}
        while True:
            masks = vec.env_method("action_masks")[0]
            action, _ = model.predict(
                obs, action_masks=masks, deterministic=True
            )
            obs, _, done, infos_ = vec.step(action)
            info = infos_[0]
            if done[0]:
                break
        infos.append(info)

    scores, _ = zip(*[transport_score(i) for i in infos])
    combined = float(np.mean(scores))

    print(f"\n=== Phase 2 평가 | n_amrs={n_amrs} | test_seeds={seeds} ===")
    print(f"transport_score : {combined:>8.2f}  (0에 가까울수록 우수)")
    print(f"throughput      : {_avg(infos, 'throughput'):>8.1f}")
    print(f"depleted_per_1k : {_avg(infos, 'depleted_per_1k'):>8.3f}")
    print(f"hot_sla_rate    : {_avg(infos, 'hot_sla_rate'):>8.3f}")
    print(f"energy/delivery : {_avg(infos, 'energy_per_delivery'):>8.3f}")
    print(f"dist_per_done   : {_avg(infos, 'dist_per_done'):>8.1f}")
    print(f"deadlock_per_1k : {_avg(infos, 'deadlock_per_1k'):>8.3f}")
    print(f"utilization     : {_avg(infos, 'utilization'):>8.1f}%")
    print("--- Attribution (결정 유형 비율) ---")
    print(f"wait_rate       : {_avg(infos, 'wait_rate'):>8.3f}")
    print(f"charge_rate     : {_avg(infos, 'charge_rate'):>8.3f}")
    print(f"dispatch_rate   : {_avg(infos, 'dispatch_rate'):>8.3f}")
    return combined


# ── 메인 ──────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-amrs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--battery-scale", type=float, default=8.0,
        help="배터리 레짐 (>=8 = 충전이 무는 빡빡한 레짐)",
    )
    ap.add_argument(
        "--days", type=float, default=DAYS_DEFAULT,
        help="wall-clock 학습 시간(일). 시뮬 기간 아님. --smoke 시 무시됨",
    )
    ap.add_argument(
        "--max-steps", type=int, default=MAX_STEPS,
        help=(
            "에피소드 길이(sim-step). 기본=5000(1.85h). "
            "battery_scale=8 첫 방전 임계=약 8800 step. "
            "SoC 패널티로 5000도 동작하나, 근본 검증은 --max-steps 50000."
        ),
    )
    ap.add_argument(
        "--n-envs", type=int, default=0,
        help="병렬 환경 수 (0=auto: cpu_count-1)",
    )
    ap.add_argument(
        "--ent-coef", type=float, default=0.1,
        help="엔트로피 보너스. Discrete(7)+masking 환경에선 0.1 권장 (Phase1=0.05)",
    )
    ap.add_argument(
        "--coupled", action="store_true",
        help="SMT2020 실공정 수요로 학습",
    )
    ap.add_argument(
        "--sim-days", type=float, default=7.0,
        help="공장 시뮬 기간(일). --coupled 시에만 유효. "
             "에피소드 시작 시각이 [0, sim_days) 내 랜덤 → 7일치 패턴 고르게 학습",
    )
    ap.add_argument(
        "--routing", default="diversify",
        choices=["reservation", "congestion", "diversify"],
        help="라우팅 모드. diversify=경로분산(권장), congestion=밀도페널티, reservation=기본",
    )
    ap.add_argument(
        "--w-cong-route", type=float, default=3.0,
        help="congestion/diversify 모드 혼잡 가중치 (기본 3.0)",
    )
    ap.add_argument(
        "--smoke", action="store_true",
        help="2분 빠른 검증",
    )
    args = ap.parse_args()

    if args.smoke:
        max_steps = 300
        n_steps = 64
        max_sec = 120.0
        n_envs = 2
    else:
        max_steps = args.max_steps  # 기본 5000, --max-steps로 조정 가능
        n_steps = 2048  # 512→2048: 배터리 7000-step 사이클의 29%를 한 rollout에서 봄
        max_sec = args.days * 86400
        n_envs = args.n_envs or max(1, os.cpu_count() - 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out_model, out_vecnorm, tag = out_paths(
        args.n_amrs, args.seed, args.battery_scale, args.coupled, args.routing
    )

    mode = "smoke 2분" if args.smoke else f"{args.days}일 학습"
    req = "coupled(SMT2020)" if args.coupled else "합성요청"
    print("=" * 60)
    print("[Phase 2] DispatchChargingEnv — 배정 + 충전 통합 RL")
    print(
        f"  n_amrs={args.n_amrs}  "
        f"battery_scale={args.battery_scale:g}  n_envs={n_envs}"
    )
    print(f"  {mode}  |  {req}")
    print("  action: Discrete(7) = 대기/충전/요청x5  obs: 33-dim")
    print(f"  train_seeds={TRAIN_SEEDS}")
    print(f"  test_seeds={TEST_SEEDS}")
    print("=" * 60)

    sim_days = args.sim_days if args.coupled else 30.0
    fns = [
        make_env(
            args.n_amrs, max_steps,
            TRAIN_SEEDS[i % len(TRAIN_SEEDS)],
            args.battery_scale, args.coupled,
            sim_days=sim_days,
            random_start=args.coupled,  # coupled 시 7일치 랜덤 구간 학습
            routing=args.routing,
            w_cong_route=args.w_cong_route,
        )
        for i in range(n_envs)
    ]
    vec = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_reward=10.0)

    tb_log = str(OUT_DIR / "logs")
    model = MaskablePPO(
        "MlpPolicy", vec,
        learning_rate=LR,
        n_steps=n_steps,
        batch_size=256,
        gamma=0.99,
        ent_coef=args.ent_coef,
        seed=args.seed,
        verbose=1,
        tensorboard_log=tb_log,
        policy_kwargs=dict(net_arch=[64, 64]),
    )

    callbacks = [
        TimeLimitCallback(max_sec),
        CheckpointCallback(
            save_freq=max(200_000 // n_envs, n_steps),
            save_path=str(CKPT_DIR),
            name_prefix=f"{tag}_ckpt",
            verbose=1,
        ),
    ]

    # total_timesteps 사실상 무한 → TimeLimitCallback이 시간으로 종료
    model.learn(total_timesteps=int(1e12), callback=callbacks)

    model.save(out_model)
    vec.save(out_vecnorm)
    print("\n[저장 완료]")
    print(f"  모델   : {out_model}.zip")
    print(f"  VecNorm: {out_vecnorm}")

    n_ep = 2 if args.smoke else 4
    evaluate(
        model, out_vecnorm, args.n_amrs, max_steps,
        args.battery_scale, n_ep=n_ep, coupled=args.coupled,
        sim_days=sim_days, routing=args.routing, w_cong_route=args.w_cong_route,
    )


if __name__ == "__main__":
    main()
