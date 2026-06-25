"""
충전 RL 학습 스크립트 (M3) — ChargingFabEnv를 MaskablePPO로 scratch 학습.

배경 (roadmap_transport_layer.md M3):
    측정으로 디스패칭-RL 기각 후 RL의 자리를 '충전' 하나로 확정.
    dispatch = 고정 휴리스틱 / routing = 예약 A* / deadlock = 구조적 /
    charging  = RL ★ (이 스크립트가 학습하는 유일 레이어)

방식 (FA의 fine-tune과 달리 scratch):
    충전 결정은 새 레이어라 사전학습 정책이 없다 → MaskablePPO를 처음부터 학습한다.
    · obs 13-dim (soc·충전소거리/여유·대기요청·fleet SoC분포·진행도)
    · action Discrete(2): 0=서빙, 1=충전 (불가능 행동은 env.action_masks()로 차단)
    · Light reward (Bischoff 교훈: Basic 붕괴·Light 최고) — env._reward()

대수별 + multi-seed:
    AMR 대수가 바뀌면 fleet 혼잡 레짐이 바뀌므로 대수별로 정책을 따로 학습한다.
    (--n-amrs). 결판 sweep(M4)에서 {16·20·30·50·76}×seed로 비교.

산출물: experiments/charging/charging_{n}_seed{s}_model.zip + _vecnorm.pkl

실행:
    conda activate ai-mcs
    python experiments/train_charging.py --smoke                  # 2k 스텝 빠른 검증
    python experiments/train_charging.py --n-amrs 20 --total 1000000
    python experiments/train_charging.py --n-amrs 50 --seed 1
"""

import sys
import argparse
import functools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from simulator.amr.charging_env import ChargingFabEnv

# ── 경로 ──────────────────────────────────────────────────
HERE       = Path(__file__).parent
OUT_DIR    = HERE / "charging"
CKPT_DIR   = OUT_DIR / "checkpoints"

# ── 실험 설정 ─────────────────────────────────────────────
MAX_STEPS     = 5_000       # sim-step 기준 에피소드 길이 (event-triggered → 결정 수는 가변)
DEFAULT_TOTAL = 1_000_000   # 총 학습 스텝(=결정 수)
LR            = 3e-4        # scratch 표준 LR


def _env_factory(n_amrs, max_steps, seed, battery_scale, coupled):
    """모듈 레벨 팩토리 — SubprocVecEnv 피클링 가능(중첩 클로저 불가)."""
    return Monitor(ChargingFabEnv(n_amrs=n_amrs, max_steps=max_steps, seed=seed,
                                  battery_scale=battery_scale, coupled=coupled))


def make_env(n_amrs, max_steps, seed, battery_scale=1.0, coupled=False):
    return functools.partial(_env_factory, n_amrs, max_steps, seed, battery_scale, coupled)


def out_paths(n_amrs, seed, battery_scale, coupled=False):
    cp = "_coupled" if coupled else ""
    tag = f"charging_n{n_amrs}_bs{battery_scale:g}{cp}_seed{seed}"
    return str(OUT_DIR / f"{tag}_model"), str(OUT_DIR / f"{tag}_vecnorm.pkl"), tag


def evaluate(model, vecnorm_path, n_amrs, max_steps, battery_scale, n_episodes=3, coupled=False):
    eval_vec = DummyVecEnv([make_env(n_amrs, max_steps, seed=0,
                                     battery_scale=battery_scale, coupled=coupled)])
    eval_vec = VecNormalize.load(vecnorm_path, eval_vec)
    eval_vec.training    = False
    eval_vec.norm_reward = False

    infos = []
    for _ in range(n_episodes):
        obs = eval_vec.reset()
        while True:
            masks = eval_vec.env_method("action_masks")[0]
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, _, done, info = eval_vec.step(action)
            if done[0]:
                infos.append(info[0])
                break

    def avg(k):
        return float(np.mean([i.get(k, np.nan) for i in infos]))

    print(f"\n=== 충전 RL 평가 (n_amrs={n_amrs}, {n_episodes} 에피소드) ===")
    print(f"throughput:        {avg('throughput'):.1f}")
    print(f"depleted_per_1k:   {avg('depleted_per_1k'):.3f}   (← 0 목표)")
    print(f"hot_sla_rate:      {avg('hot_sla_rate'):.3f}")
    print(f"energy_per_deliv:  {avg('energy_per_delivery'):.3f}")
    print(f"dist_per_done:     {avg('dist_per_done'):.1f}")
    print(f"deadlock_per_1k:   {avg('deadlock_per_1k'):.3f}")
    print(f"utilization:       {avg('utilization'):.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-amrs", type=int, default=20, help="AMR 대수 (대수별 scratch)")
    ap.add_argument("--seed", type=int, default=42, help="학습 시드 (multi-seed)")
    ap.add_argument("--battery-scale", type=float, default=8.0,
                    help="유효용량 1/scale. 충전이 무는 레짐(≥8 권장, 프로브 근거)")
    ap.add_argument("--n-envs", type=int, default=8, help="병렬 환경 수 (SubprocVecEnv)")
    ap.add_argument("--ent-coef", type=float, default=0.01,
                    help="엔트로피 보너스 (조기 결정론 붕괴 방지, >0 권장)")
    ap.add_argument("--coupled", action="store_true",
                    help="SMT2020 실공정 수요로 학습 (합성 랜덤요청 대신)")
    ap.add_argument("--total", type=int, default=DEFAULT_TOTAL, help="총 학습 스텝")
    ap.add_argument("--smoke", action="store_true", help="2k 스텝·짧은 에피소드 빠른 검증")
    args = ap.parse_args()

    total_steps = 2_000 if args.smoke else args.total
    max_steps   = 1_000 if args.smoke else MAX_STEPS
    n_steps     = 128   if args.smoke else 512        # per-env 롤아웃 (총=n_steps×n_envs)
    n_envs      = 2     if args.smoke else args.n_envs

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out_model, out_vecnorm, tag = out_paths(args.n_amrs, args.seed, args.battery_scale, args.coupled)

    print(f"[charging] ChargingFabEnv scratch 학습 | n_amrs={args.n_amrs} "
          f"battery_scale={args.battery_scale:g} seed={args.seed} | n_envs={n_envs} "
          f"| {'COUPLED(SMT2020 실수요)' if args.coupled else '합성요청'}")
    print(f"[charging] 에피소드={max_steps:,} sim-step | 총 {total_steps:,} 결정스텝 "
          f"| 롤아웃 {n_steps}×{n_envs}={n_steps*n_envs} | LR={LR}")

    fns = [make_env(args.n_amrs, max_steps, seed=args.seed + i,
                    battery_scale=args.battery_scale, coupled=args.coupled)
           for i in range(n_envs)]
    vec_env = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    model = MaskablePPO(
        "MlpPolicy", vec_env,
        learning_rate=LR,
        n_steps=n_steps,
        batch_size=256,
        gamma=0.99,
        ent_coef=args.ent_coef,
        seed=args.seed,
        verbose=1,
        policy_kwargs=dict(net_arch=[64, 64]),
    )

    # CheckpointCallback은 per-env 호출 기준 → n_envs로 나눠야 실제 저장됨
    checkpoint_cb = CheckpointCallback(
        save_freq=max(200_000 // n_envs, n_steps),
        save_path=str(CKPT_DIR),
        name_prefix=f"{tag}_ckpt",
        verbose=1,
    )

    model.learn(total_timesteps=total_steps, callback=checkpoint_cb)

    model.save(out_model)
    vec_env.save(out_vecnorm)
    print(f"[charging] 저장 완료: {out_model}.zip, {out_vecnorm}")

    evaluate(model, out_vecnorm, args.n_amrs, max_steps, args.battery_scale,
             n_episodes=2 if args.smoke else 3, coupled=args.coupled)


if __name__ == "__main__":
    main()
