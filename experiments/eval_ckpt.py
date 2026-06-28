"""체크포인트 빠른 평가 — 저장된 .zip + vecnorm.pkl로 transport_score 측정."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from experiments.train_dispatch_rl import evaluate

CKPT_DIR = Path(__file__).parent / "dispatch_rl" / "checkpoints"

steps_list = [100_352, 200_704, 301_056, 401_408, 501_760]

print(f"{'step':>10}  {'score':>8}  {'vs FIFO':>10}  hot_sla  dep/1k  dl/1k")
print("-" * 65)
for step in steps_list:
    model_path = CKPT_DIR / f"drl_n35_bs8_coupled_lanes_seed0_ckpt_{step}_steps.zip"
    vn_path    = CKPT_DIR / f"vecnorm_{step}.pkl"
    if not model_path.exists() or not vn_path.exists():
        print(f"{step:>10,}  (파일 없음)")
        continue

    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(str(model_path))

    # n_ep=1로 빠르게 (seed=100 고정)
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from simulator.amr.dispatch_rl_env import DispatchRLEnv
    import numpy as np
    from experiments.eval_baseline import transport_score

    def _factory():
        return DispatchRLEnv(
            n_amrs=35, max_steps=5000, seed=100,
            battery_scale=8, coupled=True, sim_days=7.0,
            random_start=False, routing="lanes",
            w_cong_route=1.0, charger_slots=3, top_k=5,
        )

    vec = DummyVecEnv([_factory])
    vec = VecNormalize.load(str(vn_path), vec)
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

    score, _ = transport_score(info)
    gap = score - (-202.22)
    sign = "▲WIN" if gap > 0 else "▼"
    hot = info.get("hot_sla_rate", 0)
    dep = info.get("depleted_per_1k", 0)
    dl  = info.get("deadlock_per_1k", 0)
    print(f"{step:>10,}  {score:>8.2f}  {gap:>+10.2f} {sign}  {hot:.3f}  {dep:.3f}  {dl:.3f}")
    sys.stdout.flush()
