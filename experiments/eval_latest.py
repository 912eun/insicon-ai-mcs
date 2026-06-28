"""최신 체크포인트만 빠르게 평가."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from simulator.amr.dispatch_rl_env import DispatchRLEnv
from experiments.eval_baseline import transport_score

CKPT_DIR = Path(__file__).parent / "dispatch_rl" / "checkpoints"
STEP = 501_760

model = MaskablePPO.load(str(CKPT_DIR / f"drl_n35_bs8_coupled_lanes_seed0_ckpt_{STEP}_steps.zip"))
vn_path = str(CKPT_DIR / f"vecnorm_{STEP}.pkl")

scores = []
for seed in [100, 101]:
    def _factory(s=seed):
        return DispatchRLEnv(
            n_amrs=35, max_steps=5000, seed=s,
            battery_scale=8, coupled=True, sim_days=7.0,
            random_start=False, routing="lanes",
            w_cong_route=1.0, charger_slots=3, top_k=5,
        )
    vec = DummyVecEnv([_factory])
    vec = VecNormalize.load(vn_path, vec)
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

    sc, _ = transport_score(info)
    scores.append(sc)
    print(f"seed={seed}  score={sc:.2f}  hot_sla={info.get('hot_sla_rate',0):.3f}"
          f"  dep/1k={info.get('depleted_per_1k',0):.3f}"
          f"  dl/1k={info.get('deadlock_per_1k',0):.3f}"
          f"  dist={info.get('dist_per_done',0):.1f}")
    sys.stdout.flush()

mean = float(np.mean(scores))
gap = mean - (-202.22)
print(f"\n=== step={STEP:,} 평균 transport_score: {mean:.2f} ===")
print(f"vs FIFO(-202.22): {gap:+.2f}  {'▲ FIFO 이김!' if gap > 0 else '▼ 아직 못 이김'}")
