import datetime
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('c:/Users/안가은/insicon-ai-mcs/experiments/dispatch_rl/logs/PPO_8')
ea.Reload()
scalars = ea.Tags().get('scalars', [])
print("available:", scalars)
if 'time/fps' in scalars:
    fps = ea.Scalars('time/fps')
    print(f"iters={len(fps)}", end="")
    if fps:
        print(f"  latest_step={fps[-1].step:,}  fps={fps[-1].value:.0f}")
    else:
        print()
if 'rollout/ep_rew_mean' in scalars:
    rew = ea.Scalars('rollout/ep_rew_mean')
    for e in rew:
        t = datetime.datetime.fromtimestamp(e.wall_time).strftime('%H:%M')
        print(f"  step={e.step:>7,}  rew={e.value:>8.0f}  @{t}")
if 'train/entropy_loss' in scalars:
    ent = ea.Scalars('train/entropy_loss')
    if ent:
        print(f"  entropy_loss={ent[-1].value:.3f}")
