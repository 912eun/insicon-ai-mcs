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
from simulator.amr.simulator import HOT_SLA_STEPS

DATASET_DIR = str(Path(__file__).parents[1] / "external" / "PySCFabSim" / "datasets" / "SMT2020_LVHM")
RAIL_GRID   = Path(__file__).parents[1] / "rware_simulator" / "output" / "rail_grid.npy"


def resolve_strategy(name):
    """전략 inner env 클래스 해결 — 공유 도구가 로컬 전략에 *하드 의존* 안 하도록.

    base → None (공유 FabAMREnv, 누구나 사용 가능)
    fa   → 로컬 전용 FlowAwareFabEnv (없으면 명확히 안내). 팀원 환경엔 fa 없음.
    """
    if name == "fa":
        try:
            from mcs.dispatcher.strategies.fa import FlowAwareFabEnv
            return FlowAwareFabEnv
        except ImportError:
            raise SystemExit("❌ --strategy fa: FA 전략 코드는 로컬 전용이라 없음. "
                             "기본 env로 평가하려면 옵션 생략(=base).")
    return None


def make_coupled(n_amrs, max_steps, seed, inner_env_cls=None, early_replan=True,
                 oht=False, closed_loop=False):
    grid_map = np.load(RAIL_GRID) if oht else None   # OHT=레일 그리드, AMR=기본 floor
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


def _nearest_cost(sim, amr, gx, gy, w_cong=0.0):
    """기하 거리만 (순수 nearest)."""
    return abs(amr.x - gx) + abs(amr.y - gy)


def _congestion_cost(sim, amr, gx, gy, w_cong=3.0):
    """거리 + corridor 혼잡 패널티 = 학습 없는 Jang형 cost-greedy.

    후보 AMR→픽업 사이 bounding-box 안의 '바쁜(비유휴) 다른 AMR' 수를 혼잡 프록시로 가산.
    가까워도 경로가 막혔으면 손해 → 좀 멀어도 빈 경로의 차를 고른다(Jang Q의 결정론적 근사).
    """
    dist = abs(amr.x - gx) + abs(amr.y - gy)
    x0, x1 = (amr.x, gx) if amr.x <= gx else (gx, amr.x)
    y0, y1 = (amr.y, gy) if amr.y <= gy else (gy, amr.y)
    busy = 0
    for o in sim.amrs:
        if o.id == amr.id or o.is_idle:
            continue
        if x0 <= o.x <= x1 and y0 <= o.y <= y1:
            busy += 1
    return dist + w_cong * busy


def run_heuristic_episode(env, dispatch="nearest"):
    """최우선 요청을 cost 최소 IDLE AMR에 배정.
    dispatch='nearest' → 기하 거리 / 'cost' → 거리+혼잡(똑똑한 nearest)."""
    cost_fn = _congestion_cost if dispatch == "cost" else _nearest_cost
    env.reset()
    sim = env._inner.sim
    info = {}
    while True:
        if not sim.request_queue or not sim.get_idle_amrs():
            action = 0
        else:
            req = sim.get_next_request()
            loc = sim.port_locations[req.from_port]
            gx, gy = loc["grid_x"], loc["grid_y"]
            best = min(sim.get_idle_amrs(), key=lambda a: cost_fn(sim, a, gx, gy))
            action = best.id
        _, _, done, info = env.step(action)
        if done:
            return info


def run_model_episode(env, model):
    """모델을 raw env로 직접 실행 (DummyVecEnv 미사용).

    버그 수정(2026-06-24): DummyVecEnv는 done 시 자동 reset → sim.completed_requests가
    비워져 마지막 step의 완료 요청이 split 통계에서 누락됐다(Bug A). 또 에피소드마다
    vec를 새로 만들어 model이 ep0 vec에 영구 바인딩·누수(Bug B). raw env 직접 실행으로
    auto-reset이 없어져 둘 다 해소 — split은 호출부에서 에피소드 종료 후 직접 읽는다.
    """
    obs = env.reset()
    done = False
    info = {}
    while not done:
        masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, _, done, info = env.step(int(action))
    return info


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
    ap.add_argument("--heuristic", action="store_true", help="휴리스틱 디스패처 (모델 불필요)")
    ap.add_argument("--strategy", choices=["base", "fa"], default="base",
                    help="모델 평가용 inner env: base(공유 FabAMREnv) / fa(로컬 전용)")
    ap.add_argument("--dispatch", choices=["nearest", "cost"], default="nearest",
                    help="휴리스틱 배정: nearest(기하거리) / cost(거리+혼잡, 똑똑한 nearest)")
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

    # ── ② 결합 스코어(Judge)용 가중치 초기 설정 (실비용 기하 기반) ──
    W_DL      = 5000.0  # 데드락: 영구 마비에 대한 엄청난 시간 손실 페널티
    W_BLOCK   = 10.0    # 블로킹: 일시 대기 지연(평균 약 10 스텝 지연) 페널티
    W_TRAV    = 1.0     # 이동거리: 1셀당 1 스텝 시간/이동과 1:1 환산
    W_DEPLETE = 3000.0  # 방전: 견인 회수 다운타임 기회비용 페널티
    W_ENERGY  = 26.0    # 에너지: 1 Ah 충전에 걸리는 충전소 대기 스텝 수 (30분/52Ah)

    tag = f"휴리스틱({args.dispatch})" if args.heuristic else Path(args.model).name
    grid_tag = "OHT(rail)" if args.oht else "AMR(floor)"
    print(f"=== 평가: {tag} | {grid_tag} ===")
    print(f"n_amrs={args.n_amrs} | 에피소드={args.episodes} | max_steps={args.max_steps:,} "
          f"(≈{args.max_steps*1.33/3600:.1f}h fab) | early_replan={'ON' if early else 'OFF'}\n")

    # 휴리스틱은 inner env 무관(sim 직접) → base. 모델은 --strategy로 결정(기본 base).
    strategy_cls = None if args.heuristic else resolve_strategy(args.strategy)
    infos, splits, model = [], [], None
    for ep in range(args.episodes):
        env = make_coupled(args.n_amrs, args.max_steps, args.seed + ep,
                           inner_env_cls=strategy_cls, early_replan=early, oht=args.oht,
                           closed_loop=args.closed_loop)
        if args.heuristic:
            info = run_heuristic_episode(env, dispatch=args.dispatch)
        else:
            if model is None:
                model = load_policy_only(args.model, env)
            info = run_model_episode(env, model)
        # auto-reset 없는 raw env → 에피소드 종료 후 sim에서 split 직접 읽기 (휴리스틱·모델 공통)
        split = split_kpi(env._inner.sim)
        infos.append(info)
        splits.append(split)

        # 에피소드별 단일 결합 스코어 계산
        ep_score = -(
            W_DL * info.get('deadlock_per_1k', 0) +
            W_BLOCK * info.get('blocking_per_1k', 0) +
            W_TRAV * info.get('dist_per_done', 0) +
            W_DEPLETE * info.get('depleted_per_1k', 0) +
            W_ENERGY * info.get('energy_per_delivery', 0)
        )
        print(f"  ep{ep}: throughput={info['throughput']:>4} "
              f"deadlocks={info['deadlocks']:>2} blocked={info.get('blocking', 0):>3} "
              f"depleted={info.get('depleted', 0):>2} replans={info.get('replans', 0):>3} "
              f"score={ep_score:>8.1f}")

    def avg(k): return np.mean([i.get(k, 0) for i in infos])
    per1k = lambda k: avg(k) / args.max_steps * 1000
    def savg(prio, k): return np.mean([s[prio][k] for s in splits])

    # 1차 지표 전체 평균 추출
    avg_dl_per_1k = avg('deadlock_per_1k')
    avg_block_per_1k = avg('blocking_per_1k')
    avg_dist_per_done = avg('dist_per_done')
    avg_deplete_per_1k = avg('depleted_per_1k')
    avg_energy_per_delivery = avg('energy_per_delivery')

    # 최종 가중 결합 스코어(Judge) 계산
    combined_score = -(
        W_DL * avg_dl_per_1k +
        W_BLOCK * avg_block_per_1k +
        W_TRAV * avg_dist_per_done +
        W_DEPLETE * avg_deplete_per_1k +
        W_ENERGY * avg_energy_per_delivery
    )

    print("\n=== 결합 스코어 (② Judge - 비용 기반 단일 점수) ===")
    print(f"Combined Score: {combined_score:.2f} (0에 가까울수록 우수)")
    print(f"  └ Deadlock 페널티   (-{W_DL:>6} * {avg_dl_per_1k:.3f}): {-W_DL*avg_dl_per_1k:>8.2f}")
    print(f"  └ Blocking 페널티   (-{W_BLOCK:>6} * {avg_block_per_1k:.3f}): {-W_BLOCK*avg_block_per_1k:>8.2f} (시뮬레이터 지원 시)")
    print(f"  └ Travel 거리 비용   (-{W_TRAV:>6} * {avg_dist_per_done:.1f}): {-W_TRAV*avg_dist_per_done:>8.2f}")
    print(f"  └ Deplete 방전 페널티 (-{W_DEPLETE:>6} * {avg_deplete_per_1k:.3f}): {-W_DEPLETE*avg_deplete_per_1k:>8.2f}")
    print(f"  └ Energy 소모 비용   (-{W_ENERGY:>6} * {avg_energy_per_delivery:.4f}): {-W_ENERGY*avg_energy_per_delivery:>8.2f}")

    # transport-layer 1차 지표 (problem-reframe: fab 처리량 아님, AMR 운영 효율).
    # deadlock_per_1k 등 정규화 값은 get_kpi()가 실제 스텝수 기준으로 계산(에피소드 조기종료 무관).
    print("\n=== transport-layer 1차 지표 (평균) ===")
    print(f"① 강건성   deadlock/1k:   {avg_dl_per_1k:.3f}   "
          f"(데드락 {avg('deadlocks'):.1f}건)")
    print(f"           blocking/1k:   {avg_block_per_1k:.3f}   "
          f"(대기 blocking {avg('blocking'):.1f}건)")
    print(f"           혼잡 replans:   {avg('replans'):.1f}  aborts: {avg('aborts'):.1f}  "
          f"(합 {avg('replans')+avg('aborts'):.1f})")
    print(f"② fleet효율 이동거리/완료: {avg_dist_per_done:.1f} 셀   "
          f"(누적 {avg('total_travel_dist'):.0f} 셀, 에너지 프록시)")
    print(f"           utilization:    {avg('utilization'):.1f}%")
    print(f"③ HOT SLA  충족률(≤{HOT_SLA_STEPS}step): {avg('hot_sla_rate')*100:.1f}%")
    print(f"--- 서비스 베이스라인 ---")
    print(f"throughput:        {avg('throughput'):.1f}  "
          f"(HOT {avg('hot_done'):.0f} / NORMAL {avg('normal_done'):.0f})")
    print(f"avg_queue_time:    {avg('avg_queue_time'):.1f}   avg_travel_time: {avg('avg_travel_time'):.1f}")

    print("\n=== HOT / NORMAL 분리 (FA 우선순위 설계 검증) ===")
    print(f"{'priority':<10}{'완료':>8}{'큐대기(평균)':>14}{'큐대기 p95':>12}{'이동시간':>12}{'미완료대기':>12}")
    pend_avg = {p: np.mean([s['pending'][p] for s in splits]) for p in ('HOT', 'NORMAL')}
    for prio in ("HOT", "NORMAL"):
        print(f"{prio:<10}{savg(prio,'n'):>8.0f}{savg(prio,'queue'):>14.1f}"
              f"{savg(prio,'p95'):>12.1f}{savg(prio,'travel'):>12.1f}{pend_avg[prio]:>12.1f}")

    if args.closed_loop:
        fk = lambda k: np.mean([i.get(k, 0) for i in infos])
        valid_done_infos = [i for i in infos if i.get('fab_done_lots', 0) > 0]
        def fk_valid(k):
            return np.mean([i.get(k, 0) for i in valid_done_infos]) if valid_done_infos else 0.0

        print("\n=== [보조] fab 일정 준수 (closed-loop, WIP CR) — 디커플, 1차 지표 아님 ===")
        print(f"WIP lot:          {fk('fab_wip_lots'):.0f}   (완료 {fk('fab_done_lots'):.0f})")
        print(f"평균 CR:          {fk('fab_cr_mean'):.3f}   (HOT {fk('fab_hot_cr_mean'):.3f})   ※높을수록 일정 여유")
        print(f"납기 뒤처짐(CR<1): {fk('fab_behind_rate'):.1f}%   (HOT {fk('fab_hot_behind_rate'):.1f}%)")
        print(f"평균 진행 스텝:    {fk('fab_steps_done_mean'):.1f}")
        if len(valid_done_infos) > 0:
            print(f"[완료분] 정시율 {fk_valid('fab_ontime_rate'):.1f}% | 평균지각 {fk_valid('fab_tardiness_mean')/3600:.2f}h "
                  f"| 사이클 {fk_valid('fab_cycle_mean')/3600:.2f}h")


if __name__ == "__main__":
    main()