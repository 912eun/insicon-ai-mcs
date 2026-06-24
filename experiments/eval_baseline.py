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
import importlib
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

# ── ② 결합 스코어(Judge) — 실비용 grounding, 단위=배달 1건당 손실 step ──
# 각 W는 '사건/단위당 step 손실'. robustness는 raw count/배달, efficiency는 이미 배달당 →
# 모든 항이 "배달당 step 비용"으로 코헤런트(단위 통일). throughput은 분모라 자동 보상(베이스라인).
W_DL      = 5000.0   # 데드락 1건 = 영구마비 step 손실
W_BLOCK   = 10.0     # blocking 1건 = ~10 step 대기지연
W_TRAV    = 1.0      # 이동 1셀 = 1 step (dist_per_done에 직접)
W_DEPLETE = 3000.0   # 방전 1건 = 견인회수 다운타임 step
W_ENERGY  = 26.0     # 1 Ah 충전 = 충전소 대기 step (30분/52Ah)
W_SLA     = 200.0    # HOT SLA 미달분당 비용 — 소프트 제약(효율 항과 동급 스케일, 안 묻히게). 팀 튜닝 대상
HOT_SLA_MIN = 0.90   # HOT SLA 충족 목표 (이 아래면 페널티)


def transport_score(kpi: dict):
    """transport-layer 통합 Judge — 배달당 비용(step 등가). 반환 (score, breakdown). 0=최고.

    단위: 모든 항을 '배달 1건당 손실 step'으로 통일(robustness=raw/throughput, efficiency=배달당).
    HOT SLA는 제약 → 미달분만 소프트 게이트 페널티. throughput은 분모라 자동 보상(베이스라인).
    배터리 off면 energy/deplete=0이라 그대로 동작. (eval_baseline·prediction/baseline 공유)
    """
    g = lambda k: kpi.get(k) or 0          # None/결측 → 0 (배터리 off 등)
    n = max(g("throughput"), 1)
    hot_sla = kpi.get("hot_sla_rate")
    hot_sla = 1.0 if hot_sla is None else hot_sla
    bd = {
        "deadlock": W_DL      * g("deadlocks") / n,
        "blocking": W_BLOCK   * g("blocking")  / n,
        "deplete":  W_DEPLETE * g("depleted")  / n,
        "travel":   W_TRAV    * g("dist_per_done"),
        "energy":   W_ENERGY  * g("energy_per_delivery"),
        "hot_sla":  W_SLA     * max(0.0, HOT_SLA_MIN - hot_sla),
    }
    return -sum(bd.values()), bd


def resolve_strategy(spec):
    """전략 inner env 클래스 해결 — 공유 도구가 *어떤 개인 전략에도* 하드의존 안 하도록 일반화.

    spec:
        "base" (또는 빈값)        → None (공유 FabAMREnv, 누구나 사용 가능)
        "module.path:ClassName"   → 동적 import
                                     (예: mcs.dispatcher.strategies.fa:FlowAwareFabEnv)
    각 팀원은 자기 전략을 `--strategy 내모듈:내Env` 로 주입 — 코드에 개인 전략 하드코딩 없음.
    """
    if not spec or spec == "base":
        return None
    if ":" not in spec:
        raise SystemExit(f"❌ --strategy '{spec}': 'base' 또는 'module.path:ClassName' 형식이어야 함 "
                         f"(예: mcs.dispatcher.strategies.fa:FlowAwareFabEnv).")
    mod_path, cls_name = spec.rsplit(":", 1)
    try:
        mod = importlib.import_module(mod_path)
        return getattr(mod, cls_name)
    except (ImportError, AttributeError) as e:
        raise SystemExit(f"❌ --strategy '{spec}' 로드 실패: {e}")


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
    ap.add_argument("--strategy", default="base",
                    help="모델 inner env: 'base'(공유) 또는 'module.path:ClassName' "
                         "(예: mcs.dispatcher.strategies.fa:FlowAwareFabEnv)")
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

        # 에피소드별 결합 스코어 (배달당 비용, 단위 통일)
        ep_score, _ = transport_score(info)
        print(f"  ep{ep}: throughput={info['throughput']:>4} "
              f"deadlocks={info['deadlocks']:>2} blocking={info.get('blocking', 0):>4} "
              f"depleted={info.get('depleted', 0):>2} replans={info.get('replans', 0):>3} "
              f"score={ep_score:>8.2f}")

    def avg(k): return np.mean([i.get(k, 0) for i in infos])
    per1k = lambda k: avg(k) / args.max_steps * 1000
    def savg(prio, k): return np.mean([s[prio][k] for s in splits])

    # 최종 결합 스코어(Judge) — 에피소드별 score 평균 + breakdown 평균
    scores, bds = zip(*[transport_score(i) for i in infos])
    combined_score = float(np.mean(scores))
    bd = {k: float(np.mean([b[k] for b in bds])) for k in bds[0]}

    print("\n=== 결합 스코어 (② Judge - 배달당 비용, 0에 가까울수록 우수) ===")
    print(f"Combined Score: {combined_score:.2f}  (배달 1건당 손실 step 등가)")
    print(f"  └ Deadlock : {-bd['deadlock']:>8.2f}")
    print(f"  └ Blocking : {-bd['blocking']:>8.2f}")
    print(f"  └ Travel   : {-bd['travel']:>8.2f}")
    print(f"  └ Deplete  : {-bd['deplete']:>8.2f}")
    print(f"  └ Energy   : {-bd['energy']:>8.2f}")
    print(f"  └ HOT SLA  : {-bd['hot_sla']:>8.2f}  (충족률 {avg('hot_sla_rate')*100:.1f}%, 목표 {HOT_SLA_MIN*100:.0f}%)")

    # transport-layer 1차 지표 (problem-reframe: fab 처리량 아님, AMR 운영 효율).
    # deadlock_per_1k 등 정규화 값은 get_kpi()가 실제 스텝수 기준으로 계산(에피소드 조기종료 무관).
    print("\n=== transport-layer 1차 지표 (평균) ===")
    print(f"① 강건성   deadlock/1k:   {avg('deadlock_per_1k'):.3f}   "
          f"(데드락 {avg('deadlocks'):.1f}건)")
    print(f"           blocking/1k:   {avg('blocking_per_1k'):.3f}   "
          f"(대기 blocking {avg('blocking'):.1f}건)")
    print(f"           혼잡 replans:   {avg('replans'):.1f}  aborts: {avg('aborts'):.1f}  "
          f"(합 {avg('replans')+avg('aborts'):.1f})")
    print(f"② fleet효율 이동거리/완료: {avg('dist_per_done'):.1f} 셀   "
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