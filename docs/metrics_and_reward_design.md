# 지표·가중치·보상 설계 (transport-layer + 배터리)

> 작성: 2026-06-24 | 대상: 팀 합의·구현 참조
> 선행: [problem_reframe_amr_optimization.md](problem_reframe_amr_optimization.md) (목적 재정의),
> [battery_charging_design_decisions.md](battery_charging_design_decisions.md) (배터리 §2.5 합의),
> [jang_papers_review.md](jang_papers_review.md) (RL=충전으로 이전 §5.4)

## 0. 핵심: "세 개의 다른 W"를 구분하라

| 층위 | 무엇 | 역할 | 가중치 | 상태 |
|---|---|---|---|---|
| **① KPI 지표** | 측정값 | 어휘(무엇을 잰다) | 없음 | ✅ 확정 |
| **② 결합 스코어** | judge | "누가 이겼나" 단일 점수 | `W_*` | ✅ **구현됨**(`eval_baseline.transport_score`, 공유) — 숫자만 튜닝 |
| **③ PPO reward** | train | PPO가 최적화하는 신호 | `W_*` | 초안값(charging_env) |

```
①KPI(어휘) ──합쳐서──> ②judge(목적·팀합의) ──맞춰서──> ③reward(PPO 학습)
```
**원칙:** ②를 팀이 먼저 정함 → ③ = ②의 가중치를 per-step delta에 + Light shaping. 그래야 PPO가 *판정 기준 그 자체*를 최적화.

---

## 1. KPI 지표 식 (① 측정값, `simulator.get_kpi()`)

```
# ── ① 강건성 (routing/혼잡 품질) ──
deadlock_per_1k = deadlocks / step × 1000          # 완전정지(희소, 이진적)
blocking_per_1k = 충돌롤백수 / step × 1000          # 막혀 대기(연속, 민감) ← 신규, Chung&Jang DPB 핵심
congestion      = replans + aborts                  # 재경로+배정실패 (※ raw count, 정규화 검토)

# ── ② fleet 효율 (dispatch+routing 통합 비용) ──
dist_per_done   = total_travel_dist / throughput    # 배달당 이동 셀
utilization     = total_busy_steps / (n_amrs × step) × 100
energy_per_delivery = total_drain_frac × 52Ah / battery_scale / throughput  # 배달당 Ah (배터리)

# ── ③ HOT SLA (서비스 제약) ──
hot_sla_rate    = hot_sla_met / hot_done            # HOT 큐대기 ≤ HOT_SLA_STEPS(=30) 비율
hot_queue_time, normal_queue_time                   # per-priority 큐대기

# ── ④ 배터리 (A2 이후) ──
availability    = 1 − total_unavail_steps / (n_amrs × step)   # 충전·방전·저SoC묶임 제외
depleted_per_1k = depleted / step × 1000            # 방전(실패)
```

**검증 예시 (blocking의 가치):** n=20에서 `deadlock/1k=0`(완전정지 없음)인데 `blocking/1k=430` → deadlock이 놓치는 혼잡을 연속값으로 포착.

### 검토·튜닝 후보 (확정 아님)
- `HOT_SLA_STEPS=30` — 임의값. 현실 SLA로 grounding 또는 sweep.
- `congestion` — raw count라 길이 의존 → `/step×1000` 정규화 권장.
- `speed_index` (제안됨) — 우리 sim은 가감속 프로파일이 있어 "저속=혼잡"이 오염됨 + blocking과 중복. 독립 추가 대신 `1 − blocking_rate` *유도값*만 권장.
- `detour_ratio = 실제경로 / 최단거리` (제안된 avg_travel_dist_per_trip의 올바른 형태) — routing 품질 직접 측정이나 **routing은 고정(예약A*)이라 우선순위 낮음**.

---

## 2. 결합 스코어 (② judge — **구현됨: `eval_baseline.transport_score`, 공유**)

여러 KPI를 **하나의 점수**로 합쳐 "RL vs 휴리스틱 누가 이겼나" 판정. **구현된 형태 (배달 1건당 비용, 0=최고):**
```
cost = W_DL·(deadlocks/n) + W_BLOCK·(blocking/n) + W_DEPLETE·(depleted/n)
     + W_TRAV·dist_per_done + W_ENERGY·energy_per_delivery
     + W_SLA·max(0, HOT_SLA_MIN − hot_sla_rate)     # HOT SLA 소프트 게이트(제약)
score = −cost     (n=throughput, 분모라 자동 보상)
W_*: DL=5000 BLOCK=10 TRAV=1 DEPLETE=3000 ENERGY=26 SLA=200 / HOT_SLA_MIN=0.9
```
- **단위 통일**: 전부 "배달당 손실 step". robustness=raw/throughput, efficiency=이미 배달당.
- **공유 1곳**: eval_baseline·prediction/baseline 모두 import (중복 없음). prediction은 FIFO=1.0 비율로 표현.
- **배터리 포함** → 충전 RL도 같은 judge로 평가(off면 energy/deplete=0).
- **남은 합의 = W_* *숫자*만** (구조·단위 확정). 팀이 실험 돌리며 한 함수에서 튜닝.

**각 `W_`의 정체 = 교환비** ("metric X 한 단위 = Y 몇 단위만큼 나쁜가"). 이건 **가치 판단 = 목적의 정의**라 PPO로 못 정함(순환). 실비용 grounding으로 정함(아래):

| 방법 | 내용 | 장점 |
|---|---|---|
| **1. 실비용 환산** (추천) | 방전 1번 = 차량 N스텝 손실 = 배달 M건; 에너지 = 전기료 → W_가 환산계수 | 임의 아님, ROI/fleet효율 스토리와 일치 |
| 2. 제약으로 | HOT SLA ≥95%, 방전=0 을 제약으로 빼고 나머지만 가중 | W_ 개수↓, reframe(HOT=제약) 일치 |
| 3. Pareto frontier | 안 합치고 지배 여부 비교 | 지배자 있으면 가중치 불필요 |
| 4. 우선순위 | 방전 최소 → throughput 최대 → 에너지 최소 순서 | 가중치 없이 |

**합의 절차:** 짧은 회의에서 (a) 무엇이 *제약*이고 무엇이 *목적*인가, (b) 목적 항들의 교환비를 실비용으로 grounding. → 구현은 `eval`에 `transport_score(kpi)` 함수 하나.

---

## 3. PPO reward 구조 (③ train — `charging_env.py`, 초안값)

충전 RL이 *최대화*하는 per-decision 신호:
```
reward = W_DONE · Δ완료  −  W_DEPLETE · Δ방전  −  W_ENERGY · Δ에너지  +  shaping
  초안 상수: W_DONE=1, W_DEPLETE=50, W_ENERGY=0.05, SHAPE=0.2
  shaping (충전 선택했을 때만, Light):
    +SHAPE  if 대기요청=0 & soc<0.5   (한가+저soc 충전 = 좋은 타이밍)
    −SHAPE  if 대기요청≥n & soc>0.6   (수요폭주+충분soc 충전 = 나쁜 타이밍)
```

**각 항이 행동을 미는 방향:**
- `W_DONE↑` → 공격적 서빙 (throughput↑, 방전 위험↑)
- `W_DEPLETE↑` → 보수적 (일찍 충전, throughput↓) ← 현재 50이 커서 RL이 무방전·보수 수렴
- `shaping` → 학습 가속 *힌트* (지표 직접 sparse 박으면 붕괴 = Bischoff: Basic 붕괴·Light 최고)

**바꾸는 법:** `charging_env.py` 클래스 상수 + `_reward()`. 항 추가/조건 변경 자유.

**②와 정렬 원칙:** ③의 `W_DONE:W_DEPLETE:W_ENERGY`를 ②와 같은 비율로 두면 PPO가 judge 목적을 직접 최적화. 예) 팀이 "방전=배달 20건"이면 ②·③ 모두 `W_DEPLETE/W_DONE=20`.

---

## 4. 권장 지표 구성 (확정안)

```
① 강건성 :  deadlock_per_1k · blocking_per_1k · congestion(replans+aborts)
② fleet효율: dist_per_done · energy_per_delivery · utilization
③ HOT SLA :  hot_sla_rate · hot_queue_time · normal_queue_time
④ 배터리  :  availability · depleted_per_1k
방법론    :  multi-seed 필수 (단일 seed 노이즈) · "헤드룸 먼저 측정 → RL 투입"
```

## 5. 미해결 (다음 합의)
- [ ] **② 결합 스코어 W_* 고정** (실비용 grounding 1페이지 초안 → 팀 회의)
- [ ] `HOT_SLA_STEPS`, `congestion` 정규화, `battery_scale`·`CHARGER_SLOTS` 평가 레짐
- [ ] ③ 충전 reward W_* 를 ②에 맞춰 재설정 후 RL 재학습 → frontier 지배 여부 판정
