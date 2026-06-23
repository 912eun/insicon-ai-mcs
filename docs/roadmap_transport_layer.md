# 앞으로 할 일 — 시뮬레이터 / RL·지표 두 트랙

> 작성: 2026-06-23 | 대상: 팀 작업분배용
> 선행: [transport_fab_decoupling.md](transport_fab_decoupling.md) (디커플 검증),
> [problem_reframe_amr_optimization.md](problem_reframe_amr_optimization.md) (목적 재정의),
> [battery_charging_design_decisions.md](battery_charging_design_decisions.md) (배터리 결정점),
> [reward_design_review.md](reward_design_review.md) (공통/개인 레이어 합의)

> **전략 구조 (중요)**: 보상은 FA 하나가 아니다. 각 팀원이 base `FabAMREnv`
> (`simulator/amr/env.py`)를 상속해 **자기 전략 env에서 reward를 override**한다
> (FA = `FlowAwareFabEnv._fa_reward`, 그 외 s1~s4 = 각자 작성 예정).
> 따라서 transport-layer 변경은 **특정 전략(FA)에 박으면 안 되고**, `reward_design_review.md`가
> 합의한 **"공통 레이어 + 개인 레이어"** 구조를 따라 **공통 레이어**로 빼야 모든 전략에 공정하게 적용된다.

---

## 전제 (확정된 것)

- fab 처리량은 **장비병목+WIP버퍼로 반송과 디커플** → fab CR을 목적/지표로 쓰지 않는다.
- 목적 = **transport-layer 효율**: ①deadlock·혼잡 강건성 ②fleet 효율(이동·에너지·대수) ③HOT SLA(제약).
- 여기에 **배터리/충전**을 fleet 효율 축의 새 차원으로 추가(OHT엔 없는 AMR 고유 제약).
- 남은 핵심 숙제: **FA가 transport-layer 목적에서 휴리스틱을 *일관되게* 이기는 지점 찾기**
  (deadlock 우위가 50·20대 O, 76대 X로 들쭉날쭉).

## 두 트랙의 의존관계

```
트랙 A (시뮬레이터)              트랙 B (RL·지표)
─────────────────              ─────────────────
A1 transport KPI 노출   ───→   B1 지표 재설정(측정축, 전략무관 공유)  ← 먼저
A2 배터리 1단계(제약)   ───→   B2a 공통 reward 레이어(팀 합의)
A3 충전 RL 훅(2단계)    ───→   B2b 각 전략이 공통 위에 개인 shaping
                               B3 충전 RL action
                               B4 결판 실험(전략 × fleet sweep)
```
**원칙 1: 측정축(B1)을 먼저 세우고 보상(B2)을 튜닝한다.** A1→B1이 최우선.
**원칙 2: 전략은 여러 개다.** A1(KPI)·B1(eval 지표)은 전략 무관이라 한 번만 하면 모두 공유.
보상(B2)은 **공통 레이어(B2a)를 팀이 합의**하고 각자 **개인 shaping(B2b)** 만 얹는다 —
그래야 B4에서 전략끼리 같은 목적으로 공정 비교된다.

---

## 트랙 A — 시뮬레이터 수정

### A1. transport-layer KPI 노출 *(최우선, 작음)*
`simulator/amr/simulator.py :: get_kpi()` (현재 L646) 에 추가/정리:
- `deadlock_per_1k = deadlocks / step * 1000`
- 혼잡: `replans`, `aborts` 는 이미 카운트됨 → delta 노출 확인
- **per-priority 분리**: HOT/NORMAL 각각 `queue_time`·`travel_time` 평균
- `hot_sla_rate` = HOT 중 `queue_time ≤ HOT_SLA_STEPS` 비율
- (이동거리) 누적 `total_travel_dist` 필드 추가 — AMR.step() 이동 시 셀 수 누적 (현재 없음, 1줄)
- `coupled_env.py :: _fab_kpi()` 의 fab CR/tardiness는 **보조 보고로 강등** (1차 지표에서 제외)

### A2. 배터리 — 1단계(제약) *(설계 합의 후)*
> 설계 미합의 → [battery_charging_design_decisions.md](battery_charging_design_decisions.md) §5 체크리스트 먼저.
- `AMRStatus`(L36)에 `CHARGING` 추가
- `AMR`(L92)에 `soc / battery_wh / drain_move / drain_idle / charge_rate` 필드,
  `step()`에서 상태별 soc 차감·충전 회복·full→IDLE 복귀
- `FABSimulator`(L229): 충전소 위치(그리드, 구역 균형 3개), `depleted_count`,
  dispatch 후보에서 SoC 부족 AMR 제외(픽업+드롭+귀환 거리 못 버티면)
- 휴리스틱 충전 트리거(threshold<20% + opportunity) — baseline용
- 파라미터: Bischoff 포팅(52Ah 등) → 스텝 단위 환산(1 step ≈ 1.33s)

### A3. 충전 RL 훅 — 2단계 *(B3와 함께)*
- 충전을 action으로 노출(env에서 "충전 보내기/목표치") — A_binary 권장
- obs에 배터리 feature 추가 (배터리잔량·충전소거리·충전소큐) — Bischoff 19 feature 참고

---

## 트랙 B — RL 보상함수 설계 & 지표 수정

### B1. 지표 재설정 *(A1 직후, 먼저)*
`experiments/eval_baseline.py`, `experiments/evaluate_fa.py` 비교표 컬럼 교체:
- **폐기**: fab CR / tardiness (디커플이라 안 움직임)
- **신규 1차 지표**: `deadlock_per_1k`, 혼잡(replans+aborts), avg 이동거리/에너지,
  utilization, `hot_sla_rate`, (sweep에서) 필요 fleet 크기
- KPI 키 통일(reward_design_review.md §2-3 형식 계승 + 위 신규 키)

### B2. 보상함수 재설계 *(transport-layer로 질량 이동 — 공통/개인 분리)*

> 전략이 여러 개이므로 **공통 레이어를 모든 전략이 공유**하고, 각자 개인 shaping만 얹는다.
> ([reward_design_review.md](reward_design_review.md) §2·§6의 `reward = reward_common + my_extra` 계승)

**B2a. 공통 transport-layer reward 레이어 (팀 합의 + 1곳에 구현)**
- 위치: 전략 무관 공유 함수로 — 예) `simulator/amr/env.py`의 `FabAMREnv`에 `transport_reward_common()`
  메서드 추가(또는 `mcs/dispatcher/strategies/common.py`). sim 상태만 받아 스칼라 반환.
- 기존 base `_delta_reward()`(L119, throughput−wait−deadlock)와 FA `_fa_reward()`(L97,
  완료보상 20/10이 질량 대부분)의 **공통 문제**: throughput 프록시 중심 / fleet·혼잡 항 없음 /
  HOT을 보상으로 inflate. → 재정의 §6 방향식으로 공통 레이어를 교체:
```
reward_common =
  + W_DONE   · (n_hot + n_normal)          # 서비스 베이스라인 (작게, throughput 최적화 아님)
  − W_DL     · delta_deadlock              # ① 강건성 (가장 무겁게) — 기존 100 유지
  − W_CONG   · (delta_replans+delta_aborts) # ② 혼잡 (신규)
  − W_TRAV   · delta_travel_dist / N        # ③ fleet 효율 (신규)
  − W_SLA    · n_hot_over_sla               # ④ HOT SLA 위반 패널티 (보상 아닌 제약)
  − W_DEPLETE· depleted_this_step           # 배터리 방전 = deadlock급 실패 (A2 이후)
```
- 가중치 `W_*`·`HOT_SLA_STEPS`는 **팀 공통 상수로 합의·고정**(reward_design_review.md §2 방식).
- **Bischoff 교훈**: 지표를 직접 sparse reward로 박지 말고 가볍게 shaping(Basic1 붕괴, Light 최고).
  완료보상 20/10 → W_DONE·동률·축소, HOT 우대는 ④로 이전.

**B2b. 각 전략의 개인 레이어 (담당자별 자유)**
- 각 전략 env의 reward override를 `reward = self.transport_reward_common() + my_shaping` 형태로 통일.
- **FA**: 기존 PBRS `ALPHA_SHAPE·(γΦ'−Φ)`(L144)를 개인 shaping으로 유지.
- **s1~s4**: 각자 zone-aware/urgency 등 개인 shaping([reward_design_review.md](reward_design_review.md) §3-2 표).
- **마이그레이션**: 기존 `_fa_reward`의 완료보상·deadlock·wait 항은 공통으로 빠지므로,
  각 전략은 중복 제거하고 개인 항만 남긴다(reward_design_review.md §5 이중적용 주의 그대로).

### B3. 충전 RL action *(A3와 함께, 2단계)*
- action space에 충전결정 편입, 보상에 Bischoff식 충전 shaping(충전소 없는데 충전 페널티 /
  요청 밀렸는데 충전 묶으면 페널티 / 유휴+충전소 빔에 충전 보너스)
- 충전 shaping을 공통 레이어에 넣을지 개인 레이어로 둘지는 B2a 합의 때 함께 결정.

### B4. 결판 실험 — RL 전략이 일관되게 이기는 지점 *(재정의 §7)*
가설: **76대는 over-provisioned라 누구나 잘함 → RL 우위는 혼잡 레짐(차량 부족)에서 단조적.**
- **fleet-size sweep** {16·20·30·50·76대} × {**FA, s1~s4(각 팀원 전략)**, nearest-휴리스틱, OHT(레일)}
  × transport 목적(deadlock_per_1k·혼잡·HOT SLA·에너지)
- 모든 전략이 **공통 레이어(B2a) 동일 목적**으로 학습됐으므로 비교가 공정 — 전략 간 차이는 개인 shaping 효과.
- 배터리 ON/OFF 비교(A2 이후): 배터리가 가용 fleet을 줄이면 혼잡 레짐이 더 쉽게 발생 →
  RL 우위 구간이 넓어지는지
- 성공 기준: "어떤 전략이 같은 서비스를 **더 적은 차/적은 에너지**로" = fleet-sizing 주장(조작 없이 real)

---

## 권장 순서 (마일스톤)

1. **M1**: A1 + B1 — transport KPI 노출 + eval 지표 교체. (전략·배터리 무관, 한 번에 모두 공유)
2. **M2**: B2a — **공통 reward 레이어 팀 합의·구현** + 각 전략이 B2b로 마이그레이션.
   기존 sweep에서 각 전략 vs 휴리스틱 transport 목적 재측정.
3. **M3**: B4 — fleet-size sweep으로 전략별 우위 지점 확정.
4. **M4**: 배터리 설계 합의 → A2 + 공통 레이어에 방전 항 — 배터리 있는 판에서 재비교.
5. **M5**: A3 + B3 — 충전을 RL action으로 승격(2단계).

> M1~M3은 배터리와 독립이라 **지금 바로 착수 가능**. 배터리(M4~)는 설계 합의가 선행.
> **M2의 B2a(공통 레이어 합의)가 멀티전략의 핵심 동기화 지점** — 여기서 W_* 상수와 KPI 키를
> 팀 전체가 고정해야 이후 전략 비교(M3)가 성립한다.
