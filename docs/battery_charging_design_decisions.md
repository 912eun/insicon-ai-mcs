# 배터리 충전 — 설계 결정점 (팀 합의용)

> 작성: 2026-06-23 | 대상: 팀 합의용 | 상태: **설계 미합의**
> 선행 문서: [problem_reframe_amr_optimization.md](problem_reframe_amr_optimization.md) (목적 재정의),
> [reward_design_review.md](reward_design_review.md) (보상 합의 가이드)
> 내부 자료: 서영 PDF (Bischoff 2505.11136 리뷰) — *리뷰지 합의안 아님*

---

## TL;DR

- **왜 배터리?** 재정의(§6 fleet 효율/에너지)의 real한 축. fab에서 OHT/RGV는 레일 급전이라
  배터리가 없고, **배터리는 AMR을 도입할 때 비로소 생기는 고유 제약** → "AMR 전환 전제"와 직결.
  OHT엔 없는 문제를 다룬다는 점에서 비교 스토리가 깨끗하다.
- **선행연구**: 있음(창고·터미널·job-shop RL 다수). 다만 **fab+AMR+배터리 공개 시뮬/데이터는 없음**
  → novel scope.
- **이 문서의 역할**: 합의해야 할 **7개 결정점**과 선택지·근거·추천을 한 곳에 모음. 값 채우는 건 팀.

---

## 0. 왜 지금 이 결정이 필요한가

배터리를 simulator에 넣는 순간 **AMR 가용 대수가 줄고(충전 중·방전), 새 실패 모드(방전)와
새 의사결정(언제·얼마나 충전)** 이 생긴다. 이건 obs/action/reward·평가지표를 전부 건드리므로
구현 전에 범위를 못 박아야 한다. 아래 7개는 서로 의존적이라 **묶어서** 합의해야 한다.

---

## 1. 선행연구 요약 (결정의 근거)

| 출처 | 환경 | 핵심 기여 | 우리에게 |
|---|---|---|---|
| **Bischoff et al. 2025** (arxiv 2505.11136) | 창고 (SLAPStack, MIT, 공개) | AMR이 **언제·얼마나 충전할지** RL 결정. reward·action space 설계가 안정성↔일반화 트레이드오프 결정 | **골격 동일**(event-triggered). 배터리모델·obs·action·reward 그대로 참고 |
| **Fleet-Level Battery-Health-Aware Scheduling** (arxiv 2603.22731) | AMR fleet | SoC뿐 아니라 **용량 fade(노화)** 까지 반영, 가용성·에너지·**수명** 동시 최적화 | health 축의 존재 근거 (1단계엔 과할 수 있음) |
| 에너지효율 AGV DRL (Springer JoIM 2024) | job-shop | tardiness+energy **bi-objective**, dueling double DQN | 에너지를 목적에 넣는 표준 패턴 |
| AGV battery-constraint DRL / 컨테이너터미널 resilient charging (IET 2025) | 터미널·job-shop | 배터리 제약 dispatching, 충전 전략 | 제약-우선 접근의 선례 |

**공통 메시지**: 배터리/충전을 dispatching에 통합하는 RL은 활발하나 **fab은 비어있다**.

**참고 수치 (Bischoff, 포팅 가능):** 용량 52Ah · 소모 15Ah(적재)/10Ah(공차) ·
선형충전 30분→100% · 충전소 3개 등거리 · obs 19 feature · PPO(sb3).

**Bischoff 핵심 실험 결과 (보상설계에 직결):**
- Basic1 = 최적화 지표(서비스시간)를 그대로 sparse reward → **붕괴**(20만 스텝 후, 4897s).
- LightShaped(약한 shaping) = **최고**(588s). FullyShaped는 수렴 빠르나 일반화 실패.
- → "직접 지표를 reward로 박지 말고 transport-layer 신호를 **가볍게** shaping."

---

## 2. 합의해야 할 7개 결정점

각 항목: **선택지 / 근거 / 추천(논의용 출발선)**.

### ① 배터리와 RL의 관계 — *가장 먼저 합의*
- **선택지 A. 제약으로만**: 충전은 룰베이스(트리거), RL은 기존대로 dispatch만. 배터리는 가용 fleet을 줄이는 제약.
- **선택지 B. 충전도 RL action**: action space에 "충전 보내기/목표치" 추가. RL이 가용성↔기회비용 학습.
- **근거**: B가 기여 크지만 obs/action/reward 재설계 필요. A는 빠르고 측정축부터 세움.
- **추천**: **A로 1단계 → 안정화 후 B로 승격** (단계적). [problem_reframe §7의 "단계적" 정신]

### ② Action space (①이 B일 때)
- **선택지**: `A_full = [0,30,40,50,60,70,80,90,100]`(충전 목표 9단계) vs `A_binary = [0,100]`(충전/안함).
- **근거**: Bischoff — full은 탐색 자유↑이나 binary가 더 안정. action masking으로 현재<목표 비실행 제거, <20%는 강제충전.
- **추천**: **A_binary 시작** (도메인 단순화, 안정 우선).

### ③ Reward shaping 수준
- **선택지**: Basic(sparse) / Light(약한 shaping) / Fully(복합 페널티·보너스).
- **근거**: Bischoff — **Light 최고**, Basic 붕괴, Fully 일반화 실패.
- **추천**: **Light**. 우리 transport-layer 보상(아래 §3)에 충전 관련 항을 *가볍게* 추가.

### ④ 배터리 health/aging 포함 여부
- **선택지**: SoC·가용성만 / 충전 사이클 누적 → 용량 fade까지.
- **근거**: 2603.22731. 수명은 real CapEx지만 모델·튜닝 비용 큼.
- **추천**: **1단계 SoC만, health는 확장 축**으로 명시만.

### ⑤ 충전 트리거 (휴리스틱 baseline용 — ①A에서 필수, B에서도 비교군)
- **선택지**: threshold(<20% 가면 충전) / opportunity(유휴 & 대기 요청 없을 때) / HighLow(큐 길이 적응) / RL.
- **근거**: Bischoff baseline 그대로. RL을 이걸로 비교해야 정직.
- **추천**: **threshold + opportunity 둘 다 baseline**으로 구현.

### ⑥ 충전소 개수·위치
- **선택지**: 개수(예: 3) / 그리드 배치(등거리 벽면 vs 구역별).
- **근거**: Bischoff 3개 등거리. 우리 fab_layout 그리드에 매핑 필요.
- **추천**: **3개, 구역(NW/NE/SW/SE) 균형 배치** 시작 → 민감도 실험 대상.

### ⑦ 물리 파라미터
- **선택지**: Bischoff 포팅(52Ah 등 → 우리 셀/스텝 환산) / fab 실측치.
- **근거**: 실측 없으면 포팅이 합리적 출발.
- **추천**: **Bischoff 포팅, 스텝 단위로 환산** (1 AMR step = cell/v_max ≈ 1.33s 기준).

---

## 3. 지표·보상에 미치는 영향 (transport-layer 축에 결합)

배터리는 앞서 정의한 transport-layer 목적([problem_reframe §6])에 자연 결합한다.

**신규 KPI** (get_kpi에 추가):
- `availability` — 충전·방전으로 못 쓰는 AMR 비율
- `energy_per_delivery` — 반송 건당 에너지
- `depleted_per_1k` — 1000스텝당 방전 횟수 (deadlock과 동급 실패)
- (확장) `capacity_retention` — health 도입 시

**보상 항 (Light shaping, ①A 기준):**
```
reward += − W_DEPLETE · depleted_this_step      # 방전 = deadlock급 실패
          − W_ENERGY  · delta_energy / N         # fleet 에너지 (이동거리 항과 통합 가능)
# ①B(충전 RL)로 가면 Bischoff식 충전 shaping 추가:
#   충전소 없는데 충전 가면 페널티 / 요청 밀렸는데 충전 묶으면 페널티 /
#   요청 없고 충전소 비었을 때 충전하면 보너스
```

---

## 4. 단계적 구현 경로 (추천 합의안)

1. **1단계 — 제약(①A)**: AMR에 `soc`·`AMRStatus.CHARGING`, 충전소, `depleted_count` 추가.
   ⑤ threshold+opportunity 휴리스틱 충전. dispatch 후보에서 SoC 부족 AMR 제외(masking).
   → 측정축(availability·energy·depleted) 확보, 휴리스틱 vs FA를 *배터리 있는 판*에서 재비교.
2. **2단계 — 충전 RL(①B)**: action에 충전결정(②A_binary) 편입, ③Light shaping, obs에 배터리 feature.
3. **확장 — health(④)**: 용량 fade 도입, 수명을 목적에 추가.

**구현 지점** (simulator/amr/simulator.py): AMR 클래스(L92) 필드 / `AMRStatus`(L36) CHARGING /
`FABSimulator`(L229) 충전소·방전 카운터.

---

## 5. 합의 체크리스트

- [ ] ① 제약-우선(A)으로 1단계 합의?
- [ ] ② A_binary로 시작?
- [ ] ③ Light shaping?
- [ ] ④ 1단계 SoC만, health는 확장?
- [ ] ⑤ threshold+opportunity baseline?
- [ ] ⑥ 충전소 3개·구역배치?
- [ ] ⑦ Bischoff 파라미터 포팅?
- [ ] 신규 KPI 키(availability/energy_per_delivery/depleted_per_1k) 통일?
