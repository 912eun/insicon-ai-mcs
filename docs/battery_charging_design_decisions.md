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

---

## 1.5 검증 (2026-06-24) — "RL 헤드룸 있나 / 실무 효용 있나"

> 배경: 디스패칭-RL이 휴리스틱을 못 이긴다는 측정 결과([jang_papers_review.md](jang_papers_review.md) §5)
> 후, "배터리도 같은 함정(휴리스틱이면 충분)인지" 투자 전 검증. 결론 = **배터리는 통과(디스패칭과 반대)**.

**(a) RL 헤드룸 — 있음 (Bischoff 본문 실측):**
- opportunity charging이 **지배적**(핫스왑 비주류, LFP 화학) → "언제 충전" 결정이 *살아있는* 문제. non-problem 아님.
- Best RL(LightShaped, 582s) vs **Best 적응형 휴리스틱**(HighLow 40%, 702s) = **RL +17%**. 고정 threshold는 더 큰 손해.
- threshold가 지는 원리: "non-adaptive는 **가용 AMR이 너무 적어짐**" — 변동 workload·큐를 못 읽어 과/저충전.
  = **fleet 가용성의 동적 트레이드오프(비-myopic)** → 디스패칭(myopic, 헤드룸 0)과 정반대 구조.
- ⚠ 17%는 real이나 dramatic 아님 + **warehouse 결과**(fab 전이는 우리가 측정). 보상설계 민감(Basic 붕괴) → Light 필수.

**(b) 실무 효용 — 있음, 우리 목적과 직결:**
- 충전 최적화 → **fleet 크기 절감**: 전통 충전은 다운타임 메우려 **20~35% 과잉 구매**, 스마트 충전이면 **15% 작은 fleet**.
  TCO −32%, 페이백 2배. → **로드맵 B4 성공기준("더 적은 차로 같은 서비스")과 동일 지표.**
- fab: 24/7 필수, 다운타임 $3~10M/h, 가용성 95%+ 목표. 해법이 "지능형 충전 스케줄링"으로 제시. fab-특정 문서는
  얇음(신생 = novel 기회와 일치).
- **위협(반대 진영)**: wireless/en-route 급전(주행 중 충전)이 충전 결정을 *하드웨어로 제거*(= Jang OLEV 진영).
  단 비싼 인프라(바닥 개조)라 **클린룸 fab은 station 기반 opportunity charging 존속** → 우리 *소프트웨어 스케줄링* 문제 유지.
  구도: 하드웨어(무선/OLEV) vs 소프트웨어(우리 RL) → 교수님(무선)과 **상보적** 컨택 프레이밍.

**Bischoff 미탐색 = 우리 추가 헤드룸:** 선제충전(수요예측)·fleet 조율 안 함(개별 AMR을 집계 feature로만).
fab은 충전소 제약↑로 contention이 warehouse보다 클 수 있음(가설) → Bischoff 넘는 기여 여지.

**검증 결론:** 배터리 = (a) 문제 살아있음 (b) RL 헤드룸 17% 실측 (c) 비-myopic 원리 (d) fleet-size 절감으로 실무효용.
**단 첫 실험 = 우리 fab 시뮬에 threshold baseline 넣고 헤드룸 직접 측정**(Bischoff 17% 그대로 가정 금지).

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

## 2.5 합의된 결정 (2026-06-24) — A2 착수 기준

> 전제 갱신: 디스패칭-RL 기각([[dispatch-rl-rejected-battery-confirmed]], jang_papers_review §5).
> **충전이 *유일한* RL 레이어.** 그래서 원문 ①의 "RL은 dispatch만" 프레임은 폐기.

| # | 결정 | **합의값** |
|---|---|---|
| ① | 배터리-RL 관계 | **1단계 = 전부 휴리스틱**(dispatch=nearest, charge=룰, 배터리=제약) = 헤드룸 측정 baseline. **2단계 = 충전만 RL**(유일 학습 레이어). dispatch-RL 없음. |
| ② | Action space | **A_binary** `{서빙 / 100%충전}`, **event-triggered**(AMR idle 시 결정). masking: SoC<20% 강제충전, 현재≥목표 비실행 |
| ③ | Reward shaping | **Light** (Bischoff: Basic 붕괴·Light 최고). transport 보상에 충전항 *가볍게* |
| ④ | health/aging | **1단계 SoC만**, health(용량 fade)는 확장 축으로 명시만 |
| ⑤ | baseline 휴리스틱 | **threshold + opportunity + HighLow(적응형)** — ★HighLow가 Bischoff 최강 휴리스틱이므로 헤드룸은 *이것 대비* 측정해야 정직 |
| ⑥ | 충전소 | **3개, 구역(NW/NE/SW/SE) 균형** → 개수는 민감도/헤드룸 축(적을수록 contention↑→헤드룸↑) |
| ⑦ | 물리 파라미터 | **현실 Bischoff 포팅이 기본**(52Ah·15/10Ah·30분만충 → step 환산). 축소판은 **dev 전용 단축키**, 최종은 현실값 재확인. **용량은 "빡빡 배터리 레짐" sweep 축**(무거운적재/적은충전소/작은팩 fab = 현실 시나리오). 평가 에피소드는 길게(20~30k step). |
| ⑧ | **(신규) 방전 처리** | SoC 0 → 그 자리 **immobilized + `depleted_count`**, dispatch 후보 제외(deadlock급 실패). 회수/towing은 1단계 생략 → W_DEPLETE 패널티 근거 |

**⑦ 핵심 근거(2026-06-24 토론):** Bischoff 원값은 만충 ~9,400 step인데 우리 에피소드 3k~10k라
**배터리가 거의 안 물림**(over-prov 함정). 진짜 비용 동인 = *고정 학습예산 내 충전 사이클 수 = total_steps/사이클길이*
(현실값 500k→~53사이클, 축소판 →~278). → 현실값은 **~5배 학습(약 3.6h)** 필요하나 감당 가능. 축소는
*현실성 훼손*(40분 배터리)이라 **dev 전용**, 보고는 현실값. "사이클 ≪ 에피소드"가 학습 신호 조건.

**binds 조건:** 헤드룸 = (충전 묶인 fleet 비율 ~15~30%, 실무 oversizing 정합) × (수요 변동). 우리 coupled env는
util~99%+버스트라 변동 있음. 용량 sweep으로 헤드룸 최대(빡빡) vs 0(헐렁) 지점 탐색 = fleet-size sweep과 동일 논리.

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

## 5. 합의 체크리스트 — **합의 완료 (2026-06-24, §2.5)**

- [x] ① 1단계 전부 휴리스틱(dispatch=nearest) baseline → 2단계 충전만 RL (dispatch-RL 폐기 반영)
- [x] ② A_binary, event-triggered, masking
- [x] ③ Light shaping
- [x] ④ 1단계 SoC만, health는 확장
- [x] ⑤ threshold + opportunity **+ HighLow**(헤드룸은 최강 휴리스틱 대비)
- [x] ⑥ 충전소 3개·구역배치 (개수=헤드룸 축)
- [x] ⑦ 현실 Bischoff 포팅 기본 + 축소는 dev 전용 + 용량 sweep(빡빡 레짐) + 평가 에피소드 20~30k
- [x] ⑧ (신규) 방전 = immobilized + depleted_count
- [x] 신규 KPI 키(availability/energy_per_delivery/depleted_per_1k) 통일
