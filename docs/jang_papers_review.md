# 장영재 교수님 논문 vs 우리 프로젝트 — 방향성 점검

> 작성: 2026-06-23 | 대상: 컨택 전 자기점검 + reviewer 방어
> 점검 논문 3편:
> - **Q(λ)** = Hwang & Jang 2020, *Q(λ) learning-based dynamic route guidance for OHT systems in semiconductor fabs*, IJPR 58(4).
> - **Practical** = Hong, Hwang & Jang 2022, *Practical Q-Learning-Based Route-Guidance and Vehicle Assignment for OHT Systems in Semiconductor Fabs*, IEEE TSM 35(3).
> - **DPB** = Chung & Jang 2024, *Deadlock prevention and MAPF considering physical constraint for massive fleet AGV*, Applied Soft Computing 161.
> 선행: [problem_reframe_amr_optimization.md](problem_reframe_amr_optimization.md), [roadmap_transport_layer.md](roadmap_transport_layer.md)

---

## 0. 한 줄 결론 (Practical 논문 반영 후 수정됨)

**교수님 연구의 핵심 패턴 = "하나의 학습된 신호(Q=혼잡반영 travel time)를 라우팅·배정이 공유 +
그 위에 값싼 greedy 결정 + 데드락은 결정론적(DPB, RL 아님)". 무거운 재배정·최적화(Hungarian)는
연산량 O(N³)인데 이득 없다고 실험으로 기각하고 제일 단순한 QQ를 권고.** 즉 "디스패칭에 별도
정책망을 학습"하는 게 아니라 **배정은 학습된 cost로 greedy(=똑똑한 nearest), 동적 의사결정의
본체는 라우팅**에 있다. → 우리는 RL을 *디스패칭 정책망*에 넣고 라우팅은 고정인데, 이 배치를
재고해야 한다(아래 §5 설계 논의).

---

## 1. 두 논문이 실제로 한 것 (정확히)

### Q(λ) (2020) — OHT 라우팅
- **레이어 분리**: 디스패칭 = **nearest-first(고정 휴리스틱)**. RL은 **라우팅**(교차로에서 다음 노드 선택)에만.
- **node-based control**: 차량이 아니라 **노드(교차점)를 에이전트로** 모델 → **복잡도가 차량 수와 무관**.
  명시적 이유: vehicle-based multi-agent는 state space 폭발로 "5~10대"가 한계라 대규모 fab(500+대) 불가.
- **목적함수 = 평균 travel time / throughput** 직접 최소화. "congestion은 상대적·모호한 지표라 직접 안 쓴다"고 명시.
- **보상 shaping = PBRS**: F=γφ(d,j)−φ(d,i), φ=무혼잡 최단 travel time. **Ng·Harada·Russell(1999) 정리** 인용.
- Boltzmann softmax 정책으로 차량을 여러 경로에 분산(greedy는 oscillation, softmax는 균등분산→혼잡회피).
- Q-routing(Boyan&Littman) 비정상(non-stationary) Q-learning. 산학협력+AutoMod+실 fab 데이터 검증("practical").

### Practical (2022, IEEE TSM) — OHT 라우팅 **+ 차량배정** (가장 중요)
- **하나의 Q[d,i,j](=노드 i에서 목적지 d로 갈 때 다음노드 j 선택 시 추정 travel time)를 라우팅과 배정이 공유**.
  - 라우팅: 분기노드 i에서 Q 최소인 다음노드 j 선택.
  - **배정**: 새 job(노드 d)에 대해 유휴차량들(노드 i)의 c_i=min_k Q[d,i,k] 비교 → **argmin = 가장 빨리 갈 차** 배정.
- 즉 배정은 **"학습된 혼잡반영 travel time 기준 nearest"**. 기하 거리 nearest(SS)보다 우수
  (Q vs 실제 travel time 상관 **0.82**, 거리기반 0.55). **별도 정책망이 아니라 Q를 cost로 쓰는 greedy.**
- 4종 비교: QQ(basic) / QQ_ETR(event 재배정) / QQ_math·pmath(Hungarian 최적배정).
- **[연산량 결론 — 우리 질문에 직접 답] Table V**: QQ=라우팅 O(1)·배정 O(N)·재배정 없음.
  재배정 ETR=O(N²), Hungarian=O(N³~⁴). **무거운 재배정/최적화는 통계적으로 유의한 이득 없음(Table III/IV)**
  → "복잡한 재배정은 연산만 늘고 이득 적다"며 **제일 단순한 QQ를 실무 권고**. lifelong·non-steady라 makespan 아님.

### DPB (2024) — floor AGV 데드락
- **floor(지상) bidirectional 창고 AGV** = 우리 AMR과 같은 세팅(레일 아님!).
- 디스패칭 = **FCFS(고정)**. **path planner와 분리** — 어떤 planner와도 호환되는 게 "핵심 장점"이라 명시.
- **DPB(Dynamic Path Block)**: 예약기반. AGV가 경로 노드 예약 → `pathblock`(반대방향 엣지 차단=heading-on
  데드락 방지) + `pointblock`(멈춘 AGV 노드 진입엣지 제한=loop 데드락 방지). **Theorem 1로 loop-deadlock-free 증명.**
- 지표: 완료 task 수, 평균 travel time, **blocking time**(경로충돌로 정지한 시간). makespan 아님(lifelong).
- 물리제약(회전·가감속·정지거리·차체 1×1m) AutoMod로 반영. 비판: "기존 MAPF는 이산격자+순간이동 가정 →
  실하드웨어서 데드락." 벤치마크 DRR 대비 **+25%**, 차량 많아질수록 격차 확대.

---

## 2. 우리와의 정렬 / 발산 (점검표)

| 축 | 장영재 | 우리 | 판정 |
|---|---|---|---|
| **RL이 붙는 레이어** | 라우팅(Q(λ)) / 데드락(DPB) | **디스패칭(assignment)** | ⚠ 큰 차이 — 방어 필요 |
| 디스패칭 정책 | nearest / FCFS(고정) | RL이 학습 | 우리가 더 야심적 ↔ 또는 헛다리 |
| 라우팅 | RL(Q(λ)) or A*+DPB | 예약 A*(고정) | 역전됨 |
| 데드락 처리 | **구조적·증명된 free**(DPB) | **반응적 detect→replan**(early_replan) | ⚠ 우리가 약함 |
| 스케일 | node-based(차량수 무관) | **vehicle-based(obs/action ∝ n_amrs)** | ⚠ 우리 76대 들쭉날쭉의 원인일 수 있음 |
| 보상 shaping | **PBRS(Ng1999)** | FA도 **PBRS** | ✅ 강한 정렬 |
| 목적함수 | travel time·throughput | transport-layer(데드락·혼잡·HOT SLA·이동) | 부분 정렬 |
| congestion 지표 | 라우팅선 회피 / DPB는 **blocking time 사용** | replans+aborts(=blocking 류) | ✅ 선례 있음 |
| 우선순위(HOT) | 없음 | HOT SLA 제약 | 우리 차별점(novel) |
| 배터리 | 없음(단, 교수님 OLEV 전문) | 추가 예정 | 우리 차별점 |
| 물리현실성 | 회전·footprint·가감속 | 가감속(step_costs)○, 회전·footprint✗ | 부분 |

---

## 3. 액션 아이템 (방향 조정)

1. **[최중요] "왜 RL을 디스패칭에 두나"를 방어하라.** 교수님 두 논문 모두 디스패칭=nearest로 두고
   라우팅/데드락에 지능을 넣었다. 우리가 그 반대를 하는 정당한 이유(HOT 우선순위·이종 요청·fleet
   배분이 nearest로 안 풀린다)를 명시하거나, 아니면 가치 레이어를 재고. → 이게 reviewer 1번 질문.

2. **[강력 추천] 데드락을 구조적으로.** 우리 early_replan은 DPB의 *반응적 약형 사촌*이다. DPB의
   pathblock/pointblock 아이디어를 예약테이블에 이식하면 **데드락이 구조적으로 ~0**이 되고, 그러면
   RL은 "생존(데드락 회피)"이 아니라 "효율"에서 경쟁하게 된다 — reframe(§transport 효율)과도 맞음.
   최소한 **DPB를 데드락 벤치마크/관련연구로 인용**하고 우리 deadlock_per_1k를 그 위에 위치시켜라.

3. **[스케일] vehicle-based → node/edge-based 검토.** 76대에서 우위가 들쭉날쭉한 건 obs/action이
   n_amrs에 묶여서일 수 있다(Q(λ)가 정확히 이 문제를 node-based로 회피). 재설계까진 아니어도
   "스케일 한계는 알려진 vehicle-based 한계"라고 정직하게 적고 sweep 해석에 반영.

4. **[정렬 활용] PBRS는 강한 공통분모.** FA의 PBRS가 교수님 Q(λ) reward shaping과 같은 정리(Ng1999)
   기반이다. 컨택·논문서 "교수님 방식과 같은 PBRS를 디스패칭 보상에 적용" → 신뢰+연결.

5. **[목적함수] throughput 강등 방어 + congestion 정당화.** 교수님 주력지표는 travel time·throughput.
   우리가 throughput을 강등하는 이유(디커플)를 reframe으로 설명. congestion(replans+aborts)은 DPB의
   blocking time이 선례 → "모호하다"는 반론에 방어 가능.

6. **[차별점 사수]** DPB가 floor AGV를 이미 했으니 "OHT→AMR 확장"만으론 신규성 약함. **살아남는 차별점:
   RL 디스패칭 + HOT SLA 우선순위 + 배터리/충전(교수님 OLEV와 연결) + transport-layer 보상.** 이 4개를
   DPB·Q(λ) 대비 명시적으로 포지셔닝.

---

## 4. 컨택에 주는 시사

- 교수님은 "practical/현실배포"를 강하게 본다(산학·AutoMod·DAIM). 우리 약점(디커플, 완전AMR 비현실)을
  정확히 찌를 것 → **reframe(디커플 인정 + transport-layer)을 먼저 꺼내** 정직성으로 신뢰 확보.
- 던질 좋은 질문 예: ① "디스패칭을 nearest로 고정하신 이유 — 디스패칭 자체를 학습할 가치는 낮다고
  보시나요?" ② "DPB 같은 구조적 데드락-free 위에서 RL은 무엇을 최적화해야 한다고 보시나요?"
  ③ "OHT travel-time 목적 대비, AMR에서 배터리·HOT 우선순위를 넣으면 목적함수를 어떻게 두시겠습니까?"
- 1저자(황일회=Q(λ), Chang Hyun Chung=DPB)도 병렬 컨택 후보. 둘 다 DAIM 소속 이력.

---

## 5. 설계 논의 — "RL을 어느 레이어에? 연산량 곱 문제" (2026-06-23 토론)

**문제제기(영은):** 라우팅·데드락에까지 RL을 얹으면 action space가 곱(차량×경로)으로 터지고 연산량 폭증.
그리고 **"디스패칭은 원래 nearest가 맞고, RL은 경로탐색(라우팅)에 쓰는 게 자연스럽다."**

**Practical 논문이 주는 답:**
1. **곱 폭발 우려는 정당하고, 교수님도 실험으로 확인**했다 — 무거운 결합(재배정·Hungarian O(N³))은
   이득 없이 연산만 늘어 **기각**, 제일 단순한 QQ 권고. → "여러 레이어를 한꺼번에 최적화" = 안티패턴.
2. **교수님의 회피법** = ⓐ 차량이 아니라 **노드를 에이전트로**(차량수 무관, 곱 회피) ⓑ **단 하나의 Q를
   라우팅·배정이 공유**(따로 학습 안 함) ⓒ 그 위에 **greedy(argmin)** — 최적화 안 함 ⓓ 데드락은
   **결정론적 DPB**(RL 아님, 오히려 RL 부담을 *덜어줌*).
3. **"디스패칭 nearest" 정정**: 순수 *기하 거리* nearest는 baseline이고 Q기반보다 나쁘다(0.55 vs 0.82).
   교수님 배정도 nearest이긴 한데 **"학습된 혼잡반영 travel time 기준 nearest"** = 똑똑한 nearest다.
   → 영은 직관(디스패칭=nearest) 맞음. 단 *거리* nearest가 아니라 *cost(혼잡) 기반* nearest로.

**그래서 우리 설계 선택지:**
- **(A) Jang형으로 정렬**: 동적 의사결정의 본체를 **라우팅**으로 옮긴다. 배정은 (학습된)cost 기반
  greedy nearest, 데드락은 DPB(결정론), 라우팅에만 가벼운 학습. → 연산 O(1)~O(N), 1000대 스케일.
  단 우리 차별점(HOT 우선순위·배터리)을 *라우팅+배정 cost*에 녹여야 함.
- **(B) 디스패칭 RL 유지하되 정당화**: 우리가 RL을 배정에 두는 이유 = greedy-nearest로 안 풀리는
  **HOT SLA 우선순위 + fleet 배분 + (후속)배터리 SoC 제약**의 *결합* 결정. 이건 단일 travel-time cost로
  표현 안 되니 정책 학습 가치가 있다 — 가 방어 논리. 단 라우팅·데드락엔 RL 얹지 말 것(곱 방지).
- **공통(A·B 모두)**: **데드락은 절대 RL로 풀지 말고 DPB식 결정론으로 격하** → 곱 문제 원천 제거 +
  RL은 효율만 담당(reframe과 합치). 이게 영은 우려에 대한 핵심 해소책.

**잠정 결론**: "RL을 라우팅에" vs "RL을 배정에"는 *둘 다 타당*하나 **동시에 둘 다는 금물**(곱).
데드락은 RL 밖(DPB). 우리 차별점(HOT·배터리)이 *단일 cost*로 표현되면 (A), *결합 제약*이면 (B).
→ 다음 결정: 우리 가치가 cost-기반 greedy로 충분한지 vs 정책망이 필요한지를 작은 실험으로 가른다.

### 5.1 실측 — nearest vs cost-greedy (2026-06-23, seed 0, 3000 step, 1 ep)

학습 없는 **cost-greedy("가까워도 corridor 막혔으면 다른 차")** vs 순수 거리 nearest:

| fleet | deadlock/1k | 혼잡(rpln+abrt) | dist/done | HOT SLA | thru |
|---|---|---|---|---|---|
| 20 | 2.67 → **0.00** | 310→309 | 120.6→119.6 | 52→54% | 430→433 |
| 30 | 1.67 → **0.67** | 565→539 | 115.5→115.8 | 70→71% | 663→663 |
| 50 | 22.3 → **4.0** (−82%) | 1481→**1215** (−18%) | 114.2→112.8 | 73→**70%** | 1081→1105 |

**해석(설계 직결):**
- **데드락 강건성은 학습 없이 싸게 얻어진다** → "데드락=RL 밖(DPB/휴리스틱)" 실측 확인. 곱/복잡도 우려의 해소책.
- **이건 FA 스토리 경고**: "RL이 deadlock 줄인다"는 자랑을 cost-greedy가 가져가면 reviewer가 "RL 왜?" 찌름.
- **HOT SLA는 cost-greedy로 안 움직임**(오히려 50대서 73→70) → smart-greedy가 *못* 가져가는 빈칸 =
  HOT 우선순위(+미구현 배터리). **(B)의 정당화 지점이 'deadlock'이 아니라 'HOT·배터리 결합제약'으로 확정.**
- ⚠ seed 1개 빠른 신호. 결판은 multi-seed + FA-RL 추가 비교(§5.2)로.

### 5.2 결판 — {nearest, cost-greedy, FA-RL} @ n=50,76

**🚨 단일 seed는 노이즈 (가장 중요한 교훈).** seed 0→100만 바꿔도 n=50 nearest deadlock/1k가
**22.3 → 8.0**, cost는 §5.1의 "박살(−82%)"이 seed 100에선 **8.0→14.0(악화)** 로 뒤집힘.
→ **deadlock/1k는 seed에 따라 ~3배 출렁임. 단일 seed 비교로는 (A)/(B) 판단 불가.**
§5.1의 cost-greedy 우위 주장은 **철회** (운 좋은 seed였음). reviewer가 multi-seed로 한 방에 무너뜨릴 자리.

단일 seed(seed=100) 참고치(믿지 말 것):
| n | method | dl/1k | 혼잡 | dist/done | HOT% | thru |
|---|---|---|---|---|---|---|
| 50 | nearest/cost/fa | 8.0 / 14.0 / 12.3 | 1349/1706/1516 | 110/109/112 | 70/72/72 | 1115/1114/1089 |
| 76 | nearest/cost/fa | 4.7 / 5.0 / **2.3** | 2574/2474/**2366** | 102/103/107 | 72/69/**73** | 1760/1755/1695 |

희미한 신호(미확정): n=76서 FA가 deadlock·혼잡·HOT SLA 최저/최고지만 dist·thru 손해. **multi-seed로 확인 중**
(5 seed × 3 method × {50,76}). 결론은 mean±std 나온 뒤 §5.3에 기록.

### 5.3 결판 (5 seed mean±std, 3000 step) — **(B) 현 상태 미지지**

| n=50 | dl/1k | 혼잡 | dist/done | HOT% | thru |
|---|---|---|---|---|---|
| nearest | **3.73**±2.70 | **1277**±78 | **110.0** | 72.4 | **1125** |
| cost | 6.53±3.16 | 1336±38 | 110.8 | 71.9 | 1116 |
| FA | 5.00±2.34 | 1383±34 | 112.2 | 71.9 | 1099 |

| n=76 | dl/1k | 혼잡 | dist/done | HOT% | thru |
|---|---|---|---|---|---|
| nearest | 5.67±1.92 | 2522±84 | **103.1** | 71.2 | **1747** |
| cost | **3.60**±1.51 | **2440**±68 | 103.3 | **72.3** | 1750 |
| FA | 6.20±3.36 | 2543±164 | 108.3 | 72.0 | 1672 |

**결론:**
1. **deadlock 차이는 통계적으로 구분 불가**(std≈mean). 단일 seed 우열은 전부 노이즈.
2. **cost-greedy(corridor 프록시) 무효** — nearest 동급/약간 나쁨. 폐기 또는 재설계.
3. **FA-RL이 transport 지표서 nearest를 못 이김**: dist/done·throughput 확실히 **나쁨**(노이즈 밖),
   **HOT SLA 0 이득**(전부 ~72%). 원인 = nearest가 priority 큐로 이미 HOT 우선 처리 → FA 특화 헤드룸 없음
   ([[priority-label-collapse]] 데이터 재확인).

**caveat(다음 행동 결정):** 공정한 (B) 테스트 아님 — ⓐ FA는 *옛 보상*(throughput+PBRS)으로 학습(transport
보상 B2a 아님), ⓑ 50·76대는 over-provisioned. reframe 가설(RL 우위=혼잡 16~30대)을 **FA 모델 부재로 못 돌림.**
→ 공정 테스트 = **transport 보상(B2a)으로 16~30대 재학습 후 재비교.** 단 over-provisioned서 RL 헤드룸 없다는
reframe 예측은 **적중**. 데드락은 어차피 RL 무관(§5.1~5.2 + DPB) → 구조적(DPB)으로, RL은 혼잡레짐+배터리서 승부.

### 5.4 최종 판정 (2026-06-24) — 디스패칭-RL 기각, RL을 배터리로 이전

**혼잡 레짐 결판 (coupled n=20, HOT~54%, util 99% — FA를 거기서 500k scratch 학습 후 5 seed):**

| n=20 | dl/1k | 혼잡 | dist/done | HOT% | thru |
|---|---|---|---|---|---|
| nearest | 4.60±7.11 | 346.8 | **120.5** | **54.5** | **428.6** |
| cost | 2.67±2.99 | 321.4 | 120.1 | 53.0 | 431.4 |
| FA | **0.93±0.57** | 330.2 | 124.1 | 54.2 | 418.6 |

- **혼잡 레짐에서조차 FA가 HOT SLA 0 이득**(54.2 vs 54.5) + dist·thru 더 나쁨. (B)의 HOT 근거 붕괴.
- FA의 *유일* 우위 = deadlock 평균·분산 최저(0.93±0.57 vs 4.60±7.11) — **근데 그건 DPB가 맡을 축.**
- HOT 헤드룸 0의 구조적 원인 = nearest가 priority 큐로 이미 HOT 우선처리 → 어떤 방법도 못 이김(B2a도 못 구함).

**판정 = (A) 쪽. 디스패칭-RL은 어느 레짐서도 휴리스틱 못 이김(측정 일관).** RL을 버리는 게 아니라 **이전**:

| 축 | 최적 방법 | RL 자리 |
|---|---|---|
| 디스패칭 | greedy nearest | ❌ |
| 라우팅 | congestion-aware A*(결정론; Jang RL은 rail 특화·재현·비novel) | △ (§ 선택 II) |
| 데드락 | DPB 구조적 | ❌ |
| HOT SLA | priority 큐가 포화 | ❌ |
| **배터리/충전** | **threshold 17% 손해(Bischoff)** | ✅ **여기** |

→ 최종 아키텍처: dispatch=nearest, routing=congestion A*, deadlock=DPB, **charging=RL(유일 학습 레이어, novel)**.
배터리 검증은 [battery_charging_design_decisions.md](battery_charging_design_decisions.md) §1.5 (RL 헤드룸 + 실무효용 통과).
**다음 = A2(배터리 시뮬 구현) + threshold baseline으로 우리 판 헤드룸 측정.**
