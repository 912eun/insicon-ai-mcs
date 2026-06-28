# Phase 3 Dispatch RL 설계 문서

## 1. 문제 정의

반도체 공장 내 AMR(자율이동로봇) 함대를 운용할 때, 각 AMR은 두 가지 결정을 반복한다.

1. **배정 결정**: 대기 중인 운반 요청 중 어느 것을 수행할 것인가
2. **충전 결정**: 지금 충전소로 이동할 것인가, 운반을 계속할 것인가

기존 EDD(Earliest Due Date) 휴리스틱은 우선순위와 대기시간만 고려해 배정을 결정한다. 그러나 최적 배정은 다음 세 가지를 동시에 고려해야 한다.

- **거리**: 가까운 요청일수록 이동 비용(시간, 에너지)이 낮음
- **HOT SLA 긴박도**: SLA 마감이 임박한 HOT 요청은 조기 처리가 필요
- **현재 SOC**: 배터리가 낮은 상태에서 먼 요청을 배정하면 이동 중 방전 위험

이 세 요소를 jointly 학습하는 것이 RL 배정 정책의 핵심 가치다.

---

## 2. MDP 설계

### 에이전트 및 결정 시점

에이전트는 **결정이 필요한 AMR 한 대**가 의사결정 주체다. 시뮬레이터는 이벤트 기반으로 진행되며, AMR이 유휴 상태가 될 때마다 해당 AMR에 대해 정책을 호출한다.

### 2.1 Action Space — Discrete(6)

```
Action 0 ~ 4 : EDD 정렬된 대기 요청 상위 5개 중 index번째 선택 → 배정
Action 5     : 충전소로 이동
```

**TOP_K = 5 선택 근거**
- K=1: EDD 고정과 동일, RL 학습 의미 없음
- K=3: 거리 trade-off 학습은 가능하나 다양성 부족
- K=5: 충분한 커버리지 + Discrete(6)으로 학습 부담 낮음
- K≥10: action space 증가로 수렴 속도 저하

### 2.2 Observation Space — 28-dim

#### Base 13-dim (AMR 및 함대 상태)

| 차원 | 의미 | 정규화 |
|------|------|--------|
| soc | 현재 AMR 배터리 잔량 | [0, 1] |
| x/W, y/H | AMR 위치 | 그리드 크기 기준 |
| dist_charger | 가장 가까운 충전소까지 거리 | Manhattan / (W+H) |
| free_frac | 충전기 여유 비율 | [0, 1] |
| n_pend/20 | 대기 요청 수 | 20으로 나눔 |
| n_hot/10 | HOT 요청 수 | 10으로 나눔 |
| fleet_soc_mean | 함대 평균 SOC | [0, 1] |
| fleet_soc_min | 함대 최저 SOC | [0, 1] |
| low_soc_frac | SOC 임계값 이하 AMR 비율 | [0, 1] |
| charging_frac | 충전 중 AMR 비율 | [0, 1] |
| idle_frac | 유휴 AMR 비율 | [0, 1] |
| progress | 에피소드 진행률 | [0, 1] |

#### Per-request 15-dim (상위 5개 요청 × 3)

| 차원 | 의미 | 정규화 |
|------|------|--------|
| priority_bit | HOT=1, NORMAL=0 | — |
| norm_wait | 대기 시간 | waiting_time / 500 |
| norm_dist | 픽업 위치까지 거리 | Manhattan / (W+H) |

요청이 5개 미만이면 남은 슬롯은 [0, 0, 0] 패딩.

### 2.3 Action Masking

MaskablePPO를 사용해 매 스텝 유효하지 않은 action을 마스킹한다.

#### 강제 충전 (SOC 임계 하한)

```python
if soc < SOC_LOW and charger_free:
    return [False, False, False, False, False, True]  # 충전만 허용
```

SOC_LOW = 0.20. 방전 직전 상태에서는 반드시 충전소로 이동해야 한다.

#### HOT 우선 마스킹

```python
hot_set = {i for i, r in enumerate(reqs) if r.priority == Priority.HOT}
if hot_set:
    req_mask = [(i in hot_set) for i in range(TOP_K)]
else:
    req_mask = [(i < n) for i in range(TOP_K)]
```

큐에 HOT 요청이 존재하면 NORMAL 요청 슬롯을 마스킹한다. HOT 요청은 SLA 제약이 있으므로 NORMAL 요청보다 먼저 처리되어야 한다. 이를 마스킹으로 구조적으로 보장하고, RL은 **어떤 HOT 요청을 선택할지** (거리, 긴박도 고려) 학습한다.

#### 충전 가용성 마스킹

충전기가 모두 점유되어 있거나 SOC ≥ 0.99이면 충전 action을 마스킹한다.

---

## 3. 보상 설계

### 3.1 보상 함수 상속 구조

보상은 세 클래스에 걸쳐 계층적으로 구성된다.

```
ChargingFabEnv._reward()          ← Base: 완료·방전·에너지·SLA·교착 주 신호
        ↓  (super() 직접 호출, DispatchChargingEnv 우회)
DispatchRLEnv._reward()           ← Shaping 추가: HOT 보너스, 거리 보너스
```

> DispatchChargingEnv._reward()는 Phase 2의 이진 action(0=배정/1=충전)에
> 특화된 shaping을 갖는다. Phase 3에서는 action 의미가 달라지므로 직접 우회하고
> DispatchRLEnv가 동등한 shaping을 독립적으로 재구현한다.

---

### 3.2 ChargingFabEnv._reward() — 주 신호

매 결정 스텝마다 시뮬레이터 상태 변화(델타)를 측정해 보상을 계산한다.

```python
# 주 보상
d_done = Δtotal_completed          # 이번 스텝에 완료된 배송 수
d_dep  = Δdepleted_count           # 이번 스텝에 발생한 방전 수
d_eng  = Δtotal_drain_frac × BATTERY_CAP_AH / battery_scale

r = W_DONE × d_done - W_DEPLETE × d_dep - W_ENERGY × d_eng

# judge 정렬 항 (가중치 0이면 비활성)
d_block    = Δblocking_count
d_dl       = Δdeadlock_count
d_hot_over = Δhot_completed - Δhot_sla_met   # SLA 위반 완료건

r -= w_block × d_block + w_dl × d_dl + w_sla × d_hot_over

# 충전 타이밍 shaping (충전 action일 때만)
if last_action == charge:
    if n_pending == 0 and soc < 0.5:  r += SHAPE   # 한가 + 저SOC = 좋은 타이밍
    if n_pending ≥ n_amrs and soc > 0.6: r -= SHAPE # 수요 폭주 + 여유 = 나쁜 타이밍
```

| 항목 | 값 | 의미 |
|------|-----|------|
| W_DONE | 50.0 | 배송 완료 1건당 주 보상 |
| W_DEPLETE | 50.0 | 방전 1건 페널티 |
| W_ENERGY | 0.05 | 배터리 소모 단가 |
| SHAPE | 0.01 | 충전 타이밍 맥락 shaping |
| w_sla | 0.5 | HOT SLA 위반 완료건 페널티 |
| w_dl | 1.0 | 교착상태 발생 페널티 |
| w_block | 0.0 | 혼잡 blocking 페널티 (비활성) |

`W_DONE`이 지배적 신호(50 × 배송 수)이며, 나머지 항은 보조 패널티다.

---

### 3.3 DispatchRLEnv._reward() — Shaping 추가

ChargingFabEnv 보상 위에 즉각 피드백(shaping)을 얹는다. Credit assignment 지연(배정 결정 → 수십 스텝 뒤 완료 보상)을 완화하는 것이 목적이다.

```python
r = ChargingFabEnv._reward()   # 주 신호

if action < TOP_K and n_pending > 0:          # 배정 action이고 요청 있을 때
    r += SHAPE_DISPATCH_IMM                    # +0.05  즉시 선불
    if dispatched_request.priority == HOT:
        r += SHAPE_HOT                         # +1.5   HOT 우선 강화
    if chosen_dist > 0:
        r += SHAPE_NEAR × (1.0 - chosen_dist) # +최대 0.02  근거리 선호
```

| 항목 | 값 | 효과 |
|------|----|------|
| SHAPE_DISPATCH_IMM | +0.05 | 배정 결정을 즉시 강화 (수렴 속도 향상) |
| SHAPE_HOT | +1.5 | HOT 마스킹과 이중으로 hot_sla_rate 유도 |
| SHAPE_NEAR | +최대 0.02 | dist_per_done 감소 유도 |

---

## 4. RL 알고리즘 — MaskablePPO

### 4.1 기반: PPO (Proximal Policy Optimization)

PPO는 Actor-Critic 계열 on-policy 알고리즘이다. 정책 파라미터 θ를 아래 목적함수로 업데이트한다.

```
L_CLIP(θ) = E[ min( r_t(θ) · Â_t,  clip(r_t(θ), 1-ε, 1+ε) · Â_t ) ]

r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)   # probability ratio
Â_t     = GAE advantage estimate              # 일반화 이점 추정
ε       = 0.2                                  # clip 범위 (기본값)
```

클리핑은 정책이 한 번에 너무 크게 변하는 것을 방지해 학습 안정성을 확보한다.

**전체 손실함수:**

```
L(θ) = - L_CLIP(θ)                  # 정책 개선 (최대화)
      + c_vf × L_VF(θ)              # 가치함수 MSE 손실
      - c_ent × H[π_θ(·|s_t)]      # 엔트로피 보너스 (탐색 유지)
```

| 계수 | 역할 | 설정값 |
|------|------|--------|
| c_vf | 가치함수 학습 가중치 | 0.5 |
| c_ent | 엔트로피 보너스 | 0.05 |

GAE(Generalized Advantage Estimation)로 이점을 추정한다.

```
Â_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
δ_t = r_t + γ V(s_{t+1}) - V(s_t)

γ = 0.99   (할인율)
λ = 0.95   (GAE 트레이드오프: 분산↑ vs 편향↑)
```

---

### 4.2 Action Masking 통합

sb3-contrib의 `MaskablePPO`는 매 스텝 `invalid_action_masks`를 받아 정책 분포를 제한한다.

```
마스킹 전:   logits = [2.1, -0.3, 1.4, 0.8, -1.2, 0.5]
마스크:      [True,  False, True, False, True,  True ]
마스킹 후:   logits = [2.1,  -∞,  1.4,  -∞,  -1.2, 0.5]
softmax:     유효한 action에만 확률 부여
```

마스킹된 action은 확률이 0이 되어 절대 선택되지 않는다. 이는 소프트 페널티 방식보다 구조적으로 강력하며, HOT-first 보장과 강제충전 보장을 정책이 어길 수 없게 만든다.

**학습 시 영향:**  
PPO의 L_CLIP 계산에서 마스킹된 action은 확률비 r_t가 0이므로 정책 gradient에 기여하지 않는다. 정책은 **유효한 action들 사이의 분포**만 학습한다.

---

### 4.3 네트워크 구조 (MlpPolicy)

```
obs (28-dim)
    │
    ├─── Policy network (Actor)
    │        Linear(28→64) → ReLU
    │        Linear(64→64) → ReLU
    │        Linear(64→6)  → logits
    │            │ mask
    │        softmax → π(a|s)
    │
    └─── Value network (Critic)
             Linear(28→64) → ReLU
             Linear(64→64) → ReLU
             Linear(64→1)  → V(s)
```

Actor와 Critic이 별개의 MLP를 갖는 구조. 파라미터를 공유하지 않아 가치함수 학습이 정책 표현에 간섭하지 않는다.

---

### 4.4 학습 루프 요약

```
for rollout in range(total_timesteps // (n_steps × n_envs)):
    1. 7개 환경에서 n_steps=2048 스텝씩 경험 수집
       (각 스텝: obs → policy → masked logit → action)
    2. GAE로 Â_t, returns 계산
    3. VecNormalize가 obs/reward 정규화 (clip=10.0)
    4. mini-batch로 n_epochs 동안 L(θ) 업데이트
       batch_size=256, n_epochs=10
    5. 14,336 스텝마다 체크포인트 저장
```

---

## 5. 학습 설정

### 하이퍼파라미터

| 항목 | 값 |
|------|----|
| 알고리즘 | MaskablePPO |
| 네트워크 | MlpPolicy [64, 64] |
| learning_rate | 3e-4 |
| n_steps | 2048 |
| batch_size | 256 |
| gamma | 0.99 |
| ent_coef | 0.05 |

### 병렬 환경

| 항목 | 값 |
|------|----|
| n_envs | 7 (SubprocVecEnv) |
| VecNormalize | norm_obs=True, norm_reward=True, clip_reward=10.0 |
| 학습 시간 제한 | 12h (TimeLimitCallback) |

### 시뮬레이션 설정

| 항목 | 값 |
|------|----|
| n_amrs | 35 |
| battery_scale | 8 |
| coupled (SMT2020) | True |
| sim_days | 7 (에피소드당 시뮬 기간) |
| max_steps | 5,000 |
| routing | lanes |
| charger_slots | 3 |

### 체크포인트

```
save_freq = n_steps × 7 = 14,336  (rollout 경계 정렬)
```

매 7 rollout(약 2.5h)마다 모델 저장 + transport_score 즉시 평가.

---

## 6. 평가 지표 (transport_score)

에피소드 종료 시 KPI로 정책 성능을 측정한다.

```
transport_score = -(
    5000 × deadlocks / throughput
  + 3000 × depleted / throughput
  + 200  × max(0, 0.90 − hot_sla_rate)
  + 1.0  × dist_per_done
  + 26   × energy_per_delivery
)
```

0에 가까울수록 우수. 각 항의 단위는 "배달 1건당 손실 step"으로 통일되어 있다.

---

## 7. 산출물

```
experiments/dispatch_rl/
├── drl_n35_bs8_coupled_lanes_seed0_model.zip      # 최종 모델
├── drl_n35_bs8_coupled_lanes_seed0_vecnorm.pkl    # 최종 VecNorm 통계
├── checkpoints/
│   ├── *_ckpt_{step}_steps.zip                    # 체크포인트 모델
│   └── vecnorm_{step}.pkl                         # 체크포인트 VecNorm
└── logs/PPO_8/                                    # TensorBoard 학습 로그
```
