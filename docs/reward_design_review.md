# Reward 설계 리뷰 및 팀 합의 가이드

> 작성 기준: 2026-05-30 | 서영·태희 설계안 리뷰 기반

---

## 1. 두 설계가 이미 동의한 부분

| 항목 | 서영 | 태희 |
|---|---|---|
| 보상 타이밍 | Dense (매 스텝) | Dense (매 스텝) |
| 완료 보상 | HOT×20, NORMAL×10 | 기본값 상속 |
| Queue wait 패널티 | weighted_queue_wait × 0.02 | wip_penalty × 0.01 |
| 데드락 패널티 | ×100 | ×100 (base 상속) |
| Action Masking | IDLE AMR만 선택 가능 | IDLE AMR만 선택 가능 |

핵심 방향성은 일치한다. 충돌 지점은 **scale**과 **중복 적용** 두 가지다.

---

## 2. 통일할 변수 구조 (공통 최소 합의안)

### 2-1. 스텝마다 계산할 공통 변수

```python
n_hot_completed      # 이 스텝에 완료된 HOT 건수
n_normal_completed   # 이 스텝에 완료된 NORMAL 건수
weighted_queue_wait  # sum(w × waiting_time) for all pending
                     # HOT: w=3.0, NORMAL: w=1.0
n_pending            # len(sim.request_queue)
deadlocks_this_step  # sim.deadlock_count - prev_deadlock_count (delta)
```

### 2-2. 공통 보상 식

```python
reward_common = (
    20.0 * n_hot_completed
  + 10.0 * n_normal_completed
  - WAIT_SCALE * weighted_queue_wait   # ← 이 값을 팀 전체가 합의
  - 100.0 * deadlocks_this_step
)

# 각자 전략별 추가 패널티는 위에 더하는 방식
reward = reward_common + my_extra_shaping
```

### 2-3. KPI 로깅 — 무조건 동일한 키 사용

```python
# 에피소드 종료 시 모두 이 형식으로 기록해야 비교 가능
{
    "throughput":       int,    # 완료된 반송 건수
    "avg_queue_time":   float,  # 요청 생성 → AMR 배정 평균 (스텝)
    "avg_travel_time":  float,  # AMR 배정 → 완료 평균 (스텝)
    "utilization":      float,  # 이동/적재 중인 AMR 비율 (%)
    "deadlocks":        int,    # 누적 데드락 횟수
    "pending":          int,    # 미배정 대기 요청 수
}
```

---

## 3. 지금 당장 고민해야 할 것들

### 3-1. WAIT_SCALE — 가장 중요한 합의 포인트

서영 설계(0.02)를 그대로 쓰면 이렇게 된다:

```
pending 15건 × 평균 150스텝 대기 × HOT 가중치 3.0 × 0.02
= 패널티 약 135 / step

HOT 완료 보상 = +20
```

완료 전까지 **매 스텝 패널티가 완료 보상을 6배 이상 압도**한다.
에이전트 입장에서는 dispatch를 최대한 미루는 게 최적처럼 보일 수 있다.

**권장:** `0.005`로 시작하고, 첫 100 에피소드의 reward 분포를 보고 같이 조정.

```python
# 목표 reward 범위: 에피소드당 -200 ~ +500 수준
# 스텝당 평균 reward가 -1 ~ +3 정도면 학습 신호로 적절
WAIT_SCALE = 0.005  # 초기값, 실험 후 조정
```

### 3-2. 각자 추가할 수 있는 전략별 페널티

공통 레이어 위에 자유롭게 추가:

| 담당자 | 추가 항목 | 계산 방법 |
|---|---|---|
| 서영 | `zone_imbalance` | `max(zone_counts) - min(zone_counts)` |
| 서영 | `empty_dist` | `len(path_to_pickup)` at dispatch time |
| 서영 | `idle_penalty` | `n_idle × 0.2` (pending > 0일 때) |
| 태희 | (공통 레이어로도 충분, 추가 검토) | — |
| s3 | zone-aware shaping | 혼잡 구역 AMR 배정 시 감점 |
| s4 | priority urgency bonus | 오래 기다린 HOT 요청 우선 처리 시 추가 보상 |

---

## 4. 태희 코드 버그 수정 (실행 전 필수)

### 버그 1 — `waiting_time`은 메서드, 인자 필요

```python
# 현재 (틀림) — 메서드 객체 자체가 반환됨
wip_penalty += (req.waiting_time * weight)

# 수정
wip_penalty += (req.waiting_time(self.sim.current_step) * weight)
```

### 버그 2 — `request_queue`는 `(priority, id, req)` 튜플 리스트

```python
# 현재 (틀림) — req가 실제로는 튜플이라 req.priority 접근 불가
for req in pending_queue:
    weight = 2.0 if req.priority == "HOT" else 1.0

# 수정
for _, _, req in self.sim.request_queue:
    weight = 2.0 if req.priority.name == "HOT" else 1.0
```

### 버그 3 — Policy 타입 불일치

```python
# 현재 (틀림) — obs가 flat Box인데 MultiInputPolicy는 Dict obs 전용
MaskablePPO("MultiInputPolicy", env, ...)

# 수정
MaskablePPO("MlpPolicy", env, ...)
```

---

## 5. 태희 설계의 이중 적용 문제

태희 코드는 `super().step(action)`에서 `base_reward`를 받고, 거기에 WIP 패널티를 추가한다.

그런데 `FabAMREnv._delta_reward()`가 이미 대기시간을 반영하고 있다:

```python
# FabAMREnv._delta_reward() 내부
return delta_tp * 10.0 - delta_wait * 0.5 - delta_deadlock * 100.0
#                        ^^^^^^^^^^^^^^^ 이미 대기시간 포함
```

WIP 패널티를 추가하면 대기시간이 **두 번 반영**된다.

**해결 방법 (둘 중 택일):**

**A안.** `_delta_reward()`를 대기시간 없이 override:
```python
def _delta_reward(self) -> float:
    kpi = self.sim.get_kpi()
    delta_tp = self.sim.total_completed - self._prev_throughput
    delta_deadlock = kpi["deadlocks"] - self._prev_deadlocks
    self._prev_throughput = self.sim.total_completed
    self._prev_deadlocks = kpi["deadlocks"]
    return delta_tp * 10.0 - delta_deadlock * 100.0  # 대기시간 항 제거
```
그 다음 `step()` 에서 WIP 패널티 추가.

**B안.** 서영 방식처럼 `_delta_reward()` 자체를 쓰지 않고 reward 함수를 처음부터 직접 작성:
```python
def _delta_reward(self) -> float:
    return 0.0  # 비활성화

def step(self, action):
    obs, _, done, info = super().step(action)
    reward = self._my_reward()  # 직접 계산
    return obs, reward, done, info
```

---

## 6. 권장 합의 구조 요약

```
공통 레이어 (4명 모두 동일)
├── 완료 보상: HOT×20, NORMAL×10
├── WIP 패널티: weighted_queue_wait × WAIT_SCALE
│              (HOT=3.0, NORMAL=1.0, WAIT_SCALE=0.005 시작)
└── 데드락: ×100

개인 레이어 (각자 자유롭게 추가)
├── 서영: zone_imbalance × 0.05, empty_dist × 0.03, idle_penalty × 0.2
├── 태희: (공통만으로 WipAware 전략 완성, 필요 시 zone shaping 추가)
├── s3 담당: zone-aware dispatch shaping
└── s4 담당: priority + wait 기반 urgency shaping
```

---

## 7. 첫 실험 체크리스트

- [ ] `WAIT_SCALE`을 0.005로 맞추고 100 에피소드 실행
- [ ] 스텝당 평균 reward 범위 확인 (-1 ~ +3 목표)
- [ ] 태희 버그 3개 수정 후 단독 실행 확인
- [ ] `auto_dispatch=False` 확인 (필수)
- [ ] KPI 키 통일 확인 후 비교 표 작성
- [ ] sb3_contrib 설치: `pip install sb3-contrib` (MaskablePPO용)
