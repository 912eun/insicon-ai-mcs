# AMR 시뮬레이터 현황 및 RL 가이드라인

> 작성 기준: 2026-05-28 (M2 완료 시점)

---

## 1. 아키텍처 개요

```
PySCFabSim (공정 시뮬레이터)
    │  공정 완료 이벤트 (on_step_done)
    ▼
AMRTransportHook
    │  from_port / to_port 변환 (STNFAM → port_name)
    ▼
FABSimulator.add_request()
    │  Priority Queue (HOT / NORMAL)
    ▼
Dispatcher ── (현재: auto_dispatch or RL action)
    │  A* with Reservation Table
    ▼
AMR.assign(path, step_costs)
    │  MOVING → LOADING → MOVING → UNLOADING → DONE
    ▼
FABSimulator.step()  ←── gym 환경 루프
    │  충돌 감지·해소, 데드락 탐지·재경로
    ▼
KPI: throughput / avg_queue_time / avg_travel_time / utilization / deadlocks
```

**두 가지 실행 모드:**

| 환경 | 클래스 | 반송 요청 출처 | 용도 |
|---|---|---|---|
| 독립 실행 | `FabAMREnv` | 랜덤 생성 (`request_interval`) | RL 사전 학습, 단독 테스트 |
| 통합 실행 | `CoupledFabEnv` | PySCFabSim 공정 이벤트 | 실제 공정 연동, 최종 실험 |

---

## 2. 구현 컴포넌트

### 2-1. 경로 탐색 (`pathfinding.py`)

| 함수 | 설명 |
|---|---|
| `astar()` | 기본 A*. 대각선 코너 컷팅 방지 |
| `astar_with_reservation()` | Time-extended A*. 예약 테이블 충돌 회피, stay-in-place 비용 0.5 |
| `compute_step_costs()` | SMAT2022 acc/dec 기반 속도 프로파일. 경로 세그먼트별 실제 소요 스텝 반환 |

**속도 프로파일 수치 (SMAT2022 기준):**

```
cell_size = 2000 mm,  v_max = 1500 mm/s,  acc = 2000 mm/s²,  dec = 3000 mm/s²

경로 길이    첫 셀      중간 셀    마지막 셀   평균
1셀          1.469       -           -          1.469  (+46.9%)
10셀         1.281      1.000       1.188       1.052  (+5.2%)
50셀         1.281      1.000       1.188       1.010  (+1.0%)
```

### 2-2. AMR 상태 기계 (`simulator.py`)

```
IDLE ──dispatch()──▶ MOVING
                        │ pickup_idx 도달
                        ▼
                     LOADING (load_steps = 8스텝 ≈ 10초)
                        │ 완료
                        ▼
                     MOVING (드롭오프 방향)
                        │ path 끝 도달
                        ▼
                     UNLOADING (unload_steps = 8스텝)
                        │ 완료
                        ▼
                      DONE ──reset_idle()──▶ IDLE
```

**Reservation Table:**
- Vertex conflict: `(x, y, abs_t)` 중복 방지
- Edge conflict: swap 이동 방지
- `release(amr_id)`: 완료 AMR 예약 해제 (메모리 누수 방지)
- 예약 기준 시간: 절대 스텝 (현재_스텝 + 경로_상대_시간)

### 2-3. Gym 환경 (`env.py`)

**Observation** (n_amrs×3 + 10×6 + 7 = 82 for n_amrs=5):
```
AMR features:   [x/W, y/H, is_idle]  × n_amrs
Pending queue:  [fx/W, fy/H, tx/W, ty/H, priority, wait/100] × MAX_PENDING (10)
Summary:        [n_idle/N, n_pending/20, throughput/100, NW/N, NE/N, SW/N, SE/N]
```

**Action:** `Discrete(n_amrs + 1)` — 0=no-op, k=AMR k번에 큐 맨 앞 요청 배정

**Reward (dense):**
```python
Δ_throughput × 10.0
- Δ_avg_queue_time × 0.5
- Δ_deadlocks × 100.0
```

### 2-4. 공정 연동 (`coupled_env.py`, `transport_hook.py`)

- **시간 동기화**: AMR 1스텝 = 1.333초. PySCFabSim을 매 스텝 1.333초 진행
- **포트 매핑**: `SMT2020 STNFAM` → `SMAT2022 tool 필드` → `port_name` (랜덤 1개)
- **Hot Lot**: `lot.priority > 0` → `Priority.HOT`

---

## 3. 데이터 소스

| 파일 | 출처 | 내용 |
|---|---|---|
| `rware_simulator/output/grid_map.npy` | SMAT2022 LVHM_E | 86×159 그리드 (장애물 포함) |
| `rware_simulator/output/port_locations.json` | SMAT2022 | 1116개 포트 좌표 + tool 매핑 |
| `rware_simulator/output/fab_layout.json` | SMAT2022 | 차량 파라미터(acc/dec/v_max) + 메타 |
| `external/PySCFabSim/datasets/SMT2020_HVLM` | SMT2020 | 고부하·저기계 시나리오 |
| `external/PySCFabSim/datasets/SMT2020_LVHM` | SMT2020 | 저부하·고기계 시나리오 |

> SMAT2022는 OHT 기준 데이터지만, 레이아웃(그리드·포트 위치)과 차량 파라미터는  
> AMR 시뮬레이션에 그대로 재활용 가능. RWARE는 설계 참고용이었으며 실행 시 미사용.

---

## 4. 수정 이력 (버그 및 개선)

| 구분 | 위치 | 내용 |
|---|---|---|
| 버그 수정 | `simulator.py:_replan()` | 드롭오프 재경로 시 `pickup_idx`가 항상 `len(path)-1`로 세팅되던 문제. 드롭오프 재경로에서 `pickup_idx=-1`로 수정 |
| 버그 수정 | `env.py:_delta_reward()` | 데드락 누적값을 매 스텝 패널티로 사용해 RL 학습 왜곡. delta 방식으로 수정 |
| 개선 | `coupled_env.py:_advance_fab()` | 결정 포인트당 기계 1대만 배정하던 `break` 제거. PySCFabSim greedy 패턴으로 통일 |
| 개선 | `pathfinding.py` | `compute_step_costs()` 추가 — SMAT2022 acc/dec 기반 trapezoidal 속도 프로파일 |
| 개선 | `simulator.py`, `env.py` | AMR에 `_step_frac` 분수 스텝 누적 방식 추가. 가속·감속 구간의 실제 소요 시간 반영 |
| 개선 | `coupled_env.py` | `fab_layout.json`에서 차량 파라미터 자동 로딩 |

---

## 5. 개선 가능점

### 5-1. 경로 계획 (단기 — M3 전)

**① CBS(Conflict-Based Search) 미구현**
현재는 time-extended A*로 각 AMR이 독립적으로 경로를 잡고, 충돌 시 rollback으로 해소한다. CBS를 쓰면 모든 AMR의 경로를 전역적으로 최적화해 충돌이 애초에 없는 경로를 보장할 수 있다.

```python
# 현재: 각 AMR이 reservation table 보면서 독립 계획
path = astar_with_reservation(grid, start, goal, reservation)

# CBS: High-level search over conflict tree, Low-level per-agent A*
# mcs/cbs/ 디렉토리가 이미 준비되어 있음
```

**② Reservation Table의 정수 시간 가정**
가속·감속 모델 도입 후 실제 도달 시간이 분수 스텝이지만 예약 테이블은 여전히 정수 시간 기준이다. 충돌 예측 정확도가 낮아질 수 있다.

### 5-2. 물리 모델 (중기)

**③ 방향(Heading) 미반영**
실제 AMR은 방향이 있고, 회전에 시간이 걸린다. 현재는 방향 전환 비용이 없다.  
특히 직각 회전 시 감속 → 회전 → 가속 사이클이 추가됨.

**④ 최소 안전 거리 (`min_distance = 125mm`)**
`fab_layout.json`에 있지만 현재 미사용. Reservation Table이 같은 셀을 같은 시간에 두 AMR이 점유하지 못하게만 막고, 연속적인 거리 유지는 처리하지 않는다.

**⑤ 포트 동시 점유 제한**
현재 여러 AMR이 같은 포트에 동시에 LOADING/UNLOADING할 수 있다. 실제 Fab에서는 포트 1개에 AMR 1대만 접근 가능.

```python
# dispatch()에서 포트 점유 확인 추가 필요
if self._port_occupied.get(req.from_port):
    heapq.heappush(self.request_queue, ...)
    return False
```

### 5-3. 성능 (스케일)

**⑥ O(N²) 충돌 감지**
`_resolve_collisions()`가 이동 중인 모든 AMR 쌍을 검사한다. SMAT2022 기준 500대 AMR이면 약 125,000쌍 → 병목.  
공간 분할(grid hash) 방식으로 O(N) 근사 가능.

```python
# 현재
for i in range(len(moving)):
    for j in range(i+1, len(moving)):  # O(N²)

# 개선: 셀 기반 hash
cell_occupants = defaultdict(list)
for amr in moving:
    cell_occupants[(amr.x, amr.y)].append(amr)
```

**⑦ Reservation Table 메모리**
현재 예약 항목이 `(x, y, t)` 키로 무한 누적된다. 과거 시간의 항목은 의미 없으므로 `current_step - window` 이전 항목 주기적 삭제 필요.

### 5-4. 시뮬레이션 현실성

**⑧ 충돌 해소 전략 단순화**
현재 충돌 시 높은 id를 가진 AMR을 이전 셀로 rollback한다. 실제로는 대기 우선순위 (요청 우선순위, 대기 시간)을 고려해야 한다.

**⑨ 랜덤 반송 요청 → SMT2020 연동**
`FabAMREnv`의 기본 모드는 랜덤 요청 생성 (`request_interval`)이다. RL 사전 학습에는 유용하지만, 실제 Fab 부하 패턴을 반영하지 못한다. `CoupledFabEnv`를 기본으로 쓰는 것이 정확하다.

---

## 6. RL 적용 가이드라인

### 6-1. 문제 정의

우리가 RL로 푸는 문제: **"반송 요청이 생길 때마다, 어떤 IDLE AMR에 배정할 것인가?"**

- 경로 탐색(A*)과 충돌 회피(Reservation Table)는 이미 해결됨
- RL은 dispatch 결정만 담당

### 6-2. 환경 선택

```python
# 1단계: FabAMREnv로 빠른 반복 학습 (랜덤 요청, 공정 연동 없음)
from simulator.amr.env import FabAMREnv
env = FabAMREnv(n_amrs=20, max_steps=2000, seed=42)

# 2단계: CoupledFabEnv로 실제 공정 연동 학습
from simulator.fab_process.smat2022.coupled_env import CoupledFabEnv
env = CoupledFabEnv(n_amrs=20, sim_days=7, max_steps=50000)
```

### 6-3. Action Masking (필수)

IDLE이 아닌 AMR에 배정하면 dispatch가 실패한다. 이 무효 액션을 마스킹해야 학습이 안정된다.

```python
from sb3_contrib import MaskablePPO

class FabAMREnvWithMask(FabAMREnv):
    def action_masks(self) -> list[bool]:
        idle_ids = {a.id for a in self.sim.get_idle_amrs()}
        # 0 = no-op (항상 유효), k = AMR k (IDLE일 때만 유효)
        return [True] + [i in idle_ids for i in range(1, self._n_amrs + 1)]

env = FabAMREnvWithMask(n_amrs=20, max_steps=2000)
model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
```

### 6-4. Baseline 구현 (비교 기준)

```python
# FIFO: 가장 오래 기다린 요청 → 가장 가까운 IDLE AMR
def fifo_nearest(sim: FABSimulator) -> int:
    if not sim.request_queue or not sim.get_idle_amrs():
        return 0  # no-op
    req = sim.get_next_request()
    from_loc = sim.port_locations[req.from_port]
    best = min(
        sim.get_idle_amrs(),
        key=lambda a: abs(a.x - from_loc["grid_x"]) + abs(a.y - from_loc["grid_y"])
    )
    return best.id

# EDD: 마감 임박 요청 우선 (Hot Lot = 최우선)
# Priority Queue가 이미 HOT=0, NORMAL=1로 정렬하므로
# 큐 맨 앞 = EDD에 가까운 동작
```

### 6-5. 학습 전략

**단계별 커리큘럼:**

```
Phase 1: FabAMREnv, n_amrs=5, max_steps=500
         → 기본 dispatch 개념 학습

Phase 2: FabAMREnv, n_amrs=10, max_steps=1000
         → 스케일 적응

Phase 3: FabAMREnv, n_amrs=20, max_steps=2000
         → 실험 스케일

Phase 4: CoupledFabEnv, SMT2020_LVHM, 7일
         → 실제 공정 부하 패턴 적응

Phase 5: CoupledFabEnv, SMT2020_HVLM, 30일
         → 고부하 시나리오 최종 평가
```

**Reward 튜닝 포인트:**

```python
# 현재 reward
delta_tp * 10.0          # throughput 증가
- delta_wait * 0.5        # 평균 대기시간 증가
- delta_deadlock * 100.0  # 데드락 발생

# 추가 고려
- delta_pending * 0.1     # 큐 적체 방지
+ delta_utilization * 0.1 # AMR 가동률 장려
```

### 6-6. 관찰 공간 확장 (선택)

현재 관찰에 없지만 추가하면 성능이 오를 수 있는 정보:

| 피처 | 설명 | 기대 효과 |
|---|---|---|
| AMR 잔여 경로 길이 | 현재 임무 남은 스텝 수 | 빨리 끝나는 AMR에 배정 |
| 포트별 혼잡도 | 특정 구역 AMR 밀집도 | 혼잡 회피 routing |
| 요청 대기시간 분포 | 큐 상위 5개 요청의 waiting_time | 긴급 요청 우선 처리 |
| 데드락 위험 구역 | 최근 재경로 발생 구역 | 사전 회피 |

### 6-7. 팀원별 전략 분기점 (`mcs/dispatcher/strategies/`)

4인 RL 전략 비교를 위한 차별화 방향:

| 전략 | 핵심 아이디어 | 특징 |
|---|---|---|
| s1: Nearest-First | 현재 위치 기준 Manhattan 거리 최소 AMR 배정 | 단순, 빠른 수렴 |
| s2: Load-Balance | AMR 별 누적 이동 거리 균등화 | 가동률 편차 최소화 |
| s3: Zone-Aware | 구역 밀집도 + 요청 위치 기반 | 혼잡 예방 |
| s4: Priority+Wait | Hot Lot 우선 + 대기시간 가중 | 마감 준수 |

각 전략은 `BaseDispatcher.select_amr()` 구현. 동일 환경(seed 고정)에서 100 episode 실행 후 KPI 비교.

---

## 7. KPI 정의 및 측정

```python
kpi = sim.get_kpi()
# {
#   "throughput":      완료된 반송 건수
#   "avg_queue_time":  요청 생성 → AMR 배정 평균 시간 (스텝)
#   "avg_travel_time": AMR 배정 → 반송 완료 평균 시간 (스텝)
#   "avg_total_time":  queue_time + travel_time
#   "utilization":     현재 MOVING/LOADING/UNLOADING 중인 AMR 비율 (%)
#   "pending":         미배정 요청 수
#   "deadlocks":       누적 데드락 발생 횟수
# }
```

**스텝 → 실제 시간 변환:**
```
1 step = cell_size / v_max = 2000 / 1500 ≈ 1.33초
avg_travel_time 120 step ≈ 160초 ≈ 2.7분
```

---

## 8. 현재 실행 가능 범위 (M2 기준)

```bash
# FABSimulator 단독 (conda activate ai-mcs)
python -c "
from simulator.amr.simulator import FABSimulator
import numpy as np, json
grid = np.load('rware_simulator/output/grid_map.npy')
ports = json.load(open('rware_simulator/output/port_locations.json'))
sim = FABSimulator(grid, ports, n_amrs=20)
for _ in range(1000): sim.step()
print(sim.get_kpi())
"

# FabAMREnv gym 환경
python -c "
from simulator.amr.env import FabAMREnv
env = FabAMREnv(n_amrs=20, max_steps=1000)
obs = env.reset()
for _ in range(1000):
    obs, r, done, info = env.step(env.action_space.sample())
    if done: break
print(info)
"

# CoupledFabEnv 공정 연동
python -c "
from simulator.fab_process.smat2022.coupled_env import CoupledFabEnv
env = CoupledFabEnv(n_amrs=20, sim_days=7)
obs = env.reset()
for _ in range(10000):
    obs, r, done, info = env.step(env.action_space.sample())
    if done: break
print(info)
"
```

---

*다음 단계 (M3, 6/7): 각 모듈 단독 실행 확인 + RL 4인 전략 작성 시작*
