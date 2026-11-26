# 시뮬레이터 실행 가이드 (RL 팀원용)

> 작성 기준: 2026-05-30

---

## 0. 전제 조건

```bash
# 프로젝트 루트에서 conda 환경 활성화
conda activate ai-mcs
```

> 모든 명령은 **프로젝트 루트** (`insicon-ai-mcs/`)에서 실행해야 import가 정상 동작합니다.

---

## 1. 동작 확인 (빠른 스모크 테스트)

터미널에서 바로 붙여넣어 실행:

```bash
python -c "
from simulator.amr.env import FabAMREnv
env = FabAMREnv(n_amrs=5, max_steps=200)
obs = env.reset()
while True:
    obs, r, done, info = env.step(env.action_space.sample())
    if done: break
print(info)
"
```

정상 출력 예시:
```
{'step': 200, 'throughput': 3, 'avg_queue_time': 12.3, 'avg_travel_time': 45.1,
 'avg_total_time': 57.4, 'utilization': 72.0, 'pending': 8, 'deadlocks': 0}
```

---

## 2. 전략 작성 패턴

### 2-1. 기본 루프

```python
from simulator.amr.env import FabAMREnv

env = FabAMREnv(n_amrs=20, max_steps=2000, seed=42)
obs = env.reset()

while True:
    action = my_strategy(env.sim)          # ← 여기에 전략 함수 연결
    obs, reward, done, info = env.step(action)
    if done:
        break

print(info)
```

### 2-2. action 규칙

| action 값 | 의미 |
|---|---|
| `0` | no-op (아무것도 안 함) |
| `1 ~ n_amrs` | 해당 번호의 AMR에 큐 맨 앞 요청을 배정 |

### 2-3. 전략 함수가 쓸 수 있는 시뮬레이터 API

```python
sim = env.sim  # FABSimulator 객체

sim.get_idle_amrs()      # IDLE 상태인 AMR 목록 반환
sim.get_next_request()   # 큐 맨 앞 요청 반환 (from_port, to_port, priority)
sim.port_locations       # {port_name: {"grid_x": int, "grid_y": int, ...}}
sim.amrs                 # 전체 AMR 목록 (a.x, a.y, a.status, a.id 접근 가능)
sim.request_queue        # 현재 대기 중인 반송 요청 목록
sim.current_step         # 현재 스텝
```

### 2-4. 전략 함수 최소 예시 (FIFO-Nearest)

```python
def my_strategy(sim) -> int:
    idle = sim.get_idle_amrs()
    if not idle or not sim.request_queue:
        return 0  # 배정할 AMR 또는 요청이 없으면 no-op
    req = sim.get_next_request()
    loc = sim.port_locations[req.from_port]
    best = min(idle, key=lambda a: abs(a.x - loc["grid_x"]) + abs(a.y - loc["grid_y"]))
    return best.id
```

---

## 3. 내 전략 파일 위치

각자 담당 디렉토리에 작성합니다. 남의 디렉토리는 PR 없이 수정 금지.

```
mcs/dispatcher/strategies/
├── s1/   # Nearest-First
├── s2/   # Load-Balance
├── s3/   # Zone-Aware
└── s4/   # Priority+Wait
```

`__init__.py`에 전략 클래스/함수를 작성하면 됩니다.

```python
# mcs/dispatcher/strategies/s1/__init__.py 예시

from simulator.amr.simulator import FABSimulator

class MyDispatcher:
    def select(self, sim: FABSimulator) -> int:
        # 전략 로직 작성
        ...
```

---

## 4. 환경 파라미터

```python
FabAMREnv(
    n_amrs=20,           # AMR 대수 (기본 20)
    max_steps=2000,      # 에피소드 최대 스텝
    request_interval=15, # N 스텝마다 요청 1~3개 랜덤 생성
    seed=42,             # 재현성을 위한 고정 시드
)
```

- 1 step ≈ 1.33초 실제 시간
- 스텝 → 실제 시간: `steps × 1.33초`

---

## 5. KPI 항목

`info` 딕셔너리(또는 `sim.get_kpi()`)로 확인:

| 키 | 설명 | 단위 |
|---|---|---|
| `throughput` | 완료된 반송 건수 | 건 |
| `avg_queue_time` | 요청 생성 → AMR 배정 평균 대기 시간 | 스텝 |
| `avg_travel_time` | AMR 배정 → 반송 완료 평균 이동 시간 | 스텝 |
| `avg_total_time` | queue_time + travel_time | 스텝 |
| `utilization` | 이동/적재/하역 중인 AMR 비율 | % |
| `pending` | 미배정 대기 요청 수 | 건 |
| `deadlocks` | 누적 데드락 발생 횟수 | 회 |

---

## 6. 4가지 전략 방향 (문서 6-7 요약)

| 전략 | 핵심 아이디어 | 선택 기준 |
|---|---|---|
| s1: Nearest-First | from_port에 Manhattan 거리 최소 AMR | 거리 |
| s2: Load-Balance | 누적 이동량이 가장 적은 AMR | 누적 작업량 |
| s3: Zone-Aware | 요청 발생 구역과 같은 구역 내 최근접 AMR | 구역 밀집도 |
| s4: Priority+Wait | Hot Lot 우선, 동일 우선순위면 대기시간 가중 | 우선순위 + 대기시간 |

---

## 7. 트러블슈팅

**`ModuleNotFoundError: No module named 'simulator'`**
→ 프로젝트 루트에서 실행하고 있는지 확인. `cd insicon-ai-mcs` 후 재시도.

**`conda: command not found`**
→ Anaconda Prompt(Windows) 또는 터미널 재시작 후 `conda activate ai-mcs`.

**action을 줘도 배정이 안 됨**
→ IDLE AMR이 없거나 큐가 비어 있으면 `dispatch()`가 무시됩니다. `sim.get_idle_amrs()`로 먼저 확인.
