## [Simulator] SMAT2022 기반 AMR 반송 시뮬레이터 구현

### 작업 개요
SMAT2022 실제 반도체 FAB 데이터를 기반으로
AMR 자유주행 반송 시뮬레이터를 구현했습니다.
RL 팀은 FABSimulator 클래스에 PPO 모델만 연결하면 됩니다.

---

### 구현 내용

#### 1단계: SMAT2022 데이터 변환 (`04_smat_loader.ipynb`)
- SMAT2022 엑셀(LVHM_E) → fab_layout.json 변환
- Equipment(1116개), Port(22120개), Bay(42개), Vehicle(500대) 파싱
- 공장 좌표(mm) → 격자 좌표(2m 단위) 변환
- 결과: fab_layout.json

#### 2단계: AMR Grid Map 생성 (`05_grid_map.ipynb`)
- Equipment → obstacle 셀 등록
- Port → pickup/dropoff station 등록
- 나머지 빈 공간 → AMR 이동 가능 영역
- 결과: grid_map.npy (0=이동가능, 1=obstacle, 2=포트)
- 결과: port_locations.json

#### 3단계: A* 경로 탐색 (`06_path_planner.ipynb`)
- 단일 AMR A* 알고리즘 구현
- 8방향 이동 (상하좌우 + 대각선)
- 단일/다중 AMR 경로 탐색 검증

#### 4단계: 다중 AMR 충돌 회피 (`07_collision_manager.ipynb`)
- Reservation Table 기반 시공간 충돌 회피
- Vertex Conflict 방지 (같은 시간 같은 셀)
- Edge Conflict 방지 (서로 위치 교환)
- 시간축 포함 A* (astar_with_reservation) 구현

#### 5단계: 반송 요청 큐 + FABSimulator (`08_transport_queue.ipynb`)
- TransportRequest 클래스 (HOT/NORMAL 우선순위)
- AMR 클래스 (상태 관리: idle/moving/done)
- FABSimulator 클래스 구현
  - 반송 요청 자동 생성 (주기적)
  - 우선순위 큐 (HOT Lot 먼저 처리)
  - FIFO 기본 디스패처 (RL로 교체 예정)
  - KPI 측정 (처리량/대기시간/가동률/데드락)
  - RL 팀원용 인터페이스 제공

---

### FIFO Baseline 성능 (RL 비교 기준)

| 항목 | 값 |
|---|---|
| AMR 대수 | 20대 |
| 총 스텝 | 500 |
| 요청 간격 | 15스텝 |
| Throughput | 71건 |
| 평균 대기시간 | 153스텝 |
| AMR 가동률 | 80% |
| 데드락 | 0건 |

PPO 모델이 이 수치보다 개선되면 성공입니다.

---

### RL 팀 사용 가이드

#### 설치
```python
pip install numpy
```

#### 기본 사용법
```python
import numpy as np
import json

# 환경 로드
grid_map = np.load('simulator/output/grid_map.npy')
with open('simulator/output/port_locations.json') as f:
    port_locations = json.load(f)

# 시뮬레이터 생성
sim = FABSimulator(
    grid_map         = grid_map,
    port_locations   = port_locations,
    n_amrs           = 20,
    request_interval = 15,
    seed             = 42,
)

# 에피소드 루프
state = sim.reset()

for step in range(1000):
    state      = sim.get_state()
    idle_amrs  = sim.get_idle_amrs()

    if idle_amrs and sim.request_queue:
        # ← 이 부분만 PPO로 교체
        amr_id = ppo.predict(state)
        sim.dispatch(amr_id)

    state  = sim.step()
    reward = sim.get_reward()
    kpi    = sim.get_kpi()
```

#### State 구조 (PPO observation)
```python
{
  "step": 현재 스텝,
  "amrs": [
    {
      "id": 1,
      "x": 26, "y": 67,
      "status": "idle/moving",
      "is_idle": True/False
    }
  ],
  "pending_queue": [
    {
      "id": 1,
      "from_port": "EQP_A",
      "to_port": "EQP_B",
      "priority": "HOT/NORMAL",
      "waiting_time": 10
    }
  ],
  "zone_density": {
    "NW": 3, "NE": 2,
    "SW": 5, "SE": 4
  },
  "n_idle_amrs": 3,
  "n_pending": 5,
  "throughput": 10,
}
```

#### 주요 메서드
| 메서드 | 설명 |
|---|---|
| `sim.reset()` | 초기화, 초기 state 반환 |
| `sim.get_state()` | 현재 state 반환 (PPO observation) |
| `sim.get_idle_amrs()` | idle AMR 목록 반환 |
| `sim.get_next_request()` | 큐 맨 앞 요청 반환 |
| `sim.dispatch(amr_id)` | AMR 배정 실행 (PPO action) |
| `sim.step()` | 1스텝 진행 |
| `sim.get_reward()` | 보상 반환 (PPO reward) |
| `sim.get_kpi()` | KPI 딕셔너리 반환 |

#### 보상 함수 구조
```python
reward = (
    throughput * 10       # 처리량 증가 → +보상
    - avg_waiting * 0.5   # 대기시간 감소 → +보상
    - deadlocks * 100     # 데드락 → 패널티
)
```

---

### 추후 수정/개선 사항

#### 🔴 우선순위 높음

Port 좌표 정확도 개선
현재: 장비 중심 근처 빈 셀로 추정
개선: SMAT2022 Address/Rail 시트에서
실제 포트 좌표 계산
AMR 속도 모델 추가
현재: 매 스텝 1칸 이동 (단순화)
개선: SMAT2022 VehicleType 기반
실제 속도/가속도 반영
(MAX_SPEED=1500mm/s 기준)


#### 🟡 우선순위 중간

Rail Corridor Cost 추가
현재: 모든 빈 공간 동일 이동 비용
개선: Rail 근처 셀은 비용 낮게 설정
→ AMR이 주요 통로를 따라 이동
→ 더 현실적인 경로
ZCU 기반 병목 구역 표시
현재: 미반영
개선: ZCU 위치를 혼잡 위험 구역으로
grid_map에 별도 레이어 추가
→ LSTM 혼잡 예측 모델에 활용


#### 🟢 우선순위 낮음 (마일스톤 이후)

PySCFabSim 연동
현재: 반송 요청이 랜덤 생성
개선: PySCFabSim 공정 완료 이벤트
→ 실제 반송 요청 자동 생성
→ 공정 완료 시간이 동적으로 결정
→ 전체 공정 성능에 영향
CBS 알고리즘 교체
현재: Reservation Table (기본)
개선: CBS (Conflict-Based Search)
→ 더 최적의 다중 AMR 경로
→ 데드락 처리 더 강건하게
시각화 고도화
현재: matplotlib 정적 그래프
개선: 실시간 애니메이션 대시보드
AMR 위치/상태 실시간 표시


---

### 파일 구조
simulator/
├── notebooks/
│   ├── 04_smat_loader.ipynb       # SMAT2022 변환
│   ├── 05_grid_map.ipynb          # Grid Map 생성
│   ├── 06_path_planner.ipynb      # A* 경로 탐색
│   ├── 07_collision_manager.ipynb # 충돌 회피
│   └── 08_transport_queue.ipynb   # 시뮬레이터 완성
├── output/
│   ├── fab_layout.json            # FAB 레이아웃
│   ├── grid_map.npy               # Grid Map
│   ├── port_locations.json        # 포트 위치
│   └── simulator_env.json         # 환경 설정/KPI
└── README.md
데이터 다운로드:
SMAT2022 엑셀 파일은 아래에서 받아서
simulator/data/ 폴더에 넣으세요.
https://github.com/kwoo-lee/SMAT2022