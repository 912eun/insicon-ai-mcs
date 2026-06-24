# Baseline Evaluation — FAB AMR Transport

> **파일**: `prediction/baseline.py`  
> **목적**: RL 학습 전 Rule-based 디스패처 3종의 KPI를 측정하여 RL 개선 폭의 기준선을 확보한다.

---

## 개요

FABSimulator(`auto_dispatch=False`)를 직접 구동하며 3가지 휴리스틱 전략의 성능을 비교한다.  
5개 시드(seed) 평균으로 결과를 산출하므로 랜덤 초기 배치 편차가 상쇄된다.

---

## 전략 설명

### 1. FIFO-Nearest

| 항목 | 내용 |
|------|------|
| 요청 선택 | 큐 전체에서 **req.id 최소** (가장 먼저 생성된 요청, **priority 무시** = 순수 FIFO) |
| AMR 선택 | 선택된 요청의 `from_port`까지 **맨해튼 거리 가장 가까운** idle AMR 1대 |
| 배정 빈도 | **1스텝 1배정** — 매 스텝 최대 1쌍만 배정 후 `sim.step()` |
| 특징 | 가장 보수적인 기준선. 우선순위 무시 → HOT SLA에 불리 |

```python
for step in range(MAX_STEPS):
    idle = sim.get_idle_amrs()
    if idle and sim.request_queue:
        req = min(queue, key=lambda r: r.id)   # FIFO: priority 무관
        nearest = argmin(idle, manhattan(a, req.from_port))
        sim.dispatch(nearest.id, request_id=req.id)  # 1대만
    sim.step()
```

### 2. EDD-Nearest (Earliest Due Date)

| 항목 | 내용 |
|------|------|
| 요청 선택 | **HOT 우선**, 같은 priority 내에서는 **대기시간이 긴 요청** 먼저 |
| AMR 선택 | 선택된 요청의 `from_port`까지 맨해튼 거리 가장 가까운 idle AMR |
| 배정 빈도 | **매 스텝 전부** — idle AMR 이 있는 동안 반복 배정 (`while` 루프) |
| 특징 | HOT SLA 달성률 향상 기대. FIFO보다 처리량도 높을 가능성 있음 |

```python
req = min(queue, key=(priority.value, -waiting_time))
nearest = argmin(idle, manhattan(a, req.from_port))
sim.dispatch(nearest.id, request_id=req.id)
# 매 스텝 idle AMR 소진될 때까지 반복
```

> **FIFO vs EDD 배정 빈도 차이**: FIFO는 의도적으로 1스텝 1배정으로 제한(보수적 기준선),
> EDD/Random은 모든 idle AMR을 최대한 활용한다. 처리량 비교 시 이 차이를 감안할 것.

### 3. Random

| 항목 | 내용 |
|------|------|
| 요청 선택 | 우선순위 큐 맨 앞 (heapq 순서 = HOT 우선 → id 순) |
| AMR 선택 | idle AMR 중 **균등 랜덤** |
| 배정 빈도 | 매 스텝 전부 (`while` 루프) |
| 특징 | 최하위 기준선. Random보다 낮은 RL은 의미 없음 |

---

## 실험 설정

| 파라미터 | 값 |
|----------|----|
| `n_amrs` | 20 |
| `max_steps` | 500 |
| `request_interval` | 15 |
| `seeds` | [42, 99, 123, 256, 512] |
| `auto_dispatch` | False (전략이 직접 dispatch) |
| `battery_enabled` | False |

---

## KPI 항목

`FABSimulator.get_kpi()` 기반에 추가 지표를 직접 계산해 덧붙인다.

| KPI | 출처 | 설명 | 단위 | 방향 |
|-----|------|------|------|------|
| `throughput` | `get_kpi()` | 완료된 반송 건수 | 건 | ↑ |
| `avg_queue_time` | `get_kpi()` | 요청 생성 → AMR 배정 평균 대기 | step | ↓ |
| `avg_travel_time` | `get_kpi()` | AMR 배정 → 반송 완료 평균 이동 | step | ↓ |
| `avg_total_time` | `get_kpi()` | 큐 대기 + 이동 합계 | step | ↓ |
| `utilization` | `get_kpi()` | 전체 AMR-step 중 이동/적재/하역 비율 | % | ↑ |
| `pending` | `get_kpi()` | 에피소드 종료 시 미처리 요청 수 | 건 | ↓ |
| `deadlocks` | `get_kpi()` | 누적 데드락 발생 횟수 | 회 | ↓ |
| `deadlock_per_1k` | `get_kpi()` | 1000 step당 데드락 수 | — | ↓ |
| `blocking_per_1k` | `get_kpi()` | 1000 step당 충돌 롤백 누적 | — | ↓ |
| `replans_aborts` | **직접 계산** | `replan_count + abort_count` 합산 | 회 | ↓ |
| `hot_sla_rate` | **직접 계산** | HOT 완료 건 중 queue_time ≤ 30 step 비율 | 0~1 | ↑ |
| `hot_done` / `normal_done` | `get_kpi()` | 우선순위별 완료 건수 | 건 | ↑ |
| `hot_queue_time` | **직접 계산** | HOT 완료 건 평균 큐 대기 | step | ↓ |
| `normal_queue_time` | **직접 계산** | NORMAL 완료 건 평균 큐 대기 | step | ↓ |
| `hot_travel_time` | **직접 계산** | HOT 완료 건 평균 이동 시간 | step | ↓ |
| `normal_travel_time` | **직접 계산** | NORMAL 완료 건 평균 이동 시간 | step | ↓ |
| `dist_per_done` | `get_kpi()` | 완료 건당 이동 셀 수 (에너지 프록시) | cell | ↓ |
| `total_travel_dist` | `get_kpi()` | 누적 이동 셀 수 | cell | ↓ |
| `availability` | **None** | 배터리 미구현 → 측정 불가 | — | — |
| `energy_per_delivery` | **None** | 배터리 미구현 → 측정 불가 | — | — |
| `depleted_per_1k` | **None** | 배터리 미구현 → 측정 불가 | — | — |
| `reward` | `get_kpi()` | 시뮬레이터 내장 보상 합계 | — | ↑ |

> **"직접 계산"** 항목은 `sim.completed_requests` 리스트를 순회해 계산한다.  
> `replans_aborts`는 `get_kpi()`에 없는 신규 합산 키이다.  
> 배터리 관련 3개는 `battery_enabled=False`일 때 `get_kpi()`가 반환하는 `1.0`/`0` 대신
> 명시적으로 `None`으로 기록한다.

---

## transport_score

**FIFO-Nearest = 1.0** 기준. 핵심 지표 5개를 비율로 정규화한 **가중 합산** 점수.

### 정규화 방식

| 지표 | 방향 | 정규화 비율 |
|------|------|-------------|
| `throughput` | ↑ | strategy / FIFO |
| `deadlock_per_1k` | ↓ | FIFO / strategy |
| `replans_aborts` | ↓ | FIFO / strategy |
| `dist_per_done` | ↓ | FIFO / strategy |
| `utilization` | ↑ | strategy / FIFO |

모든 비율은 FIFO 대비 **1.0 = 동일**, **> 1.0 = 전략이 더 좋음** 을 의미한다.

### 계산 공식

```
raw = 1.0 × r(throughput)
    + 1.0 × r(deadlock_per_1k)
    + 0.5 × r(replans_aborts)
    + 0.3 × r(dist_per_done)
    + 0.3 × r(utilization)

normalized = raw / fifo_raw          # FIFO 자신 → raw = 3.1 → normalized = 1.0

if hot_sla_rate < 0.9:
    normalized -= 5.0 / 3.1         # ≈ -1.61 패널티 (정규화 스케일)
```

FIFO는 자기 자신과 비교하므로 모든 비율 = 1.0 → raw = 3.1 → normalized = **1.0** 이 보장된다.

> **이전 설계(기하평균 8개)와 차이**:  
> 초판은 `avg_queue_time`, `avg_total_time`, `blocking_per_1k`, `hot_sla_rate` 포함 8개 기하평균.  
> 현재는 교통지옥 핵심 5개 가중합 + SLA 패널티로 단순화.  
> 기하평균은 기준 raw가 음수일 때 정규화 부호가 역전되는 버그가 있어 양수 비율 가중합으로 교체.

---

## 출력 / 저장

| 출력 | 내용 |
|------|------|
| 콘솔 (전략별) | KPI 전 항목 + transport_score 상세 출력 |
| 콘솔 (비교표) | 전략 간 KPI 비교 테이블 |
| 파일 | `prediction/data/baseline_results.json` |

JSON 구조:
```json
{
  "config": {
    "n_amrs": 20, "max_steps": 500,
    "seeds": [42, 99, 123, 256, 512],
    "baseline_ref": "FIFO-Nearest",
    "transport_score_weights": { "throughput": 1.0, "deadlock_per_1k": 1.0, ... }
  },
  "results": {
    "FIFO-Nearest": { "throughput": ..., "replans_aborts": ..., "hot_sla_rate": ..., ... },
    "EDD-Nearest":  { ... },
    "Random":       { ... }
  },
  "transport_score": {
    "FIFO-Nearest": 1.0,
    "EDD-Nearest":  1.XXXX,
    "Random":       0.XXXX
  }
}
```

---

## 실행 방법

```bash
cd insicon-ai-mcs
python -m prediction.baseline
```

예상 소요 시간: 약 5~15분 (grid 크기·A* 경로 탐색 횟수에 따라 다름)

---

## 기대 해석 가이드

| 상황 | 해석 |
|------|------|
| EDD > FIFO (transport_score) | HOT 우선 처리가 실제로 유리함 → HOT SLA를 RL 보상에 반영할 가치 있음 |
| EDD ≈ FIFO | 이 시나리오에서 HOT 비율이 낮아 차이가 없음 |
| Random ≈ FIFO | 거리 기반 배정의 이득이 미미함 → 혼잡·예약 테이블 활용이 더 중요 |
| deadlock/1k 높음 | 밀집 환경 → RL 개입 여지 큼 |
| hot_sla_rate < 0.9 | SLA 패널티 발동 → RL은 HOT 응답 속도 개선 필수 |

> **RL 에이전트 목표**: transport_score > EDD-Nearest (가장 강한 baseline)
