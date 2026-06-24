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

## transport_score (팀 단일 공유 Judge)

> **갱신(2026-06-24)**: 팀에 결합 스코어가 둘로 갈렸던 것(이 파일의 비율정규화 ↔ eval_baseline 비용기반)을
> **하나의 공유 함수 `experiments.eval_baseline.transport_score`로 통합**. 이 baseline은 그걸 import해 쓴다.
> (구조·단위 확정, W_* 숫자는 실험 돌리며 팀이 튜닝. 설계 근거: [metrics_and_reward_design.md](metrics_and_reward_design.md))

### 공유 Judge = 배달 1건당 비용 (낮을수록 좋음, 0=최고)

모든 항을 **"배달 1건당 손실 step"**으로 통일 — robustness(deadlock/blocking/deplete)는 `raw count/throughput`,
efficiency(travel/energy)는 이미 배달당. throughput은 분모라 자동 보상(베이스라인).

```
cost = W_DL·(deadlocks/n) + W_BLOCK·(blocking/n) + W_DEPLETE·(depleted/n)
     + W_TRAV·dist_per_done + W_ENERGY·energy_per_delivery
     + W_SLA·max(0, HOT_SLA_MIN − hot_sla_rate)      # HOT SLA 소프트 게이트(제약)
score = −cost          (n = throughput)
```

| W (실비용 grounding) | 값 | 의미 |
|---|---|---|
| `W_DL` | 5000 | 데드락 1건 = 영구마비 step |
| `W_BLOCK` | 10 | blocking 1건 = ~10 step 지연 |
| `W_TRAV` | 1 | 이동 1셀 = 1 step |
| `W_DEPLETE` | 3000 | 방전 1건 = 견인 다운타임 step |
| `W_ENERGY` | 26 | 1 Ah 충전 = 충전소 대기 step |
| `W_SLA` | 200 | HOT SLA 미달분당 (소프트 제약, 효율 항과 동급 스케일) |

> **배터리 포함** → 충전 RL도 같은 judge로 평가 가능(battery off면 energy/deplete=0 자동).

### 이 baseline의 표현 (FIFO=1.0 비율)

공유 judge는 절대 비용(낮을수록 좋음)이므로, 이 스크립트는 **FIFO 대비 비율**로 표현:
```
transport_score(전략) = cost(FIFO) / cost(전략)      # 1.0=FIFO 동일, >1=전략이 더 우수
```
→ 콘솔/저장 모두 **FIFO-Nearest = 1.0 기준, 높을수록 좋음**. (절대 비용 breakdown은 `eval_baseline.py` 출력 참고.)

> **이력**: 초판은 5지표(throughput·deadlock·replans·dist·util) 비율가중합 + SLA 패널티(5.0/3.1) 였으나,
> **배터리 항이 없어 충전 RL을 평가 못 하는** 한계 → taehee 비용토대(배터리 포함)로 통합하고 비율표현만 계승.

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
    "transport_score_weights": { "deadlock": 5000, "blocking": 10, "travel": 1, "deplete": 3000, "energy": 26, "hot_sla": 200, "hot_sla_min": 0.9 }
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
