# 실제 환경 AMR 대수 분석

> 작성 기준: 2026-06-05
> 데이터셋: SMT2020_LVHM (PySCFabSim 공정 시뮬레이터)

---

## 1. 배경

지금까지 실험은 `n_amrs=20`으로 진행했으나, 이는 RL 학습용 실험 스케일이다.
실제 공정 부하(SMT2020_LVHM) 기준 적정 AMR 대수를 측정하고 그에 맞춰 실험을 재설계할 필요가 있다.

---

## 2. SMT2020_LVHM 규모

| 항목 | 수치 |
|---|---|
| 장비 종류 | 106가지 |
| 총 장비(포트) 대수 | **1,313대** |
| 초기 WIP lots | 2,157개 |
| 포트 좌표 (port_locations.json) | 1,116개 |

---

## 3. 요청 발생률 측정 (1일치)

`CoupledFabEnv`에 AMR 1대만 두고 항상 `no-op`으로 진행, 요청만 쌓이도록 함.

| 시각 | 누적 요청 |
|---|---|
| 1시간 | 2,359건 |
| 5시간 | 5,263건 |
| 9시간 | 8,702건 |
| 13시간 | 12,084건 |
| 17시간 | 15,671건 |
| 21시간 | 19,193건 |
| **24시간** | **21,915건** |

- **시간당 평균 발생률**: 약 913건/시간
- **분당 발생률**: 약 15건/분

---

## 4. 적정 AMR 대수 계산

```
AMR 1대 일 처리량
  = 86,400초 / (avg_travel_time × 1.33초/스텝)
  = 86,400 / (180 × 1.33)
  ≈ 360건/대/일

적정 AMR 수
  = 일 요청량 / (AMR 1대 처리량 × 목표 가동률)
  = 21,915 / (360 × 0.80)
  ≈ 76대
```

| 시나리오 | AMR 수 | 의도 |
|---|---|---|
| **부족 (병목)** | 50대 | 큐가 쌓이는 상황 |
| **권장 (적정)** | **76대** | 80% 가동률 |
| **여유** | 100대 | 데드락 최소화 검증 |

---

## 5. 차량 수 데이터는 별도로 없다

PySCFabSim 데이터셋 (SMT2020_LVHM)에는 아래만 있고 **AMR/차량 대수 정보는 없다**.

```
attach.txt    — lot 라우팅
order.txt     — lot 주문
part.txt      — 부품 정보
route_*.txt   — 공정 경로
tool.txt.1l   — 장비 정보 (장비 1,313대)
WIP.txt       — 현재 진행 중인 lot
fromto.txt    — 장비 간 이동 시간 (상수 가정)
```

원래 PySCFabSim은 **반송 시간을 fromto.txt의 상수로 처리**한다. AMR 대수 개념 자체가 없다.

README 인용:
> "기존 연구는 반송 시간을 상수로 가정했다. 우리는 AMR 위치·혼잡도·데드락에 따라 이동 시간이 동적으로 결정되는 환경을 직접 시뮬레이션한다. **이것이 차별점이다.**"

→ 우리 프로젝트의 핵심 contribution이 "AMR 대수와 동적 이동 시간을 모델링하는 것"이므로,
   AMR 수는 데이터셋이 아니라 우리가 정해야 한다.

---

## 6. 다음 단계

- [ ] `n_amrs=76`으로 학습 환경 변경
- [ ] FA 모델 재학습 (200k → 1M, 체크포인트 저장)
- [ ] 부족 (50대) / 적정 (76대) / 여유 (100대) 3개 시나리오 비교
- [ ] CoupledFabEnv 통합 검증 (Phase 4)

---

## 7. 측정 스크립트

재현용 코드:

```python
from simulator.fab_process.smat2022.coupled_env import CoupledFabEnv

env = CoupledFabEnv(
    dataset_dir='external/PySCFabSim/datasets/SMT2020_LVHM',
    n_amrs=1,          # AMR 수 무관, 요청만 측정
    sim_days=1,
    max_steps=200000,
    seed=42,
)
obs = env.reset()
while True:
    obs, r, done, info = env.step(0)  # 항상 no-op
    if done or info['fab_time_hours'] >= 24:
        break

total_requests = len(env._inner.sim.request_queue) + info['throughput']
print(f"24시간 누적 요청: {total_requests}건")
```
