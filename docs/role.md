# 역할 분담

## 현재 상황 (5/25 기준)

Git 이전 단계. 세 명이 로컬에서 환경 탐색 중.

| 사람 | 현재 작업 | 상태 |
|------|---------|------|
| 팀원 1 | RWARE 연동 방법 탐색 | 로컬 탐색 중 |
| 팀원 2 | PySCFabSim 로컬 실행 | 라이브러리 버전 충돌 해결 중 |
| 팀원 3 | SMAT2022 반송 데이터 → PySCFabSim 연동 탐색 | 로컬 탐색 중 |

세 가지 다 해결되면 git으로 합치고, **RL은 4명이 각자 다른 전략으로 구현 후 최적 선택**.

---

## Phase 0 — 로컬 탐색 (현재 ~ M2, 5/28)

Git 없이 각자 로컬에서 진행. M2에 환경 세팅 완료 후 git에 올림.

### 팀원 1 — RWARE

탐색 목표:
- RWARE 기본 환경 실행
- 맵 커스터마이징 방법 파악 (Fab 레이아웃 반영 가능한지)
- PettingZoo 연동 방식 확인 (멀티에이전트 RL)

M2 결과물: RWARE 세팅 방법 + 커스터마이징 범위 문서화

### 팀원 2 — PySCFabSim

탐색 목표:
- 라이브러리 버전 충돌 해결 (`pip install pyscfabsim` 또는 소스 빌드)
- SMT2020 데이터셋으로 시뮬레이션 실행
- 공정 완료 이벤트가 어떻게 발생하는지 확인

M2 결과물: 실행 가능한 환경 + 이벤트 발생 구조 파악

### 팀원 3 — SMAT2022 반송 데이터

탐색 목표:
- SMAT2022 데이터 구조 파악 (장비 위치, 레이아웃 정보)
- SMT2020(공정) + SMAT2022(반송) 어떻게 연결할지 방법 탐색
- RWARE 맵으로 변환 가능한지 확인

M2 결과물: 데이터 구조 문서 + 연결 방안 초안

---

## Phase 1 — Git 합치기 (M2, 5/28)

세 탐색 결과를 git에 올리고 공유.

```
main
└── develop
     ├── feat/amr-rware          # 팀원 1 탐색 결과
     ├── feat/fab-process        # 팀원 2 탐색 결과
     └── feat/smat2022-data      # 팀원 3 탐색 결과
```

각자 브랜치로 올린 후 팀 리뷰 → 인터페이스 계약 합의 → develop merge.

---

## Phase 2 — RL 4인 병렬 구현 (M2 ~ M3, ~6/7)

**4명 전원이 각자 다른 RL 전략으로 구현 → M3에서 비교 → 최적 채택.**

### 디렉토리 구조

```
mcs/dispatcher/
├── base_agent.py          # 공통 인터페이스 — 먼저 합의 필수
├── strategies/
│   ├── s1/                # 팀원 1 전략
│   ├── s2/                # 팀원 2 전략
│   ├── s3/                # 팀원 3 전략
│   └── s4/                # 팀원 4 전략
└── agent.py               # M3 비교 후 최적 채택본
```

### base_agent.py 먼저 합의

4명이 다른 구현을 해도 **같은 인터페이스**를 써야 나중에 비교와 연동이 가능함.

```python
class BaseDispatcher:
    def select_amr(
        self,
        request: TransportRequest,
        amrs: list[AMRStatus]
    ) -> DispatchCommand:
        raise NotImplementedError

    def update(self, obs, reward, done):
        raise NotImplementedError
```

이 인터페이스가 확정된 후 각자 `strategies/s1~s4/` 안에서 자유롭게 구현.

### 각자 다르게 시도할 수 있는 것

| 전략 | 차별화 포인트 |
|------|------------|
| 알고리즘 | PPO / DQN / A2C / SAC |
| Reward | 데드락 패널티 가중치 조정, 처리량 vs 대기시간 비율 |
| State | 혼잡도 표현 방식, 큐 길이 포함 여부 |
| Action | AMR 배정 방식, action masking 범위 |

### M3 비교 기준

| 항목 | 측정 방법 |
|------|---------|
| Throughput | 단위 시간당 처리 Lot 수 |
| 평균 대기시간 | 반송 요청 ~ AMR 도착 |
| 데드락 발생률 | 전체 스텝 대비 발생 횟수 |
| 학습 수렴 속도 | 동일 스텝 수 기준 reward 곡선 |

비교 결과 자체가 실험 데이터 → 논문 ablation study에 활용 가능.

---

## Phase 3 — 통합 및 고도화 (M3 이후)

- 채택된 RL → `mcs/dispatcher/agent.py`로 승격
- CBS 연동: 배정 결정 → 충돌 없는 경로 계산
- 대시보드 구현
- LSTM 예측 모델 개발 (시험 끝나고 M4 이후)
- LLM 운영자 지원 구현 (M5 이후)

---

## 전체 흐름 요약

```
지금 (5/25)
  로컬: RWARE 탐색 / PySCFabSim 버전 충돌 해결 / SMAT2022 데이터 탐색
        ↓ 셋 다 해결
M2 (5/28)
  Git 합치기: feat 브랜치 3개 → develop
  인터페이스 계약 합의 (base_agent.py 포함)
        ↓
M2 ~ M3
  RL 4인 병렬: strategies/s1~s4 각자 구현
  CBS 구현, 실험 프레임워크 준비
        ↓ 성능 비교
M3 (6/7)
  최적 RL 채택 → agent.py
  각 모듈 독립 실행 가능 상태
        ↓
M5 (6/24)
  전체 통합 + LSTM + LLM
```
