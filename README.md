# AI MCS — Autonomous FAB Material Control System

> 반도체 Fab에서 수백 대의 AMR이 교통지옥 없이 웨이퍼를 나를 수 있도록,
> Digital Twin 위에서 학습한 AI 관제탑을 만든다.

---

## 배경

반도체 공장(FAB)에서는 웨이퍼를 수백 개의 공정 장비 사이로 끊임없이 이송해야 한다.
기존에는 천장 레일을 따라 움직이는 **OHT**가 주력이었고, 단순한 규칙(FIFO)으로 충분했다.
하지만 이제 바닥을 자유롭게 돌아다니는 **AMR(자율이동로봇)**이 도입되면서 문제가 생긴다.

| | OHT | AMR |
|--|-----|-----|
| 이동 방식 | 고정 레일 | 자유 경로 |
| 장점 | 제어 단순 | 유연하고 확장 가능 |
| 문제 | 융통성 없음 | **수백 대가 얽히면 데드락 발생** |

기존 Rule-based MCS(FIFO, EDD 등)는 AMR 환경의 동적 혼잡과 데드락을 제어하지 못한다.

---

## 목표

**기존 Rule-based MCS를 AI MCS로 교체한다.**

```
PySCFabSim (공정 시뮬레이터)
        ↓  공정 완료 이벤트
    AI MCS  ←  이 프로젝트
  ┌───┴────┐
  RL 디스패처   혼잡 예측 (LSTM)   LLM 운영자 지원
  "어떤 AMR?"   "어디가 막힐까?"   "지금 상황 요약"
        ↓
    CBS 알고리즘 (충돌 없는 경로)
        ↓
    AMR 이동
        ↓
PySCFabSim (다음 공정 시작)
```

기존 연구는 반송 시간을 상수로 가정했다.
우리는 **AMR 위치·혼잡도·데드락에 따라 이동 시간이 동적으로 결정**되는 환경을 직접 시뮬레이션한다. 이것이 차별점이다.

---

## 시스템 구성

### 시뮬레이터 레이어

| 컴포넌트 | 역할 | 도구 |
|---------|------|------|
| 공정 시뮬레이터 | Lot 투입·공정 순서·기계 배정 | PySCFabSim + SMT2020 |
| AMR 시뮬레이터 | 그리드 맵 위 AMR N대 이동 | RWARE + SMAT2022 레이아웃 |
| 인터페이스 | 공정 완료 이벤트 → 반송 요청 → AMR 이동 → 다음 공정 | 직접 구현 (핵심 contribution) |

### AI 레이어

| 컴포넌트 | 역할 | 방법 |
|---------|------|------|
| RL 디스패처 | 반송 요청 발생 시 최적 AMR 배정 | PPO (stable-baselines3), Action Masking |
| CBS | 배정된 AMR의 충돌 없는 경로 계산 | Conflict-Based Search |
| 혼잡 예측 | 5·15·30분 후 병목 구역 예측 | LSTM / XGBoost |
| LLM 지원 | 이상 상황 자연어 요약 + 조치 권고 | Claude API, Human-in-the-loop |

### 실험 기준

- **Baseline:** FIFO, Nearest Vehicle, EDD
- **시나리오:** HV/LM (고부하/저기계), LV/HM (저부하/고기계)
- **KPI:** Throughput, 평균 대기시간, 가동률, 데드락 발생률
- **반복:** 100 random seed

---

## 디렉토리 구조

```
ai-mcs/
├── external/
│   └── PySCFabSim/               # git submodule (prosysscience/PySCFabSim-release)
│
├── simulator/
│   ├── fab_process/              # PySCFabSim 래퍼
│   │   ├── smt2020/              # SMT2020 데이터셋 설정
│   │   └── events.py             # 공정 완료 이벤트 추출
│   ├── amr/                      # RWARE 기반 AMR 시뮬레이터
│   │   └── maps/                 # SMAT2022 Fab 레이아웃
│   └── interface/                # ★ 핵심 contribution
│       ├── schemas/              # 모듈 간 데이터 포맷 계약
│       ├── event_bridge.py       # PySCFabSim → RWARE 변환
│       └── feedback.py           # AMR 완료 → 다음 공정 시작
│
├── mcs/
│   ├── dispatcher/
│   │   ├── base_agent.py         # 4인 RL 공통 인터페이스
│   │   ├── strategies/           # 팀원별 RL 전략 (비교 후 최적 채택)
│   │   │   ├── s1/ s2/ s3/ s4/
│   │   └── agent.py              # 최종 채택본
│   └── cbs/                      # Conflict-Based Search 경로 계산
│
├── prediction/                   # LSTM / XGBoost 혼잡 예측
├── llm/                          # LLM 운영자 지원
├── experiments/
│   ├── baselines/                # FIFO, EDD, Nearest Vehicle
│   ├── scenarios/                # HV/LM, LV/HM 시나리오
│   └── runner.py                 # 100 seed 실험 프레임워크
├── dashboard/                    # 실시간 모니터링
├── data/
│   ├── smat2022/                 # Fab 레이아웃 원본 (gitignore)
│   └── pipeline/                 # 데이터 수집 파이프라인
├── tests/
│   ├── unit/ integration/ mock/
└── configs/
    ├── smt2020.yaml
    └── fab_layout.yaml
```

---

## 시작하기

### 1. 레포 클론

```bash
git clone --recurse-submodules https://github.com/912eun/insicon-ai-mcs.git
cd insicon-ai-mcs
```

> `--recurse-submodules` 옵션으로 PySCFabSim도 함께 받아진다.
> 이미 클론한 경우: `git submodule update --init`

### 2. 환경 세팅

**스크립트로 자동 세팅 (권장)**

```bash
# macOS / Linux
bash setup.sh

# Windows
setup.bat
```

**직접 세팅**

```bash
conda env create -f environment.yml
conda activate ai-mcs
```

> conda가 없다면 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 설치 후 진행.

### 3. PySCFabSim 실행 확인

```bash
cd external/PySCFabSim
python main.py
```

> `pyinstrument` 빨간줄이 뜨는 경우: `environment.yml`에 포함되어 있으므로 conda 환경이 활성화된 상태인지 확인.

### 4. 데이터 준비

`data/smat2022/` 원본 데이터는 gitignore 대상이므로 별도로 받아야 한다.
팀 공유 경로에서 받아서 `data/smat2022/` 아래에 넣으면 된다.

---

## 브랜치 전략

```
main        마일스톤 태그 (v0.1 / v0.2 / v1.0)
└── develop 통합 브랜치
     ├── feat/amr-rware
     ├── feat/fab-process
     ├── feat/smat2022-data
     ├── feat/rl-s1 ~ s4      팀원별 RL 전략
     ├── feat/cbs
     ├── feat/experiment
     ├── feat/lstm             M4 이후
     └── feat/llm              M5 이후
```

- `main`은 PR로만 merge (마일스톤 달성 시)
- `strategies/s1~s4/`는 각자 소유 디렉토리 — 남의 디렉토리 PR 없이 수정 금지
- `interface/schemas/`, `base_agent.py` 변경 시 전원 리뷰 필수

---

## 마일스톤

| | 날짜 | 목표 |
|--|------|------|
| M2 | 5/28 | 환경 세팅 완료, git 합치기, 인터페이스 계약 확정 |
| M3 | 6/7 | 각 모듈 단독 실행, RL 4인 전략 비교 및 채택 |
| M4 | 6/15 | 모듈 간 연결 테스트, 버퍼 |
| — | 6/16–6/22 | 시험 기간 (동결) |
| M5 | 6/24 | 전체 파이프라인 end-to-end 통합 |
| M6 | 6/27 | KPI 검증, baseline 비교 |
| M7 | 6/29 | 발표 |

**M3이 핵심이다.** 여기서 각 모듈이 독립 실행 가능해야 M5 통합 주에 빠르게 붙일 수 있다.

---

## 기술 스택

| 분야 | 도구 |
|------|------|
| 공정 시뮬레이터 | PySCFabSim, SMT2020 |
| AMR 시뮬레이터 | RWARE, PettingZoo |
| 강화학습 | stable-baselines3 (PPO) |
| 예측 모델 | PyTorch (LSTM), XGBoost |
| LLM | Claude API |
| 실험 관리 | Python, NumPy, Pandas |
