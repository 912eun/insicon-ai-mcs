# Git 구조

## 전제

- 현재(5/25): 로컬 탐색 단계, git 레포 아직 없음
- M2(5/28): 로컬 환경 완성 후 git 시작
- M3(6/7): 각 모듈 독립 실행 가능 → RL 비교 후 채택
- RL은 4명이 각자 전략으로 구현 후 M3에서 최적 선택

---

## PySCFabSim 포함 방식

PySCFabSim (https://github.com/prosysscience/PySCFabSim-release) 은 프로젝트 중심 시뮬레이터.
내부 수정 없이 API만 쓴다면 pip, 커스터마이징이 필요하면 submodule.

**권장: git submodule**
- 팀원마다 pip 버전이 달라지는 문제 방지
- 소스 버전을 레포에 고정 → 재현 가능한 환경 보장
- 현재 팀원 2의 버전 충돌 문제도 submodule로 해결 가능

```bash
# 레포 생성 직후 한 번만 실행
git submodule add https://github.com/prosysscience/PySCFabSim-release external/PySCFabSim

# 팀원이 클론할 때
git clone --recurse-submodules <repo-url>
```

pip으로 쓸 경우에는 submodule 없이 `requirements.txt`에 버전 고정:
```
# requirements.txt
pyscfabsim==x.x.x   # 팀원 2가 동작 확인한 버전으로 고정
```

---

## 디렉토리 구조

```
ai-mcs/
├── external/
│   └── PySCFabSim/           # git submodule (prosysscience/PySCFabSim-release)
│
├── simulator/
│   ├── fab_process/          # PySCFabSim 래퍼 — 우리 코드만
│   │   ├── smt2020/          # SMT2020 설정
│   │   └── events.py         # 공정 완료 이벤트 추출
│   ├── amr/                  # RWARE 기반 AMR
│   │   └── maps/             # SMAT2022 Fab 레이아웃
│   └── interface/            # 핵심 contribution
│       ├── schemas/          # 데이터 포맷 스펙 — 변경은 PR로만
│       ├── event_bridge.py   # PySCFabSim → RWARE 변환
│       └── feedback.py       # AMR 완료 → 다음 공정 시작
│
├── mcs/
│   ├── dispatcher/
│   │   ├── base_agent.py     # 4인 공통 인터페이스 — M2에 먼저 합의
│   │   ├── strategies/
│   │   │   ├── s1/           # 팀원 1 RL 전략
│   │   │   ├── s2/           # 팀원 2 RL 전략
│   │   │   ├── s3/           # 팀원 3 RL 전략
│   │   │   └── s4/           # 팀원 4 RL 전략
│   │   └── agent.py          # M3 비교 후 최적 채택본
│   └── cbs/                  # CBS 경로 계산
│
├── prediction/               # LSTM/XGBoost — M4 이후
├── llm/                      # LLM 운영자 지원 — M5 이후
│
├── experiments/
│   ├── baselines/            # FIFO, EDD, Nearest Vehicle
│   ├── scenarios/            # HV/LM, LV/HM
│   └── runner.py             # 100 seed 비교 프레임워크
│
├── dashboard/
├── data/
│   ├── smat2022/             # .gitignore 대상
│   └── pipeline/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── mock/                 # 시뮬레이터 없이 쓸 mock 데이터
│
├── configs/
│   ├── smt2020.yaml
│   └── fab_layout.yaml
│
└── requirements.txt          # pip 방식 선택 시 버전 고정
```

---

## 브랜치 전략

```
main                          ← 마일스톤 태그 (v0.1 / v0.2 / v1.0)
└── develop                   ← M2에 오픈
     ├── feat/amr-rware        # RWARE 탐색 결과
     ├── feat/fab-process      # PySCFabSim 탐색 결과
     ├── feat/smat2022-data    # SMAT2022 데이터 탐색 결과
     ├── feat/interface        # 인터페이스 연결 (A+B 공동)
     ├── feat/rl-s1            # 팀원 1 RL 전략
     ├── feat/rl-s2            # 팀원 2 RL 전략
     ├── feat/rl-s3            # 팀원 3 RL 전략
     ├── feat/rl-s4            # 팀원 4 RL 전략
     ├── feat/cbs
     ├── feat/experiment
     ├── feat/dashboard
     ├── feat/lstm             # M4 이후
     └── feat/llm              # M5 이후
```

---

## 단계별 git 운용

### Phase 0 — 로컬 탐색 (지금 ~ 5/28)

git 없음. 각자 로컬에서 탐색.

탐색이 끝난 사람부터 결과를 정리해두기:
- 어떤 방법으로 실행했는지
- 어떤 버전 조합이 동작하는지
- 인터페이스로 뭘 뽑아낼 수 있는지

M2에 레포 생성하고 한꺼번에 올림.

---

### Phase 1 — 환경 합치기 (M2, 5/28)

**레포 생성 후 첫 커밋 순서:**

```
1. main 브랜치에 뼈대 커밋
   - 디렉토리 구조
   - .gitignore
   - README

2. PySCFabSim submodule 추가 (main에서)
   git submodule add https://github.com/prosysscience/PySCFabSim-release external/PySCFabSim
   git submodule update --init
   → 팀원 2가 동작 확인한 커밋 해시로 고정:
     cd external/PySCFabSim && git checkout <동작하는 커밋 해시>
     cd ../.. && git add external/PySCFabSim && git commit -m "fix(submodule): PySCFabSim 동작 버전 고정"

3. develop 브랜치 오픈

4. 각자 feat 브랜치 생성 후 로컬 탐색 결과 올리기
   feat/amr-rware     ← RWARE 탐색 결과
   feat/fab-process   ← PySCFabSim 래퍼 코드 (external/ 참조)
   feat/smat2022-data ← 데이터 구조 + 연결 방안

5. 팀 리뷰 후 인터페이스 계약 합의
   - simulator/interface/schemas/ 확정
   - mcs/dispatcher/base_agent.py 확정

6. 합의된 스펙 develop에 merge
```

**클론할 때 (팀원 전원):**
```bash
git clone --recurse-submodules <repo-url>
# 이미 클론한 경우
git submodule update --init
```

---

### Phase 2 — RL 4인 병렬 (M2 ~ M3, ~6/7)

각자 `feat/rl-s1~s4` 브랜치에서 독립 구현.

**충돌 방지 규칙:**
- `strategies/s1~s4/` 는 각자 소유 — 남의 디렉토리 건드리지 않음
- `base_agent.py` 는 변경 시 PR + 전원 리뷰 필수
- 학습 결과물(`.pt`, `runs/`) 은 `.gitignore` 처리

**M3 비교:**
```
feat/rl-s1~s4 → 각자 독립 실행 → 동일 환경·시드로 성능 측정
                                        ↓
                              최적 전략 채택 → agent.py로 승격
                              나머지 전략은 strategies/ 에 보존
                              (비교 결과 = 논문 ablation study)
```

---

### Phase 3 — 통합 (M3 ~ M5)

```
채택된 RL → mcs/dispatcher/agent.py merge
feat/cbs → develop merge
feat/experiment → develop merge
전체 end-to-end 연결 테스트 → tag v0.2

M4 이후: feat/lstm 오픈
M5 이후: feat/llm 오픈 → 전체 통합 → tag v1.0 (발표 버전)
```

---

## .gitignore

```gitignore
# 원본 데이터 (용량 큼)
data/smat2022/
data/raw/

# 학습 결과물
*.pt
*.pkl
*.ckpt
runs/
logs/
results/

# 환경
.env
__pycache__/
*.pyc
.venv/
```

> `data/smat2022/` 다운로드 방법은 `data/README.md`에 명시.

---

## 커밋 컨벤션

```
feat(fab-process): PySCFabSim SMT2020 실행 환경 구성
feat(rl-s1): PPO reward 가중치 v1 구현
feat(rl-s2): DQN + action masking 구현
fix(amr-rware): Fab 맵 커스터마이징 좌표 오류 수정
test(mock): 반송 요청 mock 이벤트 추가
docs(schemas): transport_request 포맷 확정
```
