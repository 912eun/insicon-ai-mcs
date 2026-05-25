# AI MCS — 반도체 Fab AMR 자율 관제 시스템

PySCFabSim(공정) + RWARE(AMR 이동)를 연결하고, RL 기반 MCS로 AMR 배정을 최적화한다.

## 구조

```
simulator/      PySCFabSim 래퍼 + RWARE AMR + 인터페이스
mcs/            RL 디스패처 (4인 전략 비교) + CBS 경로 계산
experiments/    Baseline + 실험 프레임워크
prediction/     LSTM 혼잡 예측 (M4 이후)
llm/            LLM 운영자 지원 (M5 이후)
```

자세한 내용은 `docs/` 참고.

## 환경 세팅

```bash
git clone --recurse-submodules <repo-url>
pip install -r requirements.txt
```

> PySCFabSim submodule은 팀원 2가 버전 충돌 해결 후 추가 예정.
> `external/` 디렉토리에 위치.

## 브랜치

- `main`: 마일스톤 태그 (v0.1 / v0.2 / v1.0)
- `develop`: 통합 브랜치
- `feat/*`: 기능 브랜치

## 마일스톤

| 마일스톤 | 날짜 | 목표 |
|---------|------|------|
| M2 | 5/28 | 환경 세팅 완료, git 합치기 |
| M3 | 6/7 | 각 모듈 독립 실행, RL 전략 비교 |
| M5 | 6/24 | 전체 통합 |
| M7 | 6/29 | 발표 |
