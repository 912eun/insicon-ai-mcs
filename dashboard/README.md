# FAB 디지털 트윈 · LLM 관제 대시보드

RL 학습과 **병렬로** 돌리는, human-in-the-loop 운영 관제 화면.
공장 평면도 위에서 AMR이 실시간으로 움직이는 디지털 트윈 + 이상징후 발생 시 LLM이
관리자에게 한국어로 보고하는 피드로 구성된다.

## 핵심 설계: 데이터는 이미 다 나와 있다

시뮬레이터(`FABSimulator`)가 매 스텝 두 가지를 내보낸다 — 새 계측 코드가 필요 없다.

| 출처 | 용도 |
|---|---|
| `sim.get_state()` | AMR `{id,x,y,status}`, pending 큐, zone 밀도 → **트윈 렌더링** |
| `env.step()` info = `get_kpi()` | deadlock / HOT SLA / 배터리 등 → **KPI 차트 · LLM 보고** |

## 구조 (3계층)

```
sim_source.py ──▶ recorder.py (JSONL 녹화)  ┐
   get_state()                              ├─▶ server.py ──▶ static/ (캔버스 트윈)
   get_kpi()  ──▶ (live 직접 스텝) ──────────┘     │ WS        KPI 패널
                                                  │           LLM 보고 피드 (HITL)
                              llm/watcher.py (임계 탐지) → llm/report.py (Claude 보고)
```

- **replay**: 미리 녹화한 `runs/<name>.jsonl` 재생 — 데모 안전망, 가볍다.
- **live**: `FABSimulator`를 실시간 스텝 — 진짜 트윈, 무겁다.
- 둘 다 **같은 스냅샷 스키마 + 같은 WS 계약** → 프론트는 소스를 구분하지 않는다.

## 실행

```bash
conda activate ai-mcs

# 1) (선택) replay 녹화 — 데모 안정성용
#   AMR 대수: 적을수록 가동률↑·시각적으로 덜 붐빔(40대 ≈ util 97%). 많으면 유휴가 많아 보임.
python -m dashboard.recorder --name demo --n-amrs 40 --max-steps 1200
# 배터리 데모: coupled 충전 env는 sim_days까지 돌아 수만 스텝 → --max-snaps로 자르고
#   --battery-scale로 방전을 가속(충전 🟣/방전 🔴 장면이 일찍 나타남; 너무 빨간색이면 scale↓).
python -m dashboard.recorder --name demo_batt --battery --n-amrs 40 \
    --max-steps 5000 --battery-scale 24 --max-snaps 1600

# 2) 서버
uvicorn dashboard.server:app --port 8000
#   http://localhost:8000                    (소스 드롭다운에서 선택)
#   http://localhost:8000/?source=live       (라이브 트윈 바로)
#   http://localhost:8000/?source=live:battery (라이브 배터리 — LIVE_BATTERY_SCALE 기본 24로 가속)
```

## LLM 보고 (Gemini via Vertex AI · gcloud ADC)

- 트리거: `llm/watcher.py`가 **실제 KPI 필드**로 임계 위반을 잡는다
  (deadlock_per_1k, hot_sla_rate, pending/AMR 비율, blocking_per_1k, availability).
- **역할 분담**(환각 방지): **발생 상황 진술 = 규칙 기반**(`report_alert`의 `situation` — sim이 확정한
  사실·숫자, LLM 미관여) / **LLM = 판단 도움**(`advice` — 주어진 사실+고정 옵션 위에서 무엇을 왜
  우선할지 조언만, 사실 재기술·숫자 창작 금지). 보고 카드는 `📐 규칙 감지` + `🤖 판단 도움`으로 분리 표시.
- 모델: 조언/보고는 `gemini-2.5-flash`(저비용), 관리자 후속질문(HITL)은 `gemini-2.5-pro`.
- 인증은 **gcloud ADC** — 프로젝트는 ADC 기본값 자동 사용(env로 덮어쓰기 가능). 인증 없으면 **규칙기반 폴백**(대시보드는 안 깨짐).
- 스트리밍은 논블로킹: LLM 호출은 백그라운드 태스크로 돌고 단일 writer가 송신 → 보고 생성이 트윈 프레임을 멈추지 않음.

```bash
gcloud auth application-default login     # 1회 (브라우저 로그인)
# 선택: 프로젝트/리전/모델 덮어쓰기
export GOOGLE_CLOUD_PROJECT=<proj>        # 없으면 ADC 기본 프로젝트
export GOOGLE_CLOUD_LOCATION=us-central1  # 기본값
export GEMINI_REPORT_MODEL=gemini-2.5-flash
export GEMINI_QA_MODEL=gemini-2.5-pro
```

> 카드 우측 배지: `🟢 LLM`(Gemini 응답) / `⚪ rule`(폴백). SDK: `google-genai`.

## Human-in-the-loop — 결정 루프 (핵심 차별점)

선행조사 결과: AMR 실시간 시각화는 상용 다수(ABB·KUKA·OMRON), LLM 이상탐지+보고는
논문 단계(SRE/전력망 도메인)지만 **"AMR + LLM + 관리자 의사결정"을 하나로 통합**한 사례는
시장·논문 모두 없음. 특히 ④⑤(HITL autonomous control / SRE `detect→analyze→report→select`)의
**"LLM이 조치 옵션 제시 → 관리자가 선택"** 루프를 AMR 도메인에 이식한 게 우리 기여다.

1. **질의응답**: 보고 아래 입력창에 자연어로 물으면(예: "NE 구역이 왜 막혔나?")
   LLM이 그 시점 스냅샷을 근거로 답한다.
2. **결정 루프**: 모든 보고 카드에 **조치 선택지**가 버튼으로 붙는다. 선택지는 LLM이
   자유생성하지 않고 **시뮬레이터의 실제 레버에 묶인 결정형**(`llm/report.py`의
   `ACTION_TEMPLATES` — early_replan / charge_threshold / fleet 증차 / 우선순위 가중 등).
   관리자가 버튼을 누르면 서버가 `runs/decisions.jsonl`에 기록(`/api/decisions`로 조회).

임계치(`llm/watcher.py`의 `TH`)·조치 매핑은 팀 튜닝 대상. (현재 조치는 *기록*만 — sim
파라미터 자동 적용은 다음 단계: `lever` 키를 sim 런타임에 연결.)

## 선행연구 대비 포지셔닝

| 축 | 선행 | 우리 |
|---|---|---|
| AMR 실시간 시각화 | ABB/KUKA/OMRON 상용, OpenAMR_UI(ROS·단일로봇) | 76대 fab 톱다운 트윈, 벤더무관(sim 직결) |
| LLM 이상탐지·보고 | LEMAD/AD-Copilot/SRE(타도메인) | transport KPI 기반, FAB AMR 도메인 |
| **관리자 결정 루프** | **사례 없음(AMR)** | **옵션 제시→선택→로그, 실제 레버 결합** |

> 참고 UI 패턴(OpenAMR_UI): 페이지 분리(Map/Route/Info), 원형 게이지 텔레메트리,
> 시스템 메시지 브로드캐스트 — 단 ROS·단일로봇 구조라 코드 채택이 아닌 패턴 참고.

## 파일

| 파일 | 역할 |
|---|---|
| `dashboard/sim_source.py` | 스냅샷 제너레이터 (단일 데이터 계약), 정책 주입 |
| `dashboard/recorder.py` | 에피소드 녹화 → `runs/*.jsonl` + `*.meta.json` |
| `dashboard/server.py` | FastAPI: `/api/meta`, `/api/runs`, `/ws`, HITL ask |
| `dashboard/static/` | 캔버스 트윈 + KPI 패널 + 보고 피드 (바닐라 JS) |
| `llm/watcher.py` | KPI 임계 이상징후 탐지 → `Alert` |
| `llm/report.py` | `Alert`/질문 → Claude 보고 (폴백 포함) |
