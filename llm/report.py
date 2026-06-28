"""
report — Alert/스냅샷을 관리자용 자연어 보고로 변환 (Gemini via Vertex AI).

LLM 제공자 = Google Gemini, **Vertex AI + gcloud ADC**로 인증한다.
    인증: `gcloud auth application-default login` (브라우저 로그인) 1회면 됨.
    프로젝트: 환경변수 GOOGLE_CLOUD_PROJECT, 없으면 ADC 기본 프로젝트 자동 사용.
    리전: GOOGLE_CLOUD_LOCATION (기본 us-central1).

두 경로:
    report_alert(alert, kpi)              자동 보고 — watcher 트리거 시. 빈도 높음 → Flash(저비용).
    answer_question(question, snap, kpi)  HITL 후속질문 — 관리자가 자연어로 물으면. → Pro.

인증/SDK가 없으면 규칙기반 폴백 문장을 돌려준다.
→ 데모/오프라인에서도 대시보드가 깨지지 않는다 (LLM은 enrichment, 필수 의존 아님).
"""

from __future__ import annotations

import json
import os
from typing import Optional

# 자동 보고는 빈도가 높아 저비용 모델, 관리자 Q&A는 더 똑똑한 모델.
MODEL_REPORT = os.getenv("GEMINI_REPORT_MODEL", "gemini-2.5-flash")  # 자동 보고
MODEL_QA = os.getenv("GEMINI_QA_MODEL", "gemini-2.5-pro")           # 관리자 Q&A
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

_CLIENT = None  # genai.Client 싱글톤 (재인증 비용 방지). False=초기화 실패.

# 역할 분담(사용자 방향): 발생 상황 진술 = 규칙 기반(결정적, 환각 0). LLM = '판단 도움'만.
#   → LLM은 사실을 재기술하거나 새 숫자를 만들지 않는다. 주어진 사실+고정 옵션 위에서만 조언.
_ADVICE_SYSTEM = (
    "당신은 반도체 FAB의 AMR(자율반송차) 운영 관리자를 돕는 의사결정 보조다. "
    "감지된 상황(사실)과 가능한 조치 목록이 주어진다. 사실과 숫자는 시스템이 이미 확정했으니 "
    "절대 재기술하거나 새 숫자를 지어내지 마라. 너의 역할은 '판단 도움'뿐 — 주어진 조치 중 "
    "무엇을 왜 우선해야 하는지, 주의점은 무엇인지 한국어로 2~3문장만. 목록에 없는 조치는 권하지 마라."
)

_SYSTEM = (
    "당신은 반도체 FAB의 AMR 운영 관제 보조다. 관리자의 질문에 현재 상태 데이터만 근거로 "
    "한국어로 간결히 답한다. 데이터에 없는 사실을 지어내지 말 것."
)

# ── 조치 선택지 (Human-in-the-loop 결정 루프) ─────────────────────
# 각 옵션은 시뮬레이터의 *실제 레버*에 묶인다 — LLM이 자유생성하지 않고 결정형으로 보장.
#   lever = 운영 파라미터 키 (적용부가 매핑), label/detail = 관리자에게 보이는 설명.
# 선행연구(SRE detect→analyze→report→select / HITL autonomous control)의
# "LLM이 옵션 제시 → 관리자가 선택" 루프를 AMR 도메인에 이식한 부분.
ACTION_TEMPLATES = {
    "deadlock": [
        {"lever": "early_replan", "label": "교착 구역 경로 재계산", "detail": "교착에 걸린 AMR의 경로를 즉시 다시 계산해 배정하고 지연된 예약 충돌을 해소"},
        {"lever": "zone_dispatch_hold", "label": "해당 구역 배차 일시중단", "detail": "혼잡한 구역의 신규 배차를 잠시 멈춰 적체 완화"},
        {"lever": "manual_recover", "label": "작업자 수동 회수", "detail": "교착된 AMR을 작업자가 직접 회수 (최후 수단)"},
    ],
    "hot_sla": [
        {"lever": "hot_pool_boost", "label": "HOT 전용 차량 늘리기", "detail": "노는 AMR 일부를 HOT 작업 전용으로 임시 배정"},
        {"lever": "hot_priority_weight", "label": "HOT 우선순위 상향", "detail": "배차할 때 HOT 작업의 우선순위 비중을 높임"},
        {"lever": "normal_defer", "label": "일반 작업 잠시 후순위", "detail": "NORMAL 반송을 잠시 미뤄 HOT 납기(SLA) 회복"},
    ],
    "queue": [
        {"lever": "fleet_size_up", "label": "운영 차량 대수 검토", "detail": "수요가 처리 용량을 넘어선 상황, 운영 차량 수 늘리기 검토 (근본 해법)"},
        {"lever": "inflow_throttle", "label": "반송 유입량 조절", "detail": "신규 반송 요청이 들어오는 속도를 잠시 제한해 대기열 안정화"},
        {"lever": "idle_rebalance", "label": "유휴 차량 미리 배치", "detail": "노는 AMR을 적체 구역으로 미리 옮겨 둠"},
    ],
    "blocking": [
        {"lever": "congestion_weight", "label": "혼잡 우회 강화", "detail": "경로 탐색에서 혼잡 구간을 더 피하도록 조정해 충돌 대기 감소"},
        {"lever": "corridor_cap", "label": "병목 통로 진입 제한", "detail": "병목이 생기는 통로에 동시에 들어가는 차량 수 제한"},
        {"lever": "early_replan", "label": "선제 경로 재계산", "detail": "충돌 대기는 교착의 전조이므로 미리 경로를 다시 계산해 예방"},
    ],
    "battery": [
        {"lever": "charge_threshold_up", "label": "충전 시작 기준 상향", "detail": "충전을 더 일찍 시작하도록 SoC 기준을 높여 방전 예방"},
        {"lever": "tow_recover", "label": "방전 차량 견인 회수", "detail": "멈춰 선 방전 AMR을 견인해 회수"},
        {"lever": "charger_capacity", "label": "충전소 증설 검토", "detail": "충전 슬롯이 부족해 생기는 대기(경합) 해소 (근본 해법)"},
    ],
}


def actions_for(kind: str) -> list:
    """alert 종류 → 결정형 조치 선택지. id를 붙여 반환(프론트 선택·로그용)."""
    opts = ACTION_TEMPLATES.get(kind, [])
    return [{"id": f"{kind}:{o['lever']}", **o} for o in opts]


def _resolve_project() -> Optional[str]:
    p = os.getenv("GOOGLE_CLOUD_PROJECT")
    if p:
        return p
    try:
        import google.auth
        _, p = google.auth.default()
        return p
    except Exception:  # noqa: BLE001 — ADC 없음
        return None


def _client():
    """Vertex AI Gemini 클라이언트 또는 None (gcloud ADC/SDK 없으면 None). 싱글톤."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT or None  # False → None
    try:
        from google import genai
        proj = _resolve_project()
        if not proj:
            _CLIENT = False
            return None
        _CLIENT = genai.Client(vertexai=True, project=proj, location=LOCATION)
    except Exception:  # noqa: BLE001 — SDK/인증 실패
        _CLIENT = False
        return None
    return _CLIENT


def _generate(model: str, prompt: str, max_tokens: int, system: str = _SYSTEM,
              thinking_budget: int = 0) -> str:
    """Gemini 단발 생성. Gemini 2.5는 기본 'thinking'이 max_output_tokens 예산을 먹어
    본문이 잘리므로, 짧은 보고/조언은 thinking_budget=0(끔)으로 본문 전체를 확보한다.
    (flash=0 가능, pro=최소 128). 호출부에서 예외 처리."""
    from google.genai import types
    client = _client()
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0.3,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        ),
    )
    return (resp.text or "").strip()


def _kpi_brief(kpi: dict) -> str:
    keys = [
        "step", "throughput", "deadlocks", "deadlock_per_1k", "blocking_per_1k",
        "pending", "hot_sla_rate", "hot_queue_time", "dist_per_done",
        "availability", "depleted", "utilization",
    ]
    return json.dumps({k: kpi.get(k) for k in keys if kpi.get(k) is not None},
                      ensure_ascii=False)


# ── 보고 = 규칙 상황(사실) + LLM 조언(판단 도움) ──────────────────────
def _situation(alert, kpi: dict) -> str:
    """발생 상황 진술 — 전부 규칙 기반(결정적). LLM 미관여 → 환각/오류 위험 0."""
    z = (alert.context or {}).get("zone_density")
    facts = [
        alert.title,  # 이미 결정적 (지표·값·임계 포함)
        f"처리량 {kpi.get('throughput', 0)}건 · 대기 {kpi.get('pending', 0)}건 · "
        f"HOT SLA {kpi.get('hot_sla_rate', 1.0):.0%}",
    ]
    if z:
        facts.append(f"구역 분포 {z}")
    if kpi.get("depleted"):
        facts.append(f"방전 {int(kpi['depleted'])}대")
    return " / ".join(facts)


def _advise(alert, kpi: dict, actions: list, situation: str):
    """판단 도움(조언)만 LLM이 생성. 사실·옵션은 위에서 확정 → LLM은 재기술/창작 금지.

    반환 (advice_text|None, source). LLM 불가/실패 시 규칙 폴백(첫 옵션 우선 권고)."""
    if not actions:
        return None, "none"
    client = _client()
    if client is None:
        return f"권고: '{actions[0]['label']}' 우선 검토. {actions[0]['detail']}.", "rule"
    opts = "; ".join(f"[{a['label']}] {a['detail']}" for a in actions)
    prompt = (
        f"감지된 상황(사실, 재기술 금지): {situation}\n"
        f"위반 지표: {alert.metric}={alert.value} (임계 {alert.threshold})\n"
        f"가능한 조치: {opts}\n\n"
        "위 조치 중 무엇을 왜 우선할지와 주의점만 2~3문장으로 조언하라."
    )
    try:
        # thinking_budget=0 → 짧은 조언 본문이 thinking에 먹혀 잘리는 것 방지
        return _generate(MODEL_REPORT, prompt, 600, system=_ADVICE_SYSTEM,
                         thinking_budget=0), "llm"
    except Exception as e:  # noqa: BLE001 — 네트워크/쿼터 실패 → 규칙 폴백
        return f"권고: '{actions[0]['label']}' 우선 검토. (LLM 조언 실패: {e})", "rule"


def report_alert(alert, kpi: dict) -> dict:
    """Alert → 보고 dict. situation=규칙(사실), advice=LLM(판단 도움), actions=결정형 선택지."""
    actions = actions_for(alert.kind)
    situation = _situation(alert, kpi)
    advice, advice_source = _advise(alert, kpi, actions, situation)
    return {
        "title": alert.title,
        "severity": alert.severity,
        "step": alert.step,
        "kind": alert.kind,
        "situation": situation,      # 규칙 기반 사실
        "actions": actions,          # 결정형 조치 선택지 (HITL)
        "advice": advice,            # LLM 판단 도움 (없을 수 있음)
        "source": advice_source,     # advice 출처: llm | rule | none
    }


# ── HITL 후속질문 ──────────────────────────────────────────────────
def answer_question(question: str, snapshot: dict, kpi: dict,
                    history: Optional[list] = None) -> dict:
    """관리자 자연어 질문에 현재 스냅샷 근거로 답한다."""
    client = _client()
    if client is None:
        return {"body": _fallback_answer(question, kpi), "source": "rule"}

    ctx = {
        "kpi": _kpi_brief(kpi),
        "zone_density": snapshot.get("zone_density"),
        "n_pending": snapshot.get("n_pending"),
        "top_pending": snapshot.get("pending", [])[:8],
        "n_amrs": len(snapshot.get("amrs", [])),
    }
    prompt = (f"현재 FAB 상태:\n{json.dumps(ctx, ensure_ascii=False)}\n\n"
              f"질문: {question}\n\n현재 상태 데이터만 근거로 답하라.")
    try:
        # pro는 thinking 완전 비활성 불가(최소 128) — 토큰을 넉넉히 줘 본문 truncation 방지
        return {"body": _generate(MODEL_QA, prompt, max_tokens=1500,
                                  thinking_budget=128), "source": "llm"}
    except Exception as e:  # noqa: BLE001
        return {"body": _fallback_answer(question, kpi) + f"\n(LLM 호출 실패: {e})",
                "source": "rule"}


# ── 폴백 (LLM 없이) ────────────────────────────────────────────────
def _fallback_answer(question: str, kpi: dict) -> str:
    return (f"(LLM 비활성) 질문='{question}'. 현재 KPI 요약: {_kpi_brief(kpi)}. "
            "gcloud ADC 인증 시 LLM이 근거 기반으로 답합니다.")
