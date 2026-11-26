"""
watcher — KPI 스트림 위 이상징후 탐지 (LLM 보고의 트리거).

설계 원칙: 스트림 전체를 LLM에 흘리지 않는다. 비용·노이즈 때문. watcher가 *실제 KPI 필드*로
임계 조건을 잡고, 위반이 새로 발생한 순간에만 Alert를 만든다. LLM은 그 Alert를 받아
관리자용 보고문을 쓴다(llm/report.py).

각 Alert는 실제 simulator KPI(get_kpi) 필드에 직접 묶여 있다:
    deadlock_per_1k, hot_sla_rate, pending, depleted, blocking_per_1k, availability
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Alert:
    kind: str           # 'deadlock' | 'hot_sla' | 'queue' | 'battery' | 'blocking'
    severity: str       # 'warning' | 'critical'
    title: str          # 한 줄 요약 (LLM 없이도 보이는 폴백)
    metric: str         # 위반한 KPI 키
    value: float
    threshold: float
    step: int
    context: dict = field(default_factory=dict)  # zone_density 등 LLM 보고 근거


# ── 임계치 (팀 튜닝 대상; eval_baseline의 transport_score 가중과 정합) ──
TH = {
    "deadlock_per_1k": 5.0,    # 1k 스텝당 데드락 — 0이 정상, 급증=경로/예약 문제
    "hot_sla_rate":    0.90,   # HOT SLA 충족률 하한 (이 아래 = SLA 붕괴)
    "pending_ratio":   8.0,    # pending / n_amrs — 용량 대비 큐 적체 배수
    "blocking_per_1k": 50.0,   # blocking(충돌 대기) 급증
    "availability":    0.70,   # 배터리: 가용 fleet 비율 하한
}

# 같은 종류 Alert를 매 스텝 쏟지 않도록 쿨다운(스텝)
COOLDOWN = 150


class Watcher:
    """스텝마다 update(kpi, snapshot)를 호출하면 새로 발생한 Alert만 반환."""

    def __init__(self, n_amrs: int, th: Optional[dict] = None, cooldown: int = COOLDOWN):
        self.n_amrs = max(n_amrs, 1)
        self.th = {**TH, **(th or {})}
        self.cooldown = cooldown
        self._last_fired: dict[str, int] = {}

    def _fire_ok(self, kind: str, step: int) -> bool:
        last = self._last_fired.get(kind)
        if last is not None and step - last < self.cooldown:
            return False
        self._last_fired[kind] = step
        return True

    def update(self, kpi: dict, snapshot: Optional[dict] = None) -> list[Alert]:
        step = int(kpi.get("step", 0))
        ctx = {}
        if snapshot:
            ctx = {
                "zone_density": snapshot.get("zone_density"),
                "n_pending": snapshot.get("n_pending"),
                "top_pending": snapshot.get("pending", [])[:5],
            }
        alerts: list[Alert] = []

        def g(k, default=0.0):
            v = kpi.get(k)
            return default if v is None else v

        # ① 데드락 — 가장 비싼 실패 (transport_score W_DL=5000)
        dpk = g("deadlock_per_1k")
        if dpk > self.th["deadlock_per_1k"] and self._fire_ok("deadlock", step):
            alerts.append(Alert(
                "deadlock", "critical",
                f"데드락 급증: 1k스텝당 {dpk:.1f}건 (임계 {self.th['deadlock_per_1k']})",
                "deadlock_per_1k", dpk, self.th["deadlock_per_1k"], step, ctx,
            ))

        # ② HOT SLA 붕괴 — 소프트 제약 위반
        sla = g("hot_sla_rate", 1.0)
        if sla < self.th["hot_sla_rate"] and self._fire_ok("hot_sla", step):
            alerts.append(Alert(
                "hot_sla", "critical" if sla < 0.8 else "warning",
                f"HOT SLA 충족률 {sla:.0%} (목표 {self.th['hot_sla_rate']:.0%} 미달)",
                "hot_sla_rate", sla, self.th["hot_sla_rate"], step, ctx,
            ))

        # ③ 큐 적체 — 수요>용량 신호
        pending = g("pending")
        ratio = pending / self.n_amrs
        if ratio > self.th["pending_ratio"] and self._fire_ok("queue", step):
            alerts.append(Alert(
                "queue", "warning",
                f"반송 큐 적체: {int(pending)}건 (AMR {self.n_amrs}대 대비 {ratio:.1f}배)",
                "pending", pending, self.th["pending_ratio"] * self.n_amrs, step, ctx,
            ))

        # ④ blocking 급증 — 데드락 전조
        bpk = g("blocking_per_1k")
        if bpk > self.th["blocking_per_1k"] and self._fire_ok("blocking", step):
            alerts.append(Alert(
                "blocking", "warning",
                f"충돌 대기(blocking) 급증: 1k스텝당 {bpk:.0f}건",
                "blocking_per_1k", bpk, self.th["blocking_per_1k"], step, ctx,
            ))

        # ⑤ 배터리 가용성 (배터리 on일 때만 유의미)
        avail = g("availability", 1.0)
        depleted = g("depleted")
        if (avail < self.th["availability"] or depleted > 0) and self._fire_ok("battery", step):
            sev = "critical" if depleted > 0 else "warning"
            msg = (f"AMR {int(depleted)}대 방전" if depleted > 0
                   else f"fleet 가용성 {avail:.0%}로 저하")
            alerts.append(Alert(
                "battery", sev, f"배터리 위험: {msg}",
                "availability", avail, self.th["availability"], step, ctx,
            ))

        return alerts
