"""
server — FAB 디지털 트윈 + LLM 관제 대시보드 (FastAPI + WebSocket).

같은 WS 계약 뒤에 두 소스를 둔다:
    replay : 미리 녹화한 dashboard/runs/<name>.jsonl 재생  (데모 안전망, 가벼움)
    live   : FABSimulator를 실시간 스텝               (진짜 트윈, 무거움)

WS 메시지 프로토콜
    S→C  {"type":"snapshot", "data": <snap>}
         {"type":"report",   "data": {title,body,severity,step,kind,source}}
         {"type":"answer",   "qid", "data": {body,source}}
         {"type":"end"}
    C→S  {"type":"ask", "qid", "question": "..."}
         {"type":"control", "action":"pause"|"play"|"speed", "value": <fps>}

실행 (sim 환경):
    conda activate ai-mcs
    uvicorn dashboard.server:app --reload --port 8000
    # 브라우저: http://localhost:8000  (?source=replay:demo 또는 ?source=live)
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from dashboard.sim_source import export_meta, make_env, snapshots  # noqa: E402
from llm.report import answer_question, report_alert  # noqa: E402
from llm.watcher import Watcher  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
STATIC = HERE / "static"

app = FastAPI(title="FAB Digital Twin")


@app.middleware("http")
async def _no_cache(request, call_next):
    """정적 자산(css/js/html)을 브라우저가 캐싱하지 못하게 강제 — UI 수정이 즉시 반영되도록."""
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


# ── 메타 (공장 평면도) ─────────────────────────────────────────────
@app.get("/api/meta")
def api_meta(source: str = "replay:demo"):
    """프론트 배경용 grid+ports. replay면 녹화 meta, live면 즉석 생성."""
    if source.startswith("replay:"):
        name = source.split(":", 1)[1]
        meta = RUNS / f"{name}.meta.json"
        if not meta.exists():
            return JSONResponse({"error": f"no meta for run '{name}'"}, status_code=404)
        return FileResponse(meta)
    # live: 메모리에 한 번 생성해 반환
    tmp = RUNS / "_live.meta.json"
    export_meta(tmp, oht="oht" in source)
    return FileResponse(tmp)


def _log_decision(rec: dict) -> None:
    """관리자 결정을 추적 로그에 남긴다 (runs/decisions.jsonl). 감사·평가용."""
    import time as _t
    rec = {"ts": _t.strftime("%Y-%m-%dT%H:%M:%S"), **rec}
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(RUNS / "decisions.jsonl", "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@app.get("/api/decisions")
def api_decisions():
    """관리자 결정 이력."""
    path = RUNS / "decisions.jsonl"
    if not path.exists():
        return {"decisions": []}
    with open(path) as f:
        return {"decisions": [json.loads(x) for x in f if x.strip()]}


@app.get("/api/runs")
def api_runs():
    """녹화된 replay 목록."""
    if not RUNS.exists():
        return {"runs": []}
    return {"runs": sorted(p.stem for p in RUNS.glob("*.jsonl"))}


# ── 스냅샷 소스 (replay / live) → 동기 제너레이터 ──────────────────
def _replay_gen(name: str):
    path = RUNS / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no replay '{name}' at {path}")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# 라이브 배터리: scale=1이면 방전이 느려 충전/방전이 안 보이고, 너무 크면(>~10) 충전소
# 용량(3x3)을 못 따라가 대량 방전(death-spiral)이 난다. 6 = 충전 장면은 또렷하되 방전 0.
LIVE_BATTERY_SCALE = float(os.getenv("LIVE_BATTERY_SCALE", "6"))


def _live_gen(n_amrs: int, max_steps: int, battery: bool, oht: bool):
    scale = LIVE_BATTERY_SCALE if battery else 1.0
    env = make_env(n_amrs=n_amrs, max_steps=max_steps, oht=oht, battery=battery,
                   battery_scale=scale)
    yield from snapshots(n_amrs=n_amrs, max_steps=max_steps, oht=oht,
                         battery=battery, battery_scale=scale, env=env)


def _threaded(gen, out_q: queue.Queue, stop: threading.Event):
    """동기 제너레이터(특히 무거운 live 스텝)를 스레드에서 돌려 큐로 흘린다."""
    try:
        for snap in gen:
            if stop.is_set():
                break
            out_q.put(snap)
    except Exception as e:  # noqa: BLE001
        out_q.put({"__error__": str(e)})
    finally:
        out_q.put(None)  # sentinel = 끝


def _parse_source(source: str):
    """'replay:demo' / 'live' / 'live:battery' → (gen_factory, n_amrs)."""
    if source.startswith("replay:"):
        name = source.split(":", 1)[1]
        return (lambda: _replay_gen(name)), _infer_n_amrs(name)
    # live (옵션은 ':'로: live:battery, live:oht)
    opts = source.split(":")[1:]
    battery = "battery" in opts
    oht = "oht" in opts
    n_amrs = 76
    return (lambda: _live_gen(n_amrs, 4000, battery, oht)), n_amrs


def _infer_n_amrs(name: str) -> int:
    """replay 첫 줄에서 AMR 수를 읽어 watcher 비율 임계에 사용."""
    path = RUNS / f"{name}.jsonl"
    try:
        with open(path) as f:
            return len(json.loads(f.readline())["amrs"])
    except Exception:  # noqa: BLE001
        return 76


# ── WebSocket 스트림 ───────────────────────────────────────────────
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    params = websocket.query_params
    source = params.get("source", "replay:demo")
    fps = float(params.get("fps", "8"))

    try:
        gen_factory, n_amrs = _parse_source(source)
    except Exception as e:  # noqa: BLE001
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close()
        return

    watcher = Watcher(n_amrs=n_amrs)
    state = {"paused": False, "delay": 1.0 / max(fps, 0.5),
             "last": None, "kpi": {}}

    in_q: queue.Queue = queue.Queue(maxsize=64)   # sim 스레드 → 스냅샷
    outq: asyncio.Queue = asyncio.Queue()         # → 클라이언트 (단일 writer가 직렬 송신)
    stop = threading.Event()
    worker = threading.Thread(
        target=_threaded, args=(gen_factory(), in_q, stop), daemon=True)
    worker.start()
    loop = asyncio.get_event_loop()

    # 단일 writer: 모든 송신을 직렬화 (동시 send_json 충돌 방지).
    async def writer():
        while True:
            msg = await outq.get()
            if msg is None:
                break
            await websocket.send_json(msg)

    # LLM 호출은 *백그라운드 태스크*로 — 프레임 스트리밍을 절대 막지 않는다(버벅임 원인 제거).
    async def make_report(alert, kpi):
        rep = await loop.run_in_executor(None, report_alert, alert, kpi)
        await outq.put({"type": "report", "data": rep})

    async def make_answer(question, snap, kpi, qid):
        ans = await loop.run_in_executor(None, answer_question, question, snap, kpi)
        await outq.put({"type": "answer", "qid": qid, "data": ans})

    bg_tasks: set = set()

    def _spawn(coro):
        t = asyncio.create_task(coro)
        bg_tasks.add(t)
        t.add_done_callback(bg_tasks.discard)

    async def handle_client():
        """관리자 입력(질문/제어) 수신 — 어느 것도 스트림을 막지 않음."""
        try:
            while True:
                msg = await websocket.receive_json()
                t = msg.get("type")
                if t == "control":
                    act = msg.get("action")
                    if act == "pause":
                        state["paused"] = True
                    elif act == "play":
                        state["paused"] = False
                    elif act == "speed":
                        state["delay"] = 1.0 / max(float(msg.get("value", 8)), 0.5)
                elif t == "ask":
                    _spawn(make_answer(msg.get("question", ""),
                                       state["last"] or {}, state["kpi"], msg.get("qid")))
                elif t == "decision":
                    # HITL 결정 루프: 관리자가 LLM 제시 조치 중 하나를 선택 → 로그·ack
                    rec = {
                        "step": state["kpi"].get("step"),
                        "kind": msg.get("kind"),
                        "action_id": msg.get("action_id"),
                        "label": msg.get("label"),
                        "source": source,
                    }
                    _log_decision(rec)
                    await outq.put({"type": "decision_ack", "data": rec})
        except (WebSocketDisconnect, RuntimeError):
            stop.set()

    writer_task = asyncio.create_task(writer())
    client_task = asyncio.create_task(handle_client())

    try:
        while True:
            if state["paused"]:
                await asyncio.sleep(0.1)
                continue
            snap = await loop.run_in_executor(None, in_q.get)
            if snap is None:
                await outq.put({"type": "end"})
                break
            if "__error__" in snap:
                await outq.put({"type": "error", "message": snap["__error__"]})
                break

            state["last"] = snap
            state["kpi"] = snap.get("kpi", {})
            await outq.put({"type": "snapshot", "data": snap})

            # 이상징후 → LLM 보고를 백그라운드로 띄우고 즉시 다음 프레임 진행
            for alert in watcher.update(snap["kpi"], snap):
                _spawn(make_report(alert, snap["kpi"]))

            await asyncio.sleep(state["delay"])
    except WebSocketDisconnect:
        pass
    finally:
        stop.set()
        await outq.put(None)       # writer 종료
        client_task.cancel()
        writer_task.cancel()
        for t in bg_tasks:
            t.cancel()


# ── 정적 프론트엔드 ────────────────────────────────────────────────
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
