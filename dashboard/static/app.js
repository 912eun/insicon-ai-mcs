// app.js — 디지털 트윈 캔버스 + KPI + LLM 보고 피드 (바닐라 JS, 의존성 0)

const $ = (id) => document.getElementById(id);
const canvas = $("floor");
const ctx = canvas.getContext("2d");

let ws = null;
let meta = null;          // {grid_h, grid_w, grid[][], ports[]}
let bg = null;            // 배경(grid+ports) 오프스크린 캔버스 — 매 프레임 재계산 방지
let paused = false;
let qid = 0;

// 캔버스는 다크 패널 — 어두운 배경에서 선명하게 보이는 AMR 상태색 (CSS 범례 변수와 일치)
const STATUS_COLOR = {
  idle: "#6b7280", moving: "#34d399", loading: "#fbbf24",
  unloading: "#60a5fa", charging: "#a78bfa", depleted: "#ef4444", done: "#34d399",
};

// ── 인라인 SVG 아이콘 (이모지 대체, 단색 라인 — Toss 스타일) ──────────
// [mode, paths] — mode 'fill'=채움 / 'stroke'=선
const _ICON = {
  pause:  ["stroke", '<rect x="14" y="4" width="4" height="16" rx="1"/><rect x="6" y="4" width="4" height="16" rx="1"/>'],
  play:   ["fill",   '<path d="M6 3v18l15-9z"/>'],
  bolt:   ["fill",   '<path d="M13 2 3 14h9l-1 8 10-12h-9z"/>'],
  box:    ["stroke", '<path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/>'],
  user:   ["stroke", '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>'],
  ai:     ["fill",   '<path d="M12 2l1.9 5.6L19.5 9.5l-5.6 1.9L12 17l-1.9-5.6L4.5 9.5l5.6-1.9z"/><path d="M19 15l.7 2.1 2.1.7-2.1.7L19 21l-.7-2.1-2.1-.7 2.1-.7z"/>'],
  rule:   ["stroke", '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>'],
  alert:  ["stroke", '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>'],
  info:   ["stroke", '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>'],
  send:   ["fill",   '<path d="M22 2 11 13"/><path d="M22 2 15 22l-4-9-9-4z"/>'],
  chevron:["stroke", '<path d="m9 18 6-6-6-6"/>'],
  check:  ["stroke", '<path d="M20 6 9 17l-5-5"/>'],
  bulb:   ["stroke", '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1.3.5 2.6 1.5 3.5.8.8 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>'],
  lever:  ["stroke", '<path d="M20 7h-9"/><path d="M14 17H5"/><circle cx="17" cy="17" r="3"/><circle cx="7" cy="7" r="3"/>'],
  battery:["stroke", '<rect x="2" y="7" width="16" height="10" rx="2"/><path d="M22 11v2"/>'],
};
function svg(name, cls) {
  const e = _ICON[name];
  if (!e) return "";
  const [mode, inner] = e;
  const attr = mode === "fill"
    ? 'fill="currentColor"'
    : 'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"';
  return `<span class="ic ${cls || ""}"><svg viewBox="0 0 24 24" ${attr}>${inner}</svg></span>`;
}

// 캔버스 둥근 사각형 (구형 브라우저 호환)
function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.closePath();
}

// KPI 카드 정의: [키, 라벨, 포맷, 경보판정(value)=>"" | "warn" | "alarm"]
const KPI_DEFS = [
  ["throughput", "처리량", (v) => v, () => ""],
  ["pending", "대기 큐", (v) => v, (v) => (v > 600 ? "alarm" : v > 300 ? "warn" : "")],
  ["utilization", "가동률 %", (v) => v, () => ""],
  ["deadlocks", "데드락", (v) => v, (v) => (v > 0 ? "alarm" : "")],
  ["hot_sla_rate", "HOT SLA", (v) => (v * 100).toFixed(0) + "%", (v) => (v < 0.8 ? "alarm" : v < 0.9 ? "warn" : "")],
  ["availability", "가용성", (v) => (v * 100).toFixed(0) + "%", (v) => (v < 0.7 ? "warn" : "")],
];

// ── 소스 목록 채우기 ────────────────────────────────────────────
async function loadSources() {
  const sel = $("source");
  sel.innerHTML = "";
  try {
    const r = await fetch("/api/runs");
    const { runs } = await r.json();
    for (const name of runs) {
      const o = document.createElement("option");
      o.value = `replay:${name}`; o.textContent = `replay · ${name}`;
      sel.appendChild(o);
    }
  } catch (e) { /* ignore */ }
  for (const [v, t] of [["live", "live · 76 AMR"], ["live:battery", "live · 배터리"]]) {
    const o = document.createElement("option");
    o.value = v; o.textContent = t; sel.appendChild(o);
  }
  // URL ?source= 우선
  const qs = new URLSearchParams(location.search).get("source");
  if (qs) sel.value = qs;
}

// ── 배경(평면도) 프리렌더 ───────────────────────────────────────
function buildBackground() {
  const { grid_h, grid_w, grid, ports } = meta;
  const off = document.createElement("canvas");
  off.width = grid_w; off.height = grid_h;
  const octx = off.getContext("2d");
  const img = octx.createImageData(grid_w, grid_h);
  for (let y = 0; y < grid_h; y++) {
    for (let x = 0; x < grid_w; x++) {
      const v = grid[y][x];
      const i = (y * grid_w + x) * 4;
      // 다크 패널: 0=통로(어두운 회색), 1=벽/장비(밝은 회색), 2=기타(중간)
      let c = v === 0 ? [22, 27, 34] : v === 1 ? [60, 70, 84] : [40, 50, 64];
      img.data[i] = c[0]; img.data[i + 1] = c[1]; img.data[i + 2] = c[2]; img.data[i + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  // 포트는 점으로 (블루그레이)
  octx.fillStyle = "#3b4a5e";
  for (const p of ports) octx.fillRect(p.x, p.y, 1, 1);
  bg = off;
}

function fitCanvas() {
  if (!meta) return;
  const wrap = canvas.parentElement;
  const availW = wrap.clientWidth - 20;
  const availH = wrap.clientHeight - 40;
  const scale = Math.max(1, Math.floor(Math.min(availW / meta.grid_w, availH / meta.grid_h)));
  canvas.width = meta.grid_w * scale;
  canvas.height = meta.grid_h * scale;
  canvas._scale = scale;
}

// ── 스냅샷 버퍼 + 보간 렌더 (requestAnimationFrame) ──────────────
// WS 프레임이 8fps로 뚝뚝 오더라도, 두 스냅샷 사이 AMR 위치를 시간 비율로 보간해
// 60fps로 부드럽게 그린다 → 끊김(버벅임) 제거.
let curSnap = null, prevSnap = null, curArrival = 0;
let frameInterval = 125;   // 스냅샷 간격 추정(ms, EMA)
let rafOn = false;
let selectedId = null;     // 클릭 선택된 AMR id (인스펙터)
let showZones = true;      // 혼잡 핫스팟 표시 토글

function onSnapshot(snap) {
  const now = performance.now();
  if (curSnap) {
    const dt = now - curArrival;
    if (dt > 5 && dt < 2000) frameInterval = frameInterval * 0.8 + dt * 0.2;
  }
  prevSnap = curSnap || snap;
  curSnap = snap;
  curArrival = now;
  renderKPIs(snap.kpi);   // KPI는 데이터 도착 시 갱신(보간 불필요)
  if (selectedId != null) renderInspector();  // 선택 AMR 상태 실시간 갱신
  hideOverlay();
}

function renderLoop() {
  if (curSnap && bg) {
    let alpha = 1;
    if (prevSnap && prevSnap !== curSnap)
      alpha = Math.min(1, (performance.now() - curArrival) / frameInterval);
    drawInterp(prevSnap, curSnap, alpha);
  }
  requestAnimationFrame(renderLoop);
}

// 혼잡 핫스팟 탐지 — 4분면이 아니라 'AMR이 실제로 가장 빽빽이 뭉친 한 지점'을 찾는다.
// BIN 격자로 밀도 추정 → 3×3 빈 합이 최대인 군집 중심·반경 반환. 군집이 약하면 null.
const HOT_BIN = 9;   // 빈 크기(grid cell) — 군집 판정 단위
function findHotspot(amrs) {
  if (!amrs || amrs.length < 6) return null;
  const bins = new Map();
  for (const a of amrs) {
    const k = Math.floor(a.x / HOT_BIN) + "," + Math.floor(a.y / HOT_BIN);
    let b = bins.get(k);
    if (!b) { b = { n: 0, sx: 0, sy: 0 }; bins.set(k, b); }
    b.n++; b.sx += a.x; b.sy += a.y;
  }
  let best = null;
  for (const k of bins.keys()) {
    const [bx, by] = k.split(",").map(Number);
    let n = 0, sx = 0, sy = 0;
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
      const b = bins.get((bx + dx) + "," + (by + dy));
      if (b) { n += b.n; sx += b.sx; sy += b.sy; }
    }
    if (!best || n > best.n) best = { n, x: sx / n, y: sy / n };
  }
  // 군집이 충분히 뭉쳐 있을 때만(전체의 30%+ & 6대+) 핫스팟으로 인정 → 고르게 퍼지면 표시 안 함
  if (!best || best.n < Math.max(6, amrs.length * 0.30)) return null;
  return best;
}

// ── 조치 적용 시각효과(발표용) ──────────────────────────────────
// 클릭한 조치가 '정말 맵에 반영된 듯' 보이게: 타깃 위치 초록 펄스 + 상단 배너 +
// 혼잡 핫스팟을 빨강→초록 "혼잡 완화 중"으로 전환(숫자 감소). replay 데이터는 못 바꾸므로
// 시간이 지나며 큐가 쌓이는 '조치 후 더 혼잡해 보임'을 이 연출로 상쇄한다.
let actionFx = null;                 // { label, kind, t0, dur }
const FX_GREEN = "16,185,129";       // emerald
const FX_DUR = 8000;
const RELIEF_KINDS = new Set(["deadlock", "blocking", "queue"]);  // 혼잡 완화형

function triggerActionFx(a, kind) {
  actionFx = { label: a.label, kind, t0: performance.now(), dur: FX_DUR };
}

// 핫스팟 표기 — 평소 빨강 "혼잡 N대", 조치(혼잡완화형) 적용 시 초록 "혼잡 완화 중 N"(감소).
function drawHotspot(cur, s) {
  const hs = findHotspot(cur.amrs);
  if (!hs) return;
  const relief = actionFx && RELIEF_KINDS.has(actionFx.kind);
  const rt = relief ? Math.min(1, (performance.now() - actionFx.t0) / actionFx.dur) : 0;
  const col = relief ? FX_GREEN : "240,68,82";
  const cx = hs.x * s + s / 2, cy = hs.y * s + s / 2;
  const R = Math.max(10, HOT_BIN * 1.6 * s);   // 타이트한 반경
  const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, R);
  g.addColorStop(0, `rgba(${col},0.42)`);
  g.addColorStop(0.6, `rgba(${col},0.16)`);
  g.addColorStop(1, `rgba(${col},0)`);
  ctx.fillStyle = g;
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.fill();
  // 또렷한 링
  ctx.strokeStyle = `rgba(${col},0.9)`;
  ctx.lineWidth = Math.max(1.5, s * 0.4);
  ctx.beginPath(); ctx.arc(cx, cy, R * 0.62, 0, Math.PI * 2); ctx.stroke();
  // 라벨 — 조치 시 "혼잡 완화 중"(줄어드는 수치)
  const fs = Math.max(11, s * 2.4);
  ctx.font = `700 ${fs}px -apple-system, "Apple SD Gothic Neo", sans-serif`;
  ctx.textBaseline = "middle"; ctx.textAlign = "left";
  const shownN = relief ? Math.max(0, Math.round(hs.n * (1 - rt))) : hs.n;
  const txt = relief ? `혼잡 완화 중 ${shownN}` : `혼잡 ${hs.n}대`;
  const tw = ctx.measureText(txt).width, bw = tw + fs * 0.9, bh = fs * 1.5;
  let bx = cx - bw / 2, by = cy - R * 0.62 - bh - 4;
  if (by < 2) by = cy + R * 0.62 + 4;                 // 위 공간 없으면 아래로
  bx = Math.max(2, Math.min(bx, canvas.width - bw - 2));
  ctx.fillStyle = relief ? "#10b981" : "#F04452";
  roundRect(ctx, bx, by, bw, bh, bh * 0.34); ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.fillText(txt, bx + fs * 0.45, by + bh / 2 + 0.5);
  ctx.textAlign = "left";
}

// 조치 종류별 타깃 좌표: 혼잡=핫스팟, 배터리=충전소, HOT=HOT 대기 포트.
function fxTargets(cur, s) {
  const k = actionFx.kind, pts = [];
  if (k === "battery") {
    for (const c of (cur.chargers || [])) pts.push([c.x * s + s / 2, c.y * s + s / 2]);
  } else if (k === "hot_sla") {
    for (const p of (cur.pending || [])) if (p.priority === "HOT") {
      const port = meta._portIndex && meta._portIndex[p.from_port];
      if (port) pts.push([port.x * s + s / 2, port.y * s + s / 2]);
    }
  } else {
    const hs = findHotspot(cur.amrs);
    pts.push(hs ? [hs.x * s + s / 2, hs.y * s + s / 2]
                : [canvas.width / 2, canvas.height / 2]);  // 핫스팟 없으면 중앙
  }
  return pts.slice(0, 8);
}

// 타깃에 확산 펄스 링 + 상단 "✓ {조치} 적용됨" 배너 (FX_DUR 동안).
function drawActionFx(cur, s) {
  if (!actionFx) return;
  const t = (performance.now() - actionFx.t0) / actionFx.dur;
  if (t >= 1) { actionFx = null; return; }

  const base = Math.max(12, HOT_BIN * 1.4 * s);
  for (const [cx, cy] of fxTargets(cur, s)) {
    for (let k = 0; k < 2; k++) {
      const pp = (t * 2.2 + k * 0.5) % 1;              // 0→1 확산, 반복
      ctx.strokeStyle = `rgba(${FX_GREEN},${0.55 * (1 - pp)})`;
      ctx.lineWidth = Math.max(1.5, s * 0.5);
      ctx.beginPath(); ctx.arc(cx, cy, base * (0.3 + pp * 1.4), 0, Math.PI * 2); ctx.stroke();
    }
  }

  const ba = Math.min(1, t / 0.06, (1 - t) / 0.18);    // 페이드 인/아웃
  if (ba > 0) {
    ctx.save();
    ctx.globalAlpha = ba;
    const fs = Math.max(12, s * 2.4);
    ctx.font = `700 ${fs}px -apple-system, "Apple SD Gothic Neo", sans-serif`;
    const txt = "✓ " + actionFx.label + " 적용됨";
    const tw = ctx.measureText(txt).width, bw = tw + fs * 1.6, bh = fs * 1.9;
    const bx = (canvas.width - bw) / 2, by = Math.max(6, s * 1.5);
    ctx.fillStyle = "rgba(16,185,129,0.96)";
    roundRect(ctx, bx, by, bw, bh, bh * 0.34); ctx.fill();
    ctx.fillStyle = "#fff"; ctx.textBaseline = "middle"; ctx.textAlign = "center";
    ctx.fillText(txt, canvas.width / 2, by + bh / 2 + 0.5);
    ctx.restore();
    ctx.textAlign = "left";
  }
}

function drawInterp(prev, cur, alpha) {
  const s = canvas._scale || 1;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(bg, 0, 0, canvas.width, canvas.height);

  // HOT 대기 픽업 포트 강조 (cur 기준) — 토스 레드
  ctx.fillStyle = "rgba(240,68,82,0.45)";
  for (const p of (cur.pending || [])) {
    if (p.priority === "HOT") {
      const port = meta._portIndex && meta._portIndex[p.from_port];
      if (port) ctx.fillRect(port.x * s - s, port.y * s - s, s * 3, s * 3);
    }
  }

  // 충전소 (배터리 모드) — 노란 라운드 사각 + 그려진 번개 모양
  for (const c of (cur.chargers || [])) {
    const cx0 = c.x * s + s / 2, cy0 = c.y * s + s / 2, sz = Math.max(7, s * 3);
    const x0 = cx0 - sz / 2, y0 = cy0 - sz / 2;
    ctx.fillStyle = "#facc15";
    roundRect(ctx, x0, y0, sz, sz, Math.max(1.5, sz * 0.22));
    ctx.fill();
    // 번개 패스 (sz 기준 정규화 좌표)
    const u = sz / 24;
    ctx.fillStyle = "#3a2c05";
    ctx.beginPath();
    const pts = [[13, 3], [6, 13], [11, 13], [10, 21], [18, 10], [13, 10]];
    pts.forEach(([px, py], i) => {
      const X = x0 + px * u, Y = y0 + py * u;
      i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
    });
    ctx.closePath();
    ctx.fill();
  }

  // AMR — id로 prev↔cur 매칭해 위치 보간 (먼 점프는 보간 안 함=잔상 방지)
  const pmap = {};
  if (prev) for (const a of prev.amrs) pmap[a.id] = a;
  const r = Math.max(2, s * 0.9);
  canvas._amrPix = {};   // 클릭 히트테스트용 현재 픽셀 좌표 캐시
  let sel = null;
  for (const a of cur.amrs) {
    let x = a.x, y = a.y;
    const p = pmap[a.id];
    if (p && Math.abs(p.x - a.x) + Math.abs(p.y - a.y) <= 8) {
      x = p.x + (a.x - p.x) * alpha;
      y = p.y + (a.y - p.y) * alpha;
    }
    const cx = x * s + s / 2, cy = y * s + s / 2;
    canvas._amrPix[a.id] = [cx, cy];
    ctx.beginPath();
    ctx.fillStyle = STATUS_COLOR[a.status] || "#888";
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    if (a.id === selectedId) sel = [cx, cy];
  }
  // 선택 AMR 강조 링 (마지막에 그려 위로) — 다크 패널에선 흰색
  if (sel) {
    ctx.beginPath();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = Math.max(2, s * 0.6);
    ctx.arc(sel[0], sel[1], r + Math.max(3, s * 1.4), 0, Math.PI * 2);
    ctx.stroke();
  }

  // 혼잡 핫스팟 표기 (최상단) — 가장 붐비는 한 지점만 또렷이
  if (showZones) drawHotspot(cur, s);
  // 조치 적용 효과(펄스+배너) — 핫스팟 위에 덧그림
  drawActionFx(cur, s);
}

// ── KPI 패널 ────────────────────────────────────────────────────
function renderKPIs(kpi) {
  const box = $("kpis");
  box.innerHTML = "";
  for (const [key, label, fmt, alarmFn] of KPI_DEFS) {
    let v = kpi[key];
    if (v === undefined || v === null) v = 0;
    const cls = alarmFn(v);
    const el = document.createElement("div");
    el.className = "kpi" + (cls ? " " + cls : "");
    el.innerHTML = `<div class="v">${fmt(v)}</div><div class="k">${label}</div>`;
    box.appendChild(el);
  }
}

// 조치 lever → 적용 효과 문구 (데모: 선택 시 '운영계에 반영된 것처럼' 보여줄 메시지).
// action_id = "<kind>:<lever>" 이므로 lever로 매핑한다.
const ACTION_EFFECTS = {
  early_replan:        "교착 구역 AMR에 경로 재계산 지시 전달, 예약 충돌 해소 중",
  zone_dispatch_hold:  "해당 구역 신규 배차 일시중단 적용, 적체 완화 중",
  manual_recover:      "작업자에게 수동 회수 지시 전달됨",
  hot_pool_boost:      "유휴 AMR을 HOT 전용으로 전환, HOT 응답 빨라지는 중",
  hot_priority_weight: "배차 HOT 우선순위 상향 적용됨",
  normal_defer:        "일반(NORMAL) 반송 후순위 적용, HOT 납기(SLA) 회복 중",
  fleet_size_up:       "운영 차량 증대 요청 등록됨, 용량 검토 진행",
  inflow_throttle:     "반송 유입량 조절 적용, 대기열 안정화 중",
  idle_rebalance:      "유휴 AMR을 적체 구역으로 미리 배치 전달됨",
  congestion_weight:   "혼잡 우회 강화 적용됨, 충돌 대기 감소 중",
  corridor_cap:        "병목 통로 동시 진입 제한 적용됨",
  charge_threshold_up: "충전 시작 기준 상향 적용됨, 방전 예방",
  tow_recover:         "방전 AMR 견인 회수 지시 전달됨",
  charger_capacity:    "충전소 증설 요청 등록됨, 용량 검토 진행",
};
function effectFor(actionId) {
  const lever = (actionId || "").split(":").pop();
  return ACTION_EFFECTS[lever] || "조치가 관제 시스템에 적용되었습니다";
}

const esc = (t) => (t || "").replace(/</g, "&lt;");

// LLM 출력 마크다운 경량 포매터: **볼드** → 하이라이트 토큰. (esc 먼저 → XSS 안전)
// Gemini가 조치명을 **[유입 throttle]** 처럼 강조하는데 ** 가 그대로 보이지 않게 변환.
function fmt(t) {
  return esc(t)
    .replace(/\*\*(.+?)\*\*/g, '<mark class="hl">$1</mark>')  // **굵게** → 하이라이트
    .replace(/\*\*/g, "");                                      // 짝 안 맞는 ** 잔여 제거
}

// ── 보고 / 답변 카드 (Toss 스타일) ──────────────────────────────
// 보고 = ① 발생 상황(규칙·사실) + ② AI 판단 도움 + ③ 결정형 조치 선택 리스트
function srcChip(source) {
  if (source === "llm") return `<span class="chip ai">${svg("ai")}AI 분석</span>`;
  if (source === "rule") return `<span class="chip rule">규칙</span>`;
  return "";
}

function addReport(rep, kind) {
  const box = $("reports");
  const el = document.createElement("div");
  const stepTxt = rep.step != null ? `step ${rep.step}` : "";

  if (kind === "answer") {
    // 관리자 질문(user) / AI 답변 구분
    const isUser = rep.role === "user";
    el.className = "card qa " + (isUser ? "qa-user" : "qa-ai");
    el.innerHTML = `
      <div class="qa-row">
        <span class="qa-ic">${svg(isUser ? "user" : "ai")}</span>
        <div class="qa-main">
          <div class="qa-label">${isUser ? "관리자 질문" : "AI 답변"}${isUser ? "" : " " + srcChip(rep.source)}</div>
          <div class="qa-text">${isUser ? esc(rep.body) : fmt(rep.body)}</div>
        </div>
      </div>`;
    box.prepend(el);
    while (box.children.length > 40) box.removeChild(box.lastChild);
    return;
  }

  const sev = rep.severity === "critical" ? "critical" : "warning";
  el.className = "card report sev-" + sev;
  el.innerHTML = `
    <div class="rep-head">
      <span class="sev-ic ${sev}">${svg("alert")}</span>
      <div class="rep-head-text">
        <div class="title">${esc(rep.title)}</div>
        <div class="rep-sub">${svg("rule", "tiny")}규칙 감지${stepTxt ? " · " + stepTxt : ""}</div>
      </div>
    </div>
    <div class="rep-situation">${esc(rep.situation)}</div>
    ${rep.advice ? `
      <div class="advice-box">
        <div class="advice-label">${svg("ai")}AI 판단 도움 ${srcChip(rep.source)}</div>
        <div class="advice-text">${fmt(rep.advice)}</div>
      </div>` : ""}`;

  // 결정형 조치 선택 리스트 (감사 로그 전용)
  if (rep.actions && rep.actions.length) {
    const wrap = document.createElement("div");
    wrap.className = "rep-actions";
    wrap.innerHTML = `<div class="actions-label">권장 조치</div>`;
    const list = document.createElement("div");
    list.className = "opt-list";
    for (const a of rep.actions) {
      const b = document.createElement("button");
      b.className = "opt";
      b.innerHTML = `
        <span class="opt-ic">${svg("lever")}</span>
        <span class="opt-txt"><b>${esc(a.label)}</b><small>${esc(a.detail || "")}</small></span>
        <span class="opt-go">${svg("chevron")}</span>`;
      b.onclick = () => {
        if (b.disabled) return;
        // 감사 로그용 기록은 서버로 계속 보낸다(있으면).
        if (ws && ws.readyState === 1)
          ws.send(JSON.stringify({ type: "decision", kind: rep.kind, action_id: a.id, label: a.label }));
        list.querySelectorAll("button").forEach((x) => { x.disabled = true; x.classList.add("dim"); });
        b.classList.remove("dim"); b.classList.add("chosen", "applying");
        const go = b.querySelector(".opt-go");
        go.innerHTML = `<span class="spin"></span>`;   // '적용 중' 스피너
        // 데모: 잠깐 적용 중 → 적용 완료로 전환하고 효과 카드를 띄워 '진짜 반영된 듯' 보이게.
        setTimeout(() => {
          b.classList.remove("applying");
          go.innerHTML = svg("check");
          addActionApplied(a, rep.kind);
          triggerActionFx(a, rep.kind);   // 맵에 적용 효과(펄스+배너+혼잡완화) 표시
        }, 1100);
      };
      list.appendChild(b);
    }
    wrap.appendChild(list);
    const note = document.createElement("div");
    note.className = "actions-note";
    note.textContent = "권장 조치를 선택하면 관제 시스템에 즉시 전달됩니다";
    wrap.appendChild(note);
    el.appendChild(wrap);
  }
  box.prepend(el);
  while (box.children.length > 40) box.removeChild(box.lastChild);
}

// 조치 적용 완료 카드 — 클릭한 조치가 운영계에 반영된 것처럼 효과를 함께 보여준다(데모).
function addActionApplied(a, kind) {
  const box = $("reports");
  const el = document.createElement("div");
  el.className = "card ack";
  el.innerHTML = `
    <div class="qa-row">
      <span class="qa-ic ok">${svg("check")}</span>
      <div class="qa-main">
        <div class="qa-label">조치 적용 완료 <span class="chip ok">운영계 반영</span></div>
        <div class="qa-text"><b>${esc(a.label || "")}</b><br>${esc(effectFor(a.id))}</div>
      </div>
    </div>`;
  box.prepend(el);
  while (box.children.length > 40) box.removeChild(box.lastChild);
}

// ── WebSocket ───────────────────────────────────────────────────
async function connect() {
  if (ws) { ws.close(); ws = null; }
  const source = $("source").value;
  const fps = $("speed").value;

  showOverlay("FAB 디지털 트윈 초기화 중…");
  curSnap = prevSnap = null;   // 새 연결: 보간 버퍼 리셋

  // 메타 먼저
  const m = await fetch(`/api/meta?source=${encodeURIComponent(source)}`).then((r) => r.json());
  if (m.error) { setStatus("err", m.error); hideOverlay(); return; }
  meta = m;
  meta._portIndex = {};
  for (const p of meta.ports) meta._portIndex[p.name] = p;
  buildBackground();
  fitCanvas();
  if (!rafOn) { rafOn = true; requestAnimationFrame(renderLoop); }

  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws?source=${encodeURIComponent(source)}&fps=${fps}`);
  ws.onopen = () => { setStatus("live", source.startsWith("live") ? "LIVE" : "재생 준비"); $("playpause").disabled = false; };
  ws.onclose = () => setStatus("", "종료");
  ws.onerror = () => { setStatus("err", "연결 오류"); hideOverlay(); };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "snapshot") { onSnapshot(msg.data); setStatus("live", source.startsWith("live") ? "LIVE" : "재생 중"); }
    else if (msg.type === "report") addReport(msg.data, "report");
    else if (msg.type === "answer") addReport(msg.data.data || msg.data, "answer");
    // decision_ack: 서버가 감사 로그(decisions.jsonl)에 기록만 함 — UI 효과는 클릭 시 즉시 처리.
    else if (msg.type === "end") setStatus("", "재생 완료");
    else if (msg.type === "error") { setStatus("err", msg.message); hideOverlay(); }
  };
}

// ── 초기화 오버레이 (갑작스런 시작 완충) ─────────────────────────
let _overlayTimer = null;
function showOverlay(txt) {
  const o = $("overlay");
  if (!o) return;
  o.textContent = txt; o.classList.add("on");
  canvas.classList.remove("ready");
}
function hideOverlay() {
  const o = $("overlay");
  if (!o || !o.classList.contains("on")) return;
  // 첫 프레임 후 살짝 지연해 페이드인 (갑자기 확 뜨는 느낌 완화)
  if (_overlayTimer) return;
  _overlayTimer = setTimeout(() => {
    o.classList.remove("on"); canvas.classList.add("ready"); _overlayTimer = null;
  }, 350);
}

function setStatus(cls, txt) {
  const el = $("status");
  el.className = "pill" + (cls ? " " + cls : "");
  el.textContent = txt;
}

// ── 컨트롤 ──────────────────────────────────────────────────────
$("connect").onclick = connect;
$("playpause").onclick = () => {
  if (!ws) return;
  paused = !paused;
  ws.send(JSON.stringify({ type: "control", action: paused ? "pause" : "play" }));
  $("playpause").innerHTML = paused ? svg("play") + "재생" : svg("pause") + "일시정지";
};
$("zonebtn").onclick = () => {
  showZones = !showZones;
  $("zonebtn").classList.toggle("on", showZones);
};
$("speed").oninput = () => {
  if (ws && ws.readyState === 1)
    ws.send(JSON.stringify({ type: "control", action: "speed", value: +$("speed").value }));
};
$("send").onclick = sendQuestion;
$("question").addEventListener("keydown", (e) => { if (e.key === "Enter") sendQuestion(); });
function sendQuestion() {
  const q = $("question").value.trim();
  if (!q || !ws || ws.readyState !== 1) return;
  ws.send(JSON.stringify({ type: "ask", qid: ++qid, question: q }));
  addReport({ body: q, role: "user", step: null }, "answer");
  $("question").value = "";
}
window.addEventListener("resize", () => { fitCanvas(); });

// ── AMR 클릭 인스펙터 ───────────────────────────────────────────
const STATUS_KR = {
  idle: "유휴", moving: "이동 중", loading: "적재 중", unloading: "하역 중",
  charging: "충전 중", depleted: "방전(정지)", done: "완료",
};
const DEST_KR = { charger: "충전소", pickup: "픽업지", dropoff: "하역지" };

canvas.addEventListener("click", (ev) => {
  if (!canvas._amrPix) return;
  const rect = canvas.getBoundingClientRect();
  // 캔버스 표시 크기 ≠ 내부 픽셀 크기일 수 있어 스케일 보정
  const sx = canvas.width / rect.width, sy = canvas.height / rect.height;
  const px = (ev.clientX - rect.left) * sx, py = (ev.clientY - rect.top) * sy;
  const s = canvas._scale || 1;
  const hitR = Math.max(8, s * 2.2);   // 클릭 관용 반경
  let best = null, bestD = hitR * hitR;
  for (const [id, [cx, cy]] of Object.entries(canvas._amrPix)) {
    const d = (cx - px) ** 2 + (cy - py) ** 2;
    if (d < bestD) { bestD = d; best = +id; }
  }
  if (best != null) { selectedId = best; renderInspector(); }
  else closeInspector();
});

function renderInspector() {
  if (selectedId == null || !curSnap) return;
  const a = curSnap.amrs.find((x) => x.id === selectedId);
  if (!a) return;
  const rows = [];
  rows.push(["상태", STATUS_KR[a.status] || a.status]);
  if ("soc" in a) {
    const pct = Math.round(a.soc * 100);
    const cls = pct <= 15 ? "crit" : pct <= 35 ? "warn" : "ok";
    rows.push(["배터리", `<span class="bar"><i class="${cls}" style="width:${pct}%"></i></span> ${pct}%`]);
  }
  rows.push(["운반", a.carrying ? `${svg("box", "tiny")}적재 중` : "빈 차"]);
  if (a.priority) rows.push(["우선순위", a.priority === "HOT"
    ? `<span class="prio-hot">HOT</span>` : `<span class="prio-norm">NORMAL</span>`]);
  if (a.dest_kind) {
    const where = DEST_KR[a.dest_kind] || a.dest_kind;
    rows.push(["목적지", `${where}${a.dest_port ? ` · ${a.dest_port}` : ""}`]);
  }
  if (a.from_port && a.to_port) rows.push(["경로", `${a.from_port} → ${a.to_port}`]);
  if (a.progress != null) {
    const pp = Math.round(a.progress * 100);
    rows.push(["진행률", `<span class="bar"><i class="ok" style="width:${pp}%"></i></span> ${pp}%`]);
  }
  rows.push(["위치", `(${a.x}, ${a.y})`]);

  $("insp-body").innerHTML =
    `<div class="insp-title"><span class="dot ${a.status}"></span>AMR #${a.id}</div>` +
    rows.map(([k, v]) => `<div class="irow"><span class="ik">${k}</span><span class="iv">${v}</span></div>`).join("");
  $("inspector").classList.remove("hidden");
}

function closeInspector() {
  selectedId = null;
  $("inspector").classList.add("hidden");
}
$("insp-close").onclick = closeInspector;

// ── 초기 아이콘 주입 (이모지 대체) ──────────────────────────────
$("playpause").innerHTML = svg("pause") + "일시정지";
$("send").innerHTML = svg("send") + "질문";
const _hint = $("hint"); if (_hint) _hint.innerHTML = svg("bulb", "tiny") + _hint.textContent;
const _lc = $("legend-charger"); if (_lc) _lc.innerHTML = svg("bolt");

loadSources();
