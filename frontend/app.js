"use strict";

/* ---------- helpers ---------- */
const $ = id => document.getElementById(id);
const PIP = 1e-4, GREEN = "#2ebd85", RED = "#f6465d", BLUE = "#4c8dff", BASE_MS = 600, ANN = 252;
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (v, d = 2) => Number.isFinite(v) ? v.toFixed(d) : "—";
const sgn = (v, d = 2) => Number.isFinite(v) ? (v > 0 ? "+" : v < 0 ? "-" : "") + Math.abs(v).toFixed(d) : "—";
const usd = (v, plus) => !Number.isFinite(v) ? "—" :
  (v < 0 ? "-" : plus && v > 0 ? "+" : "") + "$" + Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const tone = (el, v) => { el.classList.toggle("up", v > 0); el.classList.toggle("dn", v < 0); };
const sum = (a, f) => a.reduce((s, x) => s + f(x), 0);

/* ---------- chart ---------- */
const chart = LightweightCharts.createChart($("chart"), {
  autoSize: true,
  layout: { background: { type: LightweightCharts.ColorType.Solid, color: "#0a0e14" }, textColor: "#8b97a7", fontFamily: "system-ui, sans-serif" },
  grid: { vertLines: { color: "#121923" }, horzLines: { color: "#121923" } },
  rightPriceScale: { borderColor: "#1f2a37" },
  timeScale: { borderColor: "#1f2a37", rightOffset: 6, barSpacing: 9 },
  crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
});
const series = chart.addSeries(LightweightCharts.CandlestickSeries, {
  upColor: GREEN, downColor: RED, borderVisible: false, wickUpColor: GREEN, wickDownColor: RED,
  priceFormat: { type: "price", precision: 5, minMove: 0.00001 },
});
const markers = LightweightCharts.createSeriesMarkers(series, []);

/* ---------- state ---------- */
let D = [], TH = {}, i = -1, mode = "auto", speed = 1, playing = false, timer = null, lastN = -1;
let dmin = 0, dmax = 1;
let S = newSim();

function newSim() {
  return { cap: Math.max(1, +$("cap").value || 10000), realized: 0, pos: null, trades: [], eq: [], markers: [], line: null };
}

/* ---------- trading engine ---------- */
const params = () => ({
  lot: Math.max(0.01, +$("lot").value || 0.2),
  sp: Math.max(0, +$("spread").value || 0),
  h: Math.max(1, Math.round(+$("hold").value || 5)),
});

function openPos(dir, bar, auto) {
  const { lot, sp, h } = params();
  S.pos = {
    dir, lot, sp, auto, prob: bar.ensemble_prob, entryDate: bar.date,
    entry: bar.close + dir * sp * PIP / 2,           // pay the ask (long) / hit the bid (short)
    exitIdx: auto ? i + h : null,
  };
  S.markers.push({
    time: bar.date, position: dir > 0 ? "belowBar" : "aboveBar", shape: dir > 0 ? "arrowUp" : "arrowDown",
    color: dir > 0 ? GREEN : RED,
  });
  markers.setMarkers(S.markers);
  S.line = series.createPriceLine({
    price: S.pos.entry, color: dir > 0 ? GREEN : RED, lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: "ENTRY",
  });
}

function closePos(bar) {
  const p = S.pos, exit = bar.close - p.dir * p.sp * PIP / 2;
  const pips = (exit - p.entry) * p.dir / PIP, pnl = pips * p.lot * 10;
  S.realized += pnl;
  S.trades.push({ entryDate: p.entryDate, exitDate: bar.date, dir: p.dir, entry: p.entry, exit, prob: p.prob, pips, usd: pnl });
  if (S.line) { series.removePriceLine(S.line); S.line = null; }
  S.pos = null;
}

function floating() {
  if (!S.pos || i < 0) return { pips: 0, usd: 0 };
  const p = S.pos, mark = D[i].close - p.dir * p.sp * PIP / 2;
  const pips = (mark - p.entry) * p.dir / PIP;
  return { pips, usd: pips * p.lot * 10 };
}
const equity = () => S.cap + S.realized + floating().usd;

/* ---------- replay ---------- */
function step() {
  if (!D.length || i >= D.length - 1) return;
  i++;
  const b = D[i], last = D.length - 1;
  series.update({ time: b.date, open: b.open, high: b.high, low: b.low, close: b.close });
  if (S.pos && S.pos.exitIdx != null && i >= S.pos.exitIdx) closePos(b);
  if (mode === "auto" && !S.pos && b.signal !== "HOLD" && i + params().h <= last) openPos(b.signal === "LONG" ? 1 : -1, b, true);
  if (i === last && S.pos) closePos(b);
  S.eq[i] = equity();
  render();
}

function loop() {
  if (!playing) return;
  if (i >= D.length - 1) return pause();
  step();
  timer = setTimeout(loop, BASE_MS / speed);
}
function play() { if (!playing && D.length) { playing = true; loop(); render(); } }
function pause() { playing = false; clearTimeout(timer); render(); }

function reset() {
  playing = false; clearTimeout(timer);
  if (S.line) series.removePriceLine(S.line);
  i = -1; S = newSim(); lastN = -1;
  series.setData([]); markers.setMarkers([]);
  render();
}

function manual(dir) {
  if (i < 0 || S.pos) return;
  openPos(dir, D[i], false);
  S.eq[i] = equity();
  render();
}

/* ---------- metrics ---------- */
function metrics() {
  const t = S.trades, fl = floating(), eq = i < 0 ? S.cap : equity();
  const gw = sum(t.filter(x => x.usd > 0), x => x.usd), gl = -sum(t.filter(x => x.usd < 0), x => x.usd);
  const path = [S.cap, ...S.eq], rets = path.slice(1).map((v, k) => v / path[k] - 1);
  let sharpe = NaN;
  if (rets.length >= 2) {
    const m = sum(rets, r => r) / rets.length, sd = Math.sqrt(sum(rets, r => (r - m) ** 2) / (rets.length - 1));
    if (sd > 0) sharpe = m / sd * Math.sqrt(ANN);
  }
  let peak = path[0], ddPct = 0, ddUsd = 0;
  for (const v of path) {
    peak = Math.max(peak, v);
    if (peak - v > ddUsd) { ddUsd = peak - v; ddPct = ddUsd / peak * 100; }
  }
  return {
    eq, net: eq - S.cap, pips: sum(t, x => x.pips) + fl.pips,
    win: t.length ? t.filter(x => x.usd > 0).length / t.length * 100 : NaN,
    pf: gl > 0 ? gw / gl : gw > 0 ? Infinity : NaN, sharpe, ddPct, ddUsd,
  };
}

/* ---------- rendering ---------- */
const pct = p => Math.min(100, Math.max(0, (p - dmin) / (dmax - dmin) * 100));

function applyScale() {
  const all = D.map(d => d.ensemble_prob).concat([TH.lower_10, TH.upper_90]);
  dmin = Math.max(0, Math.min(...all) - 0.02); dmax = Math.min(1, Math.max(...all) + 0.02);
  const l = pct(TH.lower_10), u = pct(TH.upper_90);
  Object.assign($("zl").style, { left: "0", width: l + "%" });
  Object.assign($("zm").style, { left: l + "%", width: u - l + "%" });
  Object.assign($("zh").style, { left: u + "%", width: 100 - u + "%" });
  Object.assign($("tl").style, { left: l + "%" }); $("tl").textContent = "L10 " + TH.lower_10.toFixed(4);
  Object.assign($("tu").style, { left: u + "%" }); $("tu").textContent = "U90 " + TH.upper_90.toFixed(4);
}

function renderSignal(b) {
  const el = $("sig");
  el.textContent = b ? b.signal : "—";
  el.className = "sig " + (b ? b.signal.toLowerCase() : "hold");
  $("ens").textContent = b ? b.ensemble_prob.toFixed(4) : "—";
  $("ptr").hidden = !b;
  if (b) $("ptr").style.left = pct(b.ensemble_prob) + "%";
  [["lgbm", "lgbm_prob"], ["cb", "cb_prob"], ["tf", "tf_prob"]].forEach(([k, f]) => {
    const v = b ? b[f] : NaN, bar = $("b-" + k);
    bar.style.width = (Number.isFinite(v) ? v * 100 : 0) + "%";
    bar.style.background = !Number.isFinite(v) ? BLUE : v >= 0.5 ? GREEN : RED;
    $("v-" + k).textContent = Number.isFinite(v) ? (v * 100).toFixed(1) + "%" : "—";
  });
}

function renderPos() {
  const p = S.pos, el = $("pos");
  if (!p) { el.innerHTML = '<div class="mu">No open position</div>'; return; }
  const f = floating(), cls = f.usd > 0 ? "up" : f.usd < 0 ? "dn" : "";
  const rem = p.exitIdx == null ? "Manual" : Math.max(0, p.exitIdx - i) + " d";
  el.innerHTML =
    `<div class="prow"><span class="pill ${p.dir > 0 ? "long" : "short"}">${p.dir > 0 ? "LONG" : "SHORT"}</span><span class="mu">${p.lot.toFixed(2)} lot · ${p.auto ? "AI" : "Manual"}</span></div>` +
    `<div class="kv"><span class="mu">Entry</span><span>${p.entry.toFixed(5)}</span></div>` +
    `<div class="kv"><span class="mu">Floating Pips</span><span class="${cls}">${sgn(f.pips, 1)}</span></div>` +
    `<div class="kv"><span class="mu">Floating P&amp;L</span><span class="${cls}">${usd(f.usd, true)}</span></div>` +
    `<div class="kv"><span class="mu">Days Remaining</span><span>${rem}</span></div>`;
}

function renderKpis() {
  const m = metrics();
  $("k-eq").textContent = usd(m.eq);
  $("k-pnl").textContent = usd(m.net, true) + " (" + sgn(m.net / S.cap * 100) + "%)"; tone($("k-pnl"), m.net);
  $("k-pips").textContent = sgn(m.pips, 1); tone($("k-pips"), m.pips);
  $("k-win").textContent = Number.isFinite(m.win) ? m.win.toFixed(1) + "%" : "—";
  $("k-pf").textContent = m.pf === Infinity ? "∞" : num(m.pf);
  $("k-sh").textContent = num(m.sharpe); tone($("k-sh"), m.sharpe);
  $("k-dd").textContent = m.ddUsd > 0 ? "-" + m.ddPct.toFixed(2) + "% (" + usd(m.ddUsd) + ")" : "0.00%";
  $("k-dd").classList.toggle("dn", m.ddUsd > 0);
}

function renderTable() {
  $("tbody").innerHTML = S.trades.length ? S.trades.slice().reverse().map(t => {
    const c = t.usd > 0 ? "up" : t.usd < 0 ? "dn" : "";
    return `<tr><td>${t.entryDate}</td><td>${t.exitDate}</td>` +
      `<td><span class="pill ${t.dir > 0 ? "long" : "short"}">${t.dir > 0 ? "LONG" : "SHORT"}</span></td>` +
      `<td>${t.entry.toFixed(5)}</td><td>${t.exit.toFixed(5)}</td><td>${t.prob.toFixed(4)}</td>` +
      `<td class="${c}">${sgn(t.pips, 1)}</td><td class="${c}">${usd(t.usd, true)}</td></tr>`;
  }).join("") : '<tr><td colspan="8" class="mu" style="text-align:center">No trades yet</td></tr>';
}

function render() {
  const b = D[i], ready = D.length > 0, last = i >= D.length - 1;
  $("date").textContent = b ? b.date : "—";
  $("price").textContent = b ? b.close.toFixed(5) : "—";
  const ch = b && i > 0 ? (b.close / D[i - 1].close - 1) * 100 : NaN;
  $("chg").textContent = Number.isFinite(ch) ? sgn(ch) + "%" : ""; tone($("chg"), ch);
  renderSignal(b); renderPos(); renderKpis();
  if (S.trades.length !== lastN) { renderTable(); lastN = S.trades.length; }
  $("play").disabled = !ready || last || playing;
  $("pause").disabled = !playing;
  $("step").disabled = !ready || last;
  $("reset").disabled = !ready;
  const man = mode === "manual" && i >= 0;
  $("buy").disabled = $("sell").disabled = !(man && !S.pos);
  $("close").disabled = !(man && S.pos);
}

/* ---------- status & data loading ---------- */
function setBadge(txt, cls) { const e = $("badge"); e.textContent = txt; e.className = "badge " + cls; }

function showBanner(st) {
  const b = $("banner");
  if (st.ready) { b.hidden = true; return; }
  let h = "";
  if (st.missing_files && st.missing_files.length)
    h += "<b>Missing files:</b> " + st.missing_files.map(f => `<code>${esc(f)}</code>`).join(" ") +
      " — add them to <code>models/</code> / <code>data/</code>; this page re-checks automatically.";
  if (st.error) h += `<div><b>Backend error:</b> ${esc(st.error)}</div>`;
  b.innerHTML = h; b.hidden = !h;
}

async function init() {
  setBadge("LOADING MODELS…", "wait");
  try {
    const st = await (await fetch("/api/status")).json();
    showBanner(st);
    if (!st.ready) {
      setBadge(st.error ? "MODEL ERROR" : `MODELS MISSING (${st.missing_files.length})`, "bad");
      return setTimeout(init, 5000);
    }
    const r = await fetch("/api/replay-data");
    if (!r.ok) throw new Error("replay-data " + r.status);
    D = await r.json(); TH = st.thresholds;
    applyScale();
    setBadge(`MODEL READY · ${D.length} DAYS`, "ok");
    reset();
  } catch (e) {
    setBadge("BACKEND OFFLINE", "bad");
    setTimeout(init, 5000);
  }
}

/* ---------- events ---------- */
$("play").onclick = play;
$("pause").onclick = pause;
$("step").onclick = () => { if (!playing) step(); };
$("reset").onclick = reset;
$("buy").onclick = () => manual(1);
$("sell").onclick = () => manual(-1);
$("close").onclick = () => { if (S.pos) { closePos(D[i]); S.eq[i] = equity(); render(); } };
$("cap").onchange = reset;
$("lot").oninput = () => { $("pipval").textContent = "$" + (params().lot * 10).toFixed(2) + " / pip"; };
$("speed").onclick = e => {
  const b = e.target.closest("button"); if (!b) return;
  speed = +b.dataset.s;
  [...$("speed").children].forEach(x => x.classList.toggle("on", x === b));
};
$("mode").onclick = e => {
  const b = e.target.closest("button"); if (!b) return;
  mode = b.dataset.m;
  [...$("mode").children].forEach(x => x.classList.toggle("on", x === b));
  render();
};
$("export").onclick = () => {
  const head = ["entry_date", "exit_date", "direction", "entry_price", "exit_price", "ai_prob", "net_pips", "net_pnl_usd"];
  const rows = S.trades.map(t => [t.entryDate, t.exitDate, t.dir > 0 ? "LONG" : "SHORT",
    t.entry.toFixed(5), t.exit.toFixed(5), t.prob.toFixed(4), t.pips.toFixed(1), t.usd.toFixed(2)].join(","));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([[head.join(","), ...rows].join("\n")], { type: "text/csv" }));
  a.download = "eurusd_trades.csv";
  a.click();
  URL.revokeObjectURL(a.href);
};

render();
init();
