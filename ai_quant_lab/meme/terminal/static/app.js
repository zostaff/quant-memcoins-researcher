"use strict";
// Meme Quant Terminal: four windows, one watchlist, one paper book.
// Token names and symbols come from the open internet: everything is escaped.

const $ = (s, el = document) => el.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const TFS = ["1m", "5m", "15m", "1h", "4h"];
const SLOTS = 4;

const state = {
  network: localStorage.getItem("mqt.net") || "solana",
  list: "trending",
  market: null,
  focus: 0,
  slots: JSON.parse(localStorage.getItem("mqt.slots") || "null") || Array(SLOTS).fill(null),
  data: Array(SLOTS).fill(null),
  strategies: [],
};

// ---------------------------------------------------------------- formatting

const SUB = "₀₁₂₃₄₅₆₇₈₉";
function price(p) {
  if (p == null || !isFinite(p)) return "—";
  if (p >= 1000) return "$" + p.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (p >= 1) return "$" + p.toFixed(p >= 100 ? 2 : 3);
  if (p >= 0.001) return "$" + p.toFixed(5);
  const s = p.toExponential(3);                      // 1.234e-7
  const [m, e] = s.split("e");
  const zeros = -parseInt(e, 10) - 1;
  const digits = m.replace(".", "").replace(/0+$/, "").slice(0, 4);
  return "$0.0" + String(zeros).split("").map((d) => SUB[d]).join("") + digits;
}
function usd(v, d = 1) {
  if (v == null || !isFinite(v)) return "—";
  const a = Math.abs(v), sign = v < 0 ? "-" : "";
  if (a >= 1e9) return sign + "$" + (a / 1e9).toFixed(d) + "B";
  if (a >= 1e6) return sign + "$" + (a / 1e6).toFixed(d) + "M";
  if (a >= 1e3) return sign + "$" + (a / 1e3).toFixed(d) + "K";
  return sign + "$" + a.toFixed(a >= 100 ? 0 : 2);
}
const money = (v) => (v == null ? "—" : (v < 0 ? "-$" : "$") + Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }));
function pct(v, d = 1, signed = true) {
  if (v == null || !isFinite(v)) return "—";
  return (signed && v > 0 ? "+" : "") + v.toFixed(d) + "%";
}
const cls = (v) => (v == null ? "m" : v > 0 ? "g" : v < 0 ? "r" : "m");
function age(h) {
  if (h == null) return "";
  if (h < 1) return Math.round(h * 60) + "m";
  if (h < 48) return Math.round(h) + "h";
  return Math.round(h / 24) + "d";
}

// ---------------------------------------------------------------- api

async function api(path, body) {
  const res = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  const data = await res.json().catch(() => ({ error: res.statusText }));
  if (!res.ok || data.error) throw new Error(data.error || res.statusText);
  return data;
}
let toastTimer;
function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (err ? " err" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 3500);
}
const save = () => localStorage.setItem("mqt.slots", JSON.stringify(state.slots));

// ---------------------------------------------------------------- market

async function loadMarket() {
  try {
    state.market = await api(`/api/market?network=${state.network}`);
  } catch (e) {
    toast("Market: " + e.message, true);
    setTimeout(loadMarket, 10_000);
    return;
  }
  renderPulse();
  renderList();
  renderHeat();
  if (state.slots.every((s) => !s)) {
    state.slots = state.market.trending.slice(0, SLOTS).map((c) => ({ network: c.network, address: c.address, tf: "5m" }));
    save();
    state.slots.forEach((_, i) => loadSlot(i));
  }
}

function renderPulse() {
  const p = state.market.pulse, c = p.components;
  const kind = { "RISK-ON": "on", "RISK-OFF": "off" }[p.regime] || "neutral";
  const hot = Object.entries(p.narratives).filter(([n]) => n !== "other").sort((a, b) => b[1].heat - a[1].heat).slice(0, 3);
  const score = p.score == null ? `<span class="kv">calibrating ${p.readings}/${p.min_history}</span>` : `<span class="kv">score <b>${p.score.toFixed(2)}</b></span>`;
  $("#pulse").innerHTML = `
    <span class="regime ${kind}" title="Market mood from breadth, buy share, volume impulse and launch pace">${esc(p.regime)}</span>
    ${score}
    <span class="kv" title="Share of listed pools up over 1h">breadth <b>${c.breadth == null ? "—" : Math.round(c.breadth * 100) + "%"}</b></span>
    <span class="kv" title="Buys as a share of all 1h transactions">buys <b>${c.buy_pressure == null ? "—" : Math.round(c.buy_pressure * 100) + "%"}</b></span>
    <span class="kv" title="1h volume against the 24h hourly average">vol impulse <b class="${cls(c.volume_impulse)}">${c.volume_impulse == null ? "—" : (Math.exp(c.volume_impulse)).toFixed(2) + "×"}</b></span>
    <span class="kv" title="New pools per hour">launches <b>${c.launch_rate == null ? "—" : c.launch_rate.toFixed(0) + "/h"}</b></span>
    <span class="kv">hot</span>${hot.map(([n, r]) => `<span class="chip narrative" title="heat ${r.heat.toFixed(2)}">${esc(n)} <span class="${cls(r.heat)}">${r.heat > 0 ? "▲" : "▼"}</span></span>`).join("")}`;
}

function renderList() {
  const rows = state.list === "trending" ? state.market.trending : state.market.new;
  const loaded = new Set(state.slots.filter(Boolean).map((s) => s.address));
  $("#list").innerHTML = rows.map((c, i) => `
    <div class="row ${loaded.has(c.address) ? "loaded" : ""}" data-i="${i}" title="${esc(c.name)} · ${esc(c.dex || "")} · ${age(c.age_hours)} old">
      <span class="rk">${state.list === "trending" ? c.rank : age(c.age_hours)}</span>
      ${c.image ? `<img src="${esc(c.image)}" alt="" loading="lazy">` : `<img alt="">`}
      <div style="min-width:0"><div class="s">${esc(c.symbol)}<small>${esc(c.narrative)}</small></div>
        <div class="n">${usd(c.liquidity_usd, 0)} TVL · ${usd(c.volume.h24, 0)} vol</div></div>
      <div class="v"><div class="${cls(c.change.h1)}">${pct(c.change.h1)}</div><div>${price(c.price_usd)}</div></div>
    </div>`).join("") || `<div class="emptyline" style="padding:12px">Nothing listed.</div>`;
}

function renderHeat() {
  const rows = Object.entries(state.market.pulse.narratives).sort((a, b) => b[1].heat - a[1].heat).slice(0, 8);
  const max = Math.max(0.01, ...rows.map(([, r]) => Math.abs(r.heat)));
  $("#heat").innerHTML = rows.map(([n, r]) => {
    const w = (Math.abs(r.heat) / max) * 50;
    const style = r.heat >= 0 ? `left:50%;width:${w}%;background:var(--green)` : `right:50%;width:${w}%;background:var(--red)`;
    return `<div class="heat"><span>${esc(n)} <span class="m">${r.tokens}</span></span><div class="bar"><i style="${style}"></i></div><span class="${cls(r.heat)}" style="text-align:right">${r.heat >= 0 ? "+" : ""}${r.heat.toFixed(2)}</span></div>`;
  }).join("");
}

// ---------------------------------------------------------------- windows

const wins = [];
function buildGrid() {
  const grid = $("#grid");
  for (let i = 0; i < SLOTS; i++) {
    const el = $("#tpl-window").content.firstElementChild.cloneNode(true);
    el.querySelectorAll(".slotno").forEach((s) => (s.textContent = i + 1));
    el.addEventListener("mousedown", () => setFocus(i));
    $(".tfs", el).innerHTML = TFS.map((t) => `<button data-tf="${t}">${t}</button>`).join("") + `<span class="src"></span>`;
    $(".tfs", el).addEventListener("click", (e) => {
      const tf = e.target.dataset.tf;
      if (!tf || !state.slots[i]) return;
      state.slots[i].tf = tf;
      save();
      loadSlot(i);
    });
    $(".x", el).addEventListener("click", (e) => {
      e.stopPropagation();
      state.slots[i] = null;
      state.data[i] = null;
      save();
      renderWin(i);
      if (state.market) renderList();
    });
    $(".usd", el).addEventListener("input", () => renderCost(i));
    $(".buy", el).addEventListener("click", () => trade(i, "buy"));
    el.querySelectorAll(".sell").forEach((b) => b.addEventListener("click", () => trade(i, "sell", +b.dataset.f)));
    grid.appendChild(el);
    wins.push(el);
    new ResizeObserver(() => drawChart(i)).observe($(".chart", el));
  }
  setFocus(0);
}

function setFocus(i) {
  state.focus = i;
  wins.forEach((w, j) => w.classList.toggle("focus", j === i));
  $("#focusHint").textContent = i + 1;
}

async function loadSlot(i) {
  const s = state.slots[i];
  if (!s) return renderWin(i);
  const el = wins[i];
  el.classList.add("loading");
  const amount = +$(".usd", el).value || 500;
  try {
    state.data[i] = await api(`/api/pool?network=${s.network}&address=${encodeURIComponent(s.address)}&tf=${s.tf || "5m"}&usd=${amount}`);
  } catch (e) {
    toast(`Window ${i + 1}: ${e.message}`, true);
    if (!state.data[i]) setTimeout(() => loadSlot(i), 10_000);
  } finally {
    el.classList.remove("loading");
  }
  renderWin(i);
}

function renderWin(i) {
  const el = wins[i], d = state.data[i], s = state.slots[i];
  $(".empty", el).hidden = !!(s && d);
  $(".body", el).hidden = !(s && d);
  if (!s || !d) return;
  const p = d.pool, info = d.info || {};
  const icon = $(".icon", el);
  if (p.image) { icon.src = p.image; icon.style.visibility = ""; } else icon.style.visibility = "hidden";
  $(".sym", el).innerHTML = `${esc(p.symbol)}<small>${esc(p.network === "robinhood" ? "Robinhood" : "Solana")} · ${esc(p.dex || "")} · ${age(p.age_hours)}</small>`;
  $(".nm", el).textContent = p.name;
  $(".narrative", el).textContent = p.narrative;
  $(".price", el).textContent = price(p.price_usd);
  $(".chg", el).innerHTML = [["5m", "m5"], ["1h", "h1"], ["6h", "h6"], ["24h", "h24"]]
    .map(([l, k]) => `<span class="${cls(p.change[k])}"><b>${l}</b>${pct(p.change[k])}</span>`).join("");
  el.querySelectorAll(".tfs button").forEach((b) => b.classList.toggle("on", b.dataset.tf === (s.tf || "5m")));
  $(".tfs .src", el).textContent = (d.missing && d.missing.length ? `API busy, waiting for ${d.missing.join(", ")} · ` : "")
    + (d.trending_rank ? `trending #${d.trending_rank}` : "");

  const buys = p.buys.h1, sells = p.sells.h1;
  const top10 = info.top10_pct;
  $(".stats", el).innerHTML = [
    ["TVL", usd(p.liquidity_usd)],
    ["MCap/FDV", usd(p.fdv_usd)],
    ["Vol 24h", usd(p.volume.h24)],
    ["Holders", info.holders == null ? "—" : info.holders.toLocaleString("en-US")],
    ["Top-10", top10 == null ? "—" : `<span class="${top10 > 50 ? "r" : ""}">${top10.toFixed(0)}%</span>`],
    ["Buys 1h", buys + sells ? `<span class="${buys > sells ? "g" : "r"}">${Math.round((100 * buys) / (buys + sells))}%</span> <span class="m">${buys + sells}</span>` : "—"],
  ].map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join("");

  const flags = [];
  if (info.mint_authority === 0) flags.push("mint off");
  if (info.freeze_authority === 0) flags.push("freeze off");
  if (info.honeypot === 0) flags.push("no honeypot");
  if (info.dev_pct != null) flags.push(`dev ${info.dev_pct.toFixed(1)}%`);
  if (d.graduation_pct != null) flags.push(`curve ${d.graduation_pct.toFixed(0)}%`);
  $(".screen", el).innerHTML = (d.screen.length
    ? `<span class="bad">✕ screen: ${d.screen.map(esc).join(" · ")}</span>`
    : `<span class="ok">✓ passes the universe screen</span>`) + flags.map((f) => `<span class="chip">${esc(f)}</span>`).join("");

  $(".sigs", el).innerHTML = d.signals.map((g) => `
    <div class="sig ${g.state}" title="${esc(g.strategy)}: ${esc(g.reason)}. Zoo verdict: ${esc(g.verdict)}">
      <span class="d"></span><span class="nm2">${esc(g.strategy)}</span><span class="val">${esc(g.value)}</span>
      <span class="chip v-${esc(g.verdict)}">${esc(g.verdict)}</span>
    </div>`).join("");

  renderCost(i);
  const pos = d.position;
  $(".pos", el).innerHTML = pos && pos.tokens > 0
    ? `Position ${money(pos.cost_basis)} in → liquidation <b>${money(pos.liquidation)}</b> <span class="${cls(pos.unrealized)}">${money(pos.unrealized)} (${pct((pos.return || 0) * 100)})</span>`
    : `<span class="m">No position. Paper fills pay fee, curve slippage, MEV and priority fee.</span>`;
  drawChart(i);
}

function renderCost(i) {
  const el = wins[i], d = state.data[i];
  if (!d) return;
  const amount = Math.max(+$(".usd", el).value || 0, 0);
  // Same formula as the backend quote: 1 - (1 - f)·Q/(Q + s), Q = TVL/2, plus MEV and fixed fee.
  const net = d.pool.network, tvl = d.pool.liquidity_usd || 0;
  const f = 0.01, mev = net === "robinhood" ? 0.001 : 0.002, fixed = net === "robinhood" ? 0.02 : 0.1;
  const q = tvl / 2;
  const curve = q > 0 ? 1 - ((1 - f) * q) / (q + amount) : 1;
  const one = amount > 0 ? Math.min(1, curve + mev + fixed / amount) : 0;
  $(".cost", el).innerHTML = amount > 0
    ? `cost <b>${(one * 100).toFixed(2)}%</b> in · round trip <b>${Math.min(200 * one, 100).toFixed(1)}%</b>`
    : "";
}

// ---------------------------------------------------------------- chart

function drawChart(i) {
  const el = wins[i], d = state.data[i];
  if (!d) return;
  const cv = $(".chart", el);
  const w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  const dpr = window.devicePixelRatio || 1;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  const all = d.candles || [];
  const axis = 58, pad = 6;
  const n = Math.min(all.length, Math.max(20, Math.floor((w - axis) / 5)));
  const rows = all.slice(-n);
  if (!rows.length) {
    g.fillStyle = "#7f8c82"; g.font = "11px Geist Mono, monospace";
    g.fillText(d.missing && d.missing.includes("candles") ? "loading candles…" : "no candles yet", 10, h / 2);
    return;
  }
  const volH = h * 0.2, priceH = h - volH - pad * 2;
  let lo = Infinity, hi = -Infinity, vmax = 0;
  for (const [, , hh, ll, , v] of rows) { lo = Math.min(lo, ll); hi = Math.max(hi, hh); vmax = Math.max(vmax, v); }
  const entry = d.position && d.position.tokens > 0 ? d.position.cost_basis / d.position.tokens : null;
  if (hi === lo) { hi *= 1.01; lo *= 0.99; }
  const y = (v) => pad + (1 - (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * priceH;
  const cw = (w - axis) / rows.length;

  // grid + axis labels
  g.font = "10px Geist Mono, monospace";
  g.textAlign = "left";
  for (let k = 0; k <= 4; k++) {
    const v = Math.exp(Math.log(lo) + ((Math.log(hi) - Math.log(lo)) * k) / 4);
    const yy = y(v);
    g.strokeStyle = "rgba(38,48,42,.6)"; g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, yy + 0.5); g.lineTo(w - axis, yy + 0.5); g.stroke();
    g.fillStyle = "#7f8c82"; g.fillText(price(v).replace("$", ""), w - axis + 6, yy + 3);
  }
  // volume + candles
  rows.forEach(([, o, hh, ll, c, v], k) => {
    const x = k * cw, up = c >= o;
    const col = up ? "#33E28A" : "#FF4D6D";
    g.fillStyle = up ? "rgba(51,226,138,.22)" : "rgba(255,77,109,.22)";
    const vh = vmax ? (v / vmax) * volH : 0;
    g.fillRect(x + cw * 0.15, h - vh, Math.max(cw * 0.7, 1), vh);
    g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 1;
    const cx = Math.round(x + cw / 2) + 0.5;
    g.beginPath(); g.moveTo(cx, y(hh)); g.lineTo(cx, y(ll)); g.stroke();
    const top = y(Math.max(o, c)), bot = y(Math.min(o, c));
    g.fillRect(x + cw * 0.18, top, Math.max(cw * 0.64, 1), Math.max(bot - top, 1));
  });
  // last price
  const last = rows[rows.length - 1][4], ly = y(last);
  g.setLineDash([3, 3]); g.strokeStyle = "rgba(200,255,61,.6)";
  g.beginPath(); g.moveTo(0, ly + 0.5); g.lineTo(w - axis, ly + 0.5); g.stroke(); g.setLineDash([]);
  g.fillStyle = "#C8FF3D"; g.fillRect(w - axis + 1, ly - 7, axis - 1, 14);
  g.fillStyle = "#071006"; g.fillText(price(last).replace("$", ""), w - axis + 5, ly + 3);
  // entry line
  if (entry && entry > lo * 0.5 && entry < hi * 2) {
    const ey = y(Math.min(Math.max(entry, lo), hi));
    g.strokeStyle = "rgba(255,194,61,.8)"; g.setLineDash([6, 4]);
    g.beginPath(); g.moveTo(0, ey + 0.5); g.lineTo(w - axis, ey + 0.5); g.stroke(); g.setLineDash([]);
    g.fillStyle = "#FFC23D"; g.fillText("entry", 4, ey - 3);
  }
}

// ---------------------------------------------------------------- paper

async function trade(i, side, fraction) {
  const s = state.slots[i];
  if (!s) return;
  try {
    const amount = +$(".usd", wins[i]).value;
    const fill = side === "buy"
      ? await api("/api/paper/buy", { network: s.network, address: s.address, usd: amount })
      : await api("/api/paper/sell", { network: s.network, address: s.address, fraction });
    toast(`${fill.side.toUpperCase()} ${fill.symbol}: ${money(fill.usd)} at ${price(fill.price)}, cost ${money(fill.cost_usd)} (${(fill.quote.cost_fraction * 100).toFixed(2)}%)`);
  } catch (e) {
    return toast(e.message, true);
  }
  await loadSlot(i);
  loadPaper();
}

async function loadPaper() {
  let a;
  try { a = await api("/api/paper"); } catch (e) { return; }
  $("#acct").innerHTML = `
    <div class="t"><span>Paper account</span><button class="reset" id="reset">reset</button></div>
    <div class="eq">${money(a.liquidation_value)}</div>
    <div class="pl ${cls(a.pnl)}">${money(a.pnl)} · ${pct(a.return * 100, 2)}</div>
    <div class="grid2">
      <div><span>Cash</span>${money(a.cash)}</div>
      <div><span>Mark value</span>${money(a.mark_value)}</div>
      <div><span>Realized</span><span class="${cls(a.realized)}" style="font:11px var(--mono);text-transform:none;letter-spacing:0">${money(a.realized)}</span></div>
      <div><span>Costs paid</span><span class="a" style="font:11px var(--mono);text-transform:none;letter-spacing:0">${money(a.costs_paid)}</span></div>
    </div>
    <div class="note">Equity is what selling everything now would return, after exit slippage. Mark value is what a screener would show.</div>`;
  $("#reset").onclick = async () => { await api("/api/paper/reset", {}); loadPaper(); state.slots.forEach((_, i) => loadSlot(i)); };
  $("#positions").innerHTML = a.positions.map((p) => `
    <div class="p"><b>${esc(p.symbol)} <span class="m">${esc(p.network)}</span></b><span class="${cls(p.unrealized)}">${money(p.unrealized)}</span>
    <span class="m">${money(p.cost_basis)} in → ${money(p.liquidation)}</span><span class="${cls(p.return)}">${pct((p.return || 0) * 100)}</span></div>`).join("")
    || `<div class="emptyline">No open positions.</div>`;
  $("#fills").innerHTML = a.fills.slice(0, 8).map((f) => `
    <div class="f"><span class="side-b ${f.side}">${f.side.toUpperCase()}</span><b>${esc(f.symbol)}</b><span>${money(f.usd)}</span>
    <span></span><span class="m">${esc(f.at.slice(11, 16))} UTC · ${price(f.price)}</span><span class="a m">−${money(f.cost_usd)}</span></div>`).join("")
    || `<div class="emptyline">No fills yet.</div>`;
}

async function loadLibrary() {
  try { state.strategies = await api("/api/strategies"); } catch (e) { return; }
  $("#lib").innerHTML = state.strategies.map((s) => `
    <div class="s" title="${esc(s.reason || "Run `zoo --json` and start the terminal with --verdicts to load verdicts")}">
      <span class="nm3">${esc(s.name)} <span class="m">${esc(s.family)}</span></span><span class="chip v-${esc(s.verdict)}">${esc(s.verdict)}</span>
      <span class="th">${esc(s.thesis)}</span></div>`).join("");
}

// ---------------------------------------------------------------- wiring

function pick(card) {
  const i = state.focus;
  state.slots[i] = { network: card.network, address: card.address, tf: (state.slots[i] && state.slots[i].tf) || "5m" };
  save();
  loadSlot(i);
  renderList();
  setFocus((i + 1) % SLOTS);
}

$("#list").addEventListener("click", (e) => {
  const row = e.target.closest(".row");
  if (!row) return;
  const rows = state.list === "trending" ? state.market.trending : state.market.new;
  pick(rows[+row.dataset.i]);
});
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  state.list = b.dataset.list;
  document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
  if (state.market) renderList();
}));
document.querySelectorAll(".nets button").forEach((b) => {
  b.classList.toggle("on", b.dataset.net === state.network);
  b.addEventListener("click", () => {
    state.network = b.dataset.net;
    localStorage.setItem("mqt.net", state.network);
    document.querySelectorAll(".nets button").forEach((x) => x.classList.toggle("on", x === b));
    loadMarket();
  });
});

let found = [];
$("#search").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#q").value.trim();
  const box = $("#results");
  if (!q) return (box.hidden = true);
  try { found = await api(`/api/search?network=${state.network}&q=${encodeURIComponent(q)}`); } catch (err) { return toast(err.message, true); }
  box.innerHTML = found.slice(0, 12).map((c, i) => `
    <div class="row" data-i="${i}"><span class="rk">${i + 1}</span>${c.image ? `<img src="${esc(c.image)}" alt="">` : `<img alt="">`}
    <div style="min-width:0"><div class="s">${esc(c.symbol)}<small>${esc(c.narrative)}</small></div><div class="n">${usd(c.liquidity_usd, 0)} TVL · ${esc(c.dex || "")}</div></div>
    <div class="v"><div class="${cls(c.change.h1)}">${pct(c.change.h1)}</div><div>${price(c.price_usd)}</div></div></div>`).join("")
    || `<div class="emptyline" style="padding:10px">No pools found.</div>`;
  box.hidden = false;
});
$("#results").addEventListener("click", (e) => {
  const row = e.target.closest(".row");
  if (!row) return;
  pick(found[+row.dataset.i]);
  $("#results").hidden = true;
  $("#q").value = "";
});
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.key >= "1" && e.key <= "4") setFocus(+e.key - 1);
  if (e.key === "Escape") $("#results").hidden = true;
});

function tick() { $("#clock").textContent = new Date().toISOString().slice(11, 19) + " UTC"; }

buildGrid();
tick();
setInterval(tick, 1000);
loadMarket().then(() => state.slots.forEach((s, i) => s && !state.data[i] && loadSlot(i)));
loadPaper();
loadLibrary();
setInterval(loadMarket, 60_000);
setInterval(() => state.slots.forEach((_, i) => loadSlot(i)), 30_000);
setInterval(loadPaper, 15_000);
