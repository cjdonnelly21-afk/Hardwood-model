"use strict";
const $ = (s, r = document) => r.querySelector(s);
const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (x, d = 1) => (x == null || isNaN(x)) ? "–" : Number(x).toFixed(d);
const sgn = (x, d = 1) => (x > 0 ? "+" : "") + fmt(x, d);

const S = { teams: [], byName: {}, games: [], model: null, acc: null, status: null, day: null, sortK: "rank", sortDir: 1 };

// normal CDF (Abramowitz-Stegun) so the browser matches scipy's norm.cdf closely
function cdf(z) {
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp(-z * z / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z > 0 ? 1 - p : p;
}

// Same formula as src/train.py (make_features + predict)
function project(home, away, neutral) {
  const h = S.byName[home], a = S.byName[away], m = S.model, T = S.meta;
  if (!h || !a || !m) return null;
  const loc = neutral ? 0 : 1;
  const poss = T.mu_p + h.t + a.t;
  const own = ((h.o - a.d) - (a.o - h.d) + 2 * T.hca * loc) * poss / 100;
  const ownTot = (2 * T.mu + (h.o - a.d) + (a.o - h.d)) * poss / 100;
  const tempo = (h.tor_t + a.tor_t) / 2;
  const tor = ((h.tor_oe - h.tor_de) - (a.tor_oe - a.tor_de) + 2 * 2.7 * loc) * tempo / 100;
  const torTot = (h.tor_oe - a.tor_de + a.tor_oe - h.tor_de + 208) * tempo / 100;
  const early = 1 / (1 + Math.min(h.gp, a.gp) / 5);
  const f = { own, tor, own_early: own * early, tor_early: tor * early, non_neutral: loc, conf_tourney: 0, postseason: 0 };
  let margin = m.margin.intercept;
  for (const k in f) margin += m.margin[k] * f[k];
  const total = m.total.intercept + m.total.own_total * ownTot + m.total.tor_total * torTot + m.total.poss_hat * poss;
  return { margin, total, p: cdf(margin / m.sigma), own, tor, poss };
}

function spreadText(g) {
  const fav = g.pred_margin >= 0 ? g.home : g.away;
  return `${fav} ${fmt(-Math.abs(g.pred_margin), 1)}`;
}

function tier(p) { return p >= .9 ? ["Strong", "t3"] : p >= .75 ? ["Solid", "t2"] : p >= .6 ? ["Lean", "t1"] : ["Toss-up", "t0"]; }

function pct(x) { return (x >= 0 ? "+" : "") + (x * 100).toFixed(1) + "%"; }
function price(n) { return (n > 0 ? "+" : "") + n; }

// the card on the right of each game: the bet to take (when we have lines) or the plain model pick
function betCard(g, fav, favP, tierName, tierCls) {
  const m = g.market, b = m && m.best;
  if (b && b.ev > 0) {
    const cls = b.ev >= .06 ? "e3" : b.ev >= .03 ? "e2" : "e1";
    const others = m.bets.filter(x => !(x.market === b.market && x.label === b.label) && x.ev > 0).map(x => `${esc(x.label)} (${pct(x.ev)})`).join(" · ");
    return `<div class="pick bet ${cls}"><div class="tag">Take</div><div class="favname">${esc(b.label)}</div>
      <div class="spread">${pct(b.ev)}</div><div class="tier">EV per $1 · ${price(b.price)} ${esc(b.book)}</div>
      <div class="tot">Model ${fmt(-Math.abs(g.pred_margin), 1)} ${esc(fav)} · market ${m.market_margin === null ? "n/a" : fmt(-Math.abs(m.market_margin), 1)} · total ${fmt(g.pred_total, 0)}/${m.market_total ?? "n/a"}</div>
      ${b.ev >= .10 ? '<div class="tot"><b>Big gap to the market: check injuries and the book\'s line before trusting it.</b></div>' : ""}
      ${others ? `<div class="tot">Also: ${others}</div>` : ""}</div>`;
  }
  if (b) {
    return `<div class="pick t0"><div class="tag">No edge vs market</div><div class="favname">${esc(fav)}</div>
      <div class="spread">${fmt(-Math.abs(g.pred_margin), 1)}</div><div class="tier">Pass · best EV ${pct(b.ev)}</div>
      <div class="tot">Market ${m.market_margin === null ? "n/a" : fmt(-Math.abs(m.market_margin), 1)} · total ${fmt(g.pred_total, 0)}/${m.market_total ?? "n/a"}</div></div>`;
  }
  return `<div class="pick ${tierCls}"><div class="tag">Model pick</div><div class="favname">${esc(fav)}</div><div class="spread">${fmt(-Math.abs(g.pred_margin), 1)}</div><div class="tier">${tierName} · ${Math.round(favP * 100)}%</div><div class="tot">Proj. total ${fmt(g.pred_total, 0)}</div></div>`;
}

function dayKey(iso) { const d = new Date(iso); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`; }
function dayLabel(k) { const [y, m, d] = k.split("-").map(Number); return new Date(y, m - 1, d).toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" }); }

function renderGames() {
  const q = $("#gsearch").value.trim().toLowerCase();
  let list = S.games.filter(g => dayKey(g.start) === S.day && (!q || (g.home + " " + g.away).toLowerCase().includes(q)));
  const mode = $("#gsort").value;
  const evOf = g => (g.market && g.market.best) ? g.market.best.ev : -9;
  if (mode === "ev") list.sort((a, b) => evOf(b) - evOf(a));
  else if (mode === "conf") list.sort((a, b) => Math.abs(b.win_prob_home - .5) - Math.abs(a.win_prob_home - .5));
  else if (mode === "close") list.sort((a, b) => Math.abs(a.pred_margin) - Math.abs(b.pred_margin));
  else list.sort((a, b) => a.start.localeCompare(b.start));
  const ol = $("#games"); ol.innerHTML = "";
  $("#gempty").hidden = list.length > 0;
  for (const g of list) {
    const hp = Math.min(.99, Math.max(.01, g.win_prob_home)), favHome = hp >= .5;
    const h = S.byName[g.home], a = S.byName[g.away];
    const li = el("li", "game");
    const time = new Date(g.start).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
    const result = g.done ? `<div class="final">Final ${fmt(g.hpts, 0)}–${fmt(g.apts, 0)}${(g.hpts - g.apts > 0) === favHome ? " · model had the winner" : " · upset"}</div>` : "";
    const fav = favHome ? g.home : g.away, favP = favHome ? hp : 1 - hp;
    const [tierName, tierCls] = tier(favP);
    const badge = `<b class="chk" title="The team the model projects to win">${S.hasLines ? "MODEL" : "PICK"}</b>`;
    li.innerHTML = `
      <div class="teams">
        <div class="tm ${favHome ? "fav" : ""}"><span>${esc(g.home)}${favHome ? badge : ""}</span><span class="rk">${h ? "#" + h.rank : ""}</span></div>
        <div class="tm ${!favHome ? "fav" : ""}"><span>${esc(g.away)}${g.neutral ? " <em>neutral</em>" : ""}${!favHome ? badge : ""}</span><span class="rk">${a ? "#" + a.rank : ""}</span></div>
        <div class="when">${g.done ? "" : time}</div>${result}
      </div>
      <div class="probwrap">
        <div class="prob" role="img" aria-label="${esc(fav)} ${Math.round(favP * 100)} percent to win"><i style="width:${(favP * 100).toFixed(1)}%"></i></div>
        <div class="probcap">${Math.round(favP * 100)}% to win</div>
      </div>
      ${betCard(g, fav, favP, tierName, tierCls)}`;
    ol.appendChild(li);
  }
}

function renderDays() {
  const keys = [...new Set(S.games.map(g => dayKey(g.start)))].sort();
  const box = $("#days"); box.innerHTML = "";
  const today = dayKey(new Date().toISOString());
  if (!S.day || !keys.includes(S.day)) S.day = keys.includes(today) ? today : (keys.find(k => k >= today) || keys[keys.length - 1]);
  for (const k of keys) {
    const b = el("button", "", k === today ? "Today" : dayLabel(k));
    b.setAttribute("aria-pressed", k === S.day);
    b.onclick = () => { S.day = k; renderDays(); renderGames(); };
    box.appendChild(b);
  }
}

function renderRankings() {
  const q = $("#rsearch").value.trim().toLowerCase(), conf = $("#rconf").value;
  const k = S.sortK, dir = S.sortDir;
  const rows = S.teams.filter(t => (!conf || t.conf === conf) && (!q || t.team.toLowerCase().includes(q)))
    .sort((a, b) => (typeof a[k] === "string" ? a[k].localeCompare(b[k]) : a[k] - b[k]) * dir);
  const tb = $("#rtable tbody"); tb.innerHTML = "";
  for (const t of rows.slice(0, 400)) {
    const tr = el("tr", "", `<td class="num">${t.rank}</td><td class="team">${esc(t.team)}</td><td>${esc(t.conf)}</td><td>${esc(t.record)}</td>
      <td class="num"><b>${sgn(t.power)}</b></td><td class="num">${sgn(t.net)}</td><td class="num ${t.o > 0 ? "pos" : "neg"}">${sgn(t.o)}</td><td class="num ${t.d > 0 ? "pos" : "neg"}">${sgn(t.d)}</td>
      <td class="num">${fmt(t.tor_t)}</td><td class="num">${t.tor_rank}</td>`);
    tb.appendChild(tr);
  }
}

function renderMatchup() {
  const h = $("#mh").value.trim(), a = $("#ma").value.trim(), out = $("#mout");
  if (!S.byName[h] || !S.byName[a] || h === a) { out.innerHTML = `<p class="empty">Pick two teams to see the projected score, spread and win chance.</p>`; return; }
  const neutral = $("#mv").value === "neutral";
  const r = project(h, a, neutral);
  const hp = Math.min(.99, Math.max(.01, r.p)), favH = r.margin >= 0, fav = favH ? h : a;
  const hs = (r.total + r.margin) / 2, as = (r.total - r.margin) / 2;
  out.innerHTML = `
    <div class="big">
      <div class="side">${esc(h)}<small>${fmt(hs, 0)} points · ${Math.round(hp * 100)}% to win</small></div>
      <div class="mid">${esc(fav)}<br>${fmt(-Math.abs(r.margin), 1)}<small>projected spread</small></div>
      <div class="side r">${esc(a)}<small>${fmt(as, 0)} points · ${Math.round((1 - hp) * 100)}% to win</small></div>
    </div>
    <div class="facts">
      <div><b>${fmt(r.total, 0)}</b><span>projected total</span></div>
      <div><b>${fmt(r.poss, 0)}</b><span>possessions</span></div>
      <div><b>${sgn(r.own)}</b><span>own ratings say (home margin)</span></div>
      <div><b>${sgn(r.tor)}</b><span>Torvik ratings say (home margin)</span></div>
    </div>
    <p class="note">One standard error is about ${fmt(S.model.sigma, 0)} points, so any single game is far from certain.</p>`;
}

function renderModel() {
  const a = S.acc || {}, bt = S.model.backtest || [];
  const mean = k => bt.reduce((s, r) => s + r[k], 0) / Math.max(bt.length, 1);
  const tiles = [];
  if (bt.length) tiles.push([fmt(mean("mae_blend"), 2), "backtest points of error per game"], [fmt(mean("win_acc") * 100, 1) + "%", "backtest winners picked"]);
  if (a.n) tiles.push([a.n.toLocaleString(), "live games graded"], [fmt(a.mae, 2), "live points of error"], [fmt(a.win_acc * 100, 1) + "%", "live winners picked"]);
  else tiles.push(["0", "live games graded so far (the season just started)"]);
  $("#live").innerHTML = tiles.map(([b, s]) => `<div><b>${b}</b><span>${s}</span></div>`).join("");
  const tb = $("#btable tbody"); tb.innerHTML = "";
  for (const r of bt) tb.appendChild(el("tr", "", `<td>${r.season - 1}–${String(r.season).slice(2)}</td><td class="num">${r.n.toLocaleString()}</td><td class="num"><b>${fmt(r.mae_blend, 2)}</b></td><td class="num">${fmt(r.mae_own, 2)}</td><td class="num">${fmt(r.mae_torvik, 2)}</td><td class="num">${fmt(r.win_acc * 100, 1)}%</td><td class="num">${fmt(r.tot_mae, 2)}</td>`));
  const useLive = a.n >= 300;
  $("#calnote").textContent = useLive ? `Live results: ${a.n.toLocaleString()} games graded this season.` : "Backtest results across 2017–18 to 2025–26. This switches to live games once 300 have been graded.";
  drawCalibration(useLive ? a.calibration : S.model.calibration);
  const m = S.model.margin;
  const dl = (title, o) => `<dl><h3>${title}</h3>${Object.entries(o).map(([k, v]) => `<dt>${k.replace(/_/g, " ")}</dt><dd>${fmt(v, 3)}</dd>`).join("")}</dl>`;
  $("#weights").innerHTML = dl("Margin weights", m) + dl("Total weights", S.model.total);
}

function drawCalibration(live) {
  const box = $("#cal");
  const pts = (live || []).filter(p => p.n >= 20);
  if (!pts.length) { box.innerHTML = `<p class="empty">The live calibration chart appears after a few hundred games have been graded.</p>`; return; }
  const W = 440, H = 440, P = 40, sx = v => P + v * (W - 2 * P), sy = v => H - P - v * (H - 2 * P);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Calibration: predicted versus actual win rate" style="width:100%;height:auto">
    <line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}" stroke="currentColor" stroke-opacity=".35" stroke-dasharray="4 4"/>`;
  for (const t of [0, .5, 1]) s += `<text x="${sx(t)}" y="${H - 14}" text-anchor="middle" font-size="12" fill="currentColor">${t * 100}%</text><text x="12" y="${sy(t) + 4}" font-size="12" fill="currentColor">${t * 100}%</text>`;
  for (const p of pts) s += `<circle cx="${sx(p.pred)}" cy="${sy(p.actual)}" r="${Math.min(14, 4 + Math.sqrt(p.n) / 3)}" fill="var(--paint)" fill-opacity=".85"><title>Predicted ${Math.round(p.pred * 100)}%, actual ${Math.round(p.actual * 100)}% (${p.n} games)</title></circle>`;
  s += `<text x="${W / 2}" y="${H}" text-anchor="middle" font-size="12" fill="currentColor">model's win chance for the home team</text></svg>`;
  box.innerHTML = s;
}

function setTab(name) {
  document.querySelectorAll("nav button").forEach(b => b.setAttribute("aria-selected", b.dataset.tab === name));
  document.querySelectorAll(".panel").forEach(p => p.hidden = p.id !== name);
  history.replaceState(null, "", "#" + name);
}

async function load() {
  const get = async f => { const r = await fetch(`data/${f}?v=${Date.now()}`); if (!r.ok) throw new Error(f); return r.json(); };
  try {
    const [teams, games, model, acc, status] = await Promise.all(["teams.json", "games.json", "model.json", "accuracy.json", "status.json"].map(get));
    S.teams = teams.teams; S.meta = teams; S.games = games; S.model = model; S.acc = acc; S.status = status;
    S.byName = Object.fromEntries(S.teams.map(t => [t.team, t]));
  } catch (e) {
    $("#updated").textContent = "Data files not found yet. Run the daily update to create them.";
    return;
  }
  const when = new Date(S.status.run_at);
  $("#updated").textContent = `${S.meta.season - 1}–${String(S.meta.season).slice(2)} season · updated ${when.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}`;
  if (S.status.warnings && S.status.warnings.length) { const w = $("#warn"); w.hidden = false; w.textContent = S.status.warnings.join(" "); }
  const hasLines = S.hasLines = S.games.some(g => g.market);
  if (hasLines) {
    $("#gsort").insertAdjacentHTML("afterbegin", '<option value="ev">Best EV</option>');
    $("#gsort").value = "ev";
    $("#legend").innerHTML = `<b>Take</b> is the single best bet on the game (spread, moneyline or total) after comparing the model with the best price across sportsbooks. <b>EV</b> is expected profit per $1 staked; 5% means about 5 cents. The model is shrunk toward the market because it is not as sharp as the closing line, so treat EV as a ranking, not a promise. Lines as of ${new Date(S.status.lines_at || S.status.run_at).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })}; check the book before betting.${S.status.totals_bias ? ` The model's totals run ${Math.abs(S.status.totals_bias).toFixed(1)} points ${S.status.totals_bias < 0 ? "below" : "above"} the market on average today; that gap is removed before totals are compared, so Over/Under picks reflect game-to-game differences only.` : ""}`;
  }
  const confs = [...new Set(S.teams.map(t => t.conf))].sort();
  $("#rconf").innerHTML = `<option value="">All</option>` + confs.map(c => `<option>${esc(c)}</option>`).join("");
  $("#tlist").innerHTML = S.teams.map(t => `<option value="${esc(t.team)}">`).join("");
  renderDays(); renderGames(); renderRankings(); renderModel();
  if (hasLines) evKey($("#evkey"), S.games.filter(g => !g.done), { min: "5% EV", why: "College lines for small schools are soft and the model has no injury data, so edges under 5% are mostly noise, while 10%+ usually means the gap needs checking." });
}

document.querySelectorAll("nav button").forEach(b => b.onclick = () => setTab(b.dataset.tab));
["gsearch", "gsort"].forEach(id => $("#" + id).addEventListener("input", renderGames));
["rsearch", "rconf"].forEach(id => $("#" + id).addEventListener("input", renderRankings));
["mh", "ma", "mv"].forEach(id => $("#" + id).addEventListener("input", renderMatchup));
document.querySelectorAll("#rtable th").forEach(th => {
  th.tabIndex = 0;
  const go = () => { const k = th.dataset.k; S.sortDir = S.sortK === k ? -S.sortDir : (k === "team" || k === "conf" || k === "rank" || k === "tor_rank" ? 1 : -1); S.sortK = k; renderRankings(); };
  th.onclick = go; th.onkeydown = e => { if (e.key === "Enter") go(); };
});
if (location.hash && $(location.hash)) setTab(location.hash.slice(1));
load();
