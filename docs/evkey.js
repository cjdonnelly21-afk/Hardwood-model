/* EV key shown on every picks page: what each EV band means, how often those bets are expected to hit, and a suggested minimum. */
function evKey(el, games, opts) {
  if (!el) return;
  const plays = games.filter(g => g.market && g.market.best && g.market.best.ev > 0).map(g => g.market.best);
  const bands = [
    ["Small", "0% to 5%", b => b.ev < .05, "Barely above break-even. Inside the noise of the model and of price differences between books."],
    ["Solid", "5% to 10%", b => b.ev >= .05 && b.ev < .10, "The model likes the price. About $5 to $10 expected profit per $100 staked."],
    ["Big gap", "10% and up", b => b.ev >= .10, "The model disagrees a lot with the market. Often a stale line or missing injury news: check before betting (these carry a CHECK badge)."]
  ];
  const avg = a => a.length ? (a.reduce((s, x) => s + x.p, 0) / a.length * 100).toFixed(0) + "%" : "-";
  const rows = bands.map(([n, r, f, d]) => { const a = plays.filter(f);
    return `<tr><td class="team">${n}</td><td>${r}</td><td class="num">${a.length}</td><td class="num">${avg(a)}</td><td style="white-space:normal">${d}</td></tr>`; }).join("");
  el.innerHTML = `<h2>EV key</h2>
    <p class="note">EV is expected profit per $1 staked. +5% EV means about $5 per $100 over many bets, not on any single one. Roughly how often each kind of bet needs to win just to break even: -110 needs 52.4%, -150 needs 60%, +100 needs 50%, +200 needs 33%, +300 needs 25%. A bet only has positive EV when the real chance to hit is higher than that.</p>
    <div class="tablewrap"><table><thead><tr><th>Level</th><th>EV</th><th class="num">Plays now</th><th class="num">Avg chance to hit</th><th>What it means</th></tr></thead><tbody>${rows}</tbody></table></div>
    <p class="note"><b>Suggested minimum: ${opts.min}.</b> ${opts.why} The chance to hit is the model's estimate, so the Track record section is the check on whether it is honest.</p>`;
}
