"""Strategy zoo: run the whole library the way an allocator would read it.

Every variant of every available strategy is backtested, and every one of
them counts as a trial. Then the best variant of each strategy gets the full
tear sheet, deflated by the total number of variants tried across the whole
zoo (not just its own family: picking the best family is a choice too), and
a verdict:

    IDLE       never found a trade on this panel
    LOSS       loses money net of fees, slippage, MEV and rugs
    UNSTABLE   one half of the sample loses: a regime, not a strategy
    BETA       no significant alpha once the meme factors are accounted for
    LUCK       the deflated Sharpe says the best of this many tries looks this good by chance
    THIN       capacity below the book size it was tested at
    DUPLICATE  more than 0.6 correlated with a strategy that already passed
    PASS       survived all of the above

PASS is not a recommendation to trade. It means the idea deserves fresh data
it has never seen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ai_quant_lab.meme.backtest import MemeBacktestConfig, meme_backtest
from ai_quant_lab.meme.factors import meme_factor_returns
from ai_quant_lab.meme.narratives import narrative_summary
from ai_quant_lab.meme.panel import TokenPanel
from ai_quant_lab.meme.strategies import STRATEGIES, StrategyContext, catalog
from ai_quant_lab.meme.tearsheet import TearSheet, tear_sheet

VERDICTS = ("PASS", "DUPLICATE", "THIN", "LUCK", "BETA", "UNSTABLE", "LOSS", "IDLE")
_ALPHA_T = 2.0
_BETA_R2 = 0.15
_DSR_P = 0.05
_MAX_CORR = 0.6


@dataclass(frozen=True)
class ZooEntry:
    strategy: str
    family: str
    params: dict
    sheet: TearSheet
    verdict: str
    reason: str


@dataclass
class ZooResult:
    trials: pd.DataFrame
    entries: list[ZooEntry]
    skipped: dict[str, list[str]]
    correlation: pd.DataFrame
    panel_stats: dict[str, object] = field(default_factory=dict)

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    def leaderboard(self) -> pd.DataFrame:
        rows = []
        for e in self.entries:
            m = e.sheet.backtest.metrics
            a = e.sheet.attribution
            rows.append({
                "strategy": e.strategy,
                "family": e.family,
                "best params": _fmt_params(e.params),
                "return": m["total_return"],
                "sharpe": m["sharpe_ratio"],
                "halves": f"{e.sheet.first_half_sharpe:+.1f} / {e.sheet.second_half_sharpe:+.1f}",
                "MKT beta": float(a.betas.get("MKT", np.nan)) if a is not None else np.nan,
                "alpha t": float(a.t_stats.get("alpha", np.nan)) if a is not None else np.nan,
                "DSR p": e.sheet.dsr.pvalue if e.sheet.dsr is not None else np.nan,
                "capacity": e.sheet.capacity_usd,
                "cost/$": m["cost_per_dollar_traded"],
                "verdict": e.verdict,
                "why": e.reason,
            })
        table = pd.DataFrame(rows)
        if table.empty:
            return table
        order = {v: i for i, v in enumerate(VERDICTS)}
        return table.sort_values(["verdict", "sharpe"], key=lambda c: c.map(order) if c.name == "verdict" else -c)

    def summary(self) -> str:
        board = self.leaderboard()
        lines = [f"{self.n_trials} variants tried across {len(self.entries)} strategies"]
        if self.skipped:
            lines.append("skipped (missing data): " + ", ".join(
                f"{name} [{', '.join(miss)}]" for name, miss in self.skipped.items()))
        lines.append("")
        header = f"{'strategy':20} {'return':>8} {'sharpe':>7} {'halves':>13} {'β MKT':>6} {'α t':>6} {'DSR p':>6}  verdict"
        lines.append(header)
        lines.append("-" * len(header))
        for _, r in board.iterrows():
            lines.append(
                f"{r['strategy']:20} {r['return']:+8.1%} {r['sharpe']:+7.2f} {r['halves']:>13} "
                f"{_num(r['MKT beta'], '+.2f'):>6} {_num(r['alpha t'], '+.2f'):>6} {_num(r['DSR p'], '.3f'):>6}  "
                f"{r['verdict']:9} {r['why']}"
            )
        passed = int((board["verdict"] == "PASS").sum()) if not board.empty else 0
        lines.append("")
        lines.append(f"{passed} of {len(self.entries)} strategies passed every gate.")
        return "\n".join(lines)

    def to_markdown(self, title: str = "Strategy zoo") -> str:
        board = self.leaderboard()
        s = self.panel_stats
        out = [f"# {title}", ""]
        if s:
            out += [
                f"- Data: {s.get('source', 'panel')}, {s.get('tokens')} tokens over {s.get('days', 0):.1f} days "
                f"({s.get('dead_fraction', 0):.0%} died in the sample)",
                f"- Book size tested: ${s.get('capital_usd', 0):,.0f}",
            ]
        out += [
            f"- Variants tried (all counted as trials): **{self.n_trials}**",
            f"- Strategies judged: {len(self.entries)}; passed every gate: "
            f"**{int((board['verdict'] == 'PASS').sum()) if not board.empty else 0}**",
            "",
            "| Strategy | Family | Best params | Return | Sharpe | Halves | β MKT | α t | DSR p | Capacity | Verdict |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for _, r in board.iterrows():
            out.append(
                f"| `{r['strategy']}` | {r['family']} | {r['best params']} | {r['return']:+.1%} | {r['sharpe']:+.2f} | "
                f"{r['halves']} | {_num(r['MKT beta'], '+.2f')} | {_num(r['alpha t'], '+.2f')} | "
                f"{_num(r['DSR p'], '.3f')} | ${r['capacity']:,.0f} | **{r['verdict']}**: {r['why']} |"
            )
        if self.skipped:
            out += ["", "Not run (data not in this panel): " + ", ".join(
                f"`{n}` needs {', '.join(m)}" for n, m in self.skipped.items())]
        if "narratives" in s:
            out += ["", "## Narratives in the sample", "", "| Narrative | Tokens | Died | Volume share |", "|---|---:|---:|---:|"]
            for name, row in s["narratives"].iterrows():  # type: ignore[union-attr]
                out.append(f"| {name} | {int(row['tokens'])} | {int(row['died'])} | {row['volume_share']:.1%} |")
        out += ["", "Verdict order: IDLE, LOSS, UNSTABLE, BETA, LUCK, THIN, DUPLICATE, PASS. "
                "The deflated Sharpe uses the total trial count of the whole zoo."]
        return "\n".join(out) + "\n"


def run_zoo(
    panel: TokenPanel,
    *,
    strategies: Sequence[str] | None = None,
    config: MemeBacktestConfig | None = None,
    capitals: tuple[float, ...] | None = None,
    source: str = "panel",
    progress: bool = False,
) -> ZooResult:
    config = config or MemeBacktestConfig()
    capitals = capitals or tuple(sorted({config.capital_usd / 2, config.capital_usd, config.capital_usd * 5,
                                         config.capital_usd * 25}))
    ctx = StrategyContext(panel)
    found = catalog(ctx, strategies)

    rows, best = [], {}
    for strategy in found.available:
        for params in strategy.variants():
            weights = strategy.weights(panel, ctx, **params)
            metrics = meme_backtest(weights, panel, config).metrics
            rows.append({
                "strategy": strategy.name, "family": strategy.family, "params": _fmt_params(params),
                "sharpe": metrics["sharpe_ratio"], "total_return": metrics["total_return"],
                "max_drawdown": metrics["max_drawdown"], "cost_per_dollar": metrics["cost_per_dollar_traded"],
            })
            if strategy.name not in best or metrics["sharpe_ratio"] > best[strategy.name][0]:
                best[strategy.name] = (metrics["sharpe_ratio"], params)
        if progress:
            print(f"  {strategy.name:20} {len(strategy.variants())} variants", flush=True)
    trials = pd.DataFrame(rows)

    factors = meme_factor_returns(panel, min_liquidity_usd=config.costs.min_liquidity_usd)
    sheets = {}
    for name, (_, params) in best.items():
        weights = STRATEGIES[name].weights(panel, ctx, **params)
        sheets[name] = tear_sheet(weights, panel, n_trials=max(len(trials), 1), config=config,
                                  capitals=capitals, factors=factors)

    returns = pd.DataFrame({n: sh.backtest.returns for n, sh in sheets.items()})
    corr = returns.corr() if not returns.empty else pd.DataFrame()

    entries: list[ZooEntry] = []
    passed: list[str] = []
    for name in sorted(sheets, key=lambda n: -sheets[n].backtest.metrics["sharpe_ratio"]):
        verdict, reason = _verdict(sheets[name], config.capital_usd)
        if verdict == "PASS":
            twins = [p for p in passed if abs(corr.loc[name, p]) > _MAX_CORR]
            if twins:
                verdict, reason = "DUPLICATE", f"{corr.loc[name, twins[0]]:.2f} correlated with {twins[0]}"
            else:
                passed.append(name)
        entries.append(ZooEntry(name, STRATEGIES[name].family, best[name][1], sheets[name], verdict, reason))

    survivors = panel.survivorship_report()
    stats: dict[str, object] = {
        "source": source,
        "tokens": int(survivors["n_tokens"]),
        "dead_fraction": survivors["dead_fraction"],
        "days": (panel.close.index[-1] - panel.close.index[0]) / pd.Timedelta(days=1),
        "capital_usd": config.capital_usd,
    }
    if panel.meta is not None:
        stats["narratives"] = narrative_summary(panel, ctx.labels).head(8)
    return ZooResult(trials, entries, found.unavailable, corr, stats)


def _verdict(sheet: TearSheet, capital_usd: float) -> tuple[str, str]:
    m = sheet.backtest.metrics
    if m["mean_invested_fraction"] == 0:
        return "IDLE", "no entry signal fired on this panel"
    if m["total_return"] <= 0 or m["sharpe_ratio"] <= 0:
        return "LOSS", f"{m['total_return']:+.0%} net, {m['cost_per_dollar_traded']:.1%} cost per $ traded"
    if sheet.first_half_sharpe < 0 or sheet.second_half_sharpe < 0:
        return "UNSTABLE", f"halves Sharpe {sheet.first_half_sharpe:+.1f} / {sheet.second_half_sharpe:+.1f}"
    a = sheet.attribution
    if a is not None and a.t_stats.get("alpha", 0.0) < _ALPHA_T and a.r_squared >= _BETA_R2:
        return "BETA", f"alpha t {a.t_stats.get('alpha', 0.0):+.1f}, factors explain R² {a.r_squared:.2f}"
    if sheet.dsr is None or sheet.dsr.pvalue >= _DSR_P:
        p = sheet.dsr.pvalue if sheet.dsr is not None else float("nan")
        n = sheet.dsr.n_trials if sheet.dsr is not None else 0
        return "LUCK", f"deflated Sharpe p {p:.2f} over {n} trials"
    if sheet.capacity_usd < capital_usd:
        return "THIN", f"capacity ${sheet.capacity_usd:,.0f} below the ${capital_usd:,.0f} book"
    return "PASS", f"alpha t {a.t_stats.get('alpha', 0.0):+.1f}, DSR p {sheet.dsr.pvalue:.3f}" if a is not None \
        else f"DSR p {sheet.dsr.pvalue:.3f}"


def _fmt_params(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def _num(value: float, spec: str) -> str:
    return "n/a" if value is None or not np.isfinite(value) else format(value, spec)
