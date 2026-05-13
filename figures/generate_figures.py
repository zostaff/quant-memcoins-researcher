"""Regenerate every figure used in the README from scratch.

Each figure is a function returning a matplotlib Figure. Calling `main()`
writes all eight to ./figures/*.png using a consistent dark theme.

The figures are illustrative — none of them claim to be real production data.
They visualize the IDEAS from the article (the three eras of quant, multiple
testing penalty, leakage examples, walk-forward layout).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle
except ImportError as exc:
    raise SystemExit("Install matplotlib: pip install matplotlib") from exc

from ai_quant_lab.validation.deflated_sharpe import _expected_max_sharpe

FIGURE_DIR = Path(__file__).resolve().parent
DPI = 130
ACCENT = "#7AD6F8"
ACCENT_2 = "#FF6B9D"
ACCENT_3 = "#FFD166"
MUTED = "#9AA5AB"


def _style() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#0E1116",
            "axes.facecolor": "#161B22",
            "axes.edgecolor": "#2A2F38",
            "axes.labelcolor": "#E0E0E0",
            "xtick.color": "#9AA5AB",
            "ytick.color": "#9AA5AB",
            "text.color": "#E0E0E0",
            "axes.titleweight": "bold",
            "grid.color": "#252A33",
            "grid.alpha": 0.4,
        }
    )


def fig_01_three_eras() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    years = np.arange(1980, 2031)

    human = np.clip(2.0 - 0.025 * (years - 1980), 0.3, 2.0)
    factor_model = np.clip(0.5 + 0.7 * np.exp(-((years - 2005) ** 2) / 200), 0.0, 1.5)
    ai_research = np.clip(0.05 * np.exp((years - 2010) / 8), 0.0, 4.0)

    ax.fill_between(years, 0, human, color=ACCENT, alpha=0.18, label="Human / classical (1980-2005)")
    ax.fill_between(years, 0, factor_model, color=ACCENT_3, alpha=0.18, label="Factor / ML (2000-2020)")
    ax.fill_between(years, 0, ai_research, color=ACCENT_2, alpha=0.18, label="AI-assisted research (2020+)")

    ax.plot(years, human, color=ACCENT, linewidth=2)
    ax.plot(years, factor_model, color=ACCENT_3, linewidth=2)
    ax.plot(years, ai_research, color=ACCENT_2, linewidth=2)

    ax.set_title("The three eras of quant research")
    ax.set_xlabel("Year")
    ax.set_ylabel("Strategies tested per researcher per week (log-scale, illustrative)")
    ax.set_yscale("log")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


def fig_02_agent_loop() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    nodes = [
        ("Hypothesis\nAgent", 0.5, 0.85, ACCENT),
        ("Critic\nAgent", 0.2, 0.6, ACCENT_2),
        ("Code\nAgent", 0.8, 0.6, ACCENT_3),
        ("Sandbox\n+ Backtest", 0.8, 0.3, ACCENT),
        ("Three Gates\n(critic · DSR · corr)", 0.5, 0.15, ACCENT_2),
        ("Research\nMemory", 0.2, 0.3, ACCENT_3),
    ]
    for label, x, y, c in nodes:
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.09, y - 0.06), 0.18, 0.12,
                boxstyle="round,pad=0.02", facecolor=c, alpha=0.18, edgecolor=c, linewidth=1.5,
            )
        )
        ax.text(x, y, label, ha="center", va="center", color="white", fontsize=10, weight="bold")

    edges = [
        (0, 1, "review"), (0, 2, "render"), (1, 2, "pass"),
        (2, 3, "execute"), (3, 4, "metrics"), (4, 5, "log every trial"),
        (5, 0, "context for next"),
    ]
    for a, b, lab in edges:
        x1, y1 = nodes[a][1], nodes[a][2]
        x2, y2 = nodes[b][1], nodes[b][2]
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2, alpha=0.6),
        )
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, lab, color=MUTED, fontsize=8, ha="center")

    ax.set_title("The research loop: generate cheaply, validate hard, remember everything")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def fig_03_time_compression() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    categories = ["Idea", "Spec", "Code", "Backtest", "Validate", "Conclude"]
    human = np.array([1.5, 2.0, 8.0, 3.0, 4.0, 1.5])  # days
    ai = np.array([0.05, 0.05, 0.1, 0.05, 0.2, 0.05])
    x = np.arange(len(categories))
    width = 0.38
    ax.bar(x - width / 2, human, width, color=ACCENT, label="Human researcher (days)")
    ax.bar(x + width / 2, ai, width, color=ACCENT_2, label="AI-assisted (days)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Wall-clock days per strategy (log)")
    ax.set_title("Same workflow, ~100× compression")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_04_multiple_testing() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    n_trials = np.arange(1, 1001)
    expected_max = np.array([_expected_max_sharpe(n, trial_variance=1.0 / 999) for n in n_trials])
    expected_max_ann = expected_max * np.sqrt(252)
    ax.plot(n_trials, expected_max_ann, color=ACCENT, linewidth=2,
            label="Expected max Sharpe under H₀ (no edge)")
    ax.axhline(1.0, color=ACCENT_3, linestyle="--", alpha=0.6, label="A typical 'good' Sharpe")
    ax.set_xscale("log")
    ax.set_xlabel("Number of strategies tested (log)")
    ax.set_ylabel("Annualized Sharpe ratio")
    ax.set_title("How much Sharpe is 'free' from multiple testing alone")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_05_hedge_fund_returns() -> plt.Figure:
    """Stylized chart: dispersion of hedge-fund alpha through time."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    years = np.arange(2000, 2026)
    rng = np.random.default_rng(0)
    alpha_p25 = -0.05 + 0.01 * np.sin(years / 3) + rng.normal(0, 0.01, len(years))
    alpha_p50 = 0.02 + 0.005 * np.sin(years / 4) + rng.normal(0, 0.008, len(years))
    alpha_p75 = 0.10 - 0.002 * (years - 2000) + rng.normal(0, 0.01, len(years))
    ax.fill_between(years, alpha_p25, alpha_p75, color=ACCENT, alpha=0.18, label="25-75th pctile")
    ax.plot(years, alpha_p50, color=ACCENT, linewidth=2, label="Median alpha")
    ax.axhline(0, color=MUTED, linestyle="--", alpha=0.4)
    ax.set_title("Hedge fund alpha dispersion — the easy edge keeps shrinking")
    ax.set_ylabel("Annual alpha vs SPX")
    ax.set_xlabel("Year")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_06_ai_vs_human() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    weeks = np.arange(1, 53)
    human = np.cumsum(np.random.default_rng(1).normal(2, 1, len(weeks)))
    ai = np.cumsum(np.random.default_rng(2).normal(60, 20, len(weeks)))
    ax.plot(weeks, human, color=ACCENT, linewidth=2, label="Human researcher")
    ax.plot(weeks, ai, color=ACCENT_2, linewidth=2, label="AI-assisted (single user)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Cumulative strategies tested")
    ax.set_yscale("log")
    ax.set_title("Throughput, one year of work")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_07_walkforward_layout() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")
    n_folds = 5
    train_len = 5
    test_len = 1
    bar_height = 0.12
    correct_y_base = 0.7
    wrong_y_base = 0.25
    for i in range(n_folds):
        # Correct: walk-forward with no peek
        train_x = i * test_len
        test_x = train_x + train_len
        ax.add_patch(Rectangle((train_x, correct_y_base), train_len, bar_height,
                               facecolor=ACCENT, alpha=0.7))
        ax.add_patch(Rectangle((test_x, correct_y_base), test_len, bar_height,
                               facecolor=ACCENT_2, alpha=0.9))
        # Wrong: random k-fold
        for j in range(train_len + test_len):
            color = ACCENT_2 if j % (train_len + 1) == i else ACCENT
            alpha = 0.9 if color == ACCENT_2 else 0.5
            ax.add_patch(Rectangle((i * test_len + j, wrong_y_base), 1, bar_height,
                                   facecolor=color, alpha=alpha))

    ax.text(-1, correct_y_base + bar_height / 2, "Correct\n(walk-forward)", ha="right", va="center",
            color="white", fontsize=10, weight="bold")
    ax.text(-1, wrong_y_base + bar_height / 2, "Wrong\n(shuffled k-fold)", ha="right", va="center",
            color="white", fontsize=10, weight="bold")
    ax.set_title("Walk-forward respects time. Shuffled k-fold doesn't.")
    ax.set_xlim(-7, n_folds * test_len + train_len + 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def fig_08_leakage_examples() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    rng = np.random.default_rng(3)
    n = 250
    returns = rng.normal(0.0002, 0.012, n)
    prices = pd.Series(100 * np.exp(np.cumsum(returns)))
    clean_signal = prices.shift(1).rolling(21).mean()
    leaky_signal = prices.rolling(21, center=True).mean()
    ax.plot(prices.values, color=MUTED, alpha=0.7, label="price")
    ax.plot(clean_signal.values, color=ACCENT, linewidth=2, label="clean signal (lagged)")
    ax.plot(leaky_signal.values, color=ACCENT_2, linewidth=2,
            label="leaky signal (centered window)", linestyle="--")
    ax.set_title("Where leakage hides: it tracks the price too well")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Level")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


FIGURES = {
    "01_three_eras.png": fig_01_three_eras,
    "02_agent_loop.png": fig_02_agent_loop,
    "03_time_compression.png": fig_03_time_compression,
    "04_multiple_testing.png": fig_04_multiple_testing,
    "05_hedge_fund_returns.png": fig_05_hedge_fund_returns,
    "06_ai_vs_human.png": fig_06_ai_vs_human,
    "07_walkforward_correct_vs_wrong.png": fig_07_walkforward_layout,
    "08_leakage_examples.png": fig_08_leakage_examples,
}


def main() -> None:
    _style()
    for filename, fn in FIGURES.items():
        fig = fn()
        path = FIGURE_DIR / filename
        fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path.relative_to(FIGURE_DIR.parent)}")


if __name__ == "__main__":
    main()
