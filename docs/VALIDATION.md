# Validation: the gate that separates edge from noise

This is the longest doc on purpose. Statistical rigor is the only feature
ai-quant-lab has that the alternatives don't.

## The problem

When you test N strategies and report the best one's Sharpe, you are NOT
estimating the true Sharpe of that strategy. You are estimating the expected
maximum of N noisy estimates — which is positive and grows with N even when
no strategy has any edge.

### Numerical intuition

Run 1000 random buy/sell strategies on a no-edge GBM series of 1000 daily
bars. The expected maximum Sharpe (annualized) is approximately:

```
E[max SR] ≈ √(Var(SR)) · ( (1 - γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e)) )

where Var(SR) under H₀ ≈ 1/(T-1) per period, and γ is the Euler-Mascheroni constant.
```

For N=1000, T=1000, this gives an annualized expected max around **1.5**.

That is: the best of 1000 random strategies will, on average, have an
annualized Sharpe of ~1.5. Which is the Sharpe of a "good" strategy. Which is
why the best-of-many naïve report is meaningless.

## The fix

The Deflated Sharpe Ratio (Bailey & López de Prado, 2014) subtracts the
expected maximum from the observed Sharpe, then re-tests for significance
under the resulting distribution. A strategy with a deflated SR p-value below
α has an observed Sharpe that's unlikely to be explained by multiple testing.

In `ai-quant-lab`, this is exposed as `validation/deflated_sharpe.py`:

```python
from ai_quant_lab.validation import deflated_sharpe

result = deflated_sharpe(strategy_returns, n_trials=100)
print(result.pvalue, result.passes(alpha=0.05))
```

`n_trials` is the **honest** count of strategies you've tested. This is the
hard part. People forget about the 50 hyperparameter sweeps. They forget
about the three feature variants. They forget about the 12 lookback windows.
`ResearchMemory` exists to track this for you.

## Walk-forward

Walk-forward evaluation slides a (train, test) window forward in time.
Concatenating the test slices gives an honest out-of-sample equity curve —
the only kind that matters.

Two knobs to remember:

- **`purge`**: number of bars between train end and test start. Set to the
  maximum label-overlap horizon for triple-barrier labels (or any label that
  looks forward more than one bar). Otherwise a clean-looking validation can
  still leak via overlapping labels.
- **`mode`**: `'rolling'` keeps the train window fixed-size; `'expanding'`
  grows it. Rolling is the right default for non-stationary markets (which is
  most markets).

## Purged combinatorial CV

When you want to estimate the _distribution_ of out-of-sample performance,
not just a single point, use `combinatorial_purged_cv`. It partitions the
series into N groups, picks K of them as test, the rest as train, and yields
C(N, K) splits. Every bar appears in multiple test sets — a richer signal
than walk-forward's single pass.

The `purge` parameter drops bars adjacent to the test block (where labels
might overlap). The `embargo` parameter drops bars _after_ the test block to
prevent the test set's information from leaking into the training of a
subsequent fold.

## Diagnostics

`validation/diagnostics.py` provides three orthogonal checks on a candidate:

- `degradation_ratio(IS_sharpe, OOS_sharpe)`: how much performance dropped
  going from in-sample to out-of-sample.
  - `> 1.0`: OOS beat IS. Suspicious unless sample size is small.
  - `0.5 – 1.0`: healthy.
  - `0 – 0.5`: marginal. Likely overfit.
  - `< 0`: sign flip. Edge was noise.
- `fold_stability(fold_sharpes)`: mean, std, fraction-positive, and t-stat
  across walk-forward folds. A good strategy has fraction-positive ≥ 0.7.
- `regime_breakdown(returns, regime_labels)`: per-regime metrics. A genuine
  edge usually shows up in a specific regime (e.g. low-vol days) rather than
  uniformly.

## The hard gate

Three constraints, all must pass:

1. CriticAgent verdict is `pass`.
2. `deflated_sharpe(...).pvalue < settings.dsr_pvalue_max` (default 0.05).
3. Maximum |correlation| with already-accepted strategies < `settings.max_correlation`
   (default 0.6).

There is no override. If the gate is wrong about your favorite strategy, the
right response is to fix the gate's input (more data, more trials counted)
or accept that the strategy probably doesn't work.

## Common mistakes

- **Forgetting to count parameter sweeps as trials.** A 21-day momentum that
  beats a 5-day momentum on the same data isn't two strategies; it's a
  strategy chosen from a family of 17 candidates. n_trials = 17.
- **Setting `purge=0` with triple-barrier labels.** Labels look forward;
  walk-forward without purge leaks.
- **Reusing the same OOS window across experiments.** Once you've made a
  decision based on a window, that window is effectively in-sample.

## Further reading

- Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
  _Journal of Portfolio Management_.
- López de Prado, M. (2018). _Advances in Financial Machine Learning_,
  chapters 7 (cross-validation) and 12 (combinatorial CV).
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). "...and the Cross-Section of
  Expected Returns." _Review of Financial Studies_. (On p-hacking in factor
  research.)
