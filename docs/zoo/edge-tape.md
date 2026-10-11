# Strategy zoo: synthetic edge tape, 42 days, seed 4

- Data: synthetic edge tape, 42 days, seed 4, 2032 tokens over 42.0 days (93% died in the sample)
- Book size tested: $2,000
- Variants tried (all counted as trials): **54**
- Strategies judged: 10; passed every gate: **1**

| Strategy | Family | Best params | Return | Sharpe | Halves | β MKT | α t | DSR p | Capacity | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `narrative_rotation` | narrative | k=2, fast=36, top_n=12 | +2680.1% | +17.62 | +21.6 / +13.6 | +0.08 | +6.67 | 0.000 | $2,000 | **PASS**: alpha t +6.7, DSR p 0.000 |
| `narrative_early` | narrative | max_saturation=3.0, top_n=16 | +2222.4% | +15.42 | +19.3 / +10.5 | +0.08 | +5.88 | 0.002 | $2,000 | **DUPLICATE**: 0.89 correlated with narrative_rotation |
| `narrative_momentum` | narrative | lookback=144, k=1 | +809.8% | +9.43 | +13.7 / +2.8 | +0.09 | +3.45 | 0.187 | $2,000 | **LUCK**: deflated Sharpe p 0.19 over 54 trials |
| `holder_momentum` | momentum | lookback=72, top_n=16 | +182.0% | +4.25 | +2.3 / +6.2 | +0.22 | +3.70 | 0.806 | $2,000 | **LUCK**: deflated Sharpe p 0.81 over 54 trials |
| `volume_breakout` | event | z=2.0, hold=144 | +73.5% | +3.38 | -0.1 / +6.9 | +0.28 | +0.73 | 0.877 | $1,000 | **UNSTABLE**: halves Sharpe -0.1 / +6.9 |
| `regime_momentum` | sentiment | lookback=36, threshold=0.5, top_n=8 | +37.2% | +2.36 | -0.8 / +5.5 | +0.05 | +0.88 | 0.933 | $1,000 | **UNSTABLE**: halves Sharpe -0.8 / +5.5 |
| `buy_pressure` | flow | threshold=0.6, top_n=8 | +7.4% | +1.62 | +4.4 / -1.4 | +0.00 | +0.72 | 0.961 | $2,000 | **UNSTABLE**: halves Sharpe +4.4 / -1.4 |
| `flow_momentum` | momentum | lookback=72, top_n=8 | -17.3% | +2.08 | -0.4 / +4.5 | +0.24 | +0.51 | 0.945 | $1,000 | **LOSS**: -17% net, 3.5% cost per $ traded |
| `quality_composite` | momentum | top_n=24 | -69.6% | -1.25 | -1.4 / -1.1 | +0.27 | -1.30 | 0.997 | $0 | **LOSS**: -70% net, 2.5% cost per $ traded |
| `dip_reversal` | reversal | depth=0.3, hold=144 | -97.6% | -7.58 | -10.5 / -2.5 | +0.13 | -3.29 | 1.000 | $0 | **LOSS**: -98% net, 2.5% cost per $ traded |

Not run (data not in this panel): `curve_graduation` needs graduation, `trending_entry` needs trending_rank

## Narratives in the sample

| Narrative | Tokens | Died | Volume share |
|---|---:|---:|---:|
| food | 187 | 178 | 14.4% |
| politics | 158 | 149 | 14.1% |
| other | 498 | 470 | 12.4% |
| animals | 155 | 147 | 11.8% |
| ai | 149 | 129 | 11.4% |
| cat | 132 | 121 | 9.9% |
| dog | 111 | 105 | 5.3% |
| frog | 104 | 87 | 4.7% |

Verdict order: IDLE, LOSS, UNSTABLE, BETA, LUCK, THIN, DUPLICATE, PASS. The deflated Sharpe uses the total trial count of the whole zoo.
