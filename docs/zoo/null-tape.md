# Strategy zoo: synthetic null tape, 42 days, seed 4

- Data: synthetic null tape, 42 days, seed 4, 2032 tokens over 42.0 days (94% died in the sample)
- Book size tested: $2,000
- Variants tried (all counted as trials): **54**
- Strategies judged: 10; passed every gate: **0**

| Strategy | Family | Best params | Return | Sharpe | Halves | β MKT | α t | DSR p | Capacity | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `buy_pressure` | flow | threshold=0.6, top_n=8 | -5.7% | -3.53 | -5.0 / +0.0 | -0.00 | -0.96 | 1.000 | $0 | **LOSS**: -6% net, 2.7% cost per $ traded |
| `narrative_rotation` | narrative | k=1, fast=36, top_n=12 | -78.9% | -3.72 | -7.3 / -0.1 | +0.07 | +1.00 | 1.000 | $0 | **LOSS**: -79% net, 2.4% cost per $ traded |
| `regime_momentum` | sentiment | lookback=144, threshold=0.5, top_n=8 | -83.3% | -6.57 | -9.5 / -3.9 | +0.04 | -2.06 | 1.000 | $0 | **LOSS**: -83% net, 2.3% cost per $ traded |
| `holder_momentum` | momentum | lookback=144, top_n=8 | -97.9% | -6.94 | -5.8 / -9.7 | +0.13 | -0.11 | 1.000 | $0 | **LOSS**: -98% net, 2.6% cost per $ traded |
| `quality_composite` | momentum | top_n=8 | -97.6% | -7.51 | -8.7 / -7.1 | +0.11 | -1.23 | 1.000 | $0 | **LOSS**: -98% net, 2.5% cost per $ traded |
| `narrative_early` | narrative | max_saturation=3.0, top_n=16 | -97.6% | -7.78 | -10.6 / -4.9 | +0.11 | +0.14 | 1.000 | $0 | **LOSS**: -98% net, 2.5% cost per $ traded |
| `narrative_momentum` | narrative | lookback=144, k=1 | -88.2% | -7.95 | -12.7 / -3.3 | +0.04 | -1.93 | 1.000 | $0 | **LOSS**: -88% net, 2.3% cost per $ traded |
| `dip_reversal` | reversal | depth=0.55, hold=144 | -98.2% | -8.18 | -8.3 / -11.4 | +0.14 | -2.40 | 1.000 | $0 | **LOSS**: -98% net, 2.5% cost per $ traded |
| `flow_momentum` | momentum | lookback=72, top_n=8 | -97.8% | -11.77 | -16.7 / +0.0 | +0.07 | -3.13 | 1.000 | $0 | **LOSS**: -98% net, 2.6% cost per $ traded |
| `volume_breakout` | event | z=2.0, hold=144 | -97.7% | -12.53 | -17.7 / +0.0 | +0.09 | -3.23 | 1.000 | $0 | **LOSS**: -98% net, 2.3% cost per $ traded |

Not run (data not in this panel): `curve_graduation` needs graduation, `trending_entry` needs trending_rank

## Narratives in the sample

| Narrative | Tokens | Died | Volume share |
|---|---:|---:|---:|
| other | 498 | 467 | 14.4% |
| food | 187 | 179 | 14.2% |
| politics | 158 | 153 | 11.5% |
| animals | 155 | 147 | 11.2% |
| cat | 132 | 122 | 8.3% |
| ai | 149 | 134 | 7.8% |
| dog | 111 | 106 | 6.2% |
| gaming | 120 | 117 | 5.8% |

Verdict order: IDLE, LOSS, UNSTABLE, BETA, LUCK, THIN, DUPLICATE, PASS. The deflated Sharpe uses the total trial count of the whole zoo.
