# Changelog

## 0.3.0

- **Strategy library**: 12 strategies in momentum, event, reversal, flow,
  narrative and sentiment families (`ai_quant_lab/meme/strategies.py`).
- **Strategy zoo**: every variant counted as a trial, best of each strategy
  judged with a full tear sheet and a verdict (IDLE, LOSS, UNSTABLE, BETA,
  LUCK, THIN, DUPLICATE, PASS); markdown report; `python -m ai_quant_lab.meme zoo`.
- **Narratives**: token themes from names and symbols, narrative heat, return,
  launch saturation (`narratives.py`).
- **Market mood**: breadth, launch rate, death rate, volume impulse, buy
  pressure, curve progress and a composite risk appetite (`sentiment.py`);
  external sentiment feeds with a publication lag.
- **Collector**: token names and symbols, GeckoTerminal categories and socials
  with first-seen time, developer share, honeypot flag, buy/sell ratio,
  trending ranks for tracked pools. Old databases migrate on open.
- **Synthetic tapes**: narratives and names; `narrative_attention` and
  `narrative_strength` switches that never change base-tape prices.
- **Research loop**: Claude receives narrative and mood fields.
- **Tests**: 176, including a look-ahead test for every library strategy.

## 0.2.0

- Robinhood Chain: multi-network collector, per-pool Uniswap fee tiers,
  chain cost presets.
- Token fields: holders, top-10 share, mint and freeze authority, bonding-curve
  progress; rug screen.
- Project website and README.

## 0.1.0

- Memecoin layer: point-in-time universe, AMM execution costs, meme factors,
  capacity, tear sheet, GeckoTerminal collector, synthetic tapes.
