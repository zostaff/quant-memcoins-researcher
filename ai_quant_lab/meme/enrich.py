"""Research inputs: panel fields plus narrative and market-mood features.

What the Claude research loop (and any hand-written strategy) receives on top
of the raw panel. Each added field is time × token and point-in-time:

    narrative_heat       the token's narrative: short vs long share of volume
    narrative_return     the token's narrative: trailing liquidity-weighted return
    launch_saturation    the token's narrative: launches now vs usual pace
    risk_appetite        market-wide mood score, same value for every token
    breadth              share of tradeable tokens up over the last 3 hours
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_quant_lab.meme import narratives as N
from ai_quant_lab.meme.panel import TokenPanel
from ai_quant_lab.meme.sentiment import market_mood, risk_appetite


def research_panel(panel: TokenPanel) -> TokenPanel:
    labels = N.narrative_labels(panel)
    cols = panel.tokens
    fields: dict[str, pd.DataFrame] = {}
    if panel.meta is not None:
        fields["narrative_heat"] = N.to_tokens(N.narrative_heat(panel, labels, fast=36), labels, cols)
        fields["narrative_return"] = N.to_tokens(N.narrative_return(panel, labels, 36), labels, cols)
        fields["launch_saturation"] = N.to_tokens(N.launch_saturation(panel, labels), labels, cols)
    mood = market_mood(panel)
    for name, series in (("risk_appetite", risk_appetite(mood=mood)), ("breadth", mood["breadth"])):
        fields[name] = pd.DataFrame(np.repeat(series.to_numpy()[:, None], len(cols), axis=1),
                                    index=panel.close.index, columns=cols)
    alive = panel.close.notna()
    return panel.with_extra(**{k: v.where(alive) for k, v in fields.items()})
