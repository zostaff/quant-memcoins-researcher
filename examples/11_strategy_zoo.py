"""Run the memecoin strategy library through every gate on a synthetic tape.

Narratives drive returns on this tape; on the null tape (`narrative_strength=0`)
they are visible in volume and launches but worth nothing. Change the tape and
watch which verdicts move. No API key or data needed; about a minute.
"""

from ai_quant_lab.meme import MemeBacktestConfig, UniverseSpec, run_zoo, simulate_universe


def main() -> None:
    panel = simulate_universe(UniverseSpec(n_days=14, seed=4, edge=0.5,
                                           narrative_attention=1.0, narrative_strength=1.0))
    result = run_zoo(
        panel,
        strategies=["narrative_rotation", "narrative_early", "holder_momentum", "regime_momentum"],
        config=MemeBacktestConfig(capital_usd=2_000),
        source="synthetic edge tape, 14 days",
    )
    print(result.summary())


if __name__ == "__main__":
    main()
