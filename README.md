# Pressure

I built this project for SupeRR Selector to answer two questions. When could the next ball meaningfully change a T20 match? And which players performed above both the standard for that situation and their own usual standard?

- [Methodology PDF](submission/SupeRR_Selector_Final_Methodology.pdf)
- [Batting order](submission/final_batters_top10.csv) and [bowling order](submission/final_bowlers_top10.csv)
- [Submission instructions and evidence](submission/README.md)

## How I measure pressure

1. **Rebuild the match one ball at a time.** Before each delivery I reconstruct the score, wickets, balls remaining and chase target. Wides and no-balls affect the score without using up a legal ball. The rebuild checks the match records for inconsistencies before using them.
2. **Estimate the chance of winning.** I fit separate models for the first innings and for chases, using earlier matches. For each ball I estimate how the win chance would move under several plausible outcomes, such as a dot, a boundary or a wicket.
3. **Combine three ingredients.** Half the pressure score comes from how much the next ball could move the win chance. Another 30% comes from urgency adjusted for how open the contest still is. The final 20% reflects the match stage. A ball counts as high pressure if it reaches the top-quarter cutoff set from earlier quarters.

This is why 22 off six can carry more pressure than 36 off six. The second chase is harder, but many plausible deliveries leave the result almost as settled as before. All 12 variants I tested agreed on that ordering. The estimated chance of winning from either position depends on the model and the available examples.

## How I compare players

I estimate runs and dismissal or wicket probabilities for comparable balls. The models account for the innings, phase, match situation, role and earlier records of the opposition. I give a wicket a run value based on how much scoring teams usually lose when they have one fewer wicket in hand.

Batters gain credit for scoring above expectation and avoiding dismissals. Bowlers gain credit for conceding fewer runs and taking wickets above expectation. I put the contributions on the same scale, runs of value per six legal balls.

A player needs positive value against comparable players and improvement over their own ordinary deliveries in the same innings, phase and competition. Strictly more than half of weighted pressure appearances must beat the player’s own matched ordinary benchmark. I require enough matched balls and appearances before ranking anyone. An appearance of fewer than six pressure balls gets proportionally less weight. I repeatedly resample whole matches and rank eligible players by the 20th percentile of their estimated pressure contribution. This favors players whose value holds up across different mixes of matches.

## What changed after testing

I tried changing the pressure weights with the match situation and adding information such as recent form, consecutive boundaries and gaps in appearances. Role and opposition improved run forecasts. The dynamic weights gave a small overall gain on one test of win-chance movement and slightly worse results in chases. The personal-history features did not produce a reliable overall improvement. I use those experiments to test the strategy and keep the fixed weights for the final ranking.

I also checked whether rankings made from earlier seasons predicted later results. That relationship was weak, especially for batters. The rankings summarize historical performance, and nearby positions have substantial uncertainty. International performances give limited evidence about how a player will fare in the IPL.

## Run it locally

Use Python 3.14. Put these six competition CSVs in the repository root: `ball_by_ball.csv`, `matches.csv`, `competitions.csv`, `teams.csv`, `playing_eleven.csv` and `competition_team_player_mapping.csv`. Keep these proprietary inputs and all match-level outputs local. Only summary selections and documentation belong in the release.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python build_submission.py --through response
python -m final_strategy.round_one
python -m pytest -q
```

If the original response and pressure artifacts already exist locally, start at the second command. The exporter recomputes both complete pools, checks names against official IDs, and writes the two dropdown lists and the compact [20-player CSV](submission/round_one_rankings.csv). Private full-pool diagnostics stay under ignored `outputs/`.

## Round 1 submission

The [report](submission/SupeRR_Selector_Final_Methodology.md) gives the approved formula, all ranks and IDs, the chronological validation and the separate auction screen. The [PDF](submission/SupeRR_Selector_Final_Methodology.pdf) is the upload copy. Availability was checked on 18 November 2026. Refresh it against the official auction list before bidding.

The underlying pressure and response implementations are unchanged. The supplementary files under `submission/evidence/` document the previous release and use its eligibility rule where applicable.
