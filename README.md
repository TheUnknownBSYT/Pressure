# Pressure

I built this project for SupeRR Selector to answer two questions. When could the next ball meaningfully change a T20 match? And which players performed above both the standard for that situation and their own usual standard?

## How I measure pressure

1. **Rebuild the match one ball at a time.** Before each delivery I reconstruct the score, wickets, balls remaining and chase target. Wides and no-balls affect the score without using up a legal ball. The rebuild checks the match records for inconsistencies before using them.
2. **Estimate the chance of winning.** I fit separate models for the first innings and for chases, using earlier matches. For each ball I estimate how the win chance would move under several plausible outcomes, such as a dot, a boundary or a wicket.
3. **Combine three ingredients.** Half the pressure score comes from how much the next ball could move the win chance. Another 30% comes from urgency adjusted for how open the contest still is. The final 20% reflects the match stage. A ball counts as high pressure if it reaches the top-quarter cutoff set from earlier quarters.

This is why 22 off six can carry more pressure than 36 off six. The second chase is harder, but many plausible deliveries leave the result almost as settled as before. All 12 variants I tested agreed on that ordering. The estimated chance of winning from either position depends on the model and the available examples.

## How I compare players

I estimate runs and dismissal or wicket probabilities for comparable balls. The models account for the innings, phase, match situation, role and earlier records of the opposition. I give a wicket a run value based on how much scoring teams usually lose when they have one fewer wicket in hand.

Batters gain credit for scoring above expectation and avoiding dismissals. Bowlers gain credit for conceding fewer runs and taking wickets above expectation. I put the contributions on the same scale, runs of value per six legal balls.

A player needs positive value against comparable players and improvement over their own ordinary deliveries in the same innings, phase and competition. I require enough matched balls and appearances before ranking anyone. An appearance of fewer than six pressure balls gets proportionally less weight. I repeatedly resample whole matches and rank eligible players by the 20th percentile of their estimated pressure contribution. This favors players whose value holds up across different mixes of matches.

## What changed after testing

I tried changing the pressure weights with the match situation and adding information such as recent form, consecutive boundaries and gaps in appearances. Role and opposition improved run forecasts. The dynamic weights gave a small overall gain on one test of win-chance movement and slightly worse results in chases. The personal-history features did not produce a reliable overall improvement. I use those experiments to test the strategy and keep the fixed weights for the final ranking.

I also checked whether rankings made from earlier seasons predicted later results. That relationship was weak, especially for batters. The rankings summarize historical performance, and nearby positions have substantial uncertainty. International performances give limited evidence about how a player will fare in the IPL.

## Run it locally

Use Python 3.14. Put these six competition CSVs in the repository root: `ball_by_ball.csv`, `matches.csv`, `competitions.csv`, `teams.csv`, `playing_eleven.csv` and `competition_team_player_mapping.csv`. The competition data and generated submission files are excluded from this repository.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python build_submission.py --through transfer
python -m pytest -q
```

The build reconstructs the matches, tests pressure definitions, fits player expectations, generates rankings and checks how results transfer between competitions. The numerical results and logs go under `outputs/`. Some regression tests are designed for archived local builds and skip when those files are absent.

For the full local publishing workflow, install LibreOffice and Poppler and make sure `soffice` and `pdftotext` are on your PATH. Then run:

```bash
python -m pip install -r requirements-documents.txt
python build_submission.py --from-stage publication
```

## Where the implementation lives

- `pressure_pipeline/` and `build_pressure_dataset.py` rebuild matches and score deliveries.
- `final_strategy/consequence.py` tests the pressure definition and its alternatives.
- `final_strategy/response_model.py` estimates expected runs and wickets.
- `final_strategy/ratings.py` compares players with their peers and their own ordinary performances.
- `final_strategy/transfer.py` checks whether earlier international performance carries over to the IPL.
- `build_submission.py` runs the stages in order.

## Round 1 release

The supplied fixtures span **22 January 2022 to 1 November 2026**. Raw files record 200,999 deliveries, with 113,031 receiving a pressure score.

- [Methodology PDF](submission/SupeRR_Selector_Final_Methodology.pdf)
- [Batting order](submission/final_batters_top10.csv) and [bowling order](submission/final_bowlers_top10.csv)
- [Submission instructions and evidence](submission/README.md)

```text
Competition files
    → Match state before each ball
    → Pressure score and historical cutoff
    → Runs and wickets above comparable expectations
    → Personal ordinary baseline and eligibility
    → Match resampling and ordered lists
```

On the matched 2026 forward sample, correlations with later pressure quality were 0.064 for 22 batters and 0.265 for 29 bowlers. Every adjacent 95% interval for the difference in pressure quality included zero. Nine selections lacked matched IPL pressure balls. Ranking by the frequency of beating a personal ordinary benchmark kept six of the ten selections in each role. The release keeps the conservative average because it credits the size of contributions. The methodology explains where that choice matters.

The `audit` stage independently checks match state, pressure thresholds, model forecasts, eligibility and rankings from local data. See [verification and reproduction](submission/evidence/verification.md). Both ordered lists are ready to enter into Unstop, and the PDF is the separate upload.
