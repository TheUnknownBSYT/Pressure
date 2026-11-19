# Round 1 factor audit

Run from the project root with the pinned project dependencies, the six competition CSVs and the frozen `submission/` release. The first command builds the original pressure and performance inputs when they are missing.

```bash
python build_submission.py --through transfer
python -m factor_audit
python -m pytest -q
```

Open [the audit and ordered recommendations](../outputs/factor_audit/audit.md). All experiment files are written to `outputs/factor_audit/`. The predictor, pressure-classification and ranking experiments can also be rerun separately with `python -m factor_audit.forecast`, `python -m factor_audit.stakes` and `python -m factor_audit.rank`, in that order. The original submission provides regression targets. The ranking stage checks the original player metrics against those targets, and the diagnostics stage verifies the hashes of the six inputs and four submission files.

- `forecast.py` estimates runs and wickets in five expanding date windows. It replays the published expectations and fits competition, opponent, location and recent-form alternatives.
- `features.py` defines country and tournament identification and builds historical team and player ratings.
- `opposition.py` tests a previous-XI proxy and lagged team win strength. `venue.py` tests prior appearances at the ground and competition-specific win calibration.
- `stakes.py` varies the importance assigned to the stage, keeping the win model fixed. It evaluates the resulting index against later win-probability movement on legal balls.
- `rank.py` applies the original match bootstrap and eligibility rules. It tests both each method's own future target and the published historical definition. `diagnostics.py` adds matched-player comparisons, recency of ordinary form, role exposure and input checks.
- `report.py` renders the findings, coverage table and recommended orders from the completed experiments. The written interpretation applies to the verified release dataset spanning 2022–26.
