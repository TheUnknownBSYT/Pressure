# Release verification

**Implementation revision** [e13c2af81cfc33690ade27dc894840a6fc7247b0](https://github.com/TheUnknownBSYT/Pressure/commit/e13c2af81cfc33690ade27dc894840a6fc7247b0). The release started from committed GitHub main at b44dfc7.

- Ran the complete pipeline from the six unchanged local input CSVs in a clean work directory using Python 3.14 and the pinned analysis dependencies. Rebuilt the report and repeated the tests and publication verifier after the final copy edits.
- Reconciled 200,999 raw delivery rows, 113,031 pressure scores and 113,017 response rows. Checked earlier-match cutoff dates, static and dynamic pressure equations, attribution, forecasts from exported parameters, matched ordinary baselines and eligibility across 1,195 player-role records. Independently reproduced the ranking scores and intervals for eligible players.
- Reproduced both top-ten CSVs byte for byte against the prior local release. Verified ten unique official IDs per role, ranks 1–10, eligibility, and the full ranking-table rows in PDF, Markdown and Word. Inspected all ten rendered pages for legibility. The curated package contains PDF, Markdown and CSVs, and the local build also generated an editable Word copy.
- **63 tests passed, no failures, five skipped.** The skips require historical results from the old final strategy, independent audit (two checks), Phase 2 and Phase 3. The release-specific checks ran separately and passed.
- All 20 adjacent 95% intervals for the quality difference included zero. Fourteen selections kept positive uplift after every evaluable single-fixture deletion. Deletions refit the ordinary baseline on retained fixtures with comparison cells held fixed. Intervals are conditional on the fitted models and historical sample.
- The prior 2026 season was already inspected in development. Forward results are out of time for the specified fits and have limited predictive value. Nine selections had no matched IPL pressure balls.

[Machine-readable verification](verification.json) contains source hashes, published file hashes and ordered official player IDs.
