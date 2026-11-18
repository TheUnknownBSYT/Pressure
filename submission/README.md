# SupeRR Selector Season 2 Round 1

Upload [SupeRR_Selector_Final_Methodology.pdf](SupeRR_Selector_Final_Methodology.pdf) in the Unstop methodology field. Enter the two ordered lists using the dropdowns, starting with rank 1.

- [Batters in rank order](final_batters_top10.csv)
- [Bowlers in rank order](final_bowlers_top10.csv)
- [Methodology in Markdown](SupeRR_Selector_Final_Methodology.md)

The CSVs give the exact dataset spelling and player ID. If Unstop displays a variant of a name, use the CSV and methodology as the reference for the intended player.

## Evidence

- [Candidate figures](evidence/candidate_metrics.csv) cover the selections and comparison candidates. Quality, ordinary, uplift and rating use run equivalents per six legal balls. Positive values favor the named player in either role. Match shares run from zero to one. Empty fields indicate unavailable estimates.
- [Frequency alternatives](evidence/reliability_frequency.csv), [single-match deletion](evidence/reliability_deletion.csv), [eligibility sensitivity](evidence/rating_gate_sensitivity.csv) and [paired rank uncertainty](evidence/rating_adjacent.csv) show where the order depends on the definition of reliability.
- [Pressure illustration](evidence/consequence_scenarios.png) and [uncertainty chart](evidence/rating_uncertainty.png) accompany the methodology. The CSVs in this directory also include model calibration, feature removal tests, forward results and international-to-IPL transfer.
- [Verification and test results](evidence/verification.md) and [file hashes](evidence/verification.json) identify the checked build. [Local reproduction instructions](../README.md) list the inputs and commands.
