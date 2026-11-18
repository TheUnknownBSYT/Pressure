"""Regression tests for matched ordinary comparisons and chronological ranking."""
import numpy as np
import pandas as pd

from final_strategy.ratings import evaluate_sample, future_targets, position, thresholds


def role_scores():
    rows = []
    for i in range(12):
        date = pd.Timestamp("2024-01-01")+pd.Timedelta(days=i)
        for high, legal, value in ((0, 10, 2.), (1, 6, 3.)):
            rows.append((101, i+1, date, "1|death|IPL", high, legal, value, legal))
    return pd.DataFrame(rows, columns=["player_id","match_id","date","comparison_cell","high",
                                     "legal","value","events"])


def test_units_uplift_and_cluster_weights():
    frame = role_scores()
    estimates, cells = evaluate_sample(frame, frame.date.max(), draws=100)
    p = estimates.loc[101]
    np.testing.assert_allclose([p.absolute, p.ordinary, p.uplift, p.rating], [3, 1.2, 1.8, 3])
    assert p["rank"] == 1 and p.pressure_matches == 12 and p.pressure_legal == 72
    # Adding later ordinary outcomes cannot affect the target benchmark frozen at prediction time.
    frame.loc[frame.high.eq(0),"value"] = 999
    outcome = future_targets(frame, cells).loc[101]
    assert np.isclose(outcome.future_uplift, 1.8)


def test_new_quarter_threshold_uses_prior_scores_only():
    dates = pd.to_datetime(["2024-03-31"]*3 + ["2024-04-01"]*2)
    data = pd.DataFrame({"match_date_parsed":dates, "pressure_static":[1,2,3,4,100]})
    flags = thresholds(data,.75)
    assert flags[:3].isna().all()
    assert flags[3:].all()


def test_strictly_positive_uplift_and_tie_resolution():
    data = role_scores()
    data.loc[data.high.eq(0),"value"] = 5
    scored, _ = evaluate_sample(data, data.date.max(), draws=100)
    assert not scored.loc[101,"eligible"] and pd.isna(scored.loc[101,"rank"])
    s = pd.Series([2,2,2], index=pd.Index([3,2,1],name="player_id"))
    absolute = pd.Series([4,3,3], index=s.index)
    assert position(s,s.notna(),absolute).index.tolist() == [3,1,2]


def test_strict_weighted_majority():
    frame = role_scores()
    frame.loc[frame.high.eq(1) & frame.match_id.le(6), "value"] = 0
    exact, _ = evaluate_sample(frame, frame.date.max(), draws=100)
    assert exact.loc[101, "positive_own_share"] == .5
    assert exact.loc[101, "uplift"] > 0 and not exact.loc[101, "eligible"]
    frame.loc[frame.high.eq(1) & frame.match_id.eq(1), "legal"] = 3
    above, _ = evaluate_sample(frame, frame.date.max(), draws=100)
    assert above.loc[101, "eligible"]
