"""Reproduce the old forward vote test using its original functions and parameters."""
from pathlib import Path

import pandas as pd

from build_independent_audit import source_for
from final_strategy.analyse import sample
from independent_audit.core import stratified_wickets
from phase2_pipeline.core import match_metrics

EVIDENCE = Path("outputs/final_submission/evidence")


def reproduce():
    frame = pd.read_parquet("outputs/phase2/phase2_player_event_metrics.parquet")
    frame, _ = stratified_wickets(frame)
    data = source_for(frame, "stratified")
    dates = data.drop_duplicates("match_id").set_index("match_id").match_date_parsed
    rows = []
    for year in (2024, 2025, 2026):
        start = pd.Timestamp(year, 1, 1)
        cutoff = data.loc[data.pressure_eligible & (data.match_date_parsed < start),
                          "pressure_score_v1"].quantile(.75)
        high = data.pressure_eligible & data.pressure_score_v1.ge(cutoff)
        for role in ("bat", "bowl"):
            groups = match_metrics(data, role, high)
            groups["date"] = groups.match_id.map(dates)
            train = groups[groups.date < start]
            test = groups[groups.date.dt.year == year]
            denom = train.loc[train.high.eq(1)].groupby("player_id")[["legal", "rr", "wr"]].sum()
            denom = denom[denom.legal >= 50]
            scale = denom[["rr", "wr"]].div(denom.legal, axis=0).mul(100).std().to_numpy()
            past, _ = sample(train, scale, role, start - pd.Timedelta(days=1), draws=300)
            future, _ = sample(test, scale, role, test.date.max(), eval_only=True)
            pairs = pd.concat([past.score.where(past.supported).rename("forecast"),
                               future.positive_share.where(future.supported).rename("observed")],
                              axis=1).dropna()
            rows.append(dict(role=role, year=year, n=len(pairs), cutoff=cutoff,
                             rho=pairs.rank().corr().iloc[0, 1] if len(pairs) >= 3 else None,
                             mse=(pairs.forecast-pairs.observed).pow(2).mean()))
    result = pd.DataFrame(rows)
    ref_path = Path("outputs/historical_vote_submission/evidence/backtest.csv")
    if ref_path.exists():
        ref = pd.read_csv(ref_path).query("cohort=='available' and method=='historical'")
        check = result.merge(ref, on=["role", "year"], suffixes=("", "_archived"),
                             validate="one_to_one")
        import numpy as np
        for col in ("n", "rho", "mse", "cutoff"):
            np.testing.assert_allclose(check[col], check[col+"_archived"], atol=1e-12, equal_nan=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    result.to_csv(EVIDENCE/"rating_legacy_forward.csv", index=False)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    reproduce()
