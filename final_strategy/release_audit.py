"""Reconcile the release from deliveries, before rendering the report."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

E = Path("outputs/final_submission/evidence")


def audit_state(state):
    raw = pd.read_csv("ball_by_ball.csv", keep_default_na=False, na_values=[""])
    raw = raw.sort_values(["match_id", "innings", "innings_ball_num"])
    state = state.sort_values(["match_id", "innings", "innings_ball_num"])
    keys = ["match_id", "innings", "innings_ball_num"]
    np.testing.assert_array_equal(raw[keys], state[keys])
    assert not raw.duplicated(keys).any()

    for (match, innings), group in raw.groupby(["match_id", "innings"]):
        runs = wickets = legal = 0
        if innings == 1:
            target = int(group[["runs", "extras1_runs", "extras_2_runs"]].sum().sum()) + 1
        idx = state.index[state.match_id.eq(match) & state.innings.eq(innings)]
        for row, idx_row in zip(group.itertuples(index=False), state.loc[idx].itertuples(index=False), strict=True):
            assert (runs, wickets, legal) == (
                idx_row.score_pre, idx_row.wickets_fallen_pre, idx_row.legal_balls_bowled_pre)
            if innings == 2 and pd.notna(idx_row.target_pre):
                assert idx_row.target_pre == target
            runs += row.runs + row.extras1_runs + row.extras_2_runs
            wickets += row.is_wicket
            legal += row.extras_type not in {"wide", "Overstep", "other"}
    return state


def audit_pressure(state, pressure):
    cols = ["delivery_id", "win_probability_pre", "leverage", "urgency", "match_stakes",
            "model_train_end_date", "model_fold_start", "legal_balls_remaining_pre",
            "recent_wickets_12_pre", "innings"]
    x = pressure.merge(state[cols], on="delivery_id", validate="one_to_one")
    assert len(x) == state.pressure_score_v1.notna().sum()
    assert (x.model_train_end_date < x.model_fold_start).all()
    assert (x.model_fold_start <= x.match_date_parsed).all()
    contest = 4*x.win_probability_pre*(1-x.win_probability_pre)
    shift = .1*(np.minimum(x.recent_wickets_12_pre/2, 1)
                - np.where(x.innings == 2, (120-x.legal_balls_remaining_pre)/120*contest, 0))
    np.testing.assert_allclose(x.weight_leverage, .5+shift)
    np.testing.assert_allclose(x.weight_urgency_contest, .3-shift)
    np.testing.assert_allclose(x.pressure_static, 50*x.leverage+30*x.urgency*contest+20*x.match_stakes)
    np.testing.assert_allclose(
        x.pressure_dynamic, x.pressure_static+100*shift*(x.leverage-x.urgency*contest))
    for fold, test in x.groupby("model_fold_start"):
        past = x[x.match_date_parsed < fold]
        for method in ["static", "dynamic"]:
            flag = test["high_"+method]
            if past.empty:
                assert flag.isna().all()
            else:
                scores = past["pressure_"+method]
                np.testing.assert_array_equal(flag, test["pressure_"+method] >= np.quantile(scores, .75))


def audit_response(state, values):
    d = values.merge(state, on=["delivery_id", "match_id", "legal_event", "match_date_parsed"],
                     validate="one_to_one")
    expected_ids = state.loc[state.pressure_eligible & ~state.bowler_concession_uncertain &
        (~state.is_wicket | state.dismissed_player_id.eq(state.striker_id) |
         state.dismissed_player_id.eq(state.non_striker_id)), "delivery_id"]
    assert set(d.delivery_id) == set(expected_ids)
    assert d.historical_legal_probability.gt(0).all() and d.historical_legal_probability.le(1).all()
    assert d.wicket_price.notna().all() and d.wicket_price.ge(0).all()
    for role in ["bat", "bowl"]:
        p = d[role+"_expected_wicket"]
        assert p.between(0, 1).all()
        assert d[role+"_expected_runs"].ge(0).all()
        if role == "bat":
            runs = d.runs
            wicket = d.is_wicket & d.dismissed_player_id.eq(d.striker_id)
            sign = 1
        else:
            runs = d.runs + np.where(d.extras_type.isin(["wide", "Overstep"]), d.extras1_runs, 0)
            wicket = d.is_wicket & d.dismissal_type.isin(["Caught", "Bowled", "LBW", "Stump", "Hit Wicket"])
            sign = -1
        baseline = (d[role+"_expected_runs"]-d.wicket_price*p)*d.legal_event/d.historical_legal_probability
        np.testing.assert_allclose(d[role+"_value"], sign*(runs-d.wicket_price*wicket-baseline), atol=1e-10)


def audit_forecasts(values):
    # Replay the exported models with their own stored coefficients and training-only scaling.
    from final_strategy.response_model import build
    events, features, _ = build()
    coef = pd.read_csv(E/"response_coefficients.csv")
    coef = coef[coef.model.eq("opposition")]
    values = values.set_index("delivery_id")
    folds = sorted(coef.test_start.unique())
    for start, stop in zip(folds, folds[1:]+["2027-01-01"], strict=True):
        prior = events.match_date_parsed.lt(start)
        test = events.match_date_parsed.ge(start) & events.match_date_parsed.lt(stop)
        test &= events.delivery_id.isin(values.index)
        assert events.loc[prior, "match_date_parsed"].max() < events.loc[test, "match_date_parsed"].min()
        observed = values.loc[events.loc[test, "delivery_id"]]
        for role, x in features.items():
            params = coef.loc[coef.test_start.eq(start) & coef.role.eq(role)].set_index("feature")
            intercept = params.loc["intercept"]
            terms = params.drop(index="intercept")
            sample = x.loc[prior, terms.index]
            np.testing.assert_allclose(sample.mean(), terms.train_mean, atol=1e-10)
            scale = sample.std(ddof=0)
            np.testing.assert_allclose(scale.mask(scale < 1e-6, 1), terms.train_sd, atol=1e-10)
            standardized = (x.loc[test, terms.index] - terms.train_mean)/terms.train_sd
            runs = np.maximum(0, standardized @ terms.run_coefficient + intercept.run_coefficient)
            wicket = expit(standardized @ terms.wicket_logodds_coefficient + intercept.wicket_logodds_coefficient)
            np.testing.assert_allclose(runs, observed[role+"_expected_runs"], atol=1e-10)
            np.testing.assert_allclose(wicket, observed[role+"_expected_wicket"], atol=1e-10)


def audit_ranking(state, pressure, values):
    d = state[["delivery_id", "match_id", "innings", "phase_pre", "competition_type",
               "striker_id", "bowler_id", "legal_event"]].merge(
        pressure[["delivery_id", "high_static"]].dropna(), validate="one_to_one").merge(
        values[["delivery_id", "bat_value", "bowl_value"]], validate="one_to_one")
    d = d[d.competition_type.isin(["IPL", "T20I"])]
    candidates = pd.read_csv(E/"rating_all_candidates.csv")
    sensitivity = []
    stability = []
    for role, label, player in [("bat", "batters", "striker_id"), ("bowl", "bowlers", "bowler_id")]:
        actual = candidates[candidates.role.eq(role)].set_index("player_id")
        official = pd.read_csv(E.parent/f"final_{label}_top10.csv").player_id.tolist()
        match_ids = np.sort(d.match_id.unique())
        draws = np.random.default_rng(6022026).multinomial(
            len(match_ids), np.repeat(1/len(match_ids), len(match_ids)), size=2000)
        match_index = dict(zip(match_ids, range(len(match_ids))))
        totals = d.groupby([player, "match_id", "innings", "phase_pre", "competition_type", "high_static"])[
            ["legal_event", role+"_value"]].sum().reset_index()
        checked = []
        for pid, group in totals.groupby(player):
            report = actual.loc[pid]
            outcomes = []
            all_high = group.loc[group.high_static, "legal_event"].sum()
            for _, bucket in group.groupby(["innings", "phase_pre", "competition_type"]):
                ordinary = bucket[~bucket.high_static]
                high = bucket[bucket.high_static]
                if (ordinary.legal_event.sum() >= 20 and (ordinary.legal_event > 0).sum() >= 3
                        and high.legal_event.sum() > 0):
                    outcomes.append((ordinary, high))
            support = False
            if outcomes:
                support = check_player(pid, role, outcomes, report, all_high, draws,
                                       match_index, checked, stability if pid in official else None)
            assert bool(report.supported) == support, (role, pid)
        scores = pd.DataFrame(checked).set_index("player_id")
        main = scores.sort_values(["rating", "absolute", "player_id"], ascending=[False, False, True])
        assert main.index[:10].tolist() == official, role
        for method, order in {
            "primary conservative average": main,
            "weighted share above own": scores.sort_values(
                ["own_share", "rating", "absolute", "player_id"], ascending=[False, False, False, True]),
            "weighted share above both": scores.sort_values(
                ["both_share", "rating", "absolute", "player_id"], ascending=[False, False, False, True]),
        }.items():
            top = order.index[:10]
            sensitivity.append({"role": role, "method": method, "overlap_top10": len(set(top)&set(official)),
                                "names_in_order": " | ".join(actual.loc[top, "player_name"])})
    pd.DataFrame(sensitivity).to_csv(E/"reliability_frequency.csv", index=False)
    pd.DataFrame(stability).to_csv(E/"reliability_deletion.csv", index=False)
    return len(candidates)


def check_player(pid, role, cells, report, all_high, draws, match_index, checked, stability):
    mids = sorted(set().union(*(set(h.match_id) | set(o.match_id) for o, h in cells)))
    value = role+"_value"
    # Columns represent supported cells. Rows represent distinct observed matches.
    def matrix(side, column):
        return np.column_stack([cell[side].set_index("match_id")[column].reindex(mids, fill_value=0)
                                for cell in cells])

    o, ov, h, hv = matrix(0, "legal_event"), matrix(0, value), matrix(1, "legal_event"), matrix(1, value)
    hb = h.sum(axis=1)
    weights = np.minimum(hb, 6)/6
    quality = np.divide(6*hv.sum(axis=1), hb, out=np.zeros(len(mids)), where=hb > 0)
    ordinary_rates = 6*ov.sum(axis=0)/o.sum(axis=0)
    benchmarks = np.divide(h@ordinary_rates, hb, out=np.zeros(len(mids)), where=hb > 0)
    absolute = np.average(quality, weights=weights)
    ordinary = np.average(benchmarks, weights=weights)
    own_share = np.average(quality > benchmarks, weights=weights)
    both_share = np.average((quality > benchmarks) & (quality > 0), weights=weights)
    coverage = hb.sum()/all_high
    support = (np.count_nonzero(hb) >= 8 and np.count_nonzero(o.sum(axis=1)) >= 8
               and hb.sum() >= 50 and o.sum() >= 100 and coverage >= .75)
    np.testing.assert_allclose(
        [absolute, ordinary, absolute-ordinary, own_share, both_share, coverage, hb.sum(), o.sum(),
         np.count_nonzero(hb), np.count_nonzero(o.sum(axis=1))],
        [report.absolute, report.ordinary, report.uplift, report.positive_own_share,
         report.positive_both_share, report.pressure_coverage, report.pressure_legal,
         report.ordinary_legal, report.pressure_matches, report.ordinary_matches], atol=1e-10)
    eligible = support and absolute > 0 and absolute > ordinary and own_share > .5
    assert eligible == bool(report.eligible), (role, pid)
    if not eligible:
        return support

    draw = draws[:, [match_index[m] for m in mids]]
    total_weight = draw @ weights
    with np.errstate(divide="ignore", invalid="ignore"):
        weighted_quality = (draw @ (quality*weights))/total_weight
    rating, lo, hi = np.nanquantile(weighted_quality, [.20, .025, .975])
    np.testing.assert_allclose([rating, lo, hi], [report.rating, report.absolute_lo95, report.absolute_hi95])
    per_ball_mix = np.divide(h, hb[:, None], out=np.zeros(h.shape), where=hb[:, None]>0)
    draw_mix = draw @ (weights[:, None] * per_ball_mix)
    draw_balls = draw @ o
    rates = np.divide(6*(draw@ov), draw_balls, out=np.zeros(draw_balls.shape), where=draw_balls>0)
    valid = (total_weight > 0) & ~((draw_balls == 0) & (draw_mix > 0)).any(axis=1)
    uplift_draw = weighted_quality[valid] - (rates[valid]*draw_mix[valid]).sum(axis=1)/total_weight[valid]
    np.testing.assert_allclose(
        np.quantile(uplift_draw, [.025, .975]), [report.uplift_lo95, report.uplift_hi95], atol=1e-10)
    checked.append(dict(player_id=pid, rating=rating, absolute=absolute,
                        own_share=own_share, both_share=both_share))
    if stability is not None:
        uplift_samples, quality_samples = [], []
        for delete in range(len(mids)):
            keep = np.arange(len(mids)) != delete
            # Hold the cell definitions fixed. Re-estimate ordinary rates after removing the fixture.
            balls = o[keep].sum(axis=0)
            if ((balls == 0) & h[keep].sum(axis=0).astype(bool)).any():
                continue
            rates = np.divide(6*ov[keep].sum(axis=0), balls, out=np.zeros(len(cells)), where=balls>0)
            peer = np.average(quality[keep], weights=weights[keep])
            per_match = np.divide(h[keep]@rates, hb[keep], out=np.zeros(keep.sum()), where=hb[keep]>0)
            uplift_samples.append(peer-np.average(per_match, weights=weights[keep]))
            quality_samples.append(peer)
        stability.append(dict(role=role, player_id=pid, player_name=report.player_name,
                              rank=int(report["rank"]), matched_matches=len(mids),
                              deletion_draws=len(uplift_samples),
                              min_absolute=min(quality_samples), min_uplift=min(uplift_samples),
                              positive_both_deletions=sum(a>0 and u>0 for a,u in zip(
                                  quality_samples, uplift_samples, strict=True))))
    return support


def main():
    state = pd.read_parquet("outputs/pressure_scored.parquet")
    pressure = pd.read_parquet(E/"consequence_events.parquet")
    values = pd.read_parquet(E/"response_events.parquet")
    audit_state(state)
    audit_pressure(state, pressure)
    audit_response(state, values)
    audit_forecasts(values)
    count = audit_ranking(state, pressure, values)
    dates = pd.to_datetime(pd.read_csv("matches.csv").match_date)
    summary = {"raw_delivery_rows_reconciled": len(state), "scored_pressure_rows_reconciled": len(pressure),
               "response_rows_reconciled": len(values), "player_role_records_checked": count,
               "first_fixture_date": str(dates.min().date()), "last_fixture_date": str(dates.max().date()),
               "checks": ["Raw score, wicket and legal-ball ledger", "Strictly earlier training and pressure cutoffs",
                          "Static and dynamic pressure equations", "Runs and wickets attribution, forecasts and response value",
                          "Matched cells, eligibility, frequency, independent bootstrap and rank order"]}
    (E/"release_audit.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
