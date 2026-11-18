"""Rank pressure contributions and evaluate player-level forecasts chronologically.

Run from the repository root after response_model and consequence have produced their
pre-delivery predictions. No player name is used by a ranking method.
"""
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd

from phase2_pipeline.core import identities
from independent_audit.core import stratified_wickets

OUT = Path("outputs/final_submission")
EVIDENCE = OUT / "evidence"
N_BOOT = 2000
ROLE_LABEL = {"bat": "batters", "bowl": "bowlers"}
NAMED = {
    "bat": ["Ravindra Jadeja", "Nitish Reddy", "Andre Russell", "Babar Azam",
            "Cameron Green", "Ishan Kishan", "George Dockrell", "Azmatullah Omarzai",
            "Shubman Gill", "Glenn Phillips", "Lokesh Rahul", "Shreyas Iyer", "Heinrich Klaasen"],
    "bowl": ["Blessing Muzarabani", "Shaheen Afridi", "Taskin Ahmed", "Jason Holder",
             "Dasun Shanaka", "Jasprit Bumrah", "Kuldeep Yadav", "Mujeeb Ur Rahman",
             "Mohammed Siraj", "Alzarri Joseph", "Sandeep Sharma"],
}


def matched_groups(events, role, high="high_static", value=None):
    value = value or role+"_value"
    id_col = "striker_id" if role == "bat" else "bowler_id"
    usable = events[events[high].notna() & events[value].notna()].copy()
    usable["high"] = usable[high].astype(int)
    groups = usable.groupby([id_col, "match_id", "match_date_parsed", "comparison_cell", "high"],
                            observed=True).agg(legal=("legal_event", "sum"), value=(value, "sum"),
                                               events=("delivery_id", "size")).reset_index()
    return groups.rename(columns={id_col: "player_id", "match_date_parsed": "date"})


def supported_cells(frame):
    ordinary = frame[frame.high.eq(0)]
    support = ordinary.groupby("comparison_cell").legal.agg(
        balls="sum", matches=lambda x: x.gt(0).sum())
    high = frame[frame.high.eq(1) & frame.legal.gt(0)].comparison_cell.unique()
    return sorted(set(support.index[(support.balls >= 20) & (support.matches >= 3)]) & set(high))


def role_array(frame, cells):
    sub = frame[frame.comparison_cell.isin(cells)]
    match_ids, mi = np.unique(sub.match_id, return_inverse=True)
    values = np.zeros((len(match_ids), len(cells), 2, 3))
    ci = pd.Index(cells).get_indexer(sub.comparison_cell)
    values[mi, ci, sub.high] = sub[["legal", "value", "events"]]
    dates = sub.drop_duplicates("match_id").set_index("match_id").loc[match_ids, "date"]
    return match_ids, dates, values


def per_match(high):
    legal = high[:, :, 0].sum(axis=1)
    # One unit of influence after six legal balls. Exclude zero-exposure matches.
    weights = np.minimum(legal / 6, 1)
    rate = np.divide(6 * high[:, :, 1].sum(axis=1), legal,
                     out=np.zeros_like(legal), where=legal > 0)
    weights_by_cell = np.divide(high[:, :, 0], legal[:, None],
                                out=np.zeros_like(high[:, :, 0]), where=legal[:, None] > 0)
    return legal, weights, rate, weights_by_cell


def counts_matrix(mids, draws):
    rng = np.random.default_rng(6022026)
    return rng.multinomial(len(mids), np.full(len(mids), 1/len(mids)), size=draws)


def estimate_player(frame, as_of, mids, counts, *, half_life=None):
    cells = supported_cells(frame)
    actual = frame[frame.high.eq(1)]
    raw_balls = actual.legal.sum()
    raw_matches = actual.groupby("match_id").legal.sum().gt(0).sum()
    row = {"pressure_legal_all": raw_balls, "pressure_matches_all": raw_matches,
           "supported": False, "selection_reason": "no pressure cells with 20 ordinary balls in three matches"}
    if not cells:
        return row, None, None
    match_ids, dates, arr = role_array(frame, cells)
    durations = (as_of - dates).dt.days.to_numpy()
    time_weight = np.ones(len(dates)) if half_life is None else np.exp2(-durations / half_life)
    high, ordinary = arr[:, :, 1], arr[:, :, 0]
    h, weight, rate, mix = per_match(high)
    weighted = weight * time_weight
    o = ordinary[:, :, 0]
    ov = ordinary[:, :, 1]
    ord_balls = time_weight @ o
    ord_rates = 6 * (time_weight @ ov) / ord_balls
    comparator = mix @ ord_rates
    scale = weighted.sum()
    q = weighted @ rate / scale
    baseline = weighted @ comparator / scale
    frac = weighted @ (rate > comparator) / scale
    hm, om = int((h > 0).sum()), int((o.sum(axis=1) > 0).sum())
    hb, ob = h.sum(), o.sum()
    overlap = hb / raw_balls
    recent = (durations <= 364) & (h > 0)
    cell_counts = pd.Series(high[:, :, 0].sum(axis=0), index=cells)
    row.update(absolute=q, ordinary=baseline, uplift=q-baseline,
               positive_own_share=frac, positive_peer_share=weighted @ (rate > 0) / scale,
               positive_both_share=weighted @ ((rate > comparator) & (rate > 0)) / scale,
               binary_vote=(4 + scale*frac) / (8+scale),
               effective_match_weight=scale, pressure_matches=hm, pressure_legal=hb,
               ordinary_matches=om, ordinary_legal=ob, pressure_coverage=overlap,
               matched_pressure_events=int(high[h > 0, :, 2].sum()),
               matched_cells=" / ".join(cells), first_pressure=str(dates[h > 0].min().date()),
               last_pressure=str(dates[h > 0].max().date()), recent_pressure_matches=recent.sum(),
               recent_pressure_legal=h[recent].sum(),
               recent_absolute=(weight[recent] @ rate[recent] / weight[recent].sum()
                                if recent.any() else np.nan))
    for group in ("IPL", "T20I", "powerplay", "middle", "death"):
        row[group+"_legal"] = cell_counts[[group in s.split("|") for s in cells]].sum()
    ord_map = dict(zip(cells, ord_rates))
    fails = [fewer for failed, fewer in [(hm < 8, "fewer than 8 pressure matches"),
                                         (om < 8, "fewer than 8 ordinary matches"),
                                         (hb < 50, "fewer than 50 pressure balls"),
                                         (ob < 100, "fewer than 100 ordinary balls"),
                                         (overlap < .75, "under 75% matched pressure coverage")]
             if failed]
    if fails:
        row["selection_reason"] = ", ".join(fails)
        if hm < 4 or hb < 20 or om < 3:
            return row, ord_map, None

    # Resample all matches jointly. Re-estimate the ordinary cell means in each draw.
    draw_weight = counts[:, np.searchsorted(mids, match_ids)] * time_weight
    mass = draw_weight @ weight
    q_total = draw_weight @ (weight * rate)
    q_draw = np.divide(q_total, mass, out=np.full_like(mass, np.nan), where=mass > 0)
    o_draw = draw_weight @ o
    v_draw = draw_weight @ ov
    weighted_mix = draw_weight @ (mix * weight[:, None])
    missing = ((o_draw == 0) & (weighted_mix > 0)).any(axis=1) | (mass == 0)
    ordinary_draw = 6*np.divide(v_draw, o_draw, out=np.zeros_like(v_draw), where=o_draw > 0)
    ord_total = (weighted_mix * ordinary_draw).sum(axis=1)
    delta = q_draw - np.divide(ord_total, mass, out=np.full_like(mass, np.nan), where=mass > 0)
    delta[missing] = np.nan
    row["bootstrap_absolute_valid"] = np.isfinite(q_draw).mean()
    row["bootstrap_uplift_valid"] = np.isfinite(delta).mean()
    row["rating"] = np.nanquantile(q_draw, .20)
    row["uplift_lower80"] = np.nanquantile(delta, .20)
    row["absolute_lo95"], row["absolute_hi95"] = np.nanquantile(q_draw, [.025, .975])
    row["uplift_lo95"], row["uplift_hi95"] = np.nanquantile(delta, [.025, .975])
    # Minimum absolute quality after dropping any one pressure appearance.
    row["minimum_leave_one_out_absolute"] = np.min(
        (q*scale-(weighted*rate)[h > 0]) / (scale-weighted[h > 0]))
    if fails:
        row["uncertainty_scope"] = "conditional on matched cells only; failed eligibility"
        return row, ord_map, None
    row["supported"] = True
    row["uncertainty_scope"] = "conditional on recorded matches, fitted models and retained cells"
    row["selection_reason"] = ("nonpositive absolute pressure contribution" if q <= 0 else
                               "nonpositive uplift over matched ordinary" if q <= baseline else
                               "weighted success at or below 50%" if frac <= .5 else
                               "outside top ten on conservative absolute quality")
    wins = rate[None, :] > ordinary_draw @ mix.T
    majority = ((draw_weight * weight) * wins).sum(axis=1) > .5 * mass
    return row, ord_map, (q_draw, delta, majority)


def position(values, mask, absolute=None):
    ordered = values[mask].to_frame("score")
    ordered["absolute_tiebreak"] = values[mask] if absolute is None else absolute[mask]
    ranked = ordered.sort_values(["score", "absolute_tiebreak", "player_id"],
                                 ascending=[False, False, True])
    return pd.Series(np.arange(1, len(ranked)+1), index=ranked.index)


def evaluate_sample(groups, as_of, *, half_life=None, draws=N_BOOT, stability=False):
    mids = np.sort(groups.match_id.unique())
    counts = counts_matrix(mids, draws)
    rows, ordinary, boot = [], {}, {}
    for pid, frame in groups.groupby("player_id"):
        row, ordinary[pid], vals = estimate_player(frame, as_of, mids, counts, half_life=half_life)
        rows.append({"player_id": pid, **row})
        if vals is not None:
            boot[pid] = vals
    result = pd.DataFrame(rows).set_index("player_id")
    for c in ("rating", "absolute", "ordinary", "uplift", "binary_vote", "positive_own_share"):
        if c not in result:
            result[c] = np.nan
    supported = result.supported
    qualifies = supported & (result.absolute > 0) & (result.uplift > 0) & (result.positive_own_share > .5)
    result["eligible"] = qualifies
    result["rank"] = position(result.rating, qualifies, result.absolute)
    result["absolute_rank"] = position(result.absolute, qualifies)
    result["uplift_rank"] = position(result.uplift, supported)
    result["combined"] = (result.absolute + result.uplift)/2
    result["combined_rank"] = position(result.combined, qualifies)
    result["vote_rank"] = position(result.binary_vote, qualifies)
    result.loc[result["rank"] <= 10, "selection_reason"] = "selected"
    if stability and boot:
        # Rank the bootstrapped point quality subject to resampled gates. Separately
        # recomputing a nested-bootstrap lower bound is not part of this diagnostic.
        scores = pd.DataFrame({p: np.where((a > 0) & (u > 0) & majority, a, np.nan)
                               for p, (a, u, majority) in boot.items()})
        r = scores.rank(axis=1, ascending=False, method="first")
        result["resample_top10_fraction_point_rule"] = r.le(10).mean()
        result["resample_positive_both_fraction"] = scores.notna().mean()
        adjacent = []
        ordered = result[result["rank"].le(11)].sort_values("rank")
        for left, right in zip(ordered.index[:-1], ordered.index[1:]):
            diff = boot[left][0]-boot[right][0]
            valid = np.isfinite(diff)
            lo, hi = np.quantile(diff[valid], [.025,.975])
            adjacent.append(dict(left_id=int(left), right_id=int(right), left_rank=int(ordered.loc[left,"rank"]),
                                 right_rank=int(ordered.loc[right,"rank"]), difference_absolute=(
                                 ordered.loc[left,"absolute"]-ordered.loc[right,"absolute"]),
                                 difference_lo95=lo, difference_hi95=hi,
                                 fraction_left_larger=(diff[valid]>0).mean(), valid_draws=valid.sum()))
        result.attrs["adjacent"] = adjacent
    return result, ordinary


def future_targets(future, ordinary):
    """Score later pressure matches against frozen player/role ordinary baselines."""
    rows = []
    for pid, data in future[future.high.eq(1)].groupby("player_id"):
        rates = ordinary.get(pid)
        if not rates:
            continue
        cells = list(rates)
        if not data.comparison_cell.isin(cells).any():
            continue
        _, _, a = role_array(data, cells)
        legal, w, q, mix = per_match(a[:, :, 1])
        if (legal > 0).sum() < 2 or legal.sum() < 12:
            continue
        benchmark = mix @ np.array(list(rates.values()))
        rows.append(dict(player_id=pid, future_absolute=w @ q / w.sum(),
                         future_uplift=w @ (q-benchmark) / w.sum(),
                         future_vote=w @ (q > benchmark) / w.sum(),
                         future_matches=(legal > 0).sum(), future_legal=legal.sum()))
    return pd.DataFrame(rows).set_index("player_id") if rows else pd.DataFrame()


def thresholds(events, quantile):
    scores = events[["match_date_parsed", "pressure_static"]].copy()
    quarter = scores.match_date_parsed.dt.to_period("Q")
    high = pd.Series(pd.NA, dtype="boolean", index=events.index)
    for q in sorted(quarter.unique()):
        history = scores.loc[quarter < q, "pressure_static"]
        if len(history):
            high.loc[quarter == q] = scores.loc[quarter == q, "pressure_static"].ge(history.quantile(quantile))
    return high


def load_events():
    path = "outputs/phase2/phase2_player_event_metrics.parquet"
    cols = ["delivery_id", "match_id", "striker_id", "bowler_id", "competition_type",
            "comparison_cell", "legal_event", "match_date_parsed", "outcome_valid",
            "phase_pre", "analysis_available", "standard_t20_eligible", "pressure_eligible",
            "bat_dismissal", "bowl_wicket", "bat_run_residual", "bowl_run_residual"]
    source = pd.read_parquet(path, columns=cols)
    source, _ = stratified_wickets(source)
    source["bat_wicket_residual"] = source.bat_stratified_wicket_rate*source.legal_event-source.bat_dismissal
    source["bowl_wicket_residual"] = source.bowl_wicket-source.bowl_stratified_wicket_rate*source.legal_event
    consequence = pd.read_parquet(EVIDENCE/"consequence_events.parquet")
    for top in (20, 30):
        consequence[f"top{top}"] = thresholds(consequence, 1-top/100)
    extra = ["delivery_id", "high_static", "high_dynamic", "top20", "top30"]
    innovations = pd.read_parquet(EVIDENCE/"response_events.parquet")
    data = source.merge(consequence[extra], validate="one_to_one", on="delivery_id")
    # Some experiments include pre-delivery identifiers alongside innovations.
    required = ["delivery_id", "bat_value", "bowl_value", "bat_no_opponent_value",
                "bowl_no_opponent_value"]
    data = data.merge(innovations[required], on="delivery_id", validate="one_to_one")
    return data[data.outcome_valid & data.competition_type.isin(["IPL", "T20I"])].copy()


def specs(data, role):
    alternatives = {name: matched_groups(data, role, field, value) for name, field, value in [
        ("primary", "high_static", role+"_value"),
        ("no_opponent", "high_static", role+"_no_opponent_value"),
        ("dynamic", "high_dynamic", role+"_value"),
        ("top20", "top20", role+"_value"),
        ("top30", "top30", role+"_value")]}
    valid = data[data.analysis_available]
    runs = matched_groups(valid, role, value=role+"_run_residual")
    wickets = matched_groups(valid, role, value=role+"_wicket_residual")
    keys = ["player_id", "match_id", "date", "comparison_cell", "high", "legal", "events"]
    alternatives["legacy_components"] = runs.merge(wickets, on=keys, suffixes=("_runs", "_wickets"),
                                                   validate="one_to_one")
    return alternatives


def standardized_components(components):
    # Recalculate both scales only among historical players present in this window.
    high = components[components.high.eq(1)].groupby("player_id")[
        ["legal", "value_runs", "value_wickets"]].sum()
    high = high[high.legal >= 50]
    sd = high[["value_runs", "value_wickets"]].div(high.legal, axis=0).std()
    frame = components.copy()
    frame["value"] = (.85*frame.value_runs/sd.value_runs +
                      .15*frame.value_wickets/sd.value_wickets)/6
    return frame


def evaluate_methods(role, groups, year):
    start = pd.Timestamp(year, 1, 1)
    end = start - pd.Timedelta(days=1)
    past = groups["primary"][groups["primary"].date < start]
    future = groups["primary"][groups["primary"].date.dt.year == year]
    historical, ordinary = evaluate_sample(past, end)
    observed = future_targets(future, ordinary)
    if observed.empty:
        return [], []
    predictors = pd.DataFrame(index=historical.index)
    for label, col in [("lower_bound", "rating"), ("absolute", "absolute"),
                       ("uplift", "uplift"), ("combined", "combined"), ("vote", "binary_vote")]:
        predictors[label] = historical[col].where(historical.supported)
    predictors["peer_zero"] = np.where(historical.supported, 0., np.nan)
    for method in ["no_opponent", "dynamic", "decay365", "decay730", "recent365", "standardized85"]:
        src = groups.get(method, groups["primary"])
        src = src[src.date < start]
        if method == "standardized85":
            components = groups["legacy_components"]
            src = standardized_components(components[components.date < start])
        if method == "recent365":
            src = src[src.date >= start-pd.Timedelta(days=365)]
        half = int(method.removeprefix("decay")) if method.startswith("decay") else None
        if src.empty:
            predictors[method] = np.nan
            continue
        score, _ = evaluate_sample(src, end, half_life=half)
        predictors[method] = score.rating.where(score.supported)
    preds = predictors.join(observed, how="inner")
    candidate_rows = preds.reset_index().assign(role=role, year=year).to_dict("records")
    metrics = []
    common = preds.dropna(subset=["lower_bound", "no_opponent", "dynamic", "decay365", "decay730"])
    for cohort, sample in [("available", preds), ("common_except_recent_only", common)]:
        for method in predictors:
            for target in ("absolute", "uplift", "vote"):
                pairs = sample[[method, "future_"+target]].dropna()
                x, y = pairs.iloc[:, 0], pairs.iloc[:, 1]
                same_unit = (target == "vote" and method == "vote") or (
                    target != "vote" and method not in ("vote", "standardized85"))
                rho = (x.rank().corr(y.rank()) if len(x) >= 3 and x.nunique() > 1 and
                       y.nunique() > 1 else np.nan)
                metrics.append(dict(role=role, year=year, cohort=cohort, method=method,
                                    target=target, n=len(x), rho=rho,
                                    mse=((x-y)**2).mean() if same_unit else np.nan))
    for row in candidate_rows:
        row["historical_rank"] = historical.loc[row["player_id"], "rank"]
    return metrics, candidate_rows


def run():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for filename in ("rating_protocol.md", "rating_notes.md"):
        source = Path(__file__).with_name(filename)
        if source.is_file():
            copyfile(source, EVIDENCE/filename)
    data = load_events()
    names = identities(Path("."))[0].set_index("player_id").player_name
    last = pd.read_csv("matches.csv", usecols=["match_date"]).match_date
    end = pd.to_datetime(last).max()
    metrics, observations, candidates, changes, scope, log, adjacent = [], [], [], [], [], [], []
    for role, label in ROLE_LABEL.items():
        groups = specs(data, role)
        primary, _ = evaluate_sample(groups["primary"], end, stability=True)
        adjacent.extend({"role":role, **p} for p in primary.attrs.pop("adjacent", []))
        primary["player_name"] = names.reindex(primary.index)
        for name in ("no_opponent", "dynamic", "top20", "top30", "decay180",
                     "decay365", "decay730", "recent365", "standardized85"):
            src = groups.get(name, groups["primary"])
            if name == "standardized85":
                src = standardized_components(groups["legacy_components"])
            if name == "recent365":
                src = src[src.date >= end-pd.Timedelta(days=364)]
            half = int(name.removeprefix("decay")) if name.startswith("decay") else None
            alt, _ = evaluate_sample(src, end, half_life=half)
            primary[name+"_rank"] = alt["rank"]
            primary[name+"_absolute"] = alt.absolute
            primary[name+"_rating"] = alt.rating
            primary[name+"_reason"] = alt.selection_reason
            base10 = set(primary.index[primary["rank"].le(10)])
            alt10 = set(alt.index[alt["rank"].le(10)])
            changes.append(dict(role=role, method=name, eligible=alt.eligible.sum(),
                                overlap_top10=len(base10 & alt10), alternatives_returned=len(alt10)))
        for method, col in [("point_absolute_with_uplift_gate", "absolute_rank"),
                            ("point_uplift", "uplift_rank"), ("combined", "combined_rank"),
                            ("binary_match_vote_run_equivalent", "vote_rank")]:
            selected = primary[col].le(10)
            changes.append(dict(role=role, method=method, eligible=primary[col].notna().sum(),
                                overlap_top10=(selected & primary["rank"].le(10)).sum(),
                                alternatives_returned=selected.sum()))
        # The previous ten in the supplied challenge brief are a historical reference
        # for the comparison only. Changing them cannot change the new rankings.
        prior_ranks = {name: i for i, name in enumerate(NAMED[role][:10], start=1)}
        primary["historical_vote_rank"] = primary.player_name.map(prior_ranks)
        previous = Path("outputs/historical_vote_submission")/f"final_{label}_top10.csv"
        if previous.exists():
            old = pd.read_csv(previous).sort_values("rank")
            assert old.player_name.tolist() == list(prior_ranks), "Historical snapshot differs from audit brief"
        changes.append(dict(role=role, method="archived_standardized_85_15_vote",
                            eligible=None, overlap_top10=primary.loc[primary["rank"].le(10),
                            "historical_vote_rank"].notna().sum(), alternatives_returned=len(prior_ranks)))
        primary["named_in_brief"] = primary.player_name.isin(NAMED[role])
        primary["transfer_scope"] = np.select(
            [primary.IPL_legal.eq(0) & primary.T20I_legal.gt(0),
             primary.T20I_legal.eq(0) & primary.IPL_legal.gt(0),
             primary.IPL_legal.gt(0) & primary.T20I_legal.gt(0)],
            ["T20I only", "IPL only", "both competitions"], default="insufficient matched exposure")
        primary["role"] = role
        primary.index.name = "player_id"
        primary = primary.sort_values(["rank", "rating", "player_id"], ascending=[True, False, True])
        top = primary[primary["rank"] <= 10].copy()
        assert len(top) == 10 and top.player_name.notna().all()
        top["rank"] = top["rank"].astype(int)
        top.reset_index()[["rank", "player_name", "player_id"]].to_csv(OUT/f"final_{label}_top10.csv", index=False)
        candidates.append(primary.reset_index())
        for year in (2024, 2025, 2026):
            out, preds = evaluate_methods(role, groups, year)
            metrics.extend(out)
            observations.extend(preds)
        scope.append(dict(role=role, source_events=len(data), classifiable_events=int(data.high_static.notna().sum()),
                          matches=groups["primary"].match_id.nunique(), candidates=len(primary),
                          supported=int(primary.supported.sum()), eligible=int(primary.eligible.sum()),
                          source_start=str(groups["primary"].date.min().date()),
                          source_end=str(groups["primary"].date.max().date()), as_of=str(end.date())))
        msg = f"{label}: {primary.supported.sum()} supported, {primary.eligible.sum()} eligible. "+", ".join(top.player_name)
        print(msg, flush=True)
        log.append(msg)
    full = pd.concat(candidates, ignore_index=True)
    gates = []
    for role, group in full.groupby("role"):
        supported = group.set_index("player_id").query("supported")
        for kind, keep in (
            ("point gates", supported.absolute.gt(0) & supported.uplift.gt(0)),
            ("majority above own ordinary", supported.eligible & (supported.positive_own_share > .5)),
            ("one-sided 80% lower limits positive", (supported.rating > 0) & (supported.uplift_lower80 > 0)),
            ("two-sided 95% lower limits positive", (supported.absolute_lo95 > 0) & (supported.uplift_lo95 > 0))):
            ranks = position(supported.rating, keep, supported.absolute)
            top = ranks.index[ranks.le(10)]
            gates.append(dict(role=role, rule=kind, candidates=int(keep.sum()), returned=len(top),
                              overlap_top10=int(supported.loc[top, "rank"].le(10).sum()),
                              names_in_order=" | ".join(supported.loc[top, "player_name"])))
    pd.DataFrame(gates).to_csv(EVIDENCE/"rating_gate_sensitivity.csv", index=False)
    full.to_csv(EVIDENCE/"rating_all_candidates.csv", index=False)
    full.loc[full["rank"].le(10)].to_csv(OUT/"component_evidence.csv", index=False)
    expanded = full[full["rank"].le(30) | full.named_in_brief].copy()
    expanded.to_csv(OUT/"expanded_candidates.csv", index=False)
    watch = full[~full.supported & full.pressure_matches.ge(4) & full.pressure_legal.ge(25)
                 & full.ordinary_legal.ge(40) & full.pressure_coverage.ge(.5) &
                 full.absolute.gt(0) & full.uplift.gt(0)]
    watch.sort_values("absolute", ascending=False).groupby("role").head(5).to_csv(
        OUT/"under_the_radar_watchlist.csv", index=False)
    pd.DataFrame(metrics).to_csv(EVIDENCE/"rating_forward.csv", index=False)
    pd.DataFrame(observations).to_csv(EVIDENCE/"rating_forward_predictions.csv", index=False)
    pd.DataFrame(changes).to_csv(EVIDENCE/"rating_sensitivity.csv", index=False)
    pd.DataFrame(adjacent).to_csv(EVIDENCE/"rating_adjacent.csv", index=False)
    pd.DataFrame(scope).to_csv(EVIDENCE/"rating_scope.csv", index=False)
    forecasts = pd.DataFrame(observations)
    selected = forecasts[forecasts.historical_rank.le(10)]
    selected.groupby(["role", "year"]).agg(forecasts_observed=("player_id", "size"),
        mean_future_absolute=("future_absolute", "mean"), mean_future_uplift=("future_uplift", "mean"),
        mean_future_vote=("future_vote", "mean")).to_csv(EVIDENCE/"rating_forward_shortlists.csv")
    (EVIDENCE/"rating_summary.log").write_text("\n".join(log)+"\n")
    (EVIDENCE/"rating_settings.json").write_text(json.dumps(
        {"bootstrap_draws": N_BOOT, "seed": 6022026, "as_of": str(end.date()),
         "primary_statistic": "20th percentile match-cluster resample of absolute quality",
         "units": "run equivalents per six legal balls; match weights min(1, balls/6)"},
         indent=2)+"\n")


if __name__ == "__main__":
    run()
