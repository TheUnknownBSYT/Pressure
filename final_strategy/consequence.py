"""Audit the legacy index, a bounded dynamic index and a fitted consequence proxy.

Run after build_pressure_dataset.py from the project root.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import log_loss

from pressure_pipeline.model import design, next_state, outcome_weights, score_components, support_counts, trajectory

OUT = Path('outputs')
E = OUT / 'final_submission/evidence'
CUT = pd.Timestamp('2026-01-01')
COLS = '''delivery_id match_id match_date_parsed innings innings_ball_num model_fold_start
model_train_end_date win_probability_pre contestability urgency leverage match_stakes pressure_score_v1
wickets_remaining_pre legal_balls_remaining_pre legal_balls_bowled_pre runs_required_pre score_pre
phase_pre competition_type striker_name bowler_name batting_team_name bowling_team_name recent_wickets_12_pre training_label_available batting_win_label_outcome
legal_event team_runs_event wicket_event pressure_eligible'''.split()
INTERACTIONS = ['leverage', 'urgency_contest', 'match_stakes', 'leverage_x_chase', 'urgency_contest_x_late_chase',
                'leverage_x_collapse']


def mass(d):
    return 1 / d.groupby('match_id').match_id.transform('size').to_numpy()


def dynamic(d):
    # Hand-set 10-point reallocations. A competitive late chase emphasizes demand;
    # two recent wickets emphasize next-ball resource loss. Both gates are in [0,1].
    chase = d.innings.eq(2).astype(float)
    progress = 1 - d.legal_balls_remaining_pre / 120
    late_chase = chase * progress * d.contestability
    collapse = (d.recent_wickets_12_pre / 2).clip(upper=1)
    delta = .10 * (collapse - late_chase)
    d['weight_leverage'] = .50 + delta
    d['weight_urgency_contest'] = .30 - delta
    d['urgency_contest'] = d.urgency * d.contestability
    d['pressure_dynamic'] = 100 * (d.weight_leverage * d.leverage
                                   + d.weight_urgency_contest * d.urgency_contest + .20 * d.match_stakes)
    d['leverage_x_chase'] = d.leverage * chase
    d['urgency_contest_x_late_chase'] = d.urgency_contest * late_chase
    d['leverage_x_collapse'] = d.leverage * collapse


def tag(d):
    d['pressure_static'] = d.pressure_score_v1
    dynamic(d)
    cutoffs = []
    for kind in ['static', 'dynamic']:
        d['high_' + kind] = pd.Series(pd.NA, index=d.index, dtype='boolean')
    for fold, z in d.groupby('model_fold_start'):
        tr = d[d.match_date_parsed < fold]
        row = dict(fold=fold, train_matches=tr.match_id.nunique(), train_events=len(tr), test_events=len(z))
        for kind in ['static', 'dynamic']:
            col = 'pressure_' + kind
            if len(tr):
                q = tr[col].quantile(.75)
                d.loc[z.index, 'high_' + kind] = z[col].ge(q)
                row['cutoff_' + kind] = q
        cutoffs.append(row)
    pd.DataFrame(cutoffs).to_csv(E / 'consequence_cutoffs.csv', index=False)
    assert (d.model_train_end_date < d.match_date_parsed).all()
    d[['delivery_id', 'match_id', 'match_date_parsed', 'pressure_static', 'pressure_dynamic',
       'weight_leverage', 'weight_urgency_contest', 'high_static', 'high_dynamic']].to_parquet(
        E / 'consequence_events.parquet', index=False)


class FittedWP:
    """Restore the Phase 1 standardized logistic regression on its original design."""
    def __init__(self, coefs, hist, fold):
        self.fold = fold
        self.coefs = coefs.loc[fold]
        self.rate = hist.team_runs_event.sum() / hist.legal_event.sum()
        self.curve = trajectory(hist)

    def __call__(self, d, inning):
        x = design(d, inning, self.curve, self.rate)
        c = self.coefs.loc[inning]
        a = c.loc[x.columns]
        eta = ((x.to_numpy()-a.training_mean.to_numpy()) / a.training_scale.to_numpy()) @ a.standardized_coefficient.to_numpy()
        p = 1 / (1 + np.exp(-np.clip(eta + c.loc['intercept', 'standardized_coefficient'], -700, 700)))
        if inning == 2:
            won = d.runs_required_pre.le(0)
            ended = d.legal_balls_remaining_pre.le(0) | d.wickets_remaining_pre.le(0)
            p = np.where(won, 1., np.where(ended, 0., p))
        return p


def updated(d, model, inning):
    # Used solely as an ex-post RESPONSE variable, never in feature generation.
    after = next_state(d, d.team_runs_event, d.wicket_event, inning)
    return np.abs(model(after, inning) - model(d, inning))


def calibration_x(p, balls, method):
    p = np.clip(p, 1e-7, 1-1e-7)
    logit = np.log(p/(1-p))
    if method == 'platt':
        return logit[:, None]
    late = np.maximum(1 - np.asarray(balls)/30, 0)
    return np.column_stack([logit, late, late * logit])


def calibrated(model, method, z, inning, wp):
    p = wp(z, inning)
    if method == 'raw':
        return p
    pred = model.predict_proba(calibration_x(p, z.legal_balls_remaining_pre, method))[:, 1]
    return np.where((p == 1) | (p == 0), p, pred)


def parameter_rows(parameters, intercept, model, fold, features):
    return [dict(fold=fold, model=model, feature=k, value=float(v))
            for k, v in zip(['intercept', *features], [intercept, *parameters])]


def wp_audit(d, all_data, coefs, log, coef_rows):
    rows, bins, models = [], [], {}
    for year in [2024, 2025, 2026]:
        cut = pd.Timestamp(f'{year}-01-01')
        hist = all_data[all_data.training_label_available & (all_data.match_date_parsed < cut)]
        wp = FittedWP(coefs, hist, cut)
        tr = d[d.training_label_available & (d.match_date_parsed < cut)]
        te = all_data[all_data.training_label_available & (all_data.match_date_parsed.dt.year == year)]
        for inning in [1, 2]:
            a, b = tr[tr.innings.eq(inning)], te[te.innings.eq(inning)]
            support = support_counts(hist[hist.innings.eq(inning)], b, inning, wp.curve)
            b = b[support.ge(10)]
            for method in ['raw', 'platt', 'horizon_platt']:
                key = f'WP innings {inning} {method}'
                model = None
                if method != 'raw':
                    features = ['logit'] if method == 'platt' else ['logit', 'last30_progress', 'logit_x_last30_progress']
                    model = LogisticRegression(C=1., max_iter=1000)
                    model.fit(calibration_x(a.win_probability_pre.to_numpy(), a.legal_balls_remaining_pre, method),
                              a.batting_win_label_outcome, sample_weight=mass(a))
                    coef_rows.extend(parameter_rows(model.coef_[0], model.intercept_[0], key, cut, features))
                if year == 2026 and inning == 2:
                    models[method] = model
                p = calibrated(model, method, b, inning, wp)
                y = b.batting_win_label_outcome
                w = mass(b)
                # Both the forecasts and the support dictionary are frozen on 1 January.
                by_bin = pd.DataFrame({'p': p, 'y': y, 'w': w, 'bin': np.minimum((10*p).astype(int), 9)})
                ece = 0.
                for k, group in by_bin.groupby('bin'):
                    pred = np.average(group.p, weights=group.w)
                    obs = np.average(group.y, weights=group.w)
                    frac = group.w.sum()/w.sum()
                    bins.append(dict(year=year, innings=inning, method=method, bin=int(k), mass=frac, pred=pred, actual=obs))
                    ece += frac * abs(pred-obs)
                rows.append(dict(year=year, innings=inning, method=method, matches=b.match_id.nunique(), events=len(b),
                                 prior_brier=np.average((hist[hist.innings.eq(inning)].drop_duplicates('match_id').batting_win_label_outcome.mean()-y)**2, weights=w),
                                 brier=np.average((p-y)**2, weights=w), log_loss=log_loss(y, p, sample_weight=w), ece10=ece))
                training = a if model is not None else hist[hist.innings.eq(inning)]
                log.append(f"{key}, trained {training.match_date_parsed.min().date()} to {training.match_date_parsed.max().date()}, "
                           f"{training.match_id.nunique()} matches / {len(training)} events, tested {b.match_date_parsed.min().date()} to "
                           f"{b.match_date_parsed.max().date()}, {b.match_id.nunique()} matches / {len(b)} events, Brier {rows[-1]['brier']:.6f}.")
    pd.DataFrame(rows).to_csv(E / 'consequence_wp_validation.csv', index=False)
    pd.DataFrame(bins).to_csv(E / 'consequence_wp_calibration.csv', index=False)
    return models


def consequence_audit(d, all_data, coefs, log, coef_rows):
    legal = d[d.legal_event.eq(1)].copy()
    for fold, z in legal.groupby('model_fold_start'):
        hist = all_data[all_data.training_label_available & (all_data.match_date_parsed < fold)]
        wp = FittedWP(coefs, hist, fold)
        for inning, group in z.groupby('innings'):
            np.testing.assert_allclose(wp(group, inning), group.win_probability_pre, atol=1e-10)
            legal.loc[group.index, 'actual_revision'] = updated(group, wp, inning)
    models = {'static': ['pressure_static'], 'dynamic': ['pressure_dynamic'],
              'leverage_only': ['leverage'], 'additive': INTERACTIONS[:3], 'interactions': INTERACTIONS, 'rrr_only': ['rrr']}
    legal['rrr'] = 6 * legal.runs_required_pre / legal.legal_balls_remaining_pre
    results, groups = [], []
    rng = np.random.default_rng(921)
    for year in [2024, 2025, 2026]:
        cut = pd.Timestamp(f'{year}-01-01')
        for sample in ['both', 'chases']:
            z = legal if sample == 'both' else legal[legal.innings.eq(2)]
            tr, te = z[z.match_date_parsed < cut], z[z.match_date_parsed.dt.year == year].copy()
            for name, features in models.items():
                if name == 'rrr_only' and sample != 'chases':
                    continue
                # Estimate the mapping from index to forecast revision on earlier matches.
                # Transfer assumptions can be audited via the slope, intercept and match errors.
                if name == 'interactions' and sample == 'chases':
                    features = [c for c in features if c != 'leverage_x_chase']
                model = Ridge(alpha=.1) if name in ['additive', 'interactions'] else LinearRegression()
                model.fit(tr[features], tr.actual_revision, sample_weight=mass(tr))
                pred = model.predict(te[features]).clip(0, 1)
                err = (pred-te.actual_revision)**2
                te[name] = err
                by_match = err.groupby(te.match_id).mean()
                diff = by_match - te.groupby('match_id').static.mean()
                ci = np.quantile(rng.choice(diff, size=(2000, len(diff))).mean(axis=1), [.025,.975])
                results.append(dict(year=year, sample=sample, model=name, train_matches=tr.match_id.nunique(), train_events=len(tr),
                                    test_matches=te.match_id.nunique(), test_events=len(te), mse=by_match.mean(), delta_mse_static=diff.mean(),
                                    delta_lo95=ci[0], delta_hi95=ci[1]))
                coef_rows.extend(parameter_rows(model.coef_,model.intercept_, f'{name}/{sample}', cut, features))
                log.append(f"Revision {name}/{sample}, features {features}, fit {tr.match_date_parsed.min().date()} to "
                           f"{tr.match_date_parsed.max().date()} ({len(tr)} balls, {tr.match_id.nunique()} matches); "
                           f"test {te.match_date_parsed.min().date()} to {te.match_date_parsed.max().date()} "
                           f"({len(te)} balls, {te.match_id.nunique()} matches); MSE {by_match.mean():.7f}, delta vs static {diff.mean():+.7f}.")
                if sample == 'both':
                    for (comp, phase), a in te.groupby(['competition_type','phase_pre']):
                        groups.append(dict(year=year, competition=comp, phase=phase, model=name, matches=a.match_id.nunique(),
                                           balls=len(a), mse=a.groupby('match_id')[name].mean().mean()))
    pd.DataFrame(results).to_csv(E / 'consequence_models.csv', index=False)
    pd.DataFrame(groups).to_csv(E / 'consequence_by_group.csv', index=False)


def scenarios(d, coefs, calibrators):
    fold = d.model_fold_start.max()
    hist = d[d.training_label_available & (d.match_date_parsed < fold)]
    wp = FittedWP(coefs, hist, fold)
    ch = hist[hist.innings.eq(2)]
    z = ch.iloc[:3].copy()
    z['runs_required_pre'] = [6.,22.,36.]
    z['wickets_remaining_pre'] = 7
    z['legal_balls_remaining_pre'] = 6
    z['legal_balls_bowled_pre'] = 114
    z['competition_type'] = 'IPL'
    cases = pd.read_csv(OUT / 'illustrative_situations.csv').iloc[:3]
    outcomes, _ = outcome_weights(ch)
    rows, counts = [], []
    for law, train in [('all_death', ch), ('ipl_death', ch[ch.competition_type.eq('IPL')]),
                       ('high_demand_death', ch[ch.runs_required_pre * 6 >= 12 * ch.legal_balls_remaining_pre])]:
        _, weights = outcome_weights(train)
        q = weights['death']
        for method, model in calibrators.items():
            p = calibrated(model, method, z, 2, wp)
            future = np.stack([calibrated(model, method, next_state(z,r,w,2),2,wp) for r,w in outcomes], axis=1)
            sd = np.sqrt(((future - (future @ q)[:, None])**2) @ q)
            scores = score_components(p, np.minimum(sd/.1,1), cases.urgency.to_numpy(), .4)
            for i,r in enumerate(z.runs_required_pre):
                rows.append(dict(law=law, model=method, runs=int(r), required_rate=r, win_p=p[i], sd_next_wp=sd[i], pressure=scores[i]))
        # Exact finite-ball calculation under stationary empirical death-over outcomes.
        # Conditions on legal balls; omits extras, strike and strategic adaptation. A
        # physical sensitivity check with explicit wicket absorption, no fitted WP.
        @lru_cache(maxsize=None)
        def finite(r,b,w):
            if r <= 0:
                return 1.
            if b == 0 or w == 0:
                return 0.
            return sum(v * finite(r-runs,b-1,w-wicket) for v,(runs,wicket) in zip(q,outcomes))
        for i,r in enumerate(z.runs_required_pre.astype(int)):
            p = finite(r,6,7)
            future = np.array([finite(r-runs,5,7-wicket) for runs,wicket in outcomes])
            sd = np.sqrt(((future-p)**2) @ q)
            score = score_components(p, min(sd/.1,1), cases.urgency.iloc[i], .4)
            rows.append(dict(law=law, model='iid_six_legal_balls', runs=r, required_rate=r, win_p=p, sd_next_wp=sd, pressure=score))
    # First event with six balls left, even if that event proves illegal. Avoid
    # selectively observing a lower requirement after an extra at the same horizon.
    first = ch[ch.legal_balls_remaining_pre.eq(6)].drop_duplicates('match_id')
    for r in [6,22,36]:
        near = first[first.runs_required_pre.between(r-2,r+2)]
        for sample,pop in [('both_comp_all_wickets', near), ('both_comp_4to7_wickets',near[near.wickets_remaining_pre.between(4,7)]),
                           ('ipl_4to7_wickets',near[near.competition_type.eq('IPL') & near.wickets_remaining_pre.between(4,7)])]:
            counts.append(dict(runs=r, cohort=sample, matches=len(pop), wins=int(pop.batting_win_label_outcome.sum()),
                               supported_oof_matches=int(pop.win_probability_pre.notna().sum()),
                               supported_oof_wins=pop.loc[pop.win_probability_pre.notna(),'batting_win_label_outcome'].sum(),
                               mean_supported_oof_wp=pop.win_probability_pre.mean()))
    scenarios = pd.DataFrame(rows)
    np.testing.assert_allclose(scenarios.query("law == 'all_death' and model == 'raw'").pressure,
                               cases.pressure_raw_illustration, atol=1e-8)
    scenarios.to_csv(E / 'consequence_scenarios.csv', index=False)
    pd.DataFrame(counts).to_csv(E / 'consequence_scenario_neighbors.csv', index=False)


def export_case(d):
    # First recorded 2026 IPL chase meeting the conditions. Only threshold
    # membership and two *earlier* wickets select it, never the next outcome.
    g = d[d.innings.eq(2) & d.competition_type.eq('IPL') & d.match_date_parsed.ge(CUT)]
    prior = g.groupby('match_id')
    at_risk = (prior.wicket_event.shift(1).eq(1) & prior.wicket_event.shift(2).eq(1)
               & prior.innings_ball_num.shift(2).eq(g.innings_ball_num-2)
               & g.recent_wickets_12_pre.eq(2) & g.legal_balls_remaining_pre.between(6,90)
               & g.high_static.eq(True))
    case = g[at_risk].iloc[0]
    seq = g[g.match_id.eq(case.match_id) & g.innings_ball_num.between(case.innings_ball_num-2,case.innings_ball_num)]
    cols = ['delivery_id','match_date_parsed','batting_team_name','bowling_team_name','striker_name','bowler_name',
            'runs_required_pre','legal_balls_remaining_pre','wickets_remaining_pre','recent_wickets_12_pre',
            'win_probability_pre','leverage','urgency_contest','match_stakes','weight_leverage','weight_urgency_contest',
            'pressure_static','pressure_dynamic','high_static','high_dynamic','team_runs_event','wicket_event']
    seq[cols].to_csv(E/'consequence_setback_example.csv', index=False)


def scenario_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    data = pd.read_csv(E/'consequence_scenarios.csv').query("law == 'all_death'")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.45), layout='constrained')
    styles = [('raw','Original logistic','#174d77','o'), ('horizon_platt','Horizon-calibrated logistic','#af5c1a','s'),
              ('iid_six_legal_balls','Six-ball outcome model','#32806a','^')]
    for model,label,color,symbol in styles:
        z = data[data.model.eq(model)]
        for ax, metric in zip(axes,['win_p','pressure']):
            ax.plot(z.runs, z[metric], marker=symbol, linewidth=1.8, color=color, label=label)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set(ylim=(-.03,1.04), title='Chase win probability')
    axes[1].set(ylim=(0,100), title='Pre-delivery pressure score')
    for ax in axes:
        ax.set_xticks([6,22,36],['6 off 6','22 off 6','36 off 6'])
        ax.spines[['right','top']].set_visible(False)
        ax.grid(axis='y', alpha=.22)
    axes[0].legend(loc='upper right', fontsize=8, frameon=False)
    fig.savefig(E/'consequence_scenarios.png', dpi=180)
    plt.close(fig)


def main():
    E.mkdir(parents=True, exist_ok=True)
    all_data = pd.read_parquet(OUT / 'pressure_scored.parquet', columns=COLS)
    d = all_data[all_data.pressure_eligible].copy()
    assert d.delivery_id.is_unique
    audit_columns = ['delivery_id', 'score_pre', 'wickets_remaining_pre', 'legal_balls_remaining_pre',
                     'runs_required_pre', 'pressure_score_v1', 'win_probability_pre', 'high_pressure_selected']
    comparison = OUT / 'phase2/phase1_reproduction/pressure_scored.parquet'
    audit = json.loads((OUT/'validation_checks.json').read_text())
    assert all(audit.values())
    manifest = json.loads((OUT/'run_manifest.json').read_text())
    assert manifest['source_hashes_unchanged']
    verification = dict(events=len(all_data), raw_inputs=manifest['source_sha256'],
                        passing_reconstruction_checks=len(audit), archived_comparison='reference absent')
    if comparison.exists():
        rebuilt = pd.read_parquet(OUT/'pressure_scored.parquet', columns=audit_columns)
        archive = pd.read_parquet(comparison, columns=audit_columns)
        pd.testing.assert_frame_equal(rebuilt, archive, check_exact=True)
        verification['archived_comparison'] = 'exact match'
        verification['compared_columns'] = audit_columns
    (E/'consequence_verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    tag(d)
    export_case(d)
    coefs = pd.read_csv(OUT / 'model_coefficients.csv', parse_dates=['fold']).set_index(['fold','innings','feature'])
    log, params = [], []
    calibrators = wp_audit(d, all_data, coefs, log, params)
    scenarios(all_data, coefs, calibrators)
    scenario_figure()
    consequence_audit(d, all_data, coefs, log, params)
    frame = pd.DataFrame(params)
    frame.to_csv(E / 'consequence_parameters.csv', index=False)
    note = ('Quarterly Phase 1 coefficient effects are in outputs/model_coefficients.csv. '
            'OOF calibration uses earlier quarterly predictions. WP validation holds the January WP model fixed throughout each year. '
            'Revision validation uses the as-issued quarterly WP estimates and freezes revision-regression coefficients throughout the test year. '
            'The outcome is a model-dependent absolute revision on realized legal balls. Entire-match QA is retrospective.')
    full = '\n'.join([note,*log,'Fitted parameters',frame.to_string(index=False)]) + '\n'
    (E / 'consequence_training.log').write_text(full)
    notes = Path(__file__).with_name('consequence_notes.md')
    if notes.is_file():
        (E / 'consequence_report.md').write_text(notes.read_text())
    print(full, flush=True)


if __name__ == '__main__':
    main()
