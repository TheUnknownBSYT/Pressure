"""Expanding chronological models. Predictor inputs are explicitly allowlisted."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


@dataclass
class ModelConfig:
    min_train_matches: int = 100
    min_class_matches: int = 25
    min_support_matches: int = 10
    fold_months: int = 3
    leverage_scale: float = .10
    seed: int = 42


def trajectory(train):
    """Mean historical first-innings score, carrying all-out totals to ball 120."""
    curves = []
    for _, z in train[train.innings.eq(1)].groupby('match_id'):
        after_balls = z.legal_balls_bowled_pre + z.legal_event
        after_score = z.score_pre + z.team_runs_event
        # Extras count toward the score at the next legal-ball checkpoint.
        curve = pd.Series(after_score.to_numpy(), index=after_balls).groupby(level=0).last()
        curve = curve.reindex(range(121)).ffill().fillna(0)
        curve.iloc[0] = 0
        curves.append(curve.to_numpy())
    return np.mean(curves, axis=0)


def expected_at(d, curve):
    return np.interp(d.legal_balls_bowled_pre.to_numpy(), np.arange(121), curve)


def design(d, inning, curve, scoring_rate):
    """No outcome, identifiers, final totals, current delivery or roster fields."""
    t = d.legal_balls_bowled_pre.to_numpy(dtype=float) / 120
    w = d.wickets_remaining_pre.to_numpy(dtype=float) / 10
    ipl = d.competition_type.eq('IPL').to_numpy(dtype=float)
    if inning == 1:
        gap = (d.score_pre.to_numpy() - expected_at(d, curve)) / np.sqrt(12 + 120 * t)
        values = [gap, w, t, gap * t, w * t, ipl]
        names = ['score_gap_scaled', 'wickets_fraction', 'progress', 'gap_x_progress', 'wickets_x_progress', 'is_ipl']
    else:
        b = d.legal_balls_remaining_pre.to_numpy(dtype=float)
        r = d.runs_required_pre.to_numpy(dtype=float)
        margin = (r - scoring_rate * b) / np.sqrt(b + 6)
        required_rate = 6 * r / np.maximum(b, 1)
        # Spare wickets lose value when there is no time left to use them.
        resources = w * np.sqrt(b / 120)
        values = [margin, required_rate, resources, np.log1p(b), ipl]
        names = ['chase_requirement_margin', 'required_run_rate', 'time_adjusted_wickets', 'log_balls_remaining', 'is_ipl']
    return pd.DataFrame(np.array(values).T, index=d.index, columns=names)


def support_keys(d, inning, curve):
    wk = pd.cut(d.wickets_remaining_pre, [-1, 3, 7, 10], labels=['1-3', '4-7', '8-10']).astype(str)
    if inning == 1:
        gap = d.score_pre - expected_at(d, curve)
        demand = pd.cut(gap, [-np.inf, -20, 20, np.inf], labels=['below', 'near', 'above']).astype(str)
    else:
        rate = 6 * d.runs_required_pre / d.legal_balls_remaining_pre.clip(lower=1)
        demand = pd.cut(rate, [-np.inf, 6, 10, 14, 20, 30, np.inf], labels=['0-6', '6-10', '10-14', '14-20', '20-30', '30+']).astype(str)
    horizon = d.phase_pre.astype(str) if inning == 1 else pd.cut(d.legal_balls_remaining_pre,
        [-1, 3, 6, 12, 30, 60, 90, 120], labels=['1-3','4-6','7-12','13-30','31-60','61-90','91-120']).astype(str)
    return horizon + '|' + wk + '|' + demand


def support_counts(train, test, inning, curve):
    keys = support_keys(train, inning, curve)
    counts = pd.DataFrame({'key': keys, 'match': train.match_id}).groupby('key').match.nunique()
    return support_keys(test, inning, curve).map(counts).fillna(0).astype(int)


def next_state(d, runs, wicket, inning):
    z = d.copy()
    z['score_pre'] = z.score_pre + runs
    z['legal_balls_bowled_pre'] = z.legal_balls_bowled_pre + 1
    z['legal_balls_remaining_pre'] = (z.legal_balls_remaining_pre - 1).clip(lower=0)
    z['wickets_remaining_pre'] = (z.wickets_remaining_pre - wicket).clip(lower=0)
    if inning == 2:
        z['runs_required_pre'] = (z.runs_required_pre - runs).clip(lower=0)
    return z


def predict(model, d, inning, curve, rate):
    p = model.predict_proba(design(d, inning, curve, rate))[:, 1]
    if inning == 2:
        won = d.runs_required_pre.le(0).to_numpy()
        ended = (d.legal_balls_remaining_pre.le(0) | d.wickets_remaining_pre.le(0)).to_numpy()
        # A regulation tie is not a win. Training excludes super-over-decided outcomes.
        p = np.where(won, 1., np.where(ended, 0., p))
    return p


def outcome_weights(train):
    z = train[train.legal_event.eq(1)]
    outcomes = [(r, 0) for r in range(7)] + [(0, 1)]
    weights = {}
    for phase in ['powerplay', 'middle', 'death']:
        a = z[z.phase_pre.eq(phase)]
        # Add-one smoothing; cap rare >6 legal run outcomes at 6, aggregate wickets
        # as zero-run dismissals. This conditional-legal-ball approximation is labelled.
        counts = np.array([((a.team_runs_event.clip(upper=6).eq(r)) & a.wicket_event.eq(w)).sum()
                           if w == 0 else a.wicket_event.sum() for r, w in outcomes], dtype=float) + 1
        weights[phase] = counts / counts.sum()
    return outcomes, weights


def leverage(model, d, inning, curve, rate, outcomes, weights, scale):
    probs = np.column_stack([predict(model, next_state(d, r, w, inning), inning, curve, rate)
                             for r, w in outcomes])
    q = np.vstack([weights[p] for p in d.phase_pre])
    mean = (probs * q).sum(axis=1)
    sd = np.sqrt((q * (probs - mean[:, None]) ** 2).sum(axis=1))
    return np.clip(sd / scale, 0, 1), sd


def urgency(d, inning, curve, capability):
    b = d.legal_balls_remaining_pre.to_numpy(dtype=float)
    w = d.wickets_remaining_pre.to_numpy(dtype=float)
    time = np.clip(1 - b / 120, 0, 1)
    if inning == 2:
        demand = np.clip((6 * d.runs_required_pre.to_numpy() / np.maximum(b, 1)) / capability, 0, 1)
    else:
        # Remaining rate needed to reach a historical successful defending total.
        gap = np.maximum(curve[-1] - d.score_pre.to_numpy(), 0)
        demand = np.clip((6 * gap / np.maximum(b, 1)) / capability, 0, 1)
    recovery = np.clip(1 - (b / 120) * (w / 10), 0, 1)
    return np.clip(.45 * time + .35 * demand + .20 * recovery, 0, 1)


def score_components(p, lev, urg, stakes):
    contest = 4 * p * (1 - p)
    return 100 * (.50 * lev + .30 * urg * contest + .20 * stakes)


def metric_row(z, p, baseline, inning, fold):
    y = z.batting_win_label_outcome.to_numpy()
    # Each match contributes equal mass, despite different innings lengths.
    sw = 1 / z.groupby('match_id').match_id.transform('size').to_numpy()
    return {'fold': fold, 'innings': inning, 'matches': z.match_id.nunique(), 'deliveries': len(z),
            'brier': brier_score_loss(y, p, sample_weight=sw),
            'historical_prior_brier': brier_score_loss(y, baseline, sample_weight=sw),
            'coinflip_brier': brier_score_loss(y, np.full(len(z), .5), sample_weight=sw),
            'log_loss': log_loss(y, np.clip(p, 1e-6, 1-1e-6), sample_weight=sw, labels=[0, 1]),
            'auc': roc_auc_score(y, p, sample_weight=sw) if len(np.unique(y)) == 2 else np.nan}


def fit_historical(d, cfg):
    d = d.copy()
    cols = ['win_probability_pre', 'difficulty_proxy', 'contestability', 'leverage',
            'leverage_sd_probability', 'urgency', 'pressure_score_v1', 'expected_score_pre',
            'score_vs_expected_pre', 'historical_first_innings_total', 'historical_defendable_total',
            'historical_capability_run_rate', 'historical_baseline_pre']
    for c in cols:
        d[c] = np.nan
    d['probability_status'] = np.where(d.standard_t20_eligible, 'insufficient_prior_matches', 'ineligible_match_state')
    d['model_train_end_date'] = pd.NaT
    d['model_fold_start'] = pd.NaT
    d['prior_train_matches'] = 0
    d['state_support_matches'] = 0
    d['leverage_method'] = 'provisional_weighted_next_legal_ball_sd_div_0.10'
    d['model_version'] = 'v1_logistic_expanding_quarterly'
    d['pressure_eligible'] = False
    folds, metrics, coefficients, weight_rows = [], [], [], []
    latest = {}
    eligible = d[d.standard_t20_eligible]
    if eligible.empty:
        return d, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    start = eligible.match_date_parsed.min().to_period('M').start_time
    stop = eligible.match_date_parsed.max() + pd.offsets.MonthBegin(cfg.fold_months + 1)
    boundaries = pd.date_range(start, stop, freq=pd.DateOffset(months=cfg.fold_months))
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        train = d[d.training_label_available & d.match_date_parsed.lt(lo)]
        test = d[d.standard_t20_eligible & d.match_date_parsed.ge(lo) & d.match_date_parsed.lt(hi)]
        if test.empty:
            continue
        nmatch = train.match_id.nunique()
        d.loc[test.index, 'prior_train_matches'] = nmatch
        fold = {'start': str(lo.date()), 'end_exclusive': str(hi.date()), 'training_matches': nmatch,
                'evaluation_matches': test.match_id.nunique(), 'training_last_date': str(train.match_date_parsed.max())}
        folds.append(fold)
        if nmatch < cfg.min_train_matches:
            fold['status'] = 'insufficient_prior_matches'
            continue
        assert train.match_date_parsed.max() < test.match_date_parsed.min()
        assert not set(train.match_id) & set(test.match_id)
        curve = trajectory(train)
        rate = train.team_runs_event.sum() / train.legal_event.sum()
        # Empirical 75th percentile of completed legal-over scoring rates, with a
        # floor of 6 only to make normalization stable on unusual future datasets.
        over = train.groupby(['match_id', 'innings', 'overid']).agg(runs=('team_runs_event', 'sum'), balls=('legal_event', 'sum'))
        capability = max(6., float(over.loc[over.balls.eq(6), 'runs'].quantile(.75)))
        first = train[train.innings.eq(1)]
        totals = first.groupby('match_id').team_runs_event.sum()
        successful = first.groupby('match_id').batting_win_label_outcome.first().eq(1)
        defendable = float(totals[successful].median())
        urgency_curve = curve.copy()
        urgency_curve[-1] = defendable
        for inning in [1, 2]:
            tr, te = train[train.innings.eq(inning)], test[test.innings.eq(inning)]
            if te.empty:
                continue
            labels = tr.drop_duplicates('match_id').batting_win_label_outcome
            if len(labels.value_counts()) != 2 or labels.value_counts().min() < cfg.min_class_matches:
                d.loc[te.index, 'probability_status'] = 'insufficient_outcome_classes'
                continue
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000, random_state=cfg.seed))
            sw = 1 / tr.groupby('match_id').match_id.transform('size').to_numpy()
            x = design(tr, inning, curve, rate)
            model.fit(x, tr.batting_win_label_outcome, logisticregression__sample_weight=sw)
            for name, value, mean, scale in zip(x.columns, model[-1].coef_[0], model[0].mean_, model[0].scale_):
                coefficients.append({'fold': str(lo.date()), 'innings': inning, 'feature': name, 'standardized_coefficient': value, 'training_mean': mean, 'training_scale': scale})
            coefficients.append({'fold': str(lo.date()), 'innings': inning, 'feature': 'intercept', 'standardized_coefficient': model[-1].intercept_[0]})
            support = support_counts(tr, te, inning, curve)
            outcomes, weights = outcome_weights(tr)
            for phase, probs in weights.items():
                for (r, w), q in zip(outcomes, probs):
                    weight_rows.append({'fold': str(lo.date()), 'innings': inning, 'phase': phase, 'runs': r, 'wicket': w, 'probability': q})
            p = predict(model, te, inning, curve, rate)
            lev, sd = leverage(model, te, inning, curve, rate, outcomes, weights, cfg.leverage_scale)
            urg = urgency(te, inning, urgency_curve, capability)
            valid = support.ge(cfg.min_support_matches).to_numpy()
            idx = te.index
            assignments = {'state_support_matches': support, 'model_train_end_date': tr.match_date_parsed.max(),
                           'model_fold_start': lo, 'expected_score_pre': expected_at(te, curve),
                           'score_vs_expected_pre': te.score_pre - expected_at(te, curve),
                           'historical_first_innings_total': curve[-1], 'historical_defendable_total': defendable,
                           'historical_capability_run_rate': capability, 'urgency': urg,
                           'historical_baseline_pre': labels.mean(),
                           'probability_status': np.where(valid, 'available_provisional', 'insufficient_state_support')}
            for c, v in assignments.items():
                d.loc[idx, c] = v
            for c, v in {'win_probability_pre': p, 'difficulty_proxy': 1-p, 'contestability': 4*p*(1-p),
                         'leverage': lev, 'leverage_sd_probability': sd,
                         'pressure_score_v1': score_components(p, lev, urg, te.match_stakes.to_numpy())}.items():
                d.loc[idx, c] = np.where(valid, v, np.nan)
            evaluation = te.training_label_available.to_numpy() & valid
            if evaluation.any():
                metrics.append(metric_row(te.iloc[np.flatnonzero(evaluation)], p[evaluation],
                                          np.full(evaluation.sum(), labels.mean()), inning, str(lo.date())))
            latest[inning] = (model, tr, curve, rate, outcomes, weights, urgency_curve, capability, lo)
        fold['status'] = 'fit'
    d['pressure_eligible'] = d.pressure_score_v1.notna()
    # Response context for later modelling: lagged pressure exposure, never a rating.
    batter = d.groupby(['match_id', 'innings', 'striker_id'])
    d['batter_previous_pressure_pre'] = batter.pressure_score_v1.shift()
    d['batter_mean_prior_pressure_pre'] = batter.pressure_score_v1.transform(lambda x: x.shift().expanding().mean())
    d['batter_pressure_change_pre'] = d.pressure_score_v1 - d.batter_previous_pressure_pre
    illustrations = []
    if 2 in latest:
        model, tr, curve, rate, outcomes, weights, uc, cap, lo = latest[2]
        template = tr.iloc[[0]].copy()
        cases = [(6, 6, 7, .4), (22, 6, 7, .4), (36, 6, 7, .4), (6, 36, 7, .4), (22, 6, 2, .4), (22, 6, 7, 1.)]
        for runs, balls, wkts, stakes in cases:
            z = template.copy()
            z['legal_balls_remaining_pre'] = balls
            z['legal_balls_bowled_pre'] = 120 - balls
            z['runs_required_pre'] = runs
            z['wickets_remaining_pre'] = wkts
            z['competition_type'] = 'IPL'
            z['phase_pre'] = 'death' if balls <= 30 else 'middle'
            z['match_stakes'] = stakes
            p = predict(model, z, 2, curve, rate)
            lev, sd = leverage(model, z, 2, curve, rate, outcomes, weights, cfg.leverage_scale)
            urg = urgency(z, 2, uc, cap)
            n = int(support_counts(tr, z, 2, curve).iloc[0])
            illustrations.append({'runs_needed': runs, 'balls_left': balls, 'wickets_left': wkts,
                'stakes': stakes, 'p_model_raw': p[0], 'difficulty': 1-p[0], 'contestability': 4*p[0]*(1-p[0]),
                'leverage_provisional': lev[0], 'urgency': urg[0],
                'pressure_raw_illustration': score_components(p, lev, urg, stakes)[0],
                'state_support_matches': n, 'production_available': n >= cfg.min_support_matches,
                'model_fold': str(lo.date())})
    return d, pd.DataFrame(folds), pd.DataFrame(metrics), pd.DataFrame(coefficients), pd.DataFrame(weight_rows), pd.DataFrame(illustrations)


def pressure_flags(d, top_percent=25):
    if not 0 < top_percent < 100:
        raise ValueError('top_percent must be between 0 and 100')
    d = d.copy()
    thresholds = {}
    for top in sorted(set([10, 20, 25, 30, top_percent])):
        cutoff = d.pressure_score_v1.quantile(1 - top / 100)
        thresholds[str(top)] = float(cutoff)
        d[f'high_pressure_top_{top:g}pct'] = d.pressure_score_v1.ge(cutoff) & d.pressure_score_v1.notna()
    d['high_pressure_selected'] = d[f'high_pressure_top_{top_percent:g}pct']
    return d, thresholds
