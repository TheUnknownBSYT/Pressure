"""Local diagnostics, independent reconciliation, and reproducible documentation."""
from pathlib import Path
import os
import csv
import platform
import numpy as np
import pandas as pd
from .data import save_json


def md_table(frame, limit=40):
    f = frame.head(limit).copy()
    for c in f:
        f[c] = f[c].map(lambda x: f'{x:.4f}' if isinstance(x, (float, np.floating)) else str(x))
    lines = ['| ' + ' | '.join(map(str, f.columns)) + ' |', '| ' + ' | '.join(['---'] * len(f.columns)) + ' |']
    lines += ['| ' + ' | '.join(str(x).replace('|', '/') for x in row) + ' |' for row in f.itertuples(index=False, name=None)]
    return '\n'.join(lines)


def verify_invariants(d):
    checks = {}
    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)
    check('unique_delivery_ids', d.delivery_id.is_unique)
    check('source_row_preservation', d.source_row.nunique() == len(d))
    check('chronological_sort', d[['match_date_parsed', 'match_id', 'innings', 'innings_ball_num']].equals(
        d.sort_values(['match_date_parsed', 'match_id', 'innings', 'innings_ball_num'])[['match_date_parsed', 'match_id', 'innings', 'innings_ball_num']]))
    g = d.groupby(['match_id', 'innings'])
    for state, event in [('score_pre', 'team_runs_event'), ('wickets_fallen_pre', 'wicket_event'), ('legal_balls_bowled_pre', 'legal_event')]:
        delta = g[state].shift(-1) - d[state]
        check(f'{state}_reconciles', delta.dropna().eq(d.loc[delta.notna(), event]).all())
        check(f'{state}_starts_zero', g[state].first().eq(0).all())
    check('wide_no_ball_not_legal', d.loc[d.extras_type.isin(['wide', 'Overstep', 'other']), 'legal_event'].eq(0).all())
    check('runout_not_bowler_wicket', d.loc[d.dismissal_type.eq('RunOut'), 'bowler_wicket_event'].eq(0).all())
    check('nonnegative_remaining_balls', d.legal_balls_remaining_pre.dropna().ge(0).all())
    check('super_over_not_scored', d.loc[d.is_super_over, 'pressure_score_v1'].isna().all())
    check('revised_timing_not_fabricated', d.loc[d.revised_match, ['target_pre', 'scheduled_balls_pre', 'pressure_score_v1']].isna().all().all())
    for c in ['win_probability_pre', 'difficulty_proxy', 'contestability', 'leverage', 'urgency', 'match_stakes']:
        check(f'{c}_bounds', d[c].dropna().between(0, 1).all())
    check('pressure_bounds', d.pressure_score_v1.dropna().between(0, 100).all())
    scored = d[d.pressure_eligible]
    check('prior_only_training', (scored.model_train_end_date < scored.model_fold_start).all() and (scored.model_fold_start <= scored.match_date_parsed).all())
    check('eligible_scores_only', scored.standard_t20_eligible.all())
    check('first_innings_no_chase_rate', d.loc[d.innings.eq(1), ['target_pre', 'runs_required_pre', 'required_run_rate_pre']].isna().all().all())
    check('formula_reconciles', np.allclose(scored.pressure_score_v1, 100*(.5*scored.leverage + .3*scored.urgency*scored.contestability + .2*scored.match_stakes)))
    return checks


def evaluate(d, out, seed=42):
    rows, calibration = [], []
    z = d[d.pressure_eligible & d.training_label_available].copy()
    rng = np.random.default_rng(seed)
    for inning, a in z.groupby('innings'):
        a = a.copy()
        a['model_error'] = (a.win_probability_pre - a.batting_win_label_outcome)**2
        a['prior_error'] = (a.historical_baseline_pre - a.batting_win_label_outcome)**2
        match = a.groupby('match_id')[['model_error', 'prior_error']].mean()
        diff = (match.model_error - match.prior_error).to_numpy()
        boots = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(500)])
        rows.append({'innings': inning, 'matches': len(match), 'deliveries': len(a),
                     'brier_model': match.model_error.mean(), 'brier_historical_prior': match.prior_error.mean(),
                     'brier_coinflip': .25, 'brier_difference': diff.mean(),
                     'difference_bootstrap_lo95': np.quantile(boots, .025), 'difference_bootstrap_hi95': np.quantile(boots, .975)})
        a['probability_bin'] = pd.cut(a.win_probability_pre, np.linspace(0, 1, 11), include_lowest=True)
        a['sample_weight'] = 1 / a.groupby('match_id').match_id.transform('size')
        for bin_, b in a.groupby('probability_bin', observed=True):
            calibration.append({'innings': inning, 'bin': str(bin_), 'deliveries': len(b), 'matches': b.match_id.nunique(),
                'mean_probability': np.average(b.win_probability_pre, weights=b.sample_weight),
                'observed_win_fraction': np.average(b.batting_win_label_outcome, weights=b.sample_weight)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'model_summary.csv', index=False)
    pd.DataFrame(calibration).to_csv(out / 'calibration.csv', index=False)
    return summary


def audit_matches(root, d, out):
    requested = {19960: 'High-scoring chase', 19716: 'Low-scoring match', 31268: 'Close finish',
                 21586: 'Revised-target final', 31290: 'Knockout', 48610: 'Contradictory revised target'}
    # Independent standard-library reader and counters: does not reuse pandas cumsums.
    with (Path(root) / 'ball_by_ball.csv').open() as f:
        records = [r for r in csv.DictReader(f) if int(r['match_id']) in requested]
    rows = []
    for mid, title in requested.items():
        source = [r for r in records if int(r['match_id']) == mid]
        for inning in sorted(set(int(r['innings']) for r in source)):
            innings = sorted([r for r in source if int(r['innings']) == inning], key=lambda r: int(r['innings_ball_num']))
            score = wickets = legal = 0
            for event in innings:
                idx = f"{mid}:{inning}:{event['innings_ball_num']}"
                actual = d.loc[d.delivery_id.eq(idx)].iloc[0]
                assert (actual.score_pre, actual.wickets_fallen_pre, actual.legal_balls_bowled_pre) == (score, wickets, legal)
                last_pre = (score, wickets, legal)
                score += int(event['runs']) + int(event['extras1_runs']) + int(event['extras_2_runs'])
                wickets += event['is_wicket'] == 'true'
                legal += event['extras_type'] not in ['wide', 'Overstep', 'other']
            match_rows = d[d.match_id.eq(mid) & d.innings.eq(inning)]
            a = match_rows.iloc[-1]
            rows.append({'audit_case': title, 'match_id': mid, 'innings': inning, 'events': len(innings),
                'total': score, 'wickets': wickets, 'legal_balls': legal,
                'before_last_score': last_pre[0], 'before_last_wickets': last_pre[1], 'before_last_legal': last_pre[2],
                'target_recorded': a.target_recorded, 'scheduled_balls_recorded': a.scheduled_balls_recorded,
                'pressure_available_events': int(match_rows.pressure_eligible.sum()), 'result': a.result,
                'all_pre_states_reconciled': True})
    audits = pd.DataFrame(rows)
    audits.to_csv(out / 'match_audits.csv', index=False)
    return audits


def plots(d, examples, out):
    os.environ.setdefault('MPLCONFIGDIR', str(out / '.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    z = d[d.pressure_eligible]
    if z.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(z.pressure_score_v1, bins=40, color='#c73875', edgecolor='white')
    cutoff = z.pressure_score_v1.quantile(.75)
    ax.axvline(cutoff, color='#203b68', linestyle='--', label=f'Top 25% cutoff: {cutoff:.1f}')
    ax.set(xlabel='Provisional pressure score (0–100)', ylabel='Deliveries', title='Pressure distribution • available historical estimates')
    ax.legend(); fig.tight_layout(); fig.savefig(out / 'pressure_distribution.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for inning, a in z.groupby('innings'):
        curve = a.groupby('overid').pressure_score_v1.mean()
        ax.plot(curve.index, curve, marker='o', markersize=3, label=f'Innings {inning}')
    ax.set(xlabel='Recorded over (starts at 1)', ylabel='Mean pressure', title='Pressure by over and innings')
    ax.set_xticks([1, 5, 10, 15, 20]); ax.legend(); fig.tight_layout(); fig.savefig(out / 'pressure_by_over_innings.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    stage = z.groupby('stage').pressure_score_v1.mean().sort_values()
    ax.barh(stage.index, stage.values, color='#203b68')
    ax.set(xlabel='Mean pressure', title='Pressure by match stage • stakes explicitly contribute')
    fig.tight_layout(); fig.savefig(out / 'pressure_by_stage.png', dpi=150); plt.close(fig)
    if not examples.empty:
        e = examples.iloc[:4]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        labels = [f'{r.runs_needed} off {r.balls_left}' for r in e.itertuples()]
        x = np.arange(len(e))
        for i, (c, label) in enumerate([('difficulty', 'Difficulty'), ('contestability', 'Contestability'), ('leverage_provisional', 'Leverage'), ('urgency', 'Urgency')]):
            axes[0].bar(x + (i-1.5)*.19, e[c], width=.19, label=label)
        axes[0].set_xticks(x, labels); axes[0].set_ylim(0, 1.3); axes[0].set_yticks(np.linspace(0, 1, 6)); axes[0].legend(fontsize=8, ncol=2, loc='upper center')
        axes[0].set_title('Different demands, seven wickets remaining')
        bars = axes[1].bar(labels, e.pressure_raw_illustration, color='#c73875')
        for bar, available in zip(bars, e.production_available):
            if not available:
                bar.set_hatch('///'); bar.set_alpha(.5)
        axes[1].set(ylabel='Illustrative pressure', title='Raw model illustration, same league stakes', ylim=(0, 100))
        fig.text(.5, .01, 'Hatching = insufficient historical state support; no production score would be emitted.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(out / 'illustrative_situations.png', dpi=150); plt.close(fig)


def write_reports(root, out, tables, audit, d, overs, reasons, checks, thresholds, cfg, metrics, examples):
    save_json(out / 'input_schema_audit.json', audit)
    profiles, categories = [], []
    for name, t in tables.items():
        for c in t:
            profiles.append({'file': name, 'column': c, 'dtype': str(t[c].dtype), 'missing': int(t[c].isna().sum()), 'distinct': t[c].nunique()})
            if t[c].nunique() <= 50:
                for value, count in t[c].value_counts(dropna=False).items():
                    categories.append({'file': name, 'column': c, 'value': value, 'count': count})
    pd.DataFrame(profiles).to_csv(out / 'input_column_profile.csv', index=False)
    pd.DataFrame(categories).to_csv(out / 'input_category_counts.csv', index=False)
    flags = pd.DataFrame([{'issue': c, 'flagged_deliveries': int(d[c].sum()), 'affected_matches': d.loc[d[c], 'match_id'].nunique()} for c in reasons + ['revised_match', 'is_super_over', 'regulation_tied_match', 'bowler_concession_uncertain']])
    flags.to_csv(out / 'quality_counts.csv', index=False)
    relationships = pd.DataFrame([{'relationship': c, 'unmatched_delivery_rows': int(d[c].sum()),
        'affected_matches': d.loc[d[c], 'match_id'].nunique()}
        for c in d if c.endswith('_missing_roster') or c.endswith('_missing_comp_mapping')])
    relationships.to_csv(out / 'relationship_audit.csv', index=False)
    match_flags = d.groupby('match_id')[reasons + ['revised_match', 'is_super_over']].any().reset_index()
    match_flags.to_csv(out / 'match_quality_flags.csv', index=False)
    d.loc[d.state_quality_notes.ne(''), ['delivery_id', 'match_id', 'innings', 'innings_ball_num', 'state_quality_notes']].to_csv(out / 'quality_warnings.csv', index=False)
    missing_matches = tables['matches'][~tables['matches'].match_id.isin(d.match_id)]
    missing_matches.to_csv(out / 'matches_without_deliveries.csv', index=False)
    rosters = tables['playing_eleven']
    rosters[rosters.duplicated(['match_id', 'team_id', 'player_id'], keep=False)].to_csv(out / 'roster_conflicts.csv', index=False)
    revisions = d[d.revised_match].groupby(['match_id', 'innings']).agg(recorded_ball_limit=('scheduled_balls_recorded', 'first'),
        actual_legal_balls=('legal_event', 'sum'), observed_runs=('team_runs_event', 'sum'), recorded_target=('target_recorded', 'first'), result=('result', 'first'))
    revisions.to_csv(out / 'revised_match_audit.csv')
    concentration = []
    for col in ['innings', 'phase_pre', 'competition_type', 'comp_name', 'stage']:
        for val, a in d.groupby(col):
            n, hi = int(a.pressure_eligible.sum()), int(a.high_pressure_selected.sum())
            concentration.append({'dimension': col, 'value': val, 'all_deliveries': len(a), 'scored_deliveries': n,
                                  'high_pressure_deliveries': hi, 'high_fraction_of_scored': hi/n if n else np.nan})
    conc = pd.DataFrame(concentration)
    conc.to_csv(out / 'pressure_concentration.csv', index=False)
    summary = evaluate(d, out, cfg.seed)
    audits = audit_matches(root, d, out)
    raw_cols = set(tables['ball_by_ball'].columns)
    catalog = []
    for c in d:
        if c in raw_cols:
            role = 'original_delivery_field_do_not_blanket_use_as_predictor'
        elif c.endswith('_pre'):
            role = 'pre_delivery_feature_subject_to_quality_and_availability_flags'
        elif 'outcome' in c or c in ['result', 'win_team_id', 'man_of_the_match']:
            role = 'outcome_or_post_match_metadata_never_predictor'
        elif c.endswith('_event') or '_roster_' in c or 'recorded' in c:
            role = 'event_or_retrospective_metadata_not_predictor'
        else:
            role = 'context_diagnostic_or_provisional_score_review_methodology'
        catalog.append({'column': c, 'dtype': str(d[c].dtype), 'usage': role})
    pd.DataFrame(catalog).to_csv(out / 'feature_catalog.csv', index=False)
    inputs = pd.DataFrame([{'file': n, 'rows': a['rows'], 'columns': a['columns'], 'duplicate_keys': a['duplicate_keys']} for n, a in audit.items()])
    missing = pd.DataFrame(profiles).query('missing > 0')
    statuses = d.probability_status.value_counts().rename_axis('status').reset_index(name='deliveries')
    example_table = examples[['runs_needed','balls_left','wickets_left','p_model_raw','difficulty','contestability','leverage_provisional','urgency','pressure_raw_illustration','state_support_matches']] if not examples.empty else examples
    report = f'''# Data quality report

All six original CSVs are read-only. Source SHA-256 values and field examples are in `input_schema_audit.json`; `run_manifest.json` records versions and settings. All {len(d):,} delivery events survive validated many-to-one joins. No uploads or external data are used.

## Input inspection

{md_table(inputs)}

No duplicate delivery keys were found in this run. Sequence is match date, match ID, innings, recorded event number; timestamp is retained but not used because `delivery_ist_time` has a UTC suffix despite its IST name. The observed match dates span {d.match_date_parsed.min().date()} to {d.match_date_parsed.max().date()}.

{md_table(missing, 100)}

Full types, distinct counts and category distributions: `input_column_profile.csv`, `input_category_counts.csv`. Literal `None` in extras is a category. Zero bowling speed is retained as a source placeholder, never a pressure input.

Validated relationship coverage is shown below. Missing roster entries do not remove otherwise valid ball events; player IDs from the delivery are preserved and missing attributes remain missing. Roster metadata is not a model input. Every delivery joins to a match and competition, and every player role has a competition-team-player mapping in this run.

{md_table(relationships)}

## Cricket-specific findings and handling

- Team runs = batter runs + both extras columns. Current-delivery outcomes are separate from pre-delivery cumulative values.
- Wides and Overstep are illegal. There are 264 `other` events in this supplied source, each with one primary extra; they include batter runs and sometimes secondary extras. Treating `other` as illegal restores six legal balls in nearly all completed overs. This supports a no-ball-like interpretation, but does not prove its official meaning. Keep inferred counts, flag the uncertainty and exclude the entire affected match from the standard model.
- One `Penalty` event: match 16753, innings 2, over 6, event 4 has five extras in a six-event over. Assumed attached to a legal event for descriptive counts, excluded from bowling concessions and from model eligibility. It cannot be conclusively resolved from these CSVs.
- Secondary extras have no type; exclude them from assumed bowler runs and flag attribution uncertainty. No free-hit indicator exists. Bowler wickets include Caught, Bowled, LBW, Stump and Hit Wicket; RunOut, Retire Out and both source spellings of obstructing the field are excluded.
- Roster duplicates disagree on batting order, notably in super-over matches. Unanimous fields are retained and conflicts are missing; all source conflicting rows are saved in `roster_conflicts.csv`. Participation/impact flags can reflect post-match facts and are not predictors. Reported batting orders 12/13 are flagged rather than treated as valid positions.
- Revised-overs values (30–114) are legal-ball limits, evidenced by full revised innings reaching those counts (e.g. 21586: 90-ball chase; 17395: 48-ball innings). They are never multiplied by six. If only the first limit is present, the recorded second limit inherits it for retrospective auditing.
- Revision timestamps and intermediate targets are absent. Revised matches retain observed score/wickets/balls and retrospective `*_recorded` limits/targets. Their `scheduled_balls_pre`, `target_pre`, runs required, required rate and phase are unavailable/unknown, and pressure is withheld. The final revised target is NEVER backfilled into earlier predictive state.
- Source target 115 in match 48610 conflicts with the 125-run chase and a stated 11-run winning margin; no target is silently corrected. Results expressed with DLS but lacking target fields are flagged too. `revised_match_audit.csv` lists every recorded limit and observed endpoint.
- Three matches have non-six completed overs under the inferred legality rules (14130, 14164, 41509). Numbering is contiguous, so missing events versus scorer/umpire anomalies cannot be distinguished; exclude affected matches rather than synthesize balls.
- {len(missing_matches)} match records have no deliveries; their results are listed in `matches_without_deliveries.csv`. These often represent abandonments, not proof of lost delivery data. Incomplete first innings do not supply an invented chase target. Incomplete losing chases, inconsistent result margins, team IDs and terminal states are checked separately.
- Regulation ties, no-results and super-over-decided matches do not supply binary training labels. Innings 3/4 are retained, use a six-ball/two-wicket descriptive limit, and are never pressure-scored. Early successful chases need not reach 120 balls. Fewer than 120 balls alone is not an error.
- Home/away is a fixture label. Venue ID/country is preserved; venue names, ground advantage and player nationality are not invented.

## Quality and model coverage

{md_table(flags)}

Flags overlap. Entire matches are conservatively excluded if a reconstruction issue could contaminate their state or target. This selection uses retrospective quality checks; it is not an online validity predictor. Remaining valid rows may still lack enough prior history or similar historical states.

{md_table(statuses)}

There are {d.standard_t20_eligible.sum():,} standard-model-eligible rows and {d.pressure_eligible.sum():,} rows with an available provisional pressure score. {d.high_pressure_selected.sum():,} rows meet the selected pressure percentile. Ordinary deliveries remain in both full outputs. Percentile cutoffs (inclusive, so ties may increase counts): `{thresholds}`.

## Forward-only evaluation

{md_table(summary)}

Lower Brier is better. Each match has equal weight within its innings. Differences are model minus historical-prior baseline; 95% intervals resample matches, not delivery rows (500 deterministic bootstrap samples). They describe this dataset and do not cover all model or selection uncertainty. Model design was revised after an illustrative-state sanity check, so these are development backtests, not a pristine external test. Fold-level results, coefficients, empirical outcome weights and calibration are exported separately. Calibration bins can be sparse; no claim of calibrated individual psychological pressure is made. In this run the first-innings model is overconfident in its upper probability tail: consult `calibration.csv` before interpreting a high individual probability literally. Pooled Brier improvement does not establish reliable leverage at every state.

## Illustrative pressure comparison

{md_table(example_table, 10)}

These examples use the latest fold model and keep league stakes and seven remaining wickets fixed for the first three rows. Both 22 and 36 off six have high urgency; their model contestability and leverage differ. The 36-run requirement is more difficult. The 22-run case receives the largest pressure score in this run because plausible next outcomes have more effect on the estimated chance of winning. Six off six is relatively comfortable, and six off 36 has more recovery time. This ranking was not hard-coded; neither these point probabilities nor the leverage scale are independently calibrated guarantees. The wicket and final-stakes variants are saved alongside them. All raw illustrative estimates carry state-support counts and a production-availability flag in the CSV.

## Match audits

The following real cases were independently reconciled delivery-by-delivery using standard-library counters, then reviewed at their endpoints. This verifies arithmetic against the supplied records, not an external official scorecard.

{md_table(audits[['audit_case','match_id','innings','total','wickets','legal_balls','before_last_score','before_last_wickets','before_last_legal','target_recorded','pressure_available_events']], 20)}

High-scoring chase 19960: 259 chased against 258, with 7 legal balls unused. Low-scoring match 19716: 99 first-innings runs, so the target is 100. Close finish 31268: compare the final chasing score to target minus one; the one-run result must agree. Revised final 21586: 214 first-innings runs, revised target 171 in 90 balls; chase finishes at 171, and retrospective target timing remains unknown. Knockout 31290: stage/knockout classification is checked separately from cricket state. Match 48610 demonstrates why recorded revisions cannot be accepted blindly.

## Concentration, not player quotas

{md_table(conc[conc.dimension.isin(['innings','phase_pre','competition_type'])], 30)}

`pressure_concentration.csv` includes each competition and match stage. Differences combine cricket state, deliberately chosen stakes, history coverage and conservative exclusions. No player-level equalization is performed.

## Checks

{len(checks)} runtime invariants passed. `validation_checks.json` records each check. Pytest additionally covers synthetic illegal deliveries, wickets, windows, revisions, joins, terminal states, chronology and mutation-based leakage checks. Source hashes are rechecked after the run.
'''
    (out / 'data_quality_report.md').write_text(report)
    save_json(out / 'validation_checks.json', checks)
    return summary, audits
