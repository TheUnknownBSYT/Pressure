from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pressure_pipeline.data import load_sources, enrich
from pressure_pipeline.state import clean_events, reconstruct, quality_flags
from pressure_pipeline.model import (ModelConfig, design, fit_historical, next_state,
                                     pressure_flags, score_components, urgency)


def events(first=None, second=None):
    if first is None:
        first = [{} for _ in range(120)]
    if second is None:
        second = [{} for _ in range(119)] + [dict(runs=0)]
    rows = []
    for inning, sequence in [(1, first), (2, second)]:
        legal = 0
        over_events = 0
        prior_over = 0
        for n, changes in enumerate(sequence, 1):
            over = legal // 6 + 1
            over_events = over_events + 1 if over == prior_over else 1
            prior_over = over
            row = dict(match_id=1, innings=inning, overid=over, over_ball_num=over_events,
                innings_ball_num=n, source_row=len(rows), delivery_id=f'1:{inning}:{n}',
                runs=1, extras1_runs=0, extras_2_runs=0, extras_type='None',
                is_four=False, is_six=False, is_wicket=False, dismissal_type='',
                dismissed_player_id=np.nan, striker_id=10*inning+1, non_striker_id=10*inning+2,
                bowler_id=10*(3-inning)+3, batting_team_id=inning, bowling_team_id=3-inning,
                first_inn_revised_overs=np.nan, second_inn_revised_overs=np.nan, revised_target=np.nan,
                result='Home Won by 1 run', home_team_id=1, away_team_id=2, win_team_id=1,
                comp_name='IPL 2024', match_format='T20', stage='League', knock_out=0,
                batting_order=1, batting_type='RHB', pace_or_spin='Pace',
                match_date='2024-01-01', match_date_parsed=pd.Timestamp('2024-01-01'),
                missing_match_metadata=False)
            row.update(changes)
            legal += row['extras_type'] not in ['wide', 'Overstep', 'other']
            rows.append(row)
    return clean_events(pd.DataFrame(rows))


def state(b):
    d, inn = reconstruct(b)
    return quality_flags(d, inn)[0]


def test_cumulative_state_and_chase_target():
    d = state(events())
    first = d[d.innings.eq(1)]
    assert first.score_pre.tolist() == list(range(120))
    assert first.legal_balls_remaining_pre.tolist() == list(range(120, 0, -1))
    chase = d[d.innings.eq(2)]
    assert chase.target_pre.eq(121).all()
    assert chase.iloc[0].runs_required_pre == 121
    assert chase.iloc[-1].runs_required_pre == 2
    assert first.required_run_rate_pre.isna().all()
    assert d.standard_t20_eligible.all()


def test_wides_overstep_other_and_secondary_extras():
    seq = [dict(runs=0, extras_type='wide', extras1_runs=5),
           dict(runs=4, extras_type='Overstep', extras1_runs=1),
           dict(runs=0, extras_type='other', extras1_runs=1, extras_2_runs=4), {}]
    d = state(events(seq, []))
    assert d.score_pre.tolist() == [0, 5, 10, 15]
    assert d.legal_balls_bowled_pre.tolist() == [0, 0, 0, 0]
    assert d.batter_runs_pre.tolist() == [0, 0, 4, 4]
    assert d.bowler_runs_conceded_assumed_pre.tolist() == [0, 5, 10, 11]
    assert d.iloc[2].extras_semantics_uncertain
    assert not d.standard_t20_eligible.any()


def test_penalty_and_byes_attribution():
    b = events([dict(runs=0, extras_type='Penalty', extras1_runs=5),
                dict(runs=0, extras_type='byes', extras1_runs=4), {}], [])
    d = state(b)
    assert d.iloc[2].score_pre == 9
    assert d.iloc[2].bowler_runs_conceded_assumed_pre == 0
    assert d.iloc[2].bowler_concession_uncertain_pre


@pytest.mark.parametrize('dismissal,credited', [('RunOut', 0), ('Retire Out', 0), ('Obstracting the field', 0),
    ('Obstructing the field', 0), ('Caught', 1), ('Bowled', 1), ('LBW', 1), ('Stump', 1), ('Hit Wicket', 1)])
def test_wickets(dismissal, credited):
    d = state(events([dict(runs=0, is_wicket=True, dismissal_type=dismissal, dismissed_player_id=12), {}], []))
    assert d.iloc[1].wickets_fallen_pre == 1
    assert d.iloc[1].wickets_remaining_pre == 9
    assert d.iloc[1].bowler_wickets_pre == credited


def test_recent_legal_window_includes_intervening_extras_and_excludes_current():
    seq = [{} for _ in range(7)]
    seq.insert(6, dict(runs=0, extras_type='wide', extras1_runs=2))
    seq.append(dict(runs=6, is_six=True))
    d = state(events(seq, []))
    assert d.iloc[7].recent_runs_6_pre == 8  # six legal singles + wide before seventh legal
    assert d.iloc[8].recent_runs_6_pre == 8  # previous six legal singles + wide
    assert d.iloc[8].recent_runs_12_pre == 9
    assert d.iloc[8].score_pre == 9


def test_player_counters_previous_over_and_team_aggregates():
    seq = [dict(runs=4, is_four=True)] + [{} for _ in range(5)] + [dict(striker_id=12), {}]
    d = state(events(seq, []))
    assert d.iloc[6].previous_over_runs_pre == 9
    assert d.iloc[6].previous_over_legal_balls_pre == 6
    assert d.iloc[6].batter_runs_pre == 0
    assert d.iloc[6].teammates_batter_runs_pre == 9
    assert d.iloc[1].bowler_after_boundary_pre
    assert d.iloc[1].consecutive_team_boundaries_pre == 1
    assert d.iloc[2].consecutive_team_boundaries_pre == 0


def test_revision_records_not_backfilled_into_predictors():
    b = events([{} for _ in range(48)], [{} for _ in range(44)])
    b['first_inn_revised_overs'] = 48
    b['second_inn_revised_overs'] = 48
    b['revised_target'] = 91
    d = state(b)
    assert d.scheduled_balls_recorded.eq(48).all()
    assert d.loc[d.innings.eq(2), 'target_recorded'].eq(91).all()
    assert d[['target_pre', 'scheduled_balls_pre', 'legal_balls_remaining_pre']].isna().all().all()
    assert not d.standard_t20_eligible.any()


def test_short_chase_is_valid_when_target_reached():
    first = [dict(runs=0) for _ in range(120)]
    d = state(events(first, [dict(runs=1)]).assign(win_team_id=2, result='Away Won by 10 wickets'))
    assert d.standard_t20_eligible.all()
    assert not d.chase_incomplete.any()


def test_incomplete_first_innings_does_not_invent_target():
    d = state(events([{} for _ in range(10)], [{}]))
    assert d.loc[d.innings.eq(2), 'target_pre'].isna().all()
    assert d.first_innings_incomplete_standard.all()


def test_super_overs_retained_and_excluded():
    b = events()
    extra = b.iloc[:6].copy()
    extra['innings'] = 3
    extra['delivery_id'] = ['1:3:'+str(i) for i in range(1, 7)]
    extra['source_row'] = range(len(b), len(b)+6)
    extra['is_super_over'] = True
    d = state(pd.concat([b, extra], ignore_index=True))
    z = d[d.innings.eq(3)]
    assert len(z) == 6
    assert z.scheduled_balls_recorded.eq(6).all()
    assert z.wicket_limit.eq(2).all()
    assert not z.standard_t20_eligible.any()
    assert not d.training_label_available.any()


def test_missing_event_and_impossible_state_flagged():
    d = state(events().drop(index=5).reset_index(drop=True))
    assert d.event_sequence_gap.any()
    assert not d.standard_t20_eligible.any()
    d = state(events([{} for _ in range(121)], [{}]))
    assert d.impossible_state.any()
    assert d.legal_balls_remaining_pre.min() == 0


def test_result_margin_contradiction_and_missing_outcome():
    d = state(events().assign(result='Home Won by 7 runs'))
    assert d.result_margin_inconsistent.all()
    assert not d.standard_t20_eligible.any()
    d = state(events().assign(result='No Result', win_team_id=np.nan))
    assert not d.training_label_available.any()
    assert d.batting_win_label_outcome.isna().all()


def test_previous_required_rate_includes_illegal_event_without_consuming_ball():
    first = [{} for _ in range(120)]
    second = [dict(runs=0, extras_type='wide', extras1_runs=1), {}]
    d = state(events(first, second))
    chase = d[d.innings.eq(2)]
    assert chase.legal_balls_remaining_pre.tolist() == [120,120]
    assert chase.runs_required_pre.tolist() == [121,120]
    assert chase.iloc[1].previous_required_run_rate_pre == pytest.approx(6.05)
    assert chase.iloc[1].required_run_rate_change_pre == pytest.approx(-.05)


def test_collapse_excludes_current_wicket():
    seq = [dict(runs=0, is_wicket=True, dismissal_type='Caught') for _ in range(3)] + [{}]
    d = state(events(seq, []))
    assert d.collapse_pre.tolist() == [False,False,False,True]


def test_current_and_future_event_mutation_leaves_pre_features_unchanged():
    b = events()
    baseline = state(b)
    altered = b.copy()
    altered.loc[50:, 'runs'] = 6
    altered.loc[50:, 'is_six'] = True
    altered.loc[50:, 'team_runs_event'] = 6
    altered.loc[50:, 'boundary_event'] = 1
    altered.loc[50:, 'bowler_runs_event_assumed'] = 6
    changed = state(altered)
    cols = [c for c in baseline if c.endswith('_pre')]
    pd.testing.assert_frame_equal(baseline.loc[:50, cols], changed.loc[:50, cols])


def test_predictor_allowlist_ignores_outcomes_and_first_innings_target():
    d = state(events()).iloc[:20]
    curve = np.arange(121) * 1.4
    x = design(d, 1, curve, 1.4)
    changed = d.assign(runs=6, is_wicket=True, win_team_id=999, result='future', target_pre=999,
                       required_run_rate_pre=100, man_of_the_match=13)
    pd.testing.assert_frame_equal(x, design(changed, 1, curve, 1.4))


def test_score_equation_threshold_ties_and_nulls():
    p = np.array([0., .5, 1.])
    assert np.allclose(score_components(p, np.ones(3), np.ones(3), np.ones(3)), [70,100,70])
    d, thresholds = pressure_flags(pd.DataFrame({'pressure_score_v1': [0., 10., 20., 30., np.nan]}))
    assert d.high_pressure_selected.tolist() == [False, False, False, True, False]
    assert thresholds['25'] == 22.5
    with pytest.raises(ValueError):
        pressure_flags(d, 100)


def test_counterfactual_transitions_and_urgency_bounds():
    d = state(events()).iloc[[230]].copy()
    z = next_state(d, 6, 0, 2)
    assert z.iloc[0].score_pre == d.iloc[0].score_pre + 6
    assert z.iloc[0].legal_balls_remaining_pre == d.iloc[0].legal_balls_remaining_pre - 1
    wicket = next_state(d, 0, 1, 2)
    assert wicket.iloc[0].wickets_remaining_pre == d.iloc[0].wickets_remaining_pre - 1
    for inning in [1, 2]:
        u = urgency(d, inning, np.arange(121)*1.5, 12)
        assert np.all((u >= 0) & (u <= 1))


def test_chronological_models_and_future_label_mutation():
    base = state(events())
    parts = []
    for i in range(12):
        p = base.iloc[::10].copy()
        p['match_id'] = i + 1
        p['match_date_parsed'] = pd.Timestamp('2020-01-01') + pd.DateOffset(months=i)
        p['training_label_available'] = True
        p['standard_t20_eligible'] = True
        p['batting_win_label_outcome'] = np.where(p.innings.eq(1), i % 2, 1-i % 2)
        parts.append(p)
    data = pd.concat(parts, ignore_index=True)
    cfg = ModelConfig(min_train_matches=4, min_class_matches=2, min_support_matches=1, fold_months=2)
    a, folds, *_ = fit_historical(data, cfg)
    changed = data.copy()
    changed.loc[changed.match_date_parsed.ge('2020-09-01'), 'batting_win_label_outcome'] = 1-changed.loc[changed.match_date_parsed.ge('2020-09-01'), 'batting_win_label_outcome']
    b, *_ = fit_historical(changed, cfg)
    early = a.match_date_parsed.lt('2020-09-01')
    pd.testing.assert_series_equal(a.loc[early, 'win_probability_pre'], b.loc[early, 'win_probability_pre'])
    scored = a[a.pressure_eligible]
    assert len(scored) > 0
    assert (scored.model_train_end_date < scored.match_date_parsed).all()
    assert a[a.match_date_parsed.lt('2020-05-01')].win_probability_pre.isna().all()


@pytest.mark.skipif(not Path('ball_by_ball.csv').exists(), reason='proprietary source not installed')
def test_actual_source_join_duplicates_do_not_multiply_rows():
    tables, audit = load_sources('.')
    d = enrich(tables)
    assert len(d) == len(tables['ball_by_ball']) == 200999
    assert d.delivery_id.is_unique
    assert audit['playing_eleven']['duplicate_keys'] == 40


@pytest.mark.skipif(not Path('outputs/run_manifest.json').exists(), reason='run pipeline first')
def test_real_outputs_and_saved_match_audits():
    import json
    from pressure_pipeline.report import verify_invariants
    from pressure_pipeline.data import sha256
    d = pd.read_parquet('outputs/pressure_scored.parquet')
    assert all(verify_invariants(d).values())
    assert len(d) == 200999
    high = pd.read_parquet('outputs/high_pressure_deliveries.parquet')
    assert set(high.delivery_id) == set(d.loc[d.high_pressure_selected, 'delivery_id'])
    audits = pd.read_csv('outputs/match_audits.csv')
    assert audits.match_id.nunique() >= 5
    assert audits.all_pre_states_reconciled.all()
    assert json.loads(Path('outputs/validation_checks.json').read_text())
    manifest = json.loads(Path('outputs/run_manifest.json').read_text())
    for name, value in manifest['source_sha256'].items():
        assert sha256(name+'.csv') == value
