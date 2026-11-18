"""Vectorized cumulative state; small per-innings loops only for variable legal windows."""
import numpy as np
import pandas as pd

INN = ['match_id', 'innings']
BOWLER_DISMISSALS = {'Caught', 'Bowled', 'LBW', 'Stump', 'Hit Wicket'}
KNOWN_EXTRAS = {'None', 'wide', 'Overstep', 'other', 'Penalty', 'byes', 'legbyes'}


def pre_sum(d, column, keys=INN):
    return d.groupby(keys, sort=False)[column].cumsum() - d[column]


def clean_events(d):
    d = d.copy()
    d['team_runs_event'] = d.runs + d.extras1_runs + d.extras_2_runs
    d['is_legal_assumed'] = ~d.extras_type.isin(['wide', 'Overstep', 'other'])
    d['legal_event'] = d.is_legal_assumed.astype(int)
    d['extras_semantics_uncertain'] = d.extras_type.isin(['other', 'Penalty']) | ~d.extras_type.isin(KNOWN_EXTRAS)
    d['legal_rule'] = np.select([d.extras_type.eq('other'), d.extras_type.eq('Penalty')],
        ['other_inferred_illegal_unverified', 'penalty_attached_to_legal_event_unverified'], default='documented_category')
    d['wicket_event'] = d.is_wicket.astype(int)
    d['bowler_wicket_event'] = (d.is_wicket & d.dismissal_type.isin(BOWLER_DISMISSALS)).astype(int)
    # Secondary extras have no type; examples with no-balls support byes/leg-byes,
    # but attribution remains explicitly uncertain.
    d['bowler_runs_event_assumed'] = d.runs + d.extras1_runs.where(d.extras_type.isin(['wide', 'Overstep', 'other']), 0)
    d['bowler_concession_uncertain'] = d.extras_semantics_uncertain | d.extras_2_runs.ne(0)
    d['boundary_event'] = (d.is_four | d.is_six).astype(int)
    d['is_super_over'] = d.innings.gt(2)
    d['source_order_changed'] = d.source_row.diff().lt(0)
    return d


def add_recent(d):
    out = {f'recent_{c}_{n}_pre': np.zeros(len(d)) for c in ['runs', 'wickets'] for n in [6, 12]}
    streak = np.zeros(len(d), dtype=int)
    for ix in d.groupby(INN, sort=False).indices.values():
        z = d.iloc[ix]
        legal = z.legal_event.to_numpy()
        completed = np.cumsum(legal) - legal
        legal_positions = np.flatnonzero(legal)
        for n in [6, 12]:
            starts = np.zeros(len(z), dtype=int)
            mask = completed > n
            starts[mask] = legal_positions[completed[mask] - n - 1] + 1
            for c, src in [('runs', 'team_runs_event'), ('wickets', 'wicket_event')]:
                pref = np.r_[0, np.cumsum(z[src].to_numpy())]
                out[f'recent_{c}_{n}_pre'][ix] = pref[np.arange(len(z))] - pref[starts]
        # Team consecutive boundaries on delivery events (illegal events can break streak).
        boundaries = z.boundary_event.to_numpy()
        last_zero = np.maximum.accumulate(np.where(boundaries == 0, np.arange(len(z)), -1))
        after_streak = np.arange(len(z)) - last_zero
        streak[ix] = np.r_[0, after_streak[:-1]]
    for c, v in out.items():
        d[c] = v
    d['consecutive_team_boundaries_pre'] = streak
    d['collapse_pre'] = d.recent_wickets_12_pre.ge(3)
    d['recent_run_rate_6_pre'] = 6 * d.recent_runs_6_pre / d.legal_balls_bowled_pre.clip(upper=6).replace(0, np.nan)
    d['recent_run_rate_12_pre'] = 6 * d.recent_runs_12_pre / d.legal_balls_bowled_pre.clip(upper=12).replace(0, np.nan)
    d['team_acceleration_pre'] = d.recent_run_rate_6_pre - d.recent_run_rate_12_pre
    return d


def reconstruct(d):
    d = d.copy()
    g = d.groupby(INN, sort=False)
    d['score_pre'] = pre_sum(d, 'team_runs_event')
    d['wickets_fallen_pre'] = pre_sum(d, 'wicket_event')
    d['legal_balls_bowled_pre'] = pre_sum(d, 'legal_event')
    d['wicket_limit'] = np.where(d.is_super_over, 2, 10)
    d['wickets_remaining_pre'] = (d.wicket_limit - d.wickets_fallen_pre).clip(lower=0)
    d['revised_match'] = d[['first_inn_revised_overs', 'second_inn_revised_overs', 'revised_target']].notna().any(axis=1) | d.result.str.contains('D/L|DLS', case=False, na=False)
    d['revised_overs_match'] = d[['first_inn_revised_overs', 'second_inn_revised_overs']].notna().any(axis=1)
    d['revised_target_match'] = d.revised_target.notna()
    # These are explicitly retrospective metadata, NOT trusted pre-delivery inputs.
    cap = np.where(d.innings.eq(1), d.first_inn_revised_overs.fillna(120),
                   d.second_inn_revised_overs.fillna(d.first_inn_revised_overs).fillna(120))
    d['scheduled_balls_recorded'] = np.where(d.is_super_over, 6, cap)
    d['balls_remaining_recorded_raw'] = d.scheduled_balls_recorded - d.legal_balls_bowled_pre
    d['balls_remaining_recorded'] = d.balls_remaining_recorded_raw.clip(lower=0)
    d['scheduled_balls_pre'] = d.scheduled_balls_recorded.where(~d.revised_match)
    d['legal_balls_remaining_pre'] = (d.scheduled_balls_pre - d.legal_balls_bowled_pre).clip(lower=0)
    d['current_run_rate_pre'] = 6 * d.score_pre / d.legal_balls_bowled_pre.replace(0, np.nan)
    summaries = g.agg(total_runs=('team_runs_event', 'sum'), wickets=('wicket_event', 'sum'),
                      legal_balls=('legal_event', 'sum'), events=('delivery_id', 'size'),
                      first_event=('innings_ball_num', 'min'), last_event=('innings_ball_num', 'max'),
                      batting_team_id=('batting_team_id', 'first'), bowling_team_id=('bowling_team_id', 'first'))
    # A completed earlier innings is known when the next innings starts, never used in innings 1.
    first_totals = summaries.xs(1, level='innings').total_runs
    super_totals = summaries[summaries.index.get_level_values('innings') == 3].total_runs.droplevel('innings')
    d['target_recorded'] = np.where(d.innings.eq(2), d.match_id.map(first_totals) + 1, np.nan)
    d.loc[d.innings.eq(2) & d.revised_target.notna(), 'target_recorded'] = d.revised_target
    d.loc[d.innings.eq(4), 'target_recorded'] = d.loc[d.innings.eq(4), 'match_id'].map(super_totals) + 1
    first_summary = summaries.xs(1, level='innings')
    first_complete = (first_summary.legal_balls.eq(120) | first_summary.wickets.eq(10)) & first_summary.first_event.eq(1) & first_summary.last_event.eq(first_summary.events)
    d['target_pre'] = d.target_recorded.where(~d.revised_match & (~d.innings.eq(2) | d.match_id.map(first_complete).fillna(False)))
    d['runs_required_pre'] = (d.target_pre - d.score_pre).clip(lower=0)
    d['required_run_rate_pre'] = 6 * d.runs_required_pre / d.legal_balls_remaining_pre.replace(0, np.nan)
    d['previous_required_run_rate_pre'] = d.groupby(INN).required_run_rate_pre.shift()
    d['required_run_rate_change_pre'] = d.required_run_rate_pre - d.previous_required_run_rate_pre
    d['phase_pre'] = np.select([d.legal_balls_bowled_pre.lt(36), d.legal_balls_bowled_pre.lt(90)], ['powerplay', 'middle'], default='death')
    d.loc[d.revised_match, 'phase_pre'] = 'unknown_revised'
    d.loc[d.is_super_over, 'phase_pre'] = 'super_over'
    d['batting_home_away_label'] = np.select([d.batting_team_id.eq(d.home_team_id), d.batting_team_id.eq(d.away_team_id)], ['home_label', 'away_label'], default='unknown')
    d['bowling_home_away_label'] = np.select([d.bowling_team_id.eq(d.home_team_id), d.bowling_team_id.eq(d.away_team_id)], ['home_label', 'away_label'], default='unknown')
    d['verified_home_ground_advantage'] = np.nan
    d['competition_type'] = np.select([d.comp_name.str.contains('IPL', case=False, na=False), d.match_format.eq('T20I')], ['IPL', 'T20I'], default='other_T20')
    tournament = d.comp_name.str.contains('Cup|IPL|League|Tri-Nation|Tri-Series|Quadrangular', case=False, na=False)
    bilateral = d.comp_name.str.contains('Series|Match', case=False, na=False)
    d['tournament_type'] = np.select([tournament, bilateral], ['tournament', 'bilateral'], default='unknown')
    stages = {'Final': 1., 'Semi Final': .9, 'Eliminator': .85, 'Qualifier': .8, 'Super 4': .65, 'League': .4}
    d['match_stakes'] = d.stage.map(stages).fillna(pd.Series(np.where(d.tournament_type.eq('bilateral'), .3, .4), index=d.index))
    d.loc[d.knock_out.eq(1), 'match_stakes'] = d.loc[d.knock_out.eq(1), 'match_stakes'].clip(lower=.8)
    d['stakes_method'] = 'stage_v1_subjective_no_season_or_international_bonus'
    d['batting_position_reported'] = d.batting_order
    d['batting_position_uncertain'] = ~d.batting_order.between(1, 11)
    d['pace_spin_matchup'] = d.batting_type + '_vs_' + d.pace_or_spin
    batter = INN + ['striker_id']
    bowler = ['match_id', 'bowler_id']
    for name, src in [('runs', 'runs'), ('legal_balls', 'legal_event'), ('boundaries', 'boundary_event')]:
        d[f'batter_{name}_pre'] = pre_sum(d, src, batter)
    d['batter_strike_rate_pre'] = 100 * d.batter_runs_pre / d.batter_legal_balls_pre.replace(0, np.nan)
    for name, src in [('legal_balls', 'legal_event'), ('runs_conceded_assumed', 'bowler_runs_event_assumed'), ('wickets', 'bowler_wicket_event')]:
        d[f'bowler_{name}_pre'] = pre_sum(d, src, bowler)
    d['bowler_economy_assumed_pre'] = 6 * d.bowler_runs_conceded_assumed_pre / d.bowler_legal_balls_pre.replace(0, np.nan)
    d['bowler_concession_uncertain_pre'] = pre_sum(d, 'bowler_concession_uncertain', bowler).gt(0)
    d['teammates_batter_runs_pre'] = pre_sum(d, 'runs') - d.batter_runs_pre
    d['teammates_batter_legal_balls_pre'] = d.legal_balls_bowled_pre - d.batter_legal_balls_pre
    for name, src, own in [('runs_conceded_assumed', 'bowler_runs_event_assumed', 'bowler_runs_conceded_assumed_pre'), ('wickets', 'bowler_wicket_event', 'bowler_wickets_pre'), ('legal_balls', 'legal_event', 'bowler_legal_balls_pre')]:
        d[f'teammates_bowler_{name}_pre'] = pre_sum(d, src, ['match_id', 'bowling_team_id']) - d[own]
    for src in ['team_runs_event', 'wicket_event', 'boundary_event', 'legal_event', 'striker_id', 'bowler_id']:
        d[f'previous_{src}_pre'] = d.groupby(INN)[src].shift()
    d['bowler_previous_conceded_pre'] = d.groupby(bowler).bowler_runs_event_assumed.shift()
    d['bowler_after_boundary_pre'] = d.groupby(bowler).boundary_event.shift().fillna(0).astype(bool)
    d['batter_previous_boundary_pre'] = d.groupby(batter).boundary_event.shift().fillna(0).astype(bool)
    for n in [6, 12]:
        d[f'batter_recent_runs_{n}_events_pre'] = d.groupby(batter).runs.transform(lambda x: x.shift().rolling(n, min_periods=1).sum())
        d[f'batter_recent_legal_balls_{n}_events_pre'] = d.groupby(batter).legal_event.transform(lambda x: x.shift().rolling(n, min_periods=1).sum())
        d[f'batter_recent_sr_{n}_events_pre'] = 100 * d[f'batter_recent_runs_{n}_events_pre'] / d[f'batter_recent_legal_balls_{n}_events_pre'].replace(0, np.nan)
    d['batter_acceleration_pre'] = d.batter_recent_sr_6_events_pre - d.batter_recent_sr_12_events_pre
    overs = d.groupby(INN + ['overid']).agg(previous_over_runs_pre=('team_runs_event', 'sum'), previous_over_wickets_pre=('wicket_event', 'sum'), previous_over_legal_balls_pre=('legal_event', 'sum')).reset_index()
    overs['overid'] += 1
    d = d.merge(overs, on=INN + ['overid'], how='left', validate='many_to_one')
    d = add_recent(d)
    return d, summaries


def quality_flags(d, summaries):
    d = d.copy()
    continuity = summaries.first_event.ne(1) | summaries.events.ne(summaries.last_event)
    d['event_sequence_gap'] = pd.MultiIndex.from_frame(d[INN]).isin(summaries.index[continuity])
    overs = d.groupby(INN + ['overid']).agg(legal=('legal_event', 'sum'), n=('delivery_id', 'size'), lo=('over_ball_num', 'min'), hi=('over_ball_num', 'max'))
    last = overs.reset_index().groupby(INN).overid.transform('max').to_numpy()
    bad_over = (overs.legal.gt(6) | (overs.legal.ne(6) & (overs.index.get_level_values('overid') != last)) | overs.lo.ne(1) | overs.n.ne(overs.hi))
    bad_inns = overs.index[bad_over].droplevel('overid').unique()
    d['over_sequence_inconsistent'] = pd.MultiIndex.from_frame(d[INN]).isin(bad_inns)
    d['over_counter_disagrees'] = (d.overid - 1).ne(d.legal_balls_bowled_pre // 6)
    # At the boundary, an extra can occur after the sixth legal ball only if it is an erroneous event.
    d['impossible_state'] = d.balls_remaining_recorded_raw.lt(0) | d.wickets_fallen_pre.ge(d.wicket_limit) | d.score_pre.lt(0) | (d.target_pre.notna() & d.runs_required_pre.eq(0)) | (d.legal_balls_remaining_pre.eq(0))
    d['unknown_dismissal'] = d.is_wicket & ~d.dismissal_type.isin(BOWLER_DISMISSALS | {'RunOut', 'Retire Out', 'Obstracting the field', 'Obstructing the field'})
    d['team_identity_inconsistent'] = d.batting_team_id.eq(d.bowling_team_id) | ~(d.batting_team_id.eq(d.home_team_id) | d.batting_team_id.eq(d.away_team_id)) | ~(d.bowling_team_id.eq(d.home_team_id) | d.bowling_team_id.eq(d.away_team_id))
    for col in ['batting_team_id', 'bowling_team_id']:
        d['team_identity_inconsistent'] |= d.groupby(INN)[col].transform('nunique').ne(1)
    # A first-innings total is usable as a chase target only with a complete record.
    first = summaries.xs(1, level='innings')
    complete = first.legal_balls.eq(120) | first.wickets.eq(10)
    d['first_innings_incomplete_standard'] = ~d.match_id.map(complete).fillna(False)
    d['regulation_tied_match'] = False
    d['result_inconsistent'] = False
    d['result_margin_inconsistent'] = False
    d['chase_incomplete'] = False
    second = summaries[summaries.index.get_level_values('innings') == 2].droplevel('innings')
    meta = d.drop_duplicates('match_id').set_index('match_id')
    for mid, row in second.iterrows():
        if mid not in first.index:
            continue
        target = meta.loc[mid, 'revised_target']
        if pd.isna(target):
            target = first.loc[mid, 'total_runs'] + 1
        tied = row.total_runs == target - 1
        chase_won = row.total_runs >= target
        expected_winner = row.batting_team_id if chase_won else row.bowling_team_id
        winner = meta.loc[mid, 'win_team_id']
        idx = d.match_id.eq(mid)
        d.loc[idx, 'regulation_tied_match'] = tied
        d.loc[idx, 'result_inconsistent'] = pd.notna(winner) and not tied and winner != expected_winner
        import re
        margin = re.search(r'by\s*(\d+)\s*(runs?|wickets?|wkts?)', str(meta.loc[mid, 'result']), flags=re.I)
        if margin and pd.notna(winner) and not tied:
            stated, unit = int(margin.group(1)), margin.group(2).lower()
            computed = (target - 1 - row.total_runs) if unit.startswith('run') else (10 - row.wickets)
            correct_side = not chase_won if unit.startswith('run') else chase_won
            d.loc[idx, 'result_margin_inconsistent'] = (not correct_side) or computed != stated
        cap = meta.loc[mid, 'second_inn_revised_overs']
        if pd.isna(cap):
            cap = meta.loc[mid, 'first_inn_revised_overs']
        if pd.isna(cap):
            cap = 120
        d.loc[idx, 'chase_incomplete'] = not chase_won and row.wickets < 10 and row.legal_balls < cap and pd.notna(winner)
    d['missing_second_innings'] = ~d.match_id.isin(second.index)
    reasons = ['missing_match_metadata', 'event_sequence_gap', 'over_sequence_inconsistent',
               'over_counter_disagrees', 'impossible_state', 'unknown_dismissal', 'team_identity_inconsistent',
               'extras_semantics_uncertain', 'result_inconsistent', 'result_margin_inconsistent', 'chase_incomplete',
               'first_innings_incomplete_standard', 'missing_second_innings']
    d['reconstruction_match_valid'] = ~d[reasons].any(axis=1).groupby(d.match_id).transform('any')
    d['standard_t20_eligible'] = d.reconstruction_match_valid & ~d.revised_match & ~d.is_super_over & d.match_format.isin(['T20', 'T20I']) & d.match_date_parsed.notna()
    d['training_label_available'] = d.standard_t20_eligible & d.win_team_id.notna() & ~d.regulation_tied_match & ~d.match_id.isin(d.loc[d.is_super_over, 'match_id'])
    d['batting_win_label_outcome'] = d.batting_team_id.eq(d.win_team_id).astype(float).where(d.training_label_available)
    d['state_quality_notes'] = ''
    for c in reasons + ['revised_match', 'is_super_over', 'regulation_tied_match']:
        d.loc[d[c], 'state_quality_notes'] += c + ';'
    return d, overs.reset_index(), reasons
