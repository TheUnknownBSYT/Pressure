"""Earlier T20I performance versus later IPL performance on common players."""
from pathlib import Path
import numpy as np
import pandas as pd

E = Path('outputs/final_submission/evidence')


def match_values(events, role):
    who = 'striker_id' if role == 'bat' else 'bowler_id'
    g = events.dropna(subset=[who, role+'_value']).groupby(
        [who, 'match_id', 'match_date_parsed', 'competition_type'], as_index=False)
    match = g.agg(value=(role+'_value', 'sum'), balls=('legal_event', 'sum')).rename(columns={who:'player_id'})
    match = match.loc[match.balls.gt(0)].copy()
    match['weight'] = match.balls.clip(upper=6) / 6
    match['weighted_quality'] = 6 * match.value / match.balls * match.weight
    return match


def player_means(matches, min_matches, min_balls):
    a = matches.groupby('player_id').agg(
        weighted_quality=('weighted_quality', 'sum'), exposure=('weight', 'sum'),
        balls=('balls', 'sum'), matches=('match_id', 'size'))
    return a.loc[a.matches.ge(min_matches) & a.balls.ge(min_balls)]


def main():
    state = pd.read_parquet('outputs/pressure_scored.parquet', columns=[
        'delivery_id','competition_type','striker_id','bowler_id','legal_event','match_id','match_date_parsed'])
    p = pd.read_parquet(E/'consequence_events.parquet', columns=['delivery_id','high_static'])
    v = pd.read_parquet(E/'response_events.parquet', columns=['delivery_id','bat_value','bowl_value'])
    state = state.merge(p.loc[p.high_static.eq(True)], validate='1:1').merge(v,validate='1:1')
    pairs, summary = [], []
    for role in ['bat','bowl']:
        matches = match_values(state, role)
        for year in [2024, 2025, 2026]:
            before = matches.loc[matches.match_date_parsed.lt(pd.Timestamp(year,1,1))]
            later = matches.loc[matches.match_date_parsed.dt.year.eq(year) & matches.competition_type.eq('IPL')]
            actual = player_means(later,2,12).add_prefix('later_')
            for comp in ['T20I','IPL']:
                historical = player_means(before.loc[before.competition_type.eq(comp)],3,18)
                z = historical.join(actual, how='inner')
                z['prediction'] = z.weighted_quality / (z.exposure+8)
                z['observed'] = z.later_weighted_quality / z.later_exposure
                z['evaluated_year'], z['source_competition'], z['role'] = year, comp, role
                pairs.append(z.reset_index())
                err = (z.prediction-z.observed).pow(2)
                summary.append(dict(year=year,role=role,prior=comp,players=len(z),
                    prior_balls=z.balls.sum(),later_ipl_balls=z.later_balls.sum(),
                    spearman=z.prediction.corr(z.observed,method='spearman') if len(z)>=3 else np.nan,
                    rmse=np.sqrt(err.mean()),zero_benchmark_rmse=np.sqrt(z.observed.pow(2).mean())))
    pd.concat(pairs,ignore_index=True).to_csv(E/'transfer_player_pairs.csv',index=False)
    results = pd.DataFrame(summary)
    results.to_csv(E/'transfer_validation.csv',index=False)
    print(results.to_string(index=False))


if __name__ == '__main__':
    main()
