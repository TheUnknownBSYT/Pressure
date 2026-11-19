"""Render the decision, coverage table and Round 1 lists from verified experiment files."""
from pathlib import Path
import pandas as pd
from .forecast import OUT

COVERAGE = [
    ('Competition','Win and response models include IPL indicators. Own comparisons and history norms match competition, innings and phase.',
     'Keep pooled fitting and within-competition comparisons. Five extra IPL slopes do not consistently reduce run or wicket error. IPL-only answers a narrower question.'),
    ('Opponent quality','Prior-date player and team run/wicket residuals shrink toward the competition and situation norm.',
     'Keep the actual opponent in expected performance. Adding them to a team-only model improves run forecasts in both roles. Decay and the prior XI give mixed results.'),
    ('Tournament importance','Stage supplies 20% of the pressure index. Finals across tournaments share one weight.',
     'Keep stage visible and publish the no-stakes comparison. The data cannot set a prestige ladder.'),
    ('Match stage','Explicit finals, playoffs, Super 4 and league labels, with a fallback for unknown rounds.',
     'Use one weight for ordinary fixtures in identified bilateral series. Keep the raw round and preserve unknown rounds in multi-team series.'),
    ('Home / away','Earlier venue scoring contributes to expected runs. Fixture home/away labels have no direct weight.',
     'Country and lagged ground experience do not support a general bonus. T20I bowling ground familiarity deserves a separate replication.'),
    ('Crowds and expectations','No audience or psychological measurement is present.',
     'Exclude from scores. Venue, team identity and knockout status cannot identify crowd size, allegiance or individual stress.'),
    ('Form and opportunity','Five-date trends, setbacks, streaks and previous roster participation were experimental. The published ranking uses the historical window.',
     'Show recent exposure and recency sensitivities. Results do not justify an individual pressure multiplier or replacing the historical target.'),
    ('Reliability','Positive quality and personal uplift, cell and exposure gates, capped match weights and a match bootstrap.',
     'Keep the transparent ranking rule. Report frequency, uncertainty and full eligibility after deleting a match alongside the order.'),
]


def table(frame):
    def fmt(v):
        if pd.isna(v): return '—'
        if isinstance(v,float): return f'{v:.3f}'.rstrip('0').rstrip('.') if v else '0'
        return str(v).replace('|',' / ')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join('---' for _ in frame.columns)+' |',
                      *['| '+' | '.join(map(fmt,row))+' |' for row in frame.itertuples(index=False,name=None)]])


def run():
    summary=pd.read_csv(OUT/'model_summary.csv')
    tops=pd.read_csv(OUT/'top10_evidence.csv')
    sides=pd.read_csv(OUT/'rankings_side_by_side.csv')
    gates=pd.read_csv(OUT/'gates.csv')
    forward=pd.read_csv(OUT/'forward_validation.csv')
    paired=pd.read_csv(OUT/'paired_player_validation.csv')
    deletion=pd.read_csv(OUT/'deletion_rechecked.csv')
    factor=pd.DataFrame(COVERAGE,columns=['Factor','Published treatment','Decision'])
    factor.to_csv(OUT/'factor_coverage.csv',index=False)
    fields={'FACTORS':table(factor)}
    for role,filename in [('bat','batters'),('bowl','bowlers')]:
        final=tops[(tops.model=='G') & (tops.role==role)].sort_values('rank').copy()
        final['rank']=final['rank'].astype(int)
        assert len(final)==10 and final.player_id.is_unique
        final[['rank','player_name','player_id']].to_csv(OUT/('recommended_'+filename+'_top10.csv'),index=False)
        ordered=sides.loc[sides.role.eq(role),['rank',*'ABCDEFG']].rename(columns={'rank':'Rank'})
        fields[role.upper()]=table(ordered)
    main=summary[summary.model.isin(list('ABCDEFG'))].copy()
    metrics=forward.query("year==2026 and target_type=='version' and cohort=='available' and target=='future_absolute'")
    main=main.merge(metrics[['role','model','players','rho']],on=['role','model'])
    strict=gates.query("gate=='strict'")[['model','role','retained']]
    main=main.merge(strict,on=['role','model'])
    fields['METRICS']=table(main[['model','role','supported','eligible','overlap_A','selected_median_matches',
        'selected_median_balls','top_median_ordinary','top_median_pressure','top_median_uplift','retained','players','rho']].rename(
        columns={'model':'Model','role':'Role','supported':'Supported','eligible':'Eligible','overlap_A':'Overlap A',
                 'selected_median_matches':'Matches','selected_median_balls':'Balls','top_median_ordinary':'Ordinary',
                 'top_median_pressure':'Pressure','top_median_uplift':'Uplift','retained':'Strict overlap','players':'Forward n','rho':'Forward ρ'}))
    g=tops[tops.model=='G'].merge(deletion[deletion.model=='G'],on=['role','player_id','rank','player_name'])
    fields['ROBUST']=table(g[['role','rank','player_name','original_rank','pressure_matches','pressure_legal',
        'IPL_legal','T20I_legal','ordinary','absolute','uplift','positive_own_share','uplift_lo95','uplift_hi95',
        'eligible','deletions']].rename(columns={'role':'Role','rank':'#','player_name':'Player','original_rank':'A rank',
        'pressure_matches':'M','pressure_legal':'Balls','IPL_legal':'IPL','T20I_legal':'T20I','ordinary':'Ord.',
        'absolute':'Press.','uplift':'Uplift','positive_own_share':'Beat own','uplift_lo95':'Low 95%','uplift_hi95':'High 95%',
        'eligible':'Eligible deletions','deletions':'Deletions'}))
    z=paired.query("year==2026 and model in ['G','E'] and target in ['future_absolute','future_vote']")
    fields['PAIRED']=table(z[['role','model','target_definition','target','n','rho','reference_rho','delta_lo','delta_hi']].rename(
        columns={'role':'Role','model':'Model','target_definition':'Later definition','target':'Later outcome','n':'Players','rho':'ρ',
                 'reference_rho':'A ρ','delta_lo':'Δ low','delta_hi':'Δ high'}))
    flags=pd.read_csv(OUT/'stage_exposure.csv').groupby('model').flipped.sum()
    fields['GFLIPS']=f"{flags['consistent_series']:,}"
    fields['GSURVIVE']=str(int(g.eligible.eq(g.deletions).sum()))
    adj=pd.read_csv(OUT/'recommended_adjacent.csv')
    fields['ADJACENT']=str(int((adj.difference_lo95.le(0) & adj.difference_hi95.ge(0)).sum()))
    comparisons=[]
    for role,z in tops.groupby('role'):
        official=set(z[z.model.eq('G')].player_id)
        for method,weight in [('G','.30'),('series_mid','.35'),('series_league','.40')]:
            alt=z[z.model.eq(method)].sort_values('rank')
            comparisons.append(dict(Role=role,Weight=weight,Leader=alt.player_name.iloc[0],
                                    **{'Overlap with G':len(official & set(alt.player_id))}))
    fields['BILATERAL']=table(pd.DataFrame(comparisons))
    players=pd.read_csv(OUT/'candidates.csv')
    movers=players[players.model.isin(list('BCDEFG')) & players.rank_gain.notna() &
                   (players['rank'].le(10) | players.original_rank.le(10))].copy()
    movers['largest_shift']=movers.rank_gain.abs()
    peak=movers.groupby(['model','role']).largest_shift.transform('max')
    movers=movers[movers.largest_shift.eq(peak)].sort_values(['model','role','player_name'])
    movers.to_csv(OUT/'largest_rank_moves.csv',index=False)
    text=Path(__file__).with_name('interpretation.md').read_text()
    for key,value in fields.items(): text=text.replace('{{'+key+'}}',value)
    assert '{{' not in text
    (OUT/'audit.md').write_text(text)


if __name__=='__main__': run()
