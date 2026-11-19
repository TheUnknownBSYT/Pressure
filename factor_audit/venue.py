"""Earlier appearances at the venue as a familiarity proxy, plus a competition-specific WP check."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from final_strategy import response_model as rm
from final_strategy.consequence import mass
from .forecast import OUT, fixtures, comparison


def familiarity(m, recorded_matches):
    keys=['match_id','match_date','venue_id']
    teams=pd.concat([m[keys+[side]].rename(columns={side:'team_id'}) for side in ['home_team_id','away_team_id']])
    teams=teams.rename(columns={'match_date':rm.DATE})
    teams[rm.DATE]=pd.to_datetime(teams[rm.DATE])
    teams['visits']=teams.match_id.isin(recorded_matches).astype(int)
    lookup=rm.date_prior(teams,['team_id','venue_id'],['visits'])
    return teams.merge(lookup,on=['team_id','venue_id',rm.DATE],validate='many_to_one').set_index(['match_id','team_id']).hist_visits


def run():
    d,x,_=rm.build()
    played=pd.read_parquet('outputs/pressure_scored.parquet',columns=['match_id']).match_id.unique()
    visits=familiarity(fixtures(),played)
    for side in ['batting','bowling']:
        keys=pd.MultiIndex.from_arrays([d.match_id,d[side+'_team_id']])
        d[side+'_visits']=np.log1p(visits.reindex(keys).to_numpy())
    assert d[['batting_visits','bowling_visits']].notna().all().all()
    out=[]
    for role in rm.IDS:
        f=x[role][rm.BASE+rm.ROLE+rm.OPP].copy()
        cols=list(f)
        f['ground_experience']=d.batting_visits-d.bowling_visits
        for start,end in rm.FOLDS:
            tr=d[rm.DATE].lt(start).to_numpy()
            te=(d[rm.DATE].between(start,end,inclusive='left') & d.pressure_eligible &
                d.competition_type.isin(['IPL','T20I'])).to_numpy()
            errors=d.loc[te,['match_id','competition_type']].copy()
            for name,features in [('base',cols),('visits',list(f))]:
                rr,pp,_=rm.model_fit(f,d[role+'_runs'].to_numpy(),d[rm.WICKET[role]].to_numpy(),tr,te,features)
                errors[name+'_run']=(rr-d.loc[te,role+'_runs'])**2
                errors[name+'_brier']=(pp-d.loc[te,rm.WICKET[role]])**2
            out.append(errors.groupby(['match_id','competition_type']).mean().reset_index().assign(role=role,year=int(start[:4])))
    results=[]
    all_errors=pd.concat(out)
    for (role,period,comp),z in all_errors.groupby(['role',all_errors.year.eq(2026).map({True:'2026',False:'development'}),'competition_type']):
        row=dict(role=role,period=period,competition=comp,matches=len(z))
        for metric in ['run','brier']:
            delta=(z['visits_'+metric]-z['base_'+metric]).to_frame('d')
            row[metric],row[metric+'_lo'],row[metric+'_hi']=comparison(delta,'d')
        results.append(row)
    pd.DataFrame(results).to_csv(OUT/'venue_experience.csv',index=False)
    wp=pd.read_parquet('outputs/pressure_scored.parquet',columns=['match_id','innings',rm.DATE,'win_probability_pre',
        'training_label_available','batting_win_label_outcome','competition_type'])
    wp=wp[wp.training_label_available & wp.win_probability_pre.notna() & wp.competition_type.isin(['IPL','T20I'])].copy()
    p=wp.win_probability_pre.clip(1e-7,1-1e-7)
    wp['logit']=np.log(p/(1-p)); wp['ipl']=wp.competition_type.eq('IPL').astype(float)
    wp['ipl_logit']=wp.ipl*wp.logit
    metrics=[]
    for (inning,year) in [(i,y) for i in [1,2] for y in [2024,2025,2026]]:
        z=wp[wp.innings.eq(inning)]
        tr=z[z[rm.DATE].dt.year<year]; te=z[z[rm.DATE].dt.year.eq(year)]
        for name,cols in [('shared',['logit']),('competition_specific',['logit','ipl','ipl_logit'])]:
            fit=LogisticRegression(C=10,max_iter=200).fit(tr[cols],tr.batting_win_label_outcome,sample_weight=mass(tr))
            pp=fit.predict_proba(te[cols])[:,1]
            err=((pp-te.batting_win_label_outcome)**2).groupby(te.match_id).mean()
            if name=='shared': ref=err
            delta,lo,hi=comparison((err-ref).to_frame('delta'),'delta')
            metrics.append(dict(innings=inning,year=year,model=name,matches=len(err),brier=err.mean(),delta=delta,lo=lo,hi=hi))
    pd.DataFrame(metrics).to_csv(OUT/'wp_competition.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=1): run()
