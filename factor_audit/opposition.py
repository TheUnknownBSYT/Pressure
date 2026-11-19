"""Can the previous XI or previous team wins add information beyond today's known matchup?"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from final_strategy import response_model as rm
from final_strategy.consequence import mass
from .forecast import OUT, fixtures, comparison


def prior_rosters(m, xi):
    # The current file records actual participants (including impact subs), not timestamped announced XIs.
    played=xi[xi.is_played].groupby(['match_id','team_id']).player_id.agg(lambda s:sorted(set(s))).to_dict()
    visits={}; rows=[]
    for day,block in m.sort_values('match_date').groupby('match_date'):
        today={}
        for r in block.itertuples():
            for team in [r.home_team_id,r.away_team_id]:
                for pid in visits.get(team,[]):
                    rows.append((r.match_id,team,pid,pd.Timestamp(day)))
                roster=played.get((r.match_id,team),[])
                # Missing or ambiguous previous-day squads stay unavailable.
                if team in today or not 11<=len(roster)<=12: today[team]=[]
                else: today[team]=roster
        visits.update(today)
    return pd.DataFrame(rows,columns=['match_id','team_id','player_id',rm.DATE])


def projected(d, members, role):
    pid,team=rm.IDS[role],rm.TEAMS[role]
    cols=['competition_type','phase_pre','innings']
    a=d[[pid,rm.DATE,*cols]].copy()
    a['n'],a['r'],a['w']=1,d[role+'_runs'],d[rm.WICKET[role]]
    h=rm.attach(a,rm.date_prior(a,cols,['n','r','w']),cols+[rm.DATE])
    a['r']=a.r-(h.hist_r+75)/(h.hist_n+60)
    a['w']=a.w-(h.hist_w+3)/(h.hist_n+60)
    sums=a.groupby([pid,rm.DATE])[['r','w','n']].sum().groupby(level=0).cumsum().reset_index().rename(columns={pid:'player_id'})
    prior=pd.merge_asof(members.sort_values(rm.DATE),sums.sort_values(rm.DATE),on=rm.DATE,
                        by='player_id',direction='backward',allow_exact_matches=False)
    for c in ['r','w']: prior[c]=prior[c].fillna(0)/(prior.n.fillna(0)+100)
    prior=prior.groupby(['match_id','team_id'])[['r','w']].mean().rename_axis(['match_id',team])
    xi=d[['match_id',team]].join(prior,on=['match_id',team])
    return pd.DataFrame({'xi_r':xi.r.fillna(0),'xi_w':xi.w.fillna(0),'xi_unknown':xi.r.isna().astype(float)})


def wp_strength(d):
    matches=d[d.training_label_available].drop_duplicates(['match_id','innings'])
    # One indicator per team and match. Decay older wins; shrink by ten even-chance matches.
    a=matches[['match_date_parsed','batting_team_id','batting_win_label_outcome']].copy()
    a['n']=np.exp2((a.match_date_parsed-a.match_date_parsed.min()).dt.days/365)
    a['w']=a.n*(a.batting_win_label_outcome-.5)
    prior=rm.date_prior(a,['batting_team_id'],['n','w']).rename(columns={'batting_team_id':'team'})
    weight=np.exp2((d.match_date_parsed-a.match_date_parsed.min()).dt.days/365)
    quality=[]
    for key in ['batting_team_id','bowling_team_id']:
        lookup=d[[key,'match_date_parsed']].rename(columns={key:'team'})
        h=lookup.merge(prior,on=['team','match_date_parsed'],how='left',sort=False,validate='many_to_one')
        quality.append(h.hist_w.to_numpy()/(h.hist_n.to_numpy()+10*weight.to_numpy()))
    return quality[0]-quality[1]


def run():
    d,x,_=rm.build()
    members=prior_rosters(fixtures(),pd.read_csv('playing_eleven.csv'))
    losses=[]
    for r in rm.IDS:
        xi=projected(d,members,rm.OTHER[r])
        base=rm.BASE+rm.ROLE+rm.OPP
        f=x[r][base].join(xi)
        for start,end in rm.FOLDS:
            train=d[rm.DATE].lt(start).to_numpy()
            test=(d[rm.DATE].between(start,end,inclusive='left') & d.pressure_eligible &
                  d.competition_type.isin(['IPL','T20I'])).to_numpy()
            out=d.loc[test,['match_id']].reset_index(drop=True)
            for name,cols in [('published',base),('previous_XI',list(f))]:
                rr,pp,_=rm.model_fit(f,d[r+'_runs'].to_numpy(),d[rm.WICKET[r]].to_numpy(),train,test,cols)
                out[name+'_run']=(rr-d.loc[test,r+'_runs'].to_numpy())**2
                out[name+'_brier']=(pp-d.loc[test,rm.WICKET[r]].to_numpy())**2
            out['roster_available']=1-xi.loc[test,'xi_unknown'].to_numpy()
            losses.append(out.groupby('match_id').mean().assign(role=r,year=int(start[:4])).reset_index())
    raw=pd.concat(losses)
    raw.to_csv(OUT/'xi_matches.csv',index=False)
    summaries=[]
    for (role,period),z in raw.groupby(['role',raw.year.eq(2026).map({True:'2026',False:'development'})]):
        row=dict(role=role,period=period,matches=len(z),mean_match_coverage=z.roster_available.mean())
        for metric in ['run','brier']:
            delta=z['previous_XI_'+metric]-z['published_'+metric]
            row[metric],row[metric+'_lo'],row[metric+'_hi']=comparison(delta.to_frame('delta'),'delta')
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(OUT/'xi_validation.csv',index=False)
    cols=['match_id','match_date_parsed','innings','win_probability_pre','training_label_available',
          'batting_win_label_outcome','batting_team_id','bowling_team_id','competition_type']
    wp=pd.read_parquet('outputs/pressure_scored.parquet',columns=cols)
    wp['strength']=wp_strength(wp)
    wp=wp[wp.training_label_available & wp.win_probability_pre.notna() & wp.competition_type.isin(['T20I','IPL'])].copy()
    assert wp.strength.notna().all()
    p=wp.win_probability_pre.clip(1e-7,1-1e-7)
    wp['logit']=np.log(p/(1-p))
    scores=[]
    for year in [2024,2025,2026]:
        for inning,block in wp.groupby('innings'):
            tr=block[block.match_date_parsed.dt.year<year]
            te=block[block.match_date_parsed.dt.year.eq(year)]
            for name,features in [('recalibrated',['logit']),('team_strength',['logit','strength'])]:
                model=LogisticRegression(C=10,max_iter=200).fit(tr[features],tr.batting_win_label_outcome,sample_weight=mass(tr))
                pred=model.predict_proba(te[features])[:,1]
                err=((te.batting_win_label_outcome-pred)**2).groupby(te.match_id).mean()
                if name=='recalibrated': ref=err
                delta,lo,hi=comparison((err-ref).to_frame('delta'),'delta')
                scores.append(dict(year=year,innings=inning,model=name,matches=len(err),brier=err.mean(),delta=delta,lo=lo,hi=hi))
    pd.DataFrame(scores).to_csv(OUT/'wp_strength.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=1): run()
