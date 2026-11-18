"""Transparent chronological partial pooling and player attribution."""
import numpy as np
import pandas as pd

BASE=['innings','phase_pre','competition_type']
CELL=BASE+['wicket_band','demand_band']

def attribute(d):
    d=d.copy()
    d['bat_dismissal']= (d.is_wicket & d.dismissed_player_id.eq(d.striker_id)).astype(int)
    d['nonstriker_dismissal']=(d.is_wicket & d.dismissed_player_id.eq(d.non_striker_id)).astype(int)
    d['dismissal_unresolved']=d.is_wicket & ~(d.dismissed_player_id.eq(d.striker_id)|d.dismissed_player_id.eq(d.non_striker_id))
    d['balls_faced_nonwide']= (~d.extras_type.eq('wide')).astype(int)
    d['bat_runs']=d.runs
    d['bowl_runs']=d.bowler_runs_event_assumed
    d['bowl_wicket']=d.bowler_wicket_event
    d['wicket_band']=pd.cut(d.wickets_remaining_pre,[-1,3,7,10],labels=['low','medium','high']).astype(str)
    d['demand_band']=pd.cut(d.required_run_rate_pre,[-np.inf,8,12,18,np.inf],labels=['low','medium','high','extreme']).astype(str)
    d.loc[d.innings.eq(1),'demand_band']='first_innings'
    d['comparison_cell']=d[BASE].astype(str).agg('|'.join,axis=1)
    d['outcome_valid']=~d.dismissal_unresolved & ~d.bowler_concession_uncertain
    return d

def pool_predict(train,test, outcome,prior=200):
    # Rates per legal ball; all recorded events contribute outcomes. Illegal
    # events consume zero exposure and retain their actual runs/wickets.
    global_rate=train[outcome].sum()/train.legal_event.sum()
    coarse=train.groupby(BASE).agg(total=(outcome,'sum'),balls=('legal_event','sum'),matches=('match_id','nunique'))
    coarse['rate']=(coarse.total+prior*global_rate)/(coarse.balls+prior)
    fine=train.groupby(CELL).agg(total=(outcome,'sum'),balls=('legal_event','sum'))
    x=test[BASE].merge(coarse[['rate','matches']],left_on=BASE,right_index=True,how='left',validate='many_to_one')
    y=test[CELL].merge(fine,left_on=CELL,right_index=True,how='left',validate='many_to_one')
    parent=x.rate.fillna(global_rate).to_numpy()
    rates=(y.total.fillna(0).to_numpy()+prior*parent)/(y.balls.fillna(0).to_numpy()+prior)
    return rates,x.matches.fillna(0).to_numpy()

def forward_expectations(d,min_matches=50):
    d=d.copy(); diagnostics=[]
    outcomes=['bat_runs','bat_dismissal','bowl_runs','bowl_wicket']
    for c in outcomes:d['expected_'+c+'_per_legal']=np.nan
    d['expected_train_end']=pd.NaT;d['expected_support_matches']=0
    valid=d.standard_t20_eligible & d.outcome_valid
    for period in sorted(d.match_date_parsed.dt.to_period('Q').unique()):
        cutoff=period.start_time
        train=d[valid & d.match_date_parsed.lt(cutoff)]
        test=d[valid & d.match_date_parsed.ge(cutoff)&d.match_date_parsed.lt((period+1).start_time)]
        if test.empty or train.match_id.nunique()<min_matches:continue
        assert not set(train.match_id)&set(test.match_id)
        for c in outcomes:
            rate,support=pool_predict(train,test,c)
            d.loc[test.index,'expected_'+c+'_per_legal']=np.where(support>=10,rate,np.nan)
            # Evaluate rate predictions at match level, keeping illegal outcomes.
            a=pd.DataFrame({'match_id':test.match_id,'actual':test[c], 'expected':rate*test.legal_event,'balls':test.legal_event})
            a=a.groupby('match_id').sum();a=a[a.balls>0]
            prior=train[c].sum()/train.legal_event.sum()
            diagnostics.append(dict(fold=str(cutoff.date()),outcome=c,train_matches=train.match_id.nunique(),test_matches=len(a),
                match_rate_mse=float(((a.actual-a.expected)/a.balls).pow(2).mean()),
                prior_match_rate_mse=float((a.actual/a.balls-prior).pow(2).mean())))
        d.loc[test.index,'expected_train_end']=train.match_date_parsed.max()
        d.loc[test.index,'expected_support_matches']=support
    d['analysis_available']=d.pressure_eligible & d.outcome_valid & d.expected_bat_runs_per_legal.notna()
    d['bat_run_residual']=d.bat_runs-d.expected_bat_runs_per_legal*d.legal_event
    d['bat_wicket_residual']=d.expected_bat_dismissal_per_legal*d.legal_event-d.bat_dismissal
    d['bowl_run_residual']=d.expected_bowl_runs_per_legal*d.legal_event-d.bowl_runs
    d['bowl_wicket_residual']=d.bowl_wicket-d.expected_bowl_wicket_per_legal*d.legal_event
    return d,pd.DataFrame(diagnostics)

def identities(root):
    m=pd.read_csv(root/'competition_team_player_mapping.csv')
    counts=m.groupby(['player_id','player_name']).size().reset_index(name='records')
    chosen=counts.sort_values(['player_id','records','player_name'],ascending=[True,False,True]).drop_duplicates('player_id')
    chosen['name_variants']=chosen.player_id.map(counts.groupby('player_id').size())
    return chosen[['player_id','player_name','name_variants']],counts

def pressure_variant(d,top=25,scale=.1,stakes=True):
    p=100*(.5*np.clip(d.leverage_sd_probability/scale,0,1)+.3*d.urgency*d.contestability+(.2*d.match_stakes if stakes else 0))
    # Same global Phase-1 scored population in every scenario, before role exclusions.
    cutoff=p[d.pressure_eligible].quantile(1-top/100)
    return p.ge(cutoff)&d.pressure_eligible,float(cutoff)

def match_metrics(d,role,high):
    a=d[d.analysis_available].copy();a['high']=high.loc[a.index].astype(int)
    a['player_id']=a.striker_id if role=='bat' else a.bowler_id
    a['run_residual']=a[role+'_run_residual'];a['wicket_residual']=a[role+'_wicket_residual']
    a['actual_runs']=a[role+'_runs'];a['actual_wickets']=a.bat_dismissal if role=='bat' else a.bowl_wicket
    return a.groupby(['player_id','match_id','comparison_cell','high']).agg(
        legal=('legal_event','sum'),events=('delivery_id','size'),balls_faced=('balls_faced_nonwide','sum'),
        runs=('actual_runs','sum'),wickets=('actual_wickets','sum'),rr=('run_residual','sum'),wr=('wicket_residual','sum')).reset_index()

def summarize(m,min_matches=8,min_balls=50,prior=8,bootstrap=0,seed=42):
    rows=[];rng=np.random.default_rng(seed)
    for pid,a in m.groupby('player_id'):
        cells=a.groupby(['comparison_cell','high']).agg(balls=('legal','sum'),matches=('match_id','nunique'))
        good=cells[(cells.index.get_level_values('high')==0)&(cells.balls>=10)&(cells.matches>=2)].index.get_level_values(0)
        total_high=a.loc[a.high.eq(1),'legal'].sum()
        good=good.intersection(pd.Index(a.loc[a.high.eq(1),'comparison_cell'].unique()))
        z=a[a.comparison_cell.isin(good)]
        h=z[z.high.eq(1)];o=z[z.high.eq(0)]
        hb=h.legal.sum();ob=o.legal.sum();hm=h.match_id.nunique();om=o.match_id.nunique()
        row=dict(player_id=pid,pressure_legal_all=total_high,pressure_legal=hb,ordinary_legal=ob,
            pressure_matches=hm,ordinary_matches=om,overlap_fraction=hb/total_high if total_high else 0,
            pressure_events=h.events.sum(),ordinary_events=o.events.sum(),pressure_runs=h.runs.sum(),ordinary_runs=o.runs.sum(),
            pressure_wickets=h.wickets.sum(),ordinary_wickets=o.wickets.sum(),pressure_balls_faced=h.balls_faced.sum(),ordinary_balls_faced=o.balls_faced.sum())
        row['eligible']=hm>=min_matches and om>=5 and hb>=min_balls and ob>=50 and row['overlap_fraction']>=.5
        if hb<=0 or ob<=0:
            rows.append(row);continue
        cells_sorted=sorted(set(h.comparison_cell)); matches=sorted(z.match_id.unique())
        # Matrix columns cell x high/ordinary x exposure/run/wicket. Resampling
        # whole matches preserves high/ordinary covariance and within-match dependence.
        arr=np.zeros((len(matches),len(cells_sorted),2,3));mi={v:i for i,v in enumerate(matches)};ci={v:i for i,v in enumerate(cells_sorted)}
        for v in z.itertuples():
            if v.comparison_cell in ci:arr[mi[v.match_id],ci[v.comparison_cell],v.high]=[v.legal,v.rr,v.wr]
        def estimate(s):
            hb=s[:,1,0].sum(); weights=s[:,1,0]/hb if hb else np.zeros(len(s))
            high=100*s[:,1,1:].sum(axis=0)/max(hb,1)
            # Only ordinary cells observed in the resample can support uplift.
            if ((s[:,0,0]==0)&(weights>0)).any():return np.r_[high,[np.nan,np.nan]]
            ordinary=100*(weights[:,None]*s[:,0,1:]/np.maximum(s[:,0,0,None],1)).sum(axis=0)
            return np.r_[high,ordinary]
        est=estimate(arr.sum(axis=0)); shrink=hm/(hm+prior)
        row.update(dict(pressure_run_adj=est[0],pressure_wicket_adj=est[1],ordinary_run_adj=est[2],ordinary_wicket_adj=est[3],
            run_uplift=est[0]-est[2],wicket_uplift=est[1]-est[3],shrinkage=shrink))
        # Match consistency and single-best-match removal in uncombined components.
        mh=h.groupby('match_id')[['legal','rr','wr']].sum();mh=mh[mh.legal>0]
        row['positive_run_match_fraction']=(mh.rr>0).mean();row['median_match_run_adj']=(100*mh.rr/mh.legal).median()
        row['largest_match_ball_share']=mh.legal.max()/hb
        best=mh.rr.idxmax();removed=arr.sum(axis=0)-arr[mi[best]];be=estimate(removed)
        row.update(dict(without_best_pressure_run_adj=be[0],without_best_pressure_wicket_adj=be[1],without_best_run_uplift=be[0]-be[2],without_best_wicket_uplift=be[1]-be[3]))
        if bootstrap and row['eligible']:
            counts=rng.multinomial(len(matches),np.full(len(matches),1/len(matches)),size=bootstrap)
            draws=np.tensordot(counts,arr,axes=(1,0));vals=np.array([estimate(s) for s in draws])
            row['_bootstrap']=vals
            for k,j in [('pressure_run_adj',0),('pressure_wicket_adj',1)]:
                row[k+'_lo95'],row[k+'_hi95']=np.nanquantile(vals[:,j],[.025,.975])
            row['bootstrap_valid_uplift_fraction']=np.isfinite(vals[:,2]).mean()
        rows.append(row)
    return pd.DataFrame(rows)

def rank(s,run_weight=.7,absolute_weight=.75,scales=None):
    s=s.copy();eligible=s.eligible.fillna(False)
    if scales is None:
        scales=[max(float(s.loc[eligible,c].std()),1e-6) for c in ['pressure_run_adj','pressure_wicket_adj']]
    def score(run,wicket,upr,upw):
        return absolute_weight*(run_weight*run/scales[0]+(1-run_weight)*wicket/scales[1])+(1-absolute_weight)*(run_weight*upr/scales[0]+(1-run_weight)*upw/scales[1])
    s['rating']=s.shrinkage*score(s.pressure_run_adj,s.pressure_wicket_adj,s.run_uplift,s.wicket_uplift)
    s['rating_without_best']=s.shrinkage*score(s.without_best_pressure_run_adj,s.without_best_pressure_wicket_adj,s.without_best_run_uplift,s.without_best_wicket_uplift)
    s['rank']=s.rating.where(eligible).rank(method='min',ascending=False)
    s['rank_without_best']=s.rating_without_best.where(eligible).rank(method='min',ascending=False)
    if '_bootstrap' in s:
        for ix,row in s[eligible].iterrows():
            b=row['_bootstrap']
            if not isinstance(b,np.ndarray):continue
            v=row.shrinkage*score(b[:,0],b[:,1],b[:,0]-b[:,2],b[:,1]-b[:,3])
            s.loc[ix,'rating_lo95'],s.loc[ix,'rating_hi95']=np.nanquantile(v,[.025,.975])
        s=s.drop(columns='_bootstrap')
    return s.sort_values(['eligible','rating','player_id'],ascending=[False,False,True]),scales
