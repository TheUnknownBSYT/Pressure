"""Targeted, read-only robustness analysis of existing model outputs."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def calibration_audit(d):
    a=d[d.innings.eq(1)&d.training_label_available&d.win_probability_pre.notna()].copy()
    a['calibrated']=np.nan;a['calibration_train_end']=pd.NaT;folds=[]
    for period in sorted(a.match_date_parsed.dt.to_period('Q').unique()):
        tr=a[a.match_date_parsed<period.start_time];te=a[a.match_date_parsed.dt.to_period('Q').eq(period)]
        if tr.match_id.nunique()<100 or tr.batting_win_label_outcome.nunique()<2:continue
        assert not set(tr.match_id)&set(te.match_id)
        def x(z):
            p=z.win_probability_pre.clip(1e-5,1-1e-5).to_numpy();return np.log(p/(1-p)).reshape(-1,1)
        w=1/tr.groupby('match_id').match_id.transform('size')
        model=LogisticRegression(C=1.,max_iter=1000).fit(x(tr),tr.batting_win_label_outcome,sample_weight=w)
        a.loc[te.index,'calibrated']=model.predict_proba(x(te))[:,1];a.loc[te.index,'calibration_train_end']=tr.match_date_parsed.max()
        folds.append(dict(fold=str(period),train_matches=tr.match_id.nunique(),test_matches=te.match_id.nunique(),train_end=str(tr.match_date_parsed.max()),intercept=model.intercept_[0],slope=model.coef_[0,0]))
    v=a[a.calibrated.notna() & a.match_date_parsed.ge('2026-01-01')].copy()
    v['w']=1/v.groupby('match_id').match_id.transform('size');y=v.batting_win_label_outcome
    metrics=[];bins=[]
    for c in ['win_probability_pre','calibrated']:
        mse=np.average((v[c]-y)**2,weights=v.w);ll=log_loss(y,v[c],sample_weight=v.w)
        b=pd.cut(v[c],np.linspace(0,1,11),include_lowest=True);ece=0
        for k,g in v.groupby(b,observed=True):
            pred=np.average(g[c],weights=g.w);actual=np.average(g.batting_win_label_outcome,weights=g.w);mass=g.w.sum()/v.w.sum();ece+=mass*abs(pred-actual)
            bins.append(dict(model=c,bin=str(k),predicted=pred,observed=actual,matches=g.match_id.nunique(),weight_mass=mass))
        metrics.append(dict(model=c,brier=mse,log_loss=ll,ece=ece,matches=v.match_id.nunique()))
    delta=((v.calibrated-y)**2-(v.win_probability_pre-y)**2).groupby(v.match_id).mean().to_numpy()
    rng=np.random.default_rng(42);boot=rng.choice(delta,(2000,len(delta)),replace=True).mean(axis=1)
    ci=np.quantile(boot,[.025,.975]);m=pd.DataFrame(metrics)
    adopt=bool(len(delta)>=60 and ci[1]<0 and m.iloc[1].log_loss<m.iloc[0].log_loss and m.iloc[1].ece<m.iloc[0].ece)
    decision=dict(adopt=adopt,heldout_start='2026-01-01',brier_difference=float(delta.mean()),bootstrap_lo95=float(ci[0]),bootstrap_hi95=float(ci[1]),rule='>=60 heldout matches; Brier difference upper 95% <0; log loss and fixed-bin ECE both lower; rule specified before evaluation')
    return a,pd.DataFrame(folds),m,pd.DataFrame(bins),decision

def prior_wickets(d):
    d=d.copy();audit=[]
    for period in sorted(d.match_date_parsed.dt.to_period('Q').unique()):
        tr=d[d.standard_t20_eligible&d.outcome_valid&d.match_date_parsed.lt(period.start_time)]
        te=d[d.analysis_available&d.match_date_parsed.dt.to_period('Q').eq(period)]
        if te.empty:continue
        for role,outcome in [('bat','bat_dismissal'),('bowl','bowl_wicket')]:
            rate=tr[outcome].sum()/tr.legal_event.sum()
            d.loc[te.index,role+'_prior_wicket_rate']=rate
            contextual=te['expected_'+outcome+'_per_legal']*te.legal_event
            p=rate*te.legal_event
            z=pd.DataFrame({'match_id':te.match_id,'balls':te.legal_event,'actual':te[outcome],'contextual':contextual,'prior':p}).groupby('match_id').sum()
            for mid,r in z[z.balls>0].iterrows():audit.append(dict(role=role,fold=str(period),match_id=mid,contextual_error=((r.actual-r.contextual)/r.balls)**2,prior_error=((r.actual-r.prior)/r.balls)**2))
    return d,pd.DataFrame(audit)

def supported_matches(m,cell_balls=20,cell_matches=3):
    # Keep all pressure cells so summarize's overlap denominator remains honest;
    # remove ordinary observations from unsupported cells, forcing their exclusion.
    support=m[m.high.eq(0)].groupby(['player_id','comparison_cell']).agg(balls=('legal','sum'),matches=('match_id','nunique'))
    good=support[(support.balls>=cell_balls)&(support.matches>=cell_matches)].index
    ok=pd.MultiIndex.from_frame(m[['player_id','comparison_cell']]).isin(good)
    return m[m.high.eq(1)|ok].copy()

def reliability(s,m,scales,run_weight=.85,q=.2,consistency_weight=.25,min_overlap=.75,min_valid=.9):
    s=s.copy();w=run_weight
    def perf(v):return .75*(w*v[:,0]/scales[0]+(1-w)*v[:,1]/scales[1])+.25*(w*(v[:,0]-v[:,2])/scales[0]+(1-w)*(v[:,1]-v[:,3])/scales[1])
    for ix,r in s.iterrows():
        b=r.get('_bootstrap');
        if not isinstance(b,np.ndarray):continue
        v=perf(b);v=v[np.isfinite(v)]
        if len(v)==0:continue
        point=perf(np.array([[r.pressure_run_adj,r.pressure_wicket_adj,r.ordinary_run_adj,r.ordinary_wicket_adj]]))[0]
        # Reconstruct exactly the matched contexts used by summarize.
        a=m[m.player_id.eq(r.player_id)];sup=a[a.high.eq(0)].groupby('comparison_cell').agg(balls=('legal','sum'),matches=('match_id','nunique'))
        good=sup[(sup.balls>=10)&(sup.matches>=2)].index
        h=a[a.high.eq(1)&a.comparison_cell.isin(good)].groupby('match_id')[['legal','rr','wr']].sum();h=h[h.legal>0]
        qualities=w*100*h.rr/h.legal/scales[0]+(1-w)*100*h.wr/h.legal/scales[1]
        fraction=float(qualities.gt(0).mean())
        # Leave EACH match out, including its ordinary contribution; find the
        # largest loss in combined performance, not just largest run total.
        vals=[]
        from phase2_pipeline.core import summarize
        for mid in h.index:
            t=summarize(a[a.match_id.ne(mid)],min_matches=1,min_balls=1)
            if t.empty or 'pressure_run_adj' not in t:continue
            x=t.iloc[0]
            vals.append(perf(np.array([[x.pressure_run_adj,x.pressure_wicket_adj,x.ordinary_run_adj,x.ordinary_wicket_adj]]))[0])
        worst=float(np.nanmin(vals)) if len(vals) else np.nan;drop=max(0,point-worst)
        lcb=float(np.quantile(v,q));lo,hi=np.quantile(v,[.025,.975])
        for level in [.1,.2,.3]:s.loc[ix,f'lcb_{level}']=float(np.quantile(v,level))
        s.loc[ix,'unshrunk_performance']=point;s.loc[ix,'performance_lo95']=lo;s.loc[ix,'performance_hi95']=hi
        s.loc[ix,'performance_lcb']=lcb;s.loc[ix,'combined_positive_match_fraction']=fraction;s.loc[ix,'worst_leave_one_match_performance']=worst;s.loc[ix,'best_match_drop']=drop
        s.loc[ix,'reliability_score']=(1-consistency_weight)*lcb+consistency_weight*(2*fraction-1)-.25*drop
    s['reliability_eligible']=s.eligible & s.overlap_fraction.ge(min_overlap)&s.ordinary_matches.ge(8)&s.ordinary_legal.ge(100)&s.bootstrap_valid_uplift_fraction.ge(min_valid)&s.reliability_score.notna()
    s['revised_rank']=s.reliability_score.where(s.reliability_eligible).rank(ascending=False,method='min')
    return s.drop(columns=['_bootstrap'],errors='ignore')
