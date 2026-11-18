"""Run from the repository root with .venv/bin/python -m final_strategy.analyse."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from phase2_pipeline.core import identities, match_metrics, pressure_variant
from independent_audit.core import stratified_wickets, player_seed, estimate, quality, performance
from build_independent_audit import source_for

OUT=Path('outputs/historical_vote_submission'); E=OUT/'evidence'
ROLES={'bat':'batters','bowl':'bowlers'}

def sample(m,scales,role,asof,*,relaxed=False,eval_only=False,decay=False,draws=0,weight=.85,prior=8):
    """Construct each cell once, summarize all player-match comparisons, then resample matches."""
    o=m[m.high.eq(0)]
    minballs,minmatches=(10,2) if relaxed or eval_only else (20,3)
    support=o.groupby(['player_id','comparison_cell']).legal.agg(['sum',lambda x:(x>0).sum()])
    good=support[(support.iloc[:,0]>=minballs)&(support.iloc[:,1]>=minmatches)].index.intersection(pd.MultiIndex.from_frame(m.loc[m.high.eq(1),['player_id','comparison_cell']].drop_duplicates()))
    z=m[pd.MultiIndex.from_frame(m[['player_id','comparison_cell']]).isin(good)].copy()
    z['time_weight']=np.exp2(-(asof-z.date).dt.days/365.25) if decay else 1.
    z['quality_total']=weight*z.rr/scales[0]+(1-weight)*z.wr/scales[1]
    rows=[];rep={};allballs=m[m.high.eq(1)].groupby('player_id').legal.sum()
    # Same global match counts across candidates for paired comparisons.
    mids=np.array(sorted(m.match_id.unique())); counts=None
    if draws: counts=np.random.default_rng(2026).multinomial(len(mids),np.full(len(mids),1/len(mids)),size=draws)
    for pid,a in z.groupby('player_id'):
        mid,mi=np.unique(a.match_id,return_inverse=True);cells,ci=np.unique(a.comparison_cell,return_inverse=True)
        arr=np.zeros((len(mid),len(cells),2,3)); arr[mi,ci,a.high]=a[['legal','rr','wr']]
        dates=a.drop_duplicates('match_id').set_index('match_id').loc[mid,'date'];tw=np.exp2(-(asof-dates).dt.days.to_numpy()/365.25) if decay else np.ones(len(mid))
        high=arr[:,:,1,:];h=high.sum(axis=1);ob=arr[:,:,0,0];pos=h[:,0]>0
        if not pos.any():continue
        est=estimate((arr*tw[:,None,None,None]).sum(axis=0));ab,ordq=quality(est[:2],scales,weight),quality(est[2:],scales,weight)
        hq=quality(h[:,1:],scales,weight);oq=quality(arr[:,:,0,1:],scales,weight)
        wp=tw*np.minimum(h[:,0]/6,1);rates=(tw@oq)/(tw@ob);success=hq>high[:,:,0]@rates
        wins=wp@success;eff=wp.sum();smoothed=(wins+prior/2)/(eff+prior);fraction=wins/eff
        hb=h[:,0].sum();om=(ob.sum(axis=1)>0).sum();hm=pos.sum();ordinary=ob.sum();overlap=hb/allballs[pid]
        failure=[]
        for bad,msg in [(hm<(2 if eval_only else 8),'pressure matches'),(hb<(12 if eval_only else 50),'pressure balls'),(ordinary<(20 if eval_only else (50 if relaxed else 100)),'ordinary balls'),(om<(2 if eval_only else (5 if relaxed else 8)),'ordinary matches'),(overlap<(.5 if relaxed or eval_only else .75),'ordinary context overlap')]:
            if bad:failure.append(msg)
        lo=hi=valid=np.nan;legacy=np.nan
        if draws and not failure:
            cnt=counts[:,np.searchsorted(mids,mid)];c=cnt*tw;sample_h=(cnt@high[:,:,0]);co=c@ob
            ok=((co>0)|(sample_h==0)).all(axis=1)&(cnt@wp>0)
            rates_b=np.divide(c@oq,co,out=np.zeros_like(co),where=co>0)
            succ=hq[None,:]>(rates_b@high[:,:,0].T)
            val=(np.sum(cnt*wp*succ,axis=1)+prior/2)/(cnt@wp+prior);val[~ok]=np.nan
            valid=ok.mean();lo,hi=np.nanquantile(val,[.025,.975]);rep[pid]=val
            if valid<.9:failure.append('ordinary uncertainty')
            boot_est=estimate(np.einsum('bm,mchk->bchk',c,arr))
            A=performance(est,scales,weight);v=performance(boot_est,scales,weight)
            dv=performance(estimate(arr.sum(axis=0)-arr[pos]),scales,weight)
            legacy=.75*np.nanquantile(v,.2)+.25*(2*(hq[pos]>0).mean()-1)-.25*max(0,A-np.nanmin(dv))
        ipl=high[:,np.char.find(cells.astype(str),'|IPL')>=0,0].sum()
        phase={p:high[:,np.char.find(cells.astype(str),'|'+p+'|')>=0,0].sum() for p in ['powerplay','middle','death']}
        rows.append(dict(player_id=pid,score=smoothed,success_mass=wins,effective_votes=eff,positive_share=fraction,absolute=ab,ordinary=ordq,uplift=ab-ordq,run_high=est[0],wicket_high=est[1],run_ord=est[2],wicket_ord=est[3],hm=hm,hb=hb,om=om,ob=ordinary,overlap=overlap,first=str(dates[pos].min().date()),last=str(dates[pos].max().date()),IPL_balls=ipl,dominant_phase=max(phase,key=phase.get),dominant_balls=max(phase.values()),lo=lo,hi=hi,valid=valid,legacy=legacy,supported=not failure,reason=', '.join(failure),eligible=not failure and ab>0 and ab>ordq))
    s=pd.DataFrame(rows).set_index('player_id')
    s['rank']=s[s.eligible].sort_values(['score','uplift','player_id'],ascending=[False,False,True]).score.expanding().count()
    return s,rep

def run():
    E.mkdir(exist_ok=True,parents=True)
    d=pd.read_parquet('outputs/phase2/phase2_player_event_metrics.parquet')
    names=identities(Path('.'))[0].set_index('player_id').player_name
    dates=d.drop_duplicates('match_id').set_index('match_id').match_date_parsed
    last=dates.max();start=last-pd.Timedelta(days=364)
    d,_=stratified_wickets(d);g=source_for(d,'stratified');globald=source_for(d,'global')
    d.groupby([d.match_date_parsed.dt.year,'competition_type']).agg(recorded_matches=('match_id','nunique'),recorded=('delivery_id','size'),scored=('pressure_eligible','sum'),analysable=('analysis_available','sum')).to_csv(E/'calendar_coverage.csv')
    a=d[d.analysis_available].copy()
    a.groupby(['innings','phase_pre','competition_type','stage']).high_pressure_selected.agg(['size','sum','mean']).to_csv(E/'pressure_concentration.csv')
    example=a[a.high_pressure_selected&a.innings.eq(2)].iloc[0]
    example[['delivery_id','match_date_parsed','runs_required_pre','legal_balls_remaining_pre','wickets_remaining_pre','win_probability_pre','contestability','leverage','urgency','match_stakes','pressure_score_v1']].to_json(E/'delivery_example.json',date_format='iso')
    assert (a.expected_train_end<a.match_date_parsed).all() and (a.model_train_end_date<a.match_date_parsed).all()
    back=[];sens=[];pairs=[];meta={}
    def metrics(source,role,high):
        m=match_metrics(source,role,high);m['date']=m.match_id.map(dates);return m
    for role,label in ROLES.items():
        print('Testing',label,flush=True)
        old=pd.read_csv(f'outputs/phase2/phase2_{label}.csv');scales=old.loc[old.eligible,['pressure_run_adj','pressure_wicket_adj']].std().to_numpy()
        meta[role]=list(scales)
        orig=pd.read_csv(f'outputs/phase3/final_{label}_top10.csv').set_index('player_id')
        default=g.high_pressure_selected;m=metrics(g,role,default)
        base,reps=sample(m,scales,role,last,draws=2000)
        recent,_=sample(m[m.date>=start],scales,role,last,draws=2000)
        earlier,_=sample(m[m.date<start],scales,role,start-pd.Timedelta(days=1),draws=2000)
        dec,_=sample(m,scales,role,last,decay=True,draws=2000)
        compare=base[['rank','score','eligible']].copy()
        for nm,s in [('recent',recent),('decay',dec)]:
            compare[nm+'_rank']=s['rank'];compare[nm+'_score']=s.score
        compare['combined_score']=(compare.score+compare.recent_score)/2
        compare['combined_rank']=compare.combined_score.where(compare.eligible&compare.recent_rank.notna()).rank(ascending=False,method='min')
        compare['phase3_rank']=pd.read_csv(f'outputs/phase3/{label}_comparison_all.csv').set_index('player_id').revised_rank
        compare.to_csv(E/f'{role}_method_comparison.csv')
        # Forward periods use train-only pressure cutoffs and scaling. All test labels and
        # baselines are calculated only for retrospective evaluation of the later window.
        for year in [2024,2025,2026]:
            cut=pd.Timestamp(year=year,month=1,day=1);pred_end=cut-pd.Timedelta(days=1)
            cutoff=d.loc[d.pressure_eligible&d.match_date_parsed.lt(cut),'pressure_score_v1'].quantile(.75)
            labels=d.pressure_eligible&d.pressure_score_v1.ge(cutoff)
            mm=metrics(g,role,labels);tr=mm[mm.date<cut];te=mm[mm.date.dt.year==year]
            h=tr[tr.high.eq(1)].groupby('player_id')[['rr','wr','legal']].sum().query('legal>=50')
            scale=(h[['rr','wr']].div(h.legal,axis=0)*100).std().to_numpy()
            future,_=sample(te,scale,role,te.date.max(),eval_only=True)
            hist,_=sample(tr,scale,role,pred_end,draws=300)
            recent_tr,_=sample(tr[tr.date>=cut-pd.Timedelta(days=365)],scale,role,pred_end,draws=300)
            decay_tr,_=sample(tr,scale,role,pred_end,decay=True,draws=300)
            gm=metrics(globald,role,labels);leg,_=sample(gm[gm.date<cut],scale,role,pred_end,draws=300)
            preds=pd.DataFrame({'historical':hist.score.where(hist.supported),'recent':recent_tr.score.where(recent_tr.supported),'decay':decay_tr.score.where(decay_tr.supported),'phase3_proxy':leg.legacy.where(leg.supported)})
            preds['combined']=(preds.historical+preds.recent)/2
            preds['observed']=future.positive_share.where(future.supported)
            for cohort,group in [('available',preds),('common',preds.dropna())]:
                for method in ['historical','recent','decay','combined','phase3_proxy']:
                    q=group[[method,'observed']].dropna();n=len(q)
                    back.append(dict(role=role,year=year,cohort=cohort,method=method,n=n,cutoff=cutoff,rho=q.rank().corr().iloc[0,1] if n>=3 else np.nan,mse=((q[method]-q.observed)**2).mean() if method!='phase3_proxy' else np.nan))
            preds.to_csv(E/f'{role}_forward_{year}.csv')
        variants=[(25,.1,True),(20,.1,True),(30,.1,True),(25,.1,False),(25,.05,True),(25,.15,True)]
        for pct,scale,stakes in variants:
            key=f'top{pct}_scale{scale}_stakes{int(stakes)}'
            if (pct,scale,stakes)==variants[0]:s=base
            else:
                high,_=pressure_variant(g,pct,scale,stakes);s,_=sample(metrics(g,role,high),scales,role,last,draws=2000)
            sens.append(s[['rank','score','supported','eligible','hb','hm']].assign(role=role,variant=key).reset_index())
        for nm,src,w,relax,prior in [('relaxed',g,.85,True,8),('runs_only',g,1,False,8),('contextual',d,.85,False,8),('global',globald,.85,False,8),('unshrunk',g,.85,False,0),('70_30',g,.7,False,8)]:
            s,_=sample(metrics(src,role,default),scales,role,last,weight=w,relaxed=relax,draws=2000,prior=prior);compare[nm+'_rank']=s['rank']
        compare.to_csv(E/f'{role}_method_comparison.csv')
        full=base.join(recent[['hm','hb','score','supported','positive_share','uplift','absolute']].add_prefix('recent_')).join(earlier[['score','supported','uplift','absolute']].add_prefix('early_'))
        srcid='striker_id' if role=='bat' else 'bowler_id'
        registry=names.index.union(d[srcid].dropna().unique())
        full=full.reindex(registry);full.index.name='player_id';full['player_name']=names.reindex(full.index)
        counts=d.groupby(srcid).agg(source_events=('delivery_id','size'),analysable_events=('analysis_available','sum'),source_first=('match_date_parsed','min'),source_last=('match_date_parsed','max'))
        full=full.join(counts);full['phase3_rank']=compare.phase3_rank
        full['recent_status']='insufficient supported recent evidence'
        both=full.recent_supported.eq(True)&full.early_supported.eq(True)
        full.loc[both,'recent_status']=np.where(full.loc[both,'recent_positive_share']>.5,'positive recent repeatability','weaker recent repeatability')
        full.loc[both&(full.recent_score>full.early_score)&(full.early_score<=.5),'recent_status']='recently improving descriptively'
        full.loc[both&(full.recent_score<full.early_score)&(full.early_score>.5)&(full.recent_score<=.5),'recent_status']='possible decline descriptively'
        full.loc[full.recent_supported.eq(True)&~both,'recent_status']='recent evidence only or early comparison sparse'
        def reason(r):
            if pd.isna(r.source_events):return 'no recorded appearance in role'
            if not r.analysable_events:return 'no trustworthy scored and outcome-supported events'
            if pd.isna(r.hm):return 'no matched pressure exposure or supported ordinary cells'
            if r.supported==False:return r.reason
            if r.absolute<=0:return 'below contextual expectation in pressure'
            if r.uplift<=0:return 'nonpositive pooled own-baseline uplift'
            return 'selected' if r['rank']<=10 else 'lower repeatability score than cutoff'
        full['selection_reason']=full.apply(reason,axis=1)
        sg=pd.concat([v for v in sens if v.role.iloc[0]==role]);full['pressure_specs_top10']=sg.assign(top=sg['rank'].le(10)).groupby('player_id').top.sum()
        full['pressure_specs_eligible']=sg.groupby('player_id').eligible.sum()
        full['confidence']=np.where(full.lo.gt(.5)&full.pressure_specs_top10.ge(5),'stronger descriptive evidence','provisional order')
        full.to_csv(E/f'{role}_all_players.csv')
        sel=full[full['rank'].le(10)].sort_values('rank').copy();assert len(sel)==10
        sel['selection_note']=sel.apply(lambda r:f"{r.dominant_phase.capitalize()} overs supply {int(r.dominant_balls)} of {int(r.hb)} matched balls. Beats own baseline in {r.positive_share:.0%} of weighted appearances.",axis=1)
        sel.to_csv(OUT/f'final_{label}_top10.csv')
        union=orig.index.union(sel.index);full.loc[union].join(compare[['global_rank','relaxed_rank','unshrunk_rank','recent_rank','decay_rank']]).to_csv(E/f'{role}_selection_changes.csv')
        watch=full[full.supported.eq(False)&full.hm.ge(4)&full.hb.ge(25)&full.ob.ge(40)&full.overlap.ge(.50)&full.absolute.gt(0)&full.uplift.gt(0)&full.positive_share.gt(.5)].nlargest(5,'score')
        watch.to_csv(E/f'{role}_watchlist.csv')
        ordered=full[full['rank'].le(13)].sort_values('rank').index
        for left,right in zip(ordered[:-1],ordered[1:]):
            v=reps[left]-reps[right];lo,hi=np.nanquantile(v,[.025,.975]);pairs.append(dict(role=role,left=left,right=right,lo=lo,hi=hi,valid=np.isfinite(v).mean()))
        # Role gate diagnostics include role participants with analysable pressure balls.
        high=a[a.high_pressure_selected].copy();high['specialist_evidence']=high.batting_position_reported.between(6,11) if role=='bat' else high.phase_pre.eq('death')
        highs=high[high.legal_event.eq(1)].groupby(srcid).specialist_evidence.agg(['mean','size']).join(full[['supported','eligible','overlap','hb']]);highs['group']=np.where(highs['mean']>=.5,'at least half specialist opportunities','other observed usage')
        highs[highs['size']>=50].groupby('group').agg(players=('size','size'),supported=('supported','sum'),eligible=('eligible','sum'),median_overlap=('overlap','median')).to_csv(E/f'{role}_specialist_support.csv')
        print(label,sel.player_name.tolist(),flush=True)
    pd.concat(sens).to_csv(E/'pressure_sensitivity.csv',index=False);pd.DataFrame(back).to_csv(E/'backtest.csv',index=False);pd.DataFrame(pairs).to_csv(E/'adjacent_differences.csv',index=False)
    (E/'settings.json').write_text(json.dumps(dict(scales=meta,latest_source_date=str(last.date()),recent_start=str(start.date()),seed=2026,bootstrap=2000),indent=2))
if __name__=='__main__':run()
