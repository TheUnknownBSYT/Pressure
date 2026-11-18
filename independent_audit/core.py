"""Fixed-support match-cluster calculations for the independent review.

No names enter this module. Quantiles condition on observed comparison support;
failed replicates are retained as missing, never as neutral performance.
"""
import hashlib
import numpy as np
import pandas as pd


def player_seed(seed, role, pid):
    return int.from_bytes(hashlib.sha256(f'{seed}|{role}|{pid}'.encode()).digest()[:8], 'little')


def estimate(s):
    """(..., cell, high/ordinary, legal/run/wicket) -> four rates per 100."""
    s = np.asarray(s, dtype=float)
    h = s[..., :, 1, :]; o = s[..., :, 0, :]
    hb = h[..., 0].sum(axis=-1)
    weights = np.divide(h[..., 0], hb[..., None], out=np.zeros_like(h[..., 0]), where=hb[..., None]>0)
    high = np.divide(100*h[..., 1:].sum(axis=-2), hb[..., None], out=np.full(h.shape[:-2]+(2,), np.nan), where=hb[..., None]>0)
    rates = np.divide(o[..., 1:], o[..., 0, None], out=np.zeros_like(o[..., 1:]), where=o[..., 0, None]>0)
    ordinary = 100*(weights[..., None]*rates).sum(axis=-2)
    invalid = (hb<=0) | ((o[..., 0]<=0) & (h[..., 0]>0)).any(axis=-1)
    ordinary = np.where(invalid[..., None], np.nan, ordinary)
    return np.concatenate([high, ordinary], axis=-1)


def quality(v, scales, weight=.85):
    return weight*v[..., 0]/scales[0] + (1-weight)*v[..., 1]/scales[1]


def performance(v, scales, weight=.85):
    return .75*quality(v[..., :2], scales, weight)+.25*quality(v[..., :2]-v[..., 2:], scales, weight)


def prepare(a, cell_balls=20, cell_matches=3, positive_match_counts=True):
    ordinary = a[a.high.eq(0)]
    support = ordinary.groupby('comparison_cell').agg(balls=('legal','sum'), matches=('match_id','nunique'))
    if positive_match_counts:
        support['matches'] = ordinary[ordinary.legal.gt(0)].groupby('comparison_cell').match_id.nunique()
    good = support[(support.balls>=cell_balls)&(support.matches>=cell_matches)].index.intersection(a.loc[a.high.eq(1),'comparison_cell'].unique())
    z = a[a.comparison_cell.isin(good)]
    matches = sorted(z.match_id.unique()); cells = sorted(good)
    arr = np.zeros((len(matches),len(cells),2,3)); mi={v:i for i,v in enumerate(matches)}; ci={v:i for i,v in enumerate(cells)}
    for r in z.itertuples():
        arr[mi[r.match_id],ci[r.comparison_cell],r.high] += [r.legal,r.rr,r.wr]
    h=z[z.high.eq(1)];o=z[z.high.eq(0)];hb=h.legal.sum();ob=o.legal.sum();total=a.loc[a.high.eq(1),'legal'].sum()
    def nmatch(t):
        return int(t.groupby('match_id').legal.sum().gt(0).sum()) if positive_match_counts else t.match_id.nunique()
    row=dict(player_id=a.player_id.iloc[0],pressure_legal_all=total,pressure_legal=hb,ordinary_legal=ob,
             pressure_matches=nmatch(h),ordinary_matches=nmatch(o),pressure_event_matches=h.match_id.nunique(),
             ordinary_event_matches=o.match_id.nunique(),overlap_fraction=hb/total if total else 0,comparison_cells=len(good))
    row['base_eligible'] = row['pressure_matches']>=8 and row['ordinary_matches']>=5 and hb>=50 and ob>=50 and row['overlap_fraction']>=.5
    return row,arr,matches,cells,z


def gate(r, overlap=.75, valid=.9):
    return bool(r['base_eligible'] and r['overlap_fraction']>=overlap and r['ordinary_matches']>=8 and r['ordinary_legal']>=100 and r.get('bootstrap_valid_fraction',0)>=valid and np.isfinite(r.get('R',np.nan)))


def exclusions(r):
    checks=[(r['pressure_matches']<8,'fewer than 8 positive-exposure pressure matches'),(r['pressure_legal']<50,'fewer than 50 matched pressure legal balls'),
            (r['ordinary_matches']<8,'fewer than 8 positive-exposure ordinary matches'),(r['ordinary_legal']<100,'fewer than 100 ordinary legal balls'),
            (r['overlap_fraction']<.75,'retained pressure overlap below 75%'),(r.get('bootstrap_valid_fraction',0)<.9,'valid bootstrap comparisons below 90%')]
    return '; '.join(label for fail,label in checks if fail)


def summarize_fixed(m,scales,role,bootstrap=2000,seed=42,cell_balls=20,cell_matches=3,weight=.85,positive_match_counts=True):
    rows=[];packs={}
    for pid,a in m.groupby('player_id',sort=True):
        r,arr,mids,cells,z=prepare(a,cell_balls,cell_matches,positive_match_counts)
        if not len(mids) or r['pressure_legal']<=0 or r['ordinary_legal']<=0:
            r.update(A=np.nan,R=np.nan,bootstrap_valid_fraction=0.,eligible=False,exclusion_reason=exclusions(r));rows.append(r);continue
        total=arr.sum(axis=0);est=estimate(total);A=float(performance(est,scales,weight))
        r.update(A=A,absolute_quality=float(quality(est[:2],scales,weight)),uplift_quality=float(quality(est[:2]-est[2:],scales,weight)),
                 pressure_run_adj=est[0],pressure_wicket_adj=est[1],ordinary_run_adj=est[2],ordinary_wicket_adj=est[3],run_uplift=est[0]-est[2],wicket_uplift=est[1]-est[3])
        h=arr[:,:,1,:].sum(axis=1);positive=h[:,0]>0
        rate=np.divide(100*h[:,1:],h[:,0,None],out=np.full((len(h),2),np.nan),where=positive[:,None]);q=quality(rate,scales,weight)
        o=total[:,0,:];orate=np.divide(o[:,1:],o[:,0,None],out=np.zeros_like(o[:,1:]),where=o[:,0,None]>0)
        mix=np.divide(arr[:,:,1,0],h[:,0,None],out=np.zeros((len(h),len(cells))),where=positive[:,None])
        ordinary_quality=quality(100*mix@orate,scales,weight);uplift=q-ordinary_quality
        r['f_absolute']=float((q[positive]>0).mean());r['f_uplift']=float((uplift[positive]>0).mean())
        r['illegal_only_pressure_matches']=int(((h[:,0]==0)&(np.abs(h[:,1:]).sum(axis=1)>0)).sum())
        for minimum in [3,6,12]:
            valid=h[:,0]>=minimum;weights=np.minimum(h[positive,0]/minimum,1)
            r[f'brief_matches_lt{minimum}']=int(((h[:,0]<minimum)&positive).sum())
            r[f'brief_balls_lt{minimum}']=float(h[(h[:,0]<minimum)&positive,0].sum())
            r[f'f_absolute_min{minimum}']=float((q[valid]>0).mean()) if valid.any() else np.nan
            r[f'f_uplift_min{minimum}']=float((uplift[valid]>0).mean()) if valid.any() else np.nan
            r[f'f_absolute_downweight{minimum}']=float(np.average(q[positive]>0,weights=weights))
            r[f'f_uplift_downweight{minimum}']=float(np.average(uplift[positive]>0,weights=weights))
        pressure_event_matches=set(z.loc[z.high.eq(1),'match_id'])
        deletion_mask=np.array([mid in pressure_event_matches for mid in mids])
        deletion=estimate(total[None,...]-arr[deletion_mask]);dv=performance(deletion,scales,weight)
        r['loo_valid_fraction']=float(np.isfinite(dv).mean());r['D']=max(0.,A-float(np.nanmin(dv))) if np.isfinite(dv).any() else np.nan
        # Diagnose the legacy 10/2 re-selection on exactly the initial strong cells.
        dyn=[];changes=0
        ordinary_presence=np.zeros(arr.shape[:2],dtype=bool)
        mid_index={v:i for i,v in enumerate(mids)};cell_index={v:i for i,v in enumerate(cells)}
        for item in z[z.high.eq(0)].itertuples():
            ordinary_presence[mid_index[item.match_id],cell_index[item.comparison_cell]]=True
        for j in np.flatnonzero(deletion_mask):
            remain=total-arr[j];n=ordinary_presence.sum(axis=0)-ordinary_presence[j]
            keep=(remain[:,0,0]>=10)&(n>=2)
            changes+=int(not keep.all())
            dyn.append(float(performance(estimate(remain[keep]),scales,weight)))
        r['loo_cohort_change_fraction']=changes/int(deletion_mask.sum())
        r['D_dynamic10_2']=max(0.,A-float(np.nanmin(dyn))) if np.isfinite(dyn).any() else np.nan
        r['simple_score']=r['pressure_matches']/(r['pressure_matches']+8)*A
        # Compute uncertainty even below eligibility so missing comparisons remain visible.
        rng=np.random.default_rng(player_seed(seed,role,pid));counts=rng.multinomial(len(mids),np.full(len(mids),1/len(mids)),size=bootstrap)
        draws=estimate(np.einsum('bn,nchk->bchk',counts,arr,optimize=False));v=performance(draws,scales,weight);finite=np.isfinite(v);f=v[finite]
        r['bootstrap_valid_fraction']=float(finite.mean());r['bootstrap_zero_pressure_fraction']=float(np.isnan(draws[:,0]).mean())
        if len(f):
            r['Q10'],r['Q20'],r['Q30']=np.quantile(f,[.1,.2,.3]);r['A_lo95'],r['A_hi95']=np.quantile(f,[.025,.975])
            uq=quality(draws[:,:2]-draws[:,2:],scales,weight);r['uplift_lo95'],r['uplift_hi95']=np.nanquantile(uq,[.025,.975])
            # Worst/best placement of unidentifiable resamples is a quantile bound,
            # NOT imputed performance and NOT a confidence interval.
            worst=np.where(finite,v,-np.inf);best=np.where(finite,v,np.inf)
            k=int(np.floor(.2*(bootstrap-1)));r['Q20_missing_worst_bound']=np.sort(worst)[k];r['Q20_missing_best_bound']=np.sort(best)[k]
            r['uncertainty_gap']=A-r['Q20'];r['R']=.75*r['Q20']+.25*(2*r['f_absolute']-1)-.25*r['D']
            r['component_point']=.75*A;r['component_uncertainty']=-.75*r['uncertainty_gap'];r['component_consistency']=.25*(2*r['f_absolute']-1);r['component_deletion']=-.25*r['D']
        else:r['R']=np.nan
        r['eligible']=gate(r);r['exclusion_reason']=exclusions(r);rows.append(r)
        packs[pid]=dict(arr=arr,matches=mids,cells=cells,draws=draws,values=v,match_quality=q,match_uplift=uplift,pressure_balls=h[:,0])
    s=pd.DataFrame(rows)
    for score in ['R','A','simple_score']:
        s[score+'_rank']=s[score].where(s.eligible).rank(method='min',ascending=False)
    return s,packs


def stratified_wickets(d,prior=200):
    """Earlier-quarter phase x competition rates pooled toward global history."""
    d=d.copy();folds=[]
    for period in sorted(d.match_date_parsed.dt.to_period('Q').unique()):
        tr=d[d.standard_t20_eligible&d.outcome_valid&d.match_date_parsed.lt(period.start_time)]
        te=d[d.analysis_available&d.match_date_parsed.dt.to_period('Q').eq(period)]
        if te.empty:continue
        assert tr.match_date_parsed.max()<te.match_date_parsed.min()
        for role,outcome in [('bat','bat_dismissal'),('bowl','bowl_wicket')]:
            global_rate=tr[outcome].sum()/tr.legal_event.sum()
            pool=tr.groupby(['phase_pre','competition_type']).agg(total=(outcome,'sum'),balls=('legal_event','sum'))
            pool['rate']=(pool.total+prior*global_rate)/(pool.balls+prior)
            rates=te[['phase_pre','competition_type']].merge(pool[['rate']],left_on=['phase_pre','competition_type'],right_index=True,how='left',validate='many_to_one').rate.fillna(global_rate)
            d.loc[te.index,role+'_stratified_wicket_rate']=rates.to_numpy()
            d.loc[te.index,role+'_global_wicket_rate']=global_rate
        folds.append(dict(fold=str(period),train_end=str(tr.match_date_parsed.max()),test_start=str(te.match_date_parsed.min()),train_matches=tr.match_id.nunique(),test_matches=te.match_id.nunique()))
    return d,pd.DataFrame(folds)
