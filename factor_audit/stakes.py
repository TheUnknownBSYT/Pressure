"""Vary only the stakes term and compare it with subsequent absolute win-probability movement."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from final_strategy import consequence as cq
from final_strategy.ratings import thresholds
from .forecast import OUT, SOURCE, fixtures


def pressure_scores(d):
    core = (.50*d.leverage + .30*d.urgency*d.contestability)/.8
    s = d.match_stakes
    bilateral = d.family.eq('bilateral') & d.stage.isin(['League','Others']) & d.knock_out.eq(0)
    scores = {'original': d.pressure_score_v1, 'no_stakes':100*core,
            'interaction':100*core*(.8+.2*s), 'half_bonus':100*(.9*core+.1*s),
            'leverage_only':100*d.leverage}
    for label, weight in [('consistent_series',.30), ('series_mid',.35), ('series_league',.40)]:
        scores[label] = 100*(.50*d.leverage+.30*d.urgency*d.contestability+.20*s.mask(bilateral,weight))
    return scores


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    full = pd.read_parquet('outputs/pressure_scored.parquet', columns=cq.COLS+['stage','knock_out'])
    d = full.loc[full.pressure_eligible].copy()
    coefs = pd.read_csv('outputs/model_coefficients.csv',parse_dates=['fold']).set_index(['fold','innings','feature'])
    for fold,block in d[d.legal_event.eq(1)].groupby('model_fold_start'):
        hist = full[full.training_label_available & full.match_date_parsed.lt(fold)]
        model = cq.FittedWP(coefs,hist,fold)
        for inning,group in block.groupby('innings'):
            d.loc[group.index,'actual'] = cq.updated(group,model,inning)
    labels=d[['delivery_id','match_id','match_date_parsed']].copy()
    d['family'] = d.match_id.map(fixtures().set_index('match_id').family)
    scores=pressure_scores(d)
    d['near_settled']=d.contestability.lt(.2)
    counts,metrics,by_match=[],[],[]
    for name,score in scores.items():
        high=thresholds(d[['match_date_parsed']].assign(pressure_static=score),.75)
        labels[name]=high
        d[name]=score
        if name=='original':
            archived=pd.read_parquet(SOURCE/'consequence_events.parquet').set_index('delivery_id')
            pd.testing.assert_series_equal(high.reset_index(drop=True),
                archived.loc[d.delivery_id,'high_static'].reset_index(drop=True),check_names=False)
        observed=d[high.notna()].copy()
        observed['high']=high.dropna().astype(int)
        observed['flipped']=high.ne(labels.original).loc[observed.index].astype(int)
        summary=observed.groupby(['family','stage','near_settled']).agg(events=('high','size'),matches=('match_id','nunique'),
            high=('high','sum'),flipped=('flipped','sum')).reset_index()
        counts.append(summary.assign(model=name))
    for year in [2024,2025,2026]:
        for sample,mask in [('all',d.innings.gt(0)),('chases',d.innings.eq(2))]:
            legal=d[d.actual.notna() & mask & d.competition_type.isin(['IPL','T20I'])]
            train=legal[legal.match_date_parsed.dt.year<year]
            test=legal[legal.match_date_parsed.dt.year.eq(year)]
            errors={}
            for name in scores:
                fit=LinearRegression().fit(train[[name]],train.actual,sample_weight=cq.mass(train))
                pred=fit.predict(test[[name]]).clip(0,1)
                errors[name]=((test.actual-pred)**2).groupby(test.match_id).mean()
            for name,err in errors.items():
                diff=err-errors['original']
                rng=np.random.default_rng(151)
                lo,hi=np.quantile(rng.choice(diff,(2000,len(diff))).mean(1),[.025,.975])
                metrics.append(dict(year=year,sample=sample,model=name,matches=len(err),balls=len(test),
                    mse=err.mean(),delta=diff.mean(),lo=lo,hi=hi))
                by_match.append(err.rename('mse').reset_index().assign(year=year,sample=sample,model=name))
    labels.to_parquet(OUT/'pressure_labels.parquet',index=False)
    pd.concat(counts).to_csv(OUT/'stage_exposure.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'stakes_validation.csv',index=False)
    pd.concat(by_match).to_csv(OUT/'stakes_matches.csv',index=False)
    full['year']=full.match_date_parsed.dt.year
    full.groupby(['year','competition_type']).agg(matches=('match_id','nunique'),events=('match_id','size'),
        scored=('pressure_eligible','sum')).reset_index().to_csv(OUT/'coverage.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=1): run()
