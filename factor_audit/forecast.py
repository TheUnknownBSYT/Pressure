"""Fit earlier-date response ablations and export paired match diagnostics."""
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from final_strategy import response_model as response
from .features import competition_family, feature_sets, locations

OUT = Path('outputs/factor_audit')
SOURCE = Path('outputs/final_submission/evidence')


def fixtures():
    m = pd.read_csv('matches.csv').merge(pd.read_csv('competitions.csv'),on='comp_id',validate='many_to_one')
    m['family'] = competition_family(m.comp_name)
    m['location'] = locations(m)
    m['stage_detail'] = m.stage.mask(m.stage.eq('Others'), 'unknown round')
    return m


def history_within_ipl(d):
    """Replace performance-history fields with estimates based solely on IPL observations."""
    sub = d.loc[d.competition_type.eq('IPL')].reset_index(drop=True)
    histories = {role: response.long_history(sub,role)[0] for role in response.IDS}
    by_role = {}
    for role in response.IDS:
        opp = histories[response.OTHER[role]]
        by_role[role] = pd.DataFrame({
            'role_phase_share': histories[role].role_phase_share, 'venue_run': histories['bat'].venue_run,
            'opponent_run':opp.individual_run, 'opponent_wicket':opp.individual_wicket,
            'opponent_team_run':opp.team_run, 'opponent_team_wicket':opp.team_wicket})
    return by_role


def comparison(frame, metric):
    delta = frame[metric].to_numpy()
    rng = np.random.default_rng(934)
    return [delta.mean(), *np.quantile(rng.choice(delta,(2000,len(delta))).mean(1),[.025,.975])]


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    m = fixtures()
    m[['match_id','comp_name','match_number','stage','stage_detail','knock_out','family','location',
       'venue_country','venue_id','home_team','away_team']].to_csv(OUT/'fixtures.csv',index=False)
    d,x,_ = response.build()
    specs = feature_sets(d,x,m)
    ipl = d.competition_type.eq('IPL').to_numpy()
    local = history_within_ipl(d)
    original = pd.read_parquet(SOURCE/'response_events.parquet').set_index('delivery_id')
    events, losses, calibrations = [], [], []
    for start,end in response.FOLDS:
        train = d[response.DATE].lt(start).to_numpy()
        test = (d[response.DATE].between(start,end,inclusive='left') & d.pressure_eligible).to_numpy()
        t = d[test].copy()
        assert d.loc[train,response.DATE].max() < t[response.DATE].min()
        context = original.loc[t.delivery_id]
        out = t[['delivery_id','match_id',response.DATE,'comparison_cell','competition_type','legal_event',
                 'striker_id','bowler_id']].reset_index(drop=True)
        for role in response.IDS:
            runs, wickets = d[role+'_runs'].to_numpy(),d[response.WICKET[role]].to_numpy()
            for name,cols in specs.items():
                tr, te, features = train, test, x[role]
                if name=='scope':
                    tr = train & d.competition_type.isin(['IPL','T20I']).to_numpy()
                if name=='ipl_only':
                    tr,te = train & ipl, test & ipl
                    if not te.any():
                        out[role+'_'+name] = np.nan
                        continue
                    features = features.copy()
                    hist = local[role]
                    features.loc[ipl,hist.columns] = hist.to_numpy()
                subset = d.loc[te]
                rr,pp,_ = response.model_fit(features,runs,wickets,tr,te,cols)
                selected = context.loc[subset.delivery_id]
                # Shared resource prices and exposure conversion isolate the response adjustment.
                exposure = subset.legal_event.to_numpy()/selected.historical_legal_probability.to_numpy()
                value = (1 if role=='bat' else -1)*(runs[te]-rr*exposure+
                    selected.wicket_price.to_numpy()*(pp*exposure-wickets[te]))
                out[role+'_'+name] = out.delivery_id.map(pd.Series(value,index=subset.delivery_id))
                if name=='published':
                    np.testing.assert_allclose(value,selected[role+'_value'],atol=1e-10)
                errors = response.errors(subset,rr,pp,role)
                errors['competition'] = subset.competition_type.to_numpy()
                errors = errors[errors.competition.isin(['IPL','T20I'])]
                measure = errors.groupby(['match_id','competition']).mean().reset_index()
                measure['fold'],measure['role'],measure['model']=start,role,name
                losses.append(measure)
                for (competition,bin_id),z in errors.groupby(['competition',pd.cut(errors.expected_wickets,
                        [0,.025,.05,.075,.1,.15,1],include_lowest=True)],observed=True):
                    calibrations.append(dict(fold=start,role=role,model=name,competition=competition,
                        bin=str(bin_id),events=len(z),predicted=z.expected_wickets.mean(),observed=z.observed_wickets.mean()))
        events.append(out)
        print('Completed response fold',start,flush=True)
    pd.concat(events).to_parquet(OUT/'predictions.parquet',index=False)
    loss = pd.concat(losses,ignore_index=True)
    loss.to_csv(OUT/'match_forecasts.csv',index=False)
    pd.DataFrame(calibrations).to_csv(OUT/'wicket_calibration.csv',index=False)
    loss['period']=np.where(loss.fold.str.startswith('2026'),'2026','development')
    rows=[]
    refs={'competition':['no_opp'],'team':['competition'],'actual':['competition'],'both':['published','team','actual','competition'],
          'decayed':['both'],'published':['no_opp'],'scope':['published'],'home':['published'],'form':['published'],'ipl_only':['published']}
    for (role,period),block in loss.groupby(['role','period']):
        for model,baselines in refs.items():
            for ref in baselines:
                for comp in ['combined','IPL','T20I']:
                    part = block if comp=='combined' else block[block.competition.eq(comp)]
                    a=part[part.model.eq(model)].set_index('match_id')
                    b=part[part.model.eq(ref)].set_index('match_id')
                    ids=a.index.intersection(b.index)
                    if not len(ids): continue
                    row=dict(role=role,period=period,model=model,reference=ref,competition=comp,matches=len(ids))
                    for metric in ['squared_run_error','brier','wicket_log_loss']:
                        key={'squared_run_error':'run','brier':'brier','wicket_log_loss':'logloss'}[metric]
                        row[key],row[key+'_lo'],row[key+'_hi']=comparison(a.loc[ids,[metric]]-b.loc[ids,[metric]],metric)
                    rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'forecast_comparison.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=1): run()
