"""Targeted Phase 3; prior outputs are read-only and hash-checked."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from phase2_pipeline.core import summarize,rank,match_metrics,identities
from phase3_pipeline.audit import calibration_audit,prior_wickets,supported_matches,reliability
from pressure_pipeline.data import sha256
from pressure_pipeline.report import md_table

def main():
    out=Path('outputs/phase3');out.mkdir(parents=True,exist_ok=True)
    protected=[p for p in Path('outputs').rglob('*') if p.is_file() and out not in p.parents]+list(Path('.').glob('*.csv'))
    hashes={str(p):sha256(p) for p in protected}
    d=pd.read_parquet('outputs/phase2/phase2_player_event_metrics.parquet')
    a,f,cal,b,decision=calibration_audit(d)
    assert not decision['adopt'], 'Adoption gate passed: full observed and counterfactual recalculation now required; do not recommend unchanged pressure.'
    a[['delivery_id','match_id','match_date_parsed','win_probability_pre','calibrated','calibration_train_end','batting_win_label_outcome']].to_parquet(out/'calibration_predictions.parquet',index=False)
    for name,t in [('calibration_folds',f),('calibration_metrics',cal),('calibration_bins',b)]:t.to_csv(out/f'{name}.csv',index=False)
    (out/'calibration_decision.json').write_text(json.dumps(decision,indent=2))
    raw=pd.read_csv('ball_by_ball.csv',keep_default_na=False,na_values=[''])
    raw['delivery_id']=raw.match_id.astype(str)+':'+raw.innings.astype(str)+':'+raw.innings_ball_num.astype(str)
    check=d[['delivery_id','bat_dismissal','bowl_wicket']].merge(raw[['delivery_id','is_wicket','dismissed_player_id','striker_id','dismissal_type']],on='delivery_id',validate='one_to_one')
    assert check.bat_dismissal.eq((check.is_wicket&check.dismissed_player_id.eq(check.striker_id)).astype(int)).all()
    assert check.bowl_wicket.eq((check.is_wicket&check.dismissal_type.isin(['Caught','Bowled','LBW','Stump','Hit Wicket'])).astype(int)).all()
    check[check.is_wicket].groupby('dismissal_type').agg(events=('delivery_id','size'),striker_dismissals=('bat_dismissal','sum'),bowler_credited=('bowl_wicket','sum')).to_csv(out/'wicket_attribution_audit.csv')
    prior,errors=prior_wickets(d);errors.to_csv(out/'wicket_model_match_errors.csv',index=False)
    validation=[];rng=np.random.default_rng(42)
    for role,g in errors.groupby('role'):
        diff=(g.contextual_error-g.prior_error).to_numpy();boot=rng.choice(diff,(2000,len(diff)),replace=True).mean(axis=1)
        validation.append(dict(role=role,matches=len(g),contextual_mse=g.contextual_error.mean(),prior_mse=g.prior_error.mean(),difference=diff.mean(),lo95=np.quantile(boot,.025),hi95=np.quantile(boot,.975)))
    validation=pd.DataFrame(validation);validation.to_csv(out/'wicket_model_validation.csv',index=False)
    names,_=identities(Path('.'));scenarios=[];results={};support=[];checks={}
    for role,label in [('bat','batters'),('bowl','bowlers')]:
        print('Auditing',label,flush=True)
        old=pd.read_csv(f'outputs/phase2/phase2_{label}.csv')
        scales=[old.loc[old.eligible,c].std() for c in ['pressure_run_adj','pressure_wicket_adj']]
        for expectation in ['contextual','prior']:
            source=d if expectation=='contextual' else prior.copy()
            if expectation=='prior':
                source['bat_wicket_residual']=source.bat_prior_wicket_rate*source.legal_event-source.bat_dismissal
                source['bowl_wicket_residual']=source.bowl_wicket-source.bowl_prior_wicket_rate*source.legal_event
            m=match_metrics(source,role,source.high_pressure_selected);s=summarize(m,bootstrap=500)
            for weight in [.7,.85,1.]:
                r,_=rank(s,run_weight=weight,scales=scales)
                for row in r.itertuples():scenarios.append(dict(role=role,scenario=f'{expectation}_runs{weight}',player_id=row.player_id,rank=row.rank,rating=row.rating,eligible=row.eligible))
                if expectation=='contextual' and weight==.7:
                    z=r[['player_id','rating']].merge(old[['player_id','rating']],on='player_id',suffixes=('_new','_old'),validate='one_to_one')
                    checks[role]=bool(np.allclose(z.rating_new,z.rating_old,equal_nan=True));assert checks[role]
                if expectation=='prior' and weight==.85:
                    r.to_csv(out/f'{label}_performance_prior85.csv',index=False);alt=r[['player_id','rank']].rename(columns={'rank':'performance_prior85_rank'})
            if expectation=='prior':primary_m=m;base=s
        strong=supported_matches(primary_m,20,3);ss=summarize(strong,bootstrap=500)
        print('Leave-one-match-out reliability',label,flush=True)
        r=reliability(ss,strong,scales)
        for setting,s in [('original_10balls_2matches',base),('strong_20balls_3matches',ss)]:
            for row in s.itertuples():support.append(dict(role=role,setting=setting,player_id=row.player_id,overlap=row.overlap_fraction,ordinary_legal=row.ordinary_legal,ordinary_matches=row.ordinary_matches,bootstrap_valid=getattr(row,'bootstrap_valid_uplift_fraction',np.nan),pressure_legal=row.pressure_legal))
        for q in [.1,.2,.3]:
            for cw in [.1,.25,.4]:
                score=(1-cw)*r[f'lcb_{q}']+cw*(2*r.combined_positive_match_fraction-1)-.25*r.best_match_drop
                rr=score.where(r.reliability_eligible).rank(ascending=False,method='min')
                for ix,row in r.iterrows():scenarios.append(dict(role=role,scenario=f'reliability_q{q}_consistency{cw}',player_id=row.player_id,rank=rr.loc[ix],rating=score.loc[ix],eligible=row.reliability_eligible))
        for overlap in [.5,.75,.9]:
            for valid in [.8,.9,.95]:
                eligible=r.eligible&r.overlap_fraction.ge(overlap)&r.ordinary_matches.ge(8)&r.ordinary_legal.ge(100)&r.bootstrap_valid_uplift_fraction.ge(valid)&r.reliability_score.notna()
                rr=r.reliability_score.where(eligible).rank(ascending=False,method='min')
                for ix,row in r.iterrows():scenarios.append(dict(role=role,scenario=f'support_overlap{overlap}_valid{valid}',player_id=row.player_id,rank=rr.loc[ix],rating=row.reliability_score,eligible=eligible.loc[ix]))
        original=old[['player_id','player_name','rank','rating','rating_lo95','rating_hi95','top10_scenario_fraction','bootstrap_valid_uplift_fraction']].rename(columns={c:'original_'+c for c in ['rank','rating','rating_lo95','rating_hi95','top10_scenario_fraction','bootstrap_valid_uplift_fraction']})
        r=r.merge(original,on='player_id',how='left',validate='one_to_one').merge(alt,on='player_id',validate='one_to_one')
        sr=pd.DataFrame(scenarios);sr=sr[sr.role.eq(role)]
        agg=sr.groupby('player_id').agg(scenario_rank_min=('rank','min'),scenario_rank_max=('rank','max'))
        agg['scenario_top10_fraction']=sr.assign(top=sr['rank'].le(10)).groupby('player_id').top.mean()
        r=r.merge(agg,on='player_id',validate='one_to_one');r['rank_improvement']=r.original_rank-r.revised_rank;r['exclusion_reason']=''
        for mask,reason in [(~r.eligible,'insufficient matched sample'),(r.overlap_fraction.lt(.75),'pressure overlap below 75%'),(r.ordinary_legal.lt(100),'ordinary balls below 100'),(r.ordinary_matches.lt(8),'ordinary matches below 8'),(r.bootstrap_valid_uplift_fraction.lt(.9),'bootstrap support below 90%')]:
            r.loc[mask,'exclusion_reason']+=reason+'; '
        r['assessment']=np.where(r.performance_lo95.gt(0),'positive conditional interval','conditional interval includes zero')
        r['change_reason']='Consistency '+r.combined_positive_match_fraction.round(2).astype(str)+'; lower quantile '+r.performance_lcb.round(2).astype(str)+'; best-match drop '+r.best_match_drop.round(2).astype(str)+'; '+r.exclusion_reason
        r=r.sort_values(['revised_rank','player_id'],na_position='last');r.to_csv(out/f'{label}_comparison_all.csv',index=False)
        r[(r.original_rank<=15)|(r.revised_rank<=15)].to_csv(out/f'{label}_comparison_top15.csv',index=False)
        selected=r[r.reliability_eligible].head(10).copy();assert len(selected)==10,f'Only {len(selected)} supported {role} candidates'
        selected['final_order']=np.arange(1,11)
        selected['position_flag']=np.where(selected.performance_lo95.le(0)|selected.scenario_top10_fraction.lt(.8),'fragile / assumption-sensitive','supported; exact order uncertain')
        selected.to_csv(out/f'final_{label}_top10.csv',index=False);results[role]=r
    scenarios=pd.DataFrame(scenarios).merge(names,on='player_id',validate='many_to_one');scenarios.to_csv(out/'ranking_scenarios.csv',index=False)
    pd.DataFrame(support).to_csv(out/'ordinary_support_investigation.csv',index=False)
    from phase3_pipeline.report import reports
    reports(out,results,validation,cal,decision,scenarios)
    assert all(sha256(p)==v for p,v in hashes.items()),'Earlier phase file changed'
    (out/'phase3_manifest.json').write_text(json.dumps(dict(previous_files_unchanged=True,protected_file_hashes=hashes,baseline_rating_reproduced=checks,calibration_adopted=decision['adopt'],seed=42,bootstrap=500,final_method='historical-prior wickets; 85/15; reliability score; stronger support'),indent=2))
    print('Phase 3 complete',flush=True)

if __name__=='__main__':main()
