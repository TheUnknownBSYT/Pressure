"""Run the independent review using saved Phase 1/2 events. Never rebuild them."""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from independent_audit.core import summarize_fixed,stratified_wickets,estimate,performance,quality,gate,prepare,exclusions
from phase2_pipeline.core import match_metrics,pressure_variant,identities,summarize,rank
from pressure_pipeline.data import sha256

OUT=Path('outputs/independent_audit')
SEEDS=[42,137,2026]


def save(frame,name):
    frame.to_csv(OUT/(name+'.csv'),index=False)


def wicket_validation(d):
    errors=[];summary=[]
    a=d[d.analysis_available].copy()
    for role,outcome in [('bat','bat_dismissal'),('bowl','bowl_wicket')]:
        for model,column in [('contextual','expected_'+outcome+'_per_legal'),('global',role+'_global_wicket_rate'),('stratified',role+'_stratified_wicket_rate')]:
            a['expected']=a[column]*a.legal_event
            for dimension in ['overall','phase_pre','competition_type']:
                keys=['match_id']+([] if dimension=='overall' else [dimension])
                g=a.groupby(keys).agg(balls=('legal_event','sum'),actual=(outcome,'sum'),expected=('expected','sum')).reset_index()
                g=g[g.balls>0];g['group']='all' if dimension=='overall' else g[dimension]
                g['signed_error']=(g.expected-g.actual)/g.balls;g['squared_error']=g.signed_error**2
                g['model']=model;g['role']=role;g['dimension']=dimension;errors.append(g[['role','model','dimension','group','match_id','balls','actual','expected','signed_error','squared_error']])
    e=pd.concat(errors,ignore_index=True);save(e,'wicket_match_errors')
    for keys,g in e.groupby(['role','dimension','group']):
        wide=g.pivot(index='match_id',columns='model',values='squared_error')
        counts=np.random.default_rng(42).multinomial(len(wide),np.full(len(wide),1/len(wide)),size=2000)
        for model in ['contextual','global','stratified']:
            t=g[g.model.eq(model)];diff=wide[model]-wide['global'];boot=counts@diff.to_numpy()/len(wide)
            summary.append(dict(role=keys[0],dimension=keys[1],group=keys[2],model=model,matches=len(t),mse=t.squared_error.mean(),signed_error_per100=100*t.signed_error.mean(),
                actual_rate_per100=100*t.actual.sum()/t.balls.sum(),expected_rate_per100=100*t.expected.sum()/t.balls.sum(),difference_vs_global=diff.mean(),diff_lo95=np.quantile(boot,.025),diff_hi95=np.quantile(boot,.975)))
    save(pd.DataFrame(summary),'wicket_fairness_comparison')


def source_for(d,model):
    if model=='contextual':return d
    a=d.copy();a['bat_wicket_residual']=a['bat_'+model+'_wicket_rate']*a.legal_event-a.bat_dismissal
    a['bowl_wicket_residual']=a.bowl_wicket-a['bowl_'+model+'_wicket_rate']*a.legal_event
    return a


def exclude_brief_pressure_matches(m,minimum=6):
    """Remove short supported pressure appearances; never relabel them ordinary."""
    drops=set()
    for pid,a in m.groupby('player_id'):
        _,_,_,_,z=prepare(a)
        exposure=z[z.high.eq(1)].groupby('match_id').legal.sum()
        drops.update((pid,mid) for mid in exposure[exposure<minimum].index)
    drop=m.high.eq(1)&pd.Series([(p,mid) in drops for p,mid in zip(m.player_id,m.match_id)],index=m.index)
    return m[~drop].copy()


def short_sample_sensitivity(m,base,scales,role,bootstrap=2000):
    shortened=exclude_brief_pressure_matches(m,6)
    s,_=summarize_fixed(shortened,scales,role,bootstrap=bootstrap)
    original=base.set_index('player_id')
    s['original_pressure_legal_all']=s.player_id.map(original.pressure_legal_all)
    s['retained_before_exclusion']=s.player_id.map(original.pressure_legal)
    s['matched_pressure_balls_lost']=s.retained_before_exclusion-s.pressure_legal
    s['original_pressure_matches']=s.player_id.map(original.pressure_matches)
    s['pressure_matches_lost']=s.original_pressure_matches-s.pressure_matches
    # Do not make coverage look better merely by deleting difficult evidence.
    s['pressure_legal_all']=s.original_pressure_legal_all
    s['overlap_fraction']=s.pressure_legal/s.pressure_legal_all.replace(0,np.nan)
    s['eligible']=s.apply(gate,axis=1);s['exclusion_reason']=s.apply(exclusions,axis=1)
    for score in ['R','A','simple_score']:
        s[score+'_rank']=s[score].where(s.eligible).rank(ascending=False,method='min')
    s['baseline_rank']=s.player_id.map(original.R_rank)
    s['role']=role;s['scenario']='exclude_supported_pressure_appearances_below_6_legal_balls'
    save(s,f'{role}_short_appearance_sample_sensitivity')
    return s


def decomposition(role,label,old,legacy,new):
    b=pd.read_csv(f'outputs/phase3/{label}_performance_prior85.csv')
    z=legacy.copy();pop=z.reliability_eligible
    # C and D use exactly the saved Phase 3 supported/eligible population.
    z['A_phase2_rank']=z.original_rank
    z=z.merge(b[['player_id','rank','rating','pressure_legal']].rename(columns={'rank':'B_prior85_original_support_rank','rating':'B_score','pressure_legal':'old_retained_pressure_balls'}),on='player_id',validate='one_to_one')
    z['C_score']=z.shrinkage*z.unshrunk_performance
    z['C_prior85_strong_support_rank']=z.C_score.where(pop).rank(ascending=False,method='min')
    z['D_phase3_reliability_rank']=z.revised_rank
    z['same_support_unshrunk_rank']=z.unshrunk_performance.where(pop).rank(ascending=False,method='min')
    z['same_support_Q20_only_rank']=z.performance_lcb.where(pop).rank(ascending=False,method='min')
    z['same_support_Q20_consistency_rank']=(.75*z.performance_lcb+.25*(2*z.combined_positive_match_fraction-1)).where(pop).rank(ascending=False,method='min')
    z['uncertainty_component']=-.75*(z.unshrunk_performance-z.performance_lcb)
    z['positive_match_component']=.25*(2*z.combined_positive_match_fraction-1)
    z['deletion_component']=-.25*z.best_match_drop
    z['scoring_rank_change_B_minus_A']=z.B_prior85_original_support_rank-z.A_phase2_rank
    z['support_and_gate_rank_change_C_minus_B']=z.C_prior85_strong_support_rank-z.B_prior85_original_support_rank
    z['reliability_rank_change_D_minus_C']=z.D_phase3_reliability_rank-z.C_prior85_strong_support_rank
    # Context change: score difference at constant original shrinkage. Gate-only
    # rank: B scores reranked on the exact C/D population before context changes.
    orig_shrink=z.player_id.map(old.set_index('player_id').shrinkage)
    z['ordinary_context_score_change_constant_shrink']=orig_shrink*z.unshrunk_performance-z.B_score
    z['B_score_on_C_population_rank']=z.B_score.where(pop).rank(ascending=False,method='min')
    z['new_retained_pressure_balls']=z.pressure_legal
    z['phase2_bootstrap_valid_fraction']=z.player_id.map(old.set_index('player_id').bootstrap_valid_uplift_fraction)
    z['phase3_bootstrap_valid_fraction']=z.bootstrap_valid_uplift_fraction
    z=z.merge(new[['player_id','R_rank','R','pressure_matches','exclusion_reason','D','f_uplift','A_lo95','A_hi95']].add_prefix('audit_').rename(columns={'audit_player_id':'player_id'}),on='player_id',how='left')
    ids=z.loc[z.A_phase2_rank.le(20)|z.D_phase3_reliability_rank.le(20),'player_id']
    cols=['player_id','player_name','A_phase2_rank','B_prior85_original_support_rank','B_score_on_C_population_rank','C_prior85_strong_support_rank','D_phase3_reliability_rank','audit_R_rank',
          'scoring_rank_change_B_minus_A','support_and_gate_rank_change_C_minus_B','reliability_rank_change_D_minus_C','B_score','C_score','ordinary_context_score_change_constant_shrink',
          'same_support_unshrunk_rank','same_support_Q20_only_rank','same_support_Q20_consistency_rank','uncertainty_component','positive_match_component','deletion_component',
          'old_retained_pressure_balls','new_retained_pressure_balls','pressure_legal_all','overlap_fraction','ordinary_legal','ordinary_matches','pressure_matches','exclusion_reason',
          'audit_pressure_matches','audit_exclusion_reason','audit_R','audit_D','audit_f_uplift','audit_A_lo95','audit_A_hi95','phase2_bootstrap_valid_fraction','phase3_bootstrap_valid_fraction']
    save(z[z.player_id.isin(ids)][cols],f'ranking_change_decomposition_{label}')


def context_diagnostics(d,role,packs,base,names):
    a=d[d.analysis_available].copy();a['player_id']=a.striker_id if role=='bat' else a.bowler_id
    a['high']=a.high_pressure_selected.astype(int)
    keys={(pid,c) for pid,p in packs.items() for c in p['cells']}
    a['retained']=[(p,c) in keys for p,c in zip(a.player_id,a.comparison_cell)]
    a['opposition']=a.bowling_team_id if role=='bat' else a.batting_team_id
    a['calendar_year']=a.match_date_parsed.dt.year.astype(str)
    a['position_proxy']=pd.cut(a.batting_position_reported,[0,3,6,11],labels=['1-3','4-6','7-11']).astype(str)
    a.loc[a.batting_position_uncertain,'position_proxy']='unknown'
    a['usage_proxy']=pd.cut(a.bowler_legal_balls_pre,[-1,5,11,17,np.inf],labels=['first_six','second_six','third_six','fourth_or_more']).astype(str)
    # Separate raw versus retained populations exposes matching exclusions.
    dims=['competition_type','phase_pre','innings','opposition','calendar_year','season','venue_id', 'position_proxy' if role=='bat' else 'usage_proxy']
    rows=[]
    for dim in dims:
        for setting,t in [('all_analysis',a),('matched',a[a.retained])]:
            g=t.groupby(['player_id',dim,'high'],dropna=False).agg(legal=('legal_event','sum'),events=('delivery_id','size'),matches=('match_id','nunique'),rr=(role+'_run_residual','sum'),wr=(role+'_wicket_residual','sum')).reset_index().rename(columns={dim:'group'})
            g['dimension']=dim;g['setting']=setting;g['role']=role
            g['run_adj_per100']=100*g.rr/g.legal.replace(0,np.nan);g['wicket_adj_per100']=100*g.wr/g.legal.replace(0,np.nan)
            g['exposure_share']=g.legal/g.groupby(['player_id','high']).legal.transform('sum').replace(0,np.nan)
            rows.append(g)
    return pd.concat(rows,ignore_index=True).merge(names,on='player_id',how='left').merge(base[['player_id','eligible','R_rank','overlap_fraction','exclusion_reason']],on='player_id',how='left')


def joint_stability(base,packs,scales,role,bootstrap):
    # Shared global match draws retain cross-player match covariance. Eligibility,
    # consistency, D and quantile offset stay fixed: this is a linearized R rank
    # diagnostic, NOT a nested bootstrap of the complete selection procedure.
    pids=base.loc[base.eligible,'player_id'].tolist();ref=base.set_index('player_id')
    mids=sorted({m for pid in packs for m in packs[pid]['matches']});mi={m:i for i,m in enumerate(mids)};rows=[];pairrows=[]
    for seed in SEEDS:
        counts=np.random.default_rng(seed).multinomial(len(mids),np.full(len(mids),1/len(mids)),size=bootstrap)
        values=[]
        for pid in pids:
            p=packs[pid];draws=estimate(np.einsum('bn,nchk->bchk',counts[:,[mi[m] for m in p['matches']]],p['arr'],optimize=False))
            values.append(.75*performance(draws,scales)+ref.loc[pid,'R']-.75*ref.loc[pid,'A'])
        v=np.array(values).T;available=np.isfinite(v);ranks=pd.DataFrame(v).rank(axis=1,ascending=False,method='min').to_numpy()
        # An invalid candidate could rank anywhere. Report inclusion bounds
        # rather than treating all invalid candidates as certainly below others.
        missing=(~available).sum(axis=1);lower=((ranks+missing[:,None])<=10)&available
        upper=(ranks<=10)|~available
        for j,pid in enumerate(pids):
            valid=available[:,j];rows.append(dict(role=role,player_id=pid,seed=seed,method='global_match_linearized_R_fixed_eligibility',replicates=bootstrap,
                valid_comparison_fraction=valid.mean(),conditional_rank_lo=np.nanquantile(ranks[:,j],.025),conditional_rank_hi=np.nanquantile(ranks[:,j],.975),
                top10_fraction_among_available=float(np.mean(ranks[valid,j]<=10)),top10_missing_worst_bound=lower[:,j].mean(),top10_missing_best_bound=upper[:,j].mean()))
        ordered=base[base.eligible].sort_values('R_rank').player_id.tolist()[:16]
        for left,right in zip(ordered[:-1],ordered[1:]):
            diff=v[:,pids.index(left)]-v[:,pids.index(right)];finite=diff[np.isfinite(diff)]
            pairrows.append(dict(role=role,seed=seed,left_id=left,right_id=right,conditional_valid_fraction=len(finite)/bootstrap,linearized_R_diff_lo95=np.quantile(finite,.025),linearized_R_diff_hi95=np.quantile(finite,.975)))
    return pd.DataFrame(rows),pd.DataFrame(pairrows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bootstrap',type=int,default=2000);args=parser.parse_args();assert args.bootstrap>=2000
    started=time.time();OUT.mkdir(parents=True,exist_ok=True)
    protected=[p for p in Path('outputs').rglob('*') if p.is_file() and OUT not in p.parents]+list(Path('.').glob('*.csv'))
    hashes={str(p):sha256(p) for p in protected}
    manifest=dict(started_utc=pd.Timestamp.now(tz='UTC').isoformat(),protected_file_hashes=hashes,bootstrap=args.bootstrap,seeds=SEEDS,pressure_variants=18,
        method='fixed support; global prior; 85/15; original R coefficients; positive-exposure match eligibility',python=platform.python_version())
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    d=pd.read_parquet('outputs/phase2/phase2_player_event_metrics.parquet');names,_=identities(Path('.'));names=names[['player_id','player_name']]
    d,folds=stratified_wickets(d);save(folds,'wicket_prior_folds');wicket_validation(d)
    g=source_for(d,'global');allpressure=[];allformula=[];allgate=[];allbootstrap=[];alljoint=[];allpairs=[];allcontext=[];allcomponents=[];allwickets=[];alllegacy=[];allappear=[];allzero=[]
    for role,label in [('bat','batters'),('bowl','bowlers')]:
        print('Baseline and decomposition:',role,flush=True)
        old=pd.read_csv(f'outputs/phase2/phase2_{label}.csv');legacy=pd.read_csv(f'outputs/phase3/{label}_comparison_all.csv')
        scales=[old.loc[old.eligible,c].std() for c in ['pressure_run_adj','pressure_wicket_adj']]
        m=match_metrics(g,role,g.high_pressure_selected)
        base,packs=summarize_fixed(m,scales,role,bootstrap=args.bootstrap)
        base=base.merge(names,on='player_id',how='left');base['role']=role;save(base,f'{label}_corrected_diagnostic')
        short_sample_sensitivity(m,base,scales,role,args.bootstrap)
        decomposition(role,label,old,legacy,base);allcomponents.append(base)
        allcontext.append(context_diagnostics(g,role,packs,base,names))
        # Reproduce original score exactly before interpreting audit differences.
        original_m=match_metrics(d,role,d.high_pressure_selected)
        reproduced,_=rank(summarize(original_m),scales=scales)
        chk=reproduced[['player_id','rating']].merge(old[['player_id','rating']],on='player_id',suffixes=('_new','_old'))
        assert np.allclose(chk.rating_new,chk.rating_old,equal_nan=True)
        # Legacy support/count semantics with stable seeding isolate count effects.
        oldcounts,_=summarize_fixed(m,scales,role,bootstrap=args.bootstrap,positive_match_counts=False)
        bridge=oldcounts[['player_id','R','R_rank','pressure_matches','eligible']].merge(base[['player_id','R','R_rank','pressure_matches','eligible']],on='player_id',suffixes=('_event_counts','_legal_counts'))
        bridge['role']=role;save(bridge,f'{label}_count_correction_bridge')
        ids=set(old.loc[old['rank'].le(20),'player_id'])|set(legacy.loc[legacy.revised_rank.le(20),'player_id'])|set(base.loc[base.R_rank.le(15),'player_id'])
        for pid in ids:
            if pid not in packs:continue
            p=packs[pid]
            for mid,balls,q,u in zip(p['matches'],p['pressure_balls'],p['match_quality'],p['match_uplift']):
                if balls>0:allappear.append(dict(role=role,player_id=pid,match_id=mid,pressure_legal=balls,absolute_quality=q,uplift_against_pooled_ordinary=u))
        z=m[(m.high==1)&m.legal.eq(0)].copy();z['role']=role;allzero.append(z)
        for seed in SEEDS:
            s=base if seed==42 else summarize_fixed(m,scales,role,bootstrap=args.bootstrap,seed=seed)[0].assign(role=role)
            cols=['role','player_id','R','R_rank','eligible','bootstrap_valid_fraction','bootstrap_zero_pressure_fraction','Q20','A_lo95','A_hi95','Q20_missing_worst_bound','Q20_missing_best_bound','uplift_lo95','uplift_hi95']
            t=s[cols].copy();t['seed']=seed;t['replicates']=args.bootstrap;t['method']='player_stable_match_multinomial_fixed_support';allbootstrap.append(t)
        joint,pairs=joint_stability(base,packs,scales,role,args.bootstrap);alljoint.append(joint);allpairs.append(pairs)
        for top in [20,25,30]:
            for stakes in [True,False]:
                for scale in [.1,.05,.15]:
                    print('Pressure sensitivity:',role,top,stakes,scale,flush=True)
                    high,cutoff=pressure_variant(d,top,scale,stakes)
                    if top==25 and stakes and scale==.1:s=base
                    else:s,_=summarize_fixed(match_metrics(g,role,high),scales,role,bootstrap=args.bootstrap)
                    t=s[['player_id','R','R_rank','eligible','pressure_matches','pressure_legal','ordinary_legal','ordinary_matches','overlap_fraction','bootstrap_valid_fraction','exclusion_reason','f_absolute','f_uplift','A_lo95','A_hi95']].copy()
                    t['role']=role;t['threshold_top_percent']=top;t['stakes']=stakes;t['leverage_scale']=scale;t['cutoff']=cutoff;t['seed']=42;t['replicates']=args.bootstrap
                    t['baseline_rank']=t.player_id.map(base.set_index('player_id').R_rank);t['rank_change']=t.R_rank-t.baseline_rank
                    allpressure.append(t)
        # Formula sensitivity is on ONE supported population. No eligibility or
        # pressure changes enter these frequencies. Six-ball choice is one over,
        # a transparent diagnostic convention, not an estimated optimum.
        formulas={'fixed_original_R':base.R,'without_D':base.R+.25*base.D,'point_no_quantile':base.R+.75*(base.A-base.Q20),
                  'Q20_only':base.Q20,'simple_M_over_Mplus8':base.simple_score,'uplift_consistency':base.R+.5*(base.f_uplift-base.f_absolute)}
        for minimum in [3,6,12]:
            formulas[f'absolute_consistency_min{minimum}']=base.R+.5*(base[f'f_absolute_min{minimum}']-base.f_absolute)
            formulas[f'absolute_consistency_downweight{minimum}']=base.R+.5*(base[f'f_absolute_downweight{minimum}']-base.f_absolute)
        for name,score in formulas.items():
            t=pd.DataFrame(dict(role=role,player_id=base.player_id,scenario=name,score=score,eligible=base.eligible));t['rank']=score.where(base.eligible).rank(ascending=False,method='min');allformula.append(t)
        for overlap in [.5,.75,.9]:
            for valid in [.8,.9,.95]:
                elig=base.apply(lambda r:gate(r,overlap,valid),axis=1)
                t=pd.DataFrame(dict(role=role,player_id=base.player_id,overlap_gate=overlap,valid_gate=valid,eligible=elig));t['rank']=base.R.where(elig).rank(ascending=False,method='min');allgate.append(t)
        for model in ['contextual','global','stratified']:
            mm=match_metrics(source_for(d,model),role,d.high_pressure_selected)
            for weight in [.7,.85,1.]:
                print('Wicket/weight sensitivity:',role,model,weight,flush=True)
                s=base if model=='global' and weight==.85 else summarize_fixed(mm,scales,role,bootstrap=args.bootstrap,weight=weight)[0]
                t=s[['player_id','R','R_rank','eligible','A','absolute_quality','uplift_quality','pressure_matches','bootstrap_valid_fraction']].copy();t['role']=role;t['wicket_model']=model;t['run_weight']=weight;allwickets.append(t)
        # Historical scenario mixing: retain separate family frequencies.
        sr=pd.read_csv('outputs/phase3/ranking_scenarios.csv');sr=sr[sr.role.eq(role)].copy()
        sr['family']=np.select([sr.scenario.str.startswith('reliability'),sr.scenario.str.startswith('support')],['reliability_formula','eligibility_gate'],default='performance_first')
        a=sr.assign(top10=sr['rank'].le(10)).groupby(['player_id','family']).agg(top10_fraction=('top10','mean'),scenarios=('scenario','size')).reset_index();a['role']=role;alllegacy.append(a)
    outputs=[(allpressure,'pressure_definition_sensitivity'),(allformula,'reliability_formula_sensitivity'),(allgate,'eligibility_sensitivity'),(allbootstrap,'bootstrap_stability'),
             (alljoint,'joint_rank_stability'),(allpairs,'adjacent_rank_uncertainty'),(allcontext,'ordinary_support_and_role_bias'),(allcomponents,'reliability_component_audit'),
             (allwickets,'wicket_weight_ranking_sensitivity'),(alllegacy,'legacy_scenario_family_audit'),(allappear,'candidate_match_appearances'),(allzero,'zero_exposure_pressure_cells')]
    for parts,name in outputs:
        t=pd.DataFrame(parts) if parts and isinstance(parts[0],dict) else pd.concat(parts,ignore_index=True)
        if 'player_id' in t and 'player_name' not in t:t=t.merge(names,on='player_id',how='left')
        save(t,name)
    # Attribution and denominators are verified against the saved event values.
    a=d[d.analysis_available];checks={}
    for role,outcome in [('bat','bat_dismissal'),('bowl','bowl_wicket')]:
        checks[role+'_run_conservation']=bool(np.isclose(match_metrics(d,role,d.high_pressure_selected).rr.sum(),a[role+'_run_residual'].sum()))
        checks[role+'_wicket_conservation']=bool(np.isclose(match_metrics(d,role,d.high_pressure_selected).wr.sum(),a[role+'_wicket_residual'].sum()))
    checks['striker_dismissal_attribution']=bool(d.bat_dismissal.eq((d.is_wicket&d.dismissed_player_id.eq(d.striker_id)).astype(int)).all())
    checks['bowler_credit_attribution']=bool(d.bowl_wicket.eq((d.is_wicket&d.dismissal_type.isin(['Caught','Bowled','LBW','Stump','Hit Wicket'])).astype(int)).all())
    assert all(checks.values())
    manifest['invariants']=checks
    save(a.groupby(['legal_event','extras_type']).agg(events=('delivery_id','size'),legal=('legal_event','sum'),bat_runs=('bat_runs','sum'),bowl_runs=('bowl_runs','sum'),bat_dismissals=('bat_dismissal','sum'),bowler_wickets=('bowl_wicket','sum')).reset_index(),'event_attribution_audit')
    result=subprocess.run([sys.executable,'-m','pytest','-q'],capture_output=True,text=True)
    (OUT/'test_results.txt').write_text(result.stdout+result.stderr);manifest['pytest_exit_code']=result.returncode
    manifest['protected_files_unchanged']=all(sha256(p)==v for p,v in hashes.items());assert manifest['protected_files_unchanged']
    manifest['elapsed_seconds']=time.time()-started;manifest['source_code_hashes']={str(p):sha256(p) for p in [Path(__file__),*Path('independent_audit').glob('*.py'),*Path('tests').glob('*.py')]}
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2));assert result.returncode==0
    from independent_audit.report import reports
    reports(OUT)
    finalize_manifest(OUT)
    print('Audit complete in',round(time.time()-started,1),'seconds',flush=True)


def finalize_manifest(out):
    import importlib.metadata
    manifest=json.loads((out/'run_manifest.json').read_text())
    manifest['source_code_hashes']={str(p):sha256(p) for p in [Path(__file__),*Path('independent_audit').glob('*.py'),*Path('tests').glob('*.py'),Path('phase2_pipeline/core.py'),Path('phase3_pipeline/audit.py'),Path('pressure_pipeline/data.py'),Path('requirements.txt')]}
    manifest['dependency_versions']={package:importlib.metadata.version(package) for package in ['numpy','pandas','matplotlib','scikit-learn','pytest','pyarrow']}
    manifest['artifact_hashes']={str(p):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='run_manifest.json'}
    manifest['protected_files_unchanged']=all(sha256(p)==v for p,v in manifest['protected_file_hashes'].items())
    assert manifest['protected_files_unchanged']
    manifest['completed_utc']=pd.Timestamp.now(tz='UTC').isoformat()
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
