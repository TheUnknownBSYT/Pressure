#!/usr/bin/env python3
"""Reproduce descriptive Phase 2 candidate analysis; no external data required."""
import argparse,json,hashlib,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
from phase2_pipeline.core import *
from pressure_pipeline.data import sha256
from pressure_pipeline.report import md_table

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=Path('outputs/phase2'));ap.add_argument('--bootstrap',type=int,default=500);ap.add_argument('--skip-reproduction',action='store_true');args=ap.parse_args()
    if args.bootstrap<500:ap.error('Use at least 500 bootstrap replicates')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    if not args.skip_reproduction:
        subprocess.run([sys.executable,'build_pressure_dataset.py','--output-dir',str(out/'phase1_reproduction')],check=True)
    before=json.loads(Path('outputs/run_manifest.json').read_text());after=json.loads((out/'phase1_reproduction/run_manifest.json').read_text())
    comparisons={k:before[k]==after[k] for k in ['rows','matches_with_deliveries','standard_eligible_deliveries','scored_deliveries','high_pressure_deliveries','thresholds','source_sha256']}
    assert all(comparisons.values()),comparisons
    for name,digest in before['source_sha256'].items():assert sha256(name+'.csv')==digest
    d=pd.read_parquet('outputs/pressure_scored.parquet');assert d.delivery_id.is_unique
    state_ids=pd.read_parquet('outputs/match_state.parquet',columns=['delivery_id']);assert len(d.merge(state_ids,on='delivery_id',validate='one_to_one'))==len(d)
    print('Attributing outcomes and fitting prior-match expectations...',flush=True)
    d,validation=forward_expectations(attribute(d));validation.to_csv(out/'expected_model_validation.csv',index=False)
    names,aliases=identities(Path('.'));aliases.to_csv(out/'identity_aliases.csv',index=False)
    assert names.player_id.is_unique
    d.to_parquet(out/'phase2_player_event_metrics.parquet',index=False)
    coverage=[]
    for col in ['match_id','season','innings','phase_pre','match_format','competition_type','comp_name','revised_match','striker_id','bowler_id']:
        for val,g in d.groupby(col):coverage.append(dict(dimension=col,value=val,all_events=len(g),phase1_scored=g.pressure_eligible.sum(),phase2_available=g.analysis_available.sum(),high_events=(g.analysis_available&g.high_pressure_selected).sum()))
    pd.DataFrame(coverage).to_csv(out/'phase2_coverage_bias.csv',index=False)
    d.loc[d.is_wicket,['delivery_id','striker_id','non_striker_id','dismissed_player_id','dismissal_type','bat_dismissal','nonstriker_dismissal','bowl_wicket','dismissal_unresolved']].to_csv(out/'dismissal_attribution_audit.csv',index=False)
    review=[]
    for label,mask in [('non_striker_dismissal',d.nonstriker_dismissal.eq(1)),('striker_dismissal',d.bat_dismissal.eq(1)),('wide',d.extras_type.eq('wide')),('no_ball_with_batter_runs',d.extras_type.eq('Overstep')&d.runs.gt(0)),('uncertain_concession',d.bowler_concession_uncertain)]:
        a=d.loc[mask,['delivery_id','striker_id','non_striker_id','dismissed_player_id','bat_dismissal','nonstriker_dismissal','legal_event','balls_faced_nonwide','runs']].head(2).copy();a['audit_case']=label;review.append(a)
    pd.concat(review).to_csv(out/'attribution_review_cases.csv',index=False)
    d.groupby('extras_type').agg(events=('delivery_id','size'),legal=('legal_event','sum'),nonwide_faced=('balls_faced_nonwide','sum'),bat_runs=('bat_runs','sum'),conceded_assumed=('bowl_runs','sum'),uncertain=('bowler_concession_uncertain','sum')).to_csv(out/'balls_faced_audit.csv')
    allmatch=[];sens=[];rankings={};elig=[];cutoffs=[]
    for role,label in [('bat','batters'),('bowl','bowlers')]:
        print('Rating',label,flush=True)
        high,cut=pressure_variant(d);m=match_metrics(d,role,high);m['role']=role;allmatch.append(m)
        summary=summarize(m,bootstrap=args.bootstrap);r,scales=rank(summary);r=r.merge(names,on='player_id',validate='one_to_one',how='left');assert r.loc[r.eligible,'player_name'].notna().all()
        for top in [20,25,30]:
            for scale in [.05,.1,.15]:
                for stakes in [True,False]:
                    h,c=pressure_variant(d,top,scale,stakes);mm=match_metrics(d,role,h)
                    ss=summarize(mm);rr,_=rank(ss,scales=scales)
                    scenario=f'top{top}_scale{scale}_stakes{stakes}'
                    for row in rr.itertuples():sens.append(dict(role=role,scenario=scenario,player_id=row.player_id,rank=row.rank,rating=row.rating,eligible=row.eligible))
                    cutoffs.append(dict(role=role,scenario=scenario,cutoff=c))
        for rw in [.5,.7,.9,1.0]:
            for aw in [.6,.75,.9]:
                rr,_=rank(summary,run_weight=rw,absolute_weight=aw,scales=scales)
                for row in rr.itertuples():sens.append(dict(role=role,scenario=f'weights_run{rw}_absolute{aw}',player_id=row.player_id,rank=row.rank,rating=row.rating,eligible=row.eligible))
        for prior in [4,8,16]:
            ss=summary.copy();ss['shrinkage']=ss.pressure_matches/(ss.pressure_matches+prior);rr,_=rank(ss,scales=scales)
            for row in rr.itertuples():sens.append(dict(role=role,scenario=f'shrink{prior}',player_id=row.player_id,rank=row.rank,rating=row.rating,eligible=row.eligible))
        for matches in [5,8,10]:
            for balls in [30,50,75]:
                ss=summary.copy();ss['eligible']=(ss.pressure_matches>=matches)&(ss.pressure_legal>=balls)&(ss.ordinary_matches>=5)&(ss.ordinary_legal>=50)&(ss.overlap_fraction>=.5)
                rr,_=rank(ss,scales=scales);elig.append(dict(role=role,min_matches=matches,min_legal=balls,candidates=int(ss.eligible.sum())))
                for row in rr.itertuples():sens.append(dict(role=role,scenario=f'eligibility_{matches}_{balls}',player_id=row.player_id,rank=row.rank,rating=row.rating,eligible=row.eligible))
        slices=[]
        for col in ['innings','competition_type','phase_pre']:
            for val in d[col].unique():
                if val in ['unknown_revised','super_over'] or (col=='innings' and val>2):continue
                ds=d[d[col].eq(val)];mm=match_metrics(ds,role,high);ss=summarize(mm)
                if ss.empty or 'pressure_run_adj' not in ss:continue
                ss['dimension']=col;ss['value']=val;slices.append(ss)
        pd.concat(slices,ignore_index=True).to_csv(out/f'{role}_context_breakdown.csv',index=False)
        sr=pd.DataFrame(sens);sr=sr[sr.role.eq(role)]
        agg=sr.groupby('player_id').agg(sensitivity_rank_min=('rank','min'),sensitivity_rank_max=('rank','max'),scenario_eligible_fraction=('eligible','mean'))
        agg['top10_scenario_fraction']=sr.assign(top10=sr['rank'].le(10)).groupby('player_id').top10.mean()
        r=r.merge(agg,on='player_id',validate='one_to_one')
        r['pressure_raw_rate_per100legal']=100*r.pressure_runs/r.pressure_legal.replace(0,np.nan)
        r['ordinary_raw_rate_per100legal']=100*r.ordinary_runs/r.ordinary_legal.replace(0,np.nan)
        r['pressure_conventional_sr_nonwide']=100*r.pressure_runs/r.pressure_balls_faced.replace(0,np.nan)
        r['pressure_economy']=6*r.pressure_runs/r.pressure_legal.replace(0,np.nan)
        r.to_csv(out/f'phase2_{label}.csv',index=False);rankings[role]=r
    pd.concat(allmatch).to_csv(out/'phase2_player_match_metrics.csv',index=False)
    pd.DataFrame(sens).to_csv(out/'ranking_sensitivity.csv',index=False);pd.DataFrame(elig).to_csv(out/'eligibility_sensitivity.csv',index=False);pd.DataFrame(cutoffs).to_csv(out/'pressure_variant_thresholds.csv',index=False)
    figures(d,rankings,pd.concat(allmatch),out)
    write_docs(d,rankings,comparisons,out,args.bootstrap)
    for name,digest in before['source_sha256'].items():assert sha256(name+'.csv')==digest
    manifest=dict(phase1_reproduction=comparisons,source_sha256=before['source_sha256'],events=len(d),analysis_events=int(d.analysis_available.sum()),bootstrap_replicates=args.bootstrap,seed=42,configuration=dict(min_pressure_matches=8,min_pressure_legal=50,min_ordinary_matches=5,min_ordinary_legal=50,min_overlap=.5,pooling_prior_legal=200,shrinkage_prior_matches=8,run_weight=.7,absolute_weight=.75,sensitivity_scenarios_per_role=42),eligible_candidates={k:int(v.eligible.sum()) for k,v in rankings.items()},status='provisional_descriptive_shortlist_not_submission_final',scripts_sha256={str(p):sha256(p) for p in [Path('build_phase2.py'),Path('phase2_pipeline/core.py')]})
    (out/'phase2_run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest['eligible_candidates']),flush=True)

def figures(d,rankings,match,out):
    import os;os.environ.setdefault('MPLCONFIGDIR',str(out/'.mplconfig'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'phase2_report_figures';folder.mkdir(exist_ok=True)
    for role,r in rankings.items():
        t=r[r.eligible].head(10).sort_values('rank',ascending=False)
        fig,ax=plt.subplots(figsize=(10,6));y=np.arange(len(t));ax.barh(y-.18,t.ordinary_run_adj,height=.35,label='Matched ordinary');ax.barh(y+.18,t.pressure_run_adj,height=.35,label='High pressure');ax.set_yticks(y,t.player_name);ax.set_xlabel('Adjusted runs per 100 legal balls (positive is better)');ax.legend();fig.tight_layout();fig.savefig(folder/f'{role}_ordinary_pressure.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,6));ax.hlines(y,t.rating_lo95,t.rating_hi95);ax.scatter(t.rating,y);ax.set_yticks(y,[f'{v.player_name} (M={v.pressure_matches})' for v in t.itertuples()]);ax.axvline(0,color='gray',linestyle='--');ax.set_xlabel('Shrunk rating and match-bootstrap 95% interval');fig.tight_layout();fig.savefig(folder/f'{role}_uncertainty.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,6));ax.hlines(y,t.sensitivity_rank_min,t.sensitivity_rank_max);ax.scatter(t['rank'],y);ax.set_yticks(y,t.player_name);ax.set_xlabel('Rank range over pressure, weights, shrinkage and eligibility scenarios');fig.tight_layout();fig.savefig(folder/f'{role}_sensitivity.png',dpi=150);plt.close(fig)
        top=t.sort_values('rank').head(3)
        fig,axes=plt.subplots(len(top),1,figsize=(10,7),squeeze=False)
        for ax,row in zip(axes[:,0],top.itertuples()):
            player=match[(match.role==role)&(match.player_id==row.player_id)]
            support=player[player.high.eq(0)].groupby('comparison_cell').agg(balls=('legal','sum'),matches=('match_id','nunique'))
            supported=support[(support.balls>=10)&(support.matches>=2)].index
            a=player[player.high.eq(1)&player.comparison_cell.isin(supported)].groupby('match_id')[['legal','rr']].sum();a=a[a.legal>0];ax.bar(np.arange(len(a)),100*a.rr/a.legal);ax.axhline(0,color='gray');ax.set_title(row.player_name);ax.set_ylabel('Runs/100')
        axes[-1,0].set_xlabel('Distinct match (ID order)');fig.tight_layout();fig.savefig(folder/f'{role}_match_consistency.png',dpi=150);plt.close(fig)
    cal=pd.read_csv('outputs/calibration.csv');fig,ax=plt.subplots(figsize=(6,5))
    for i,a in cal.groupby('innings'):ax.plot(a.mean_probability,a.observed_win_fraction,'o-',label=f'Innings {i}')
    ax.plot([0,1],[0,1],'k--');ax.set(xlabel='Mean predicted win probability',ylabel='Observed win fraction');ax.legend();fig.tight_layout();fig.savefig(folder/'phase1_calibration.png',dpi=150);plt.close(fig)
    import shutil;shutil.copy('outputs/illustrative_situations.png',folder/'pressure_illustration.png')

def write_docs(d,rankings,comparisons,out,reps):
    sections=[]
    stronger=[]
    for role,r in rankings.items():
        a=r[r.eligible & r.top10_scenario_fraction.ge(.8) & r.rating_lo95.gt(0) & r.bootstrap_valid_uplift_fraction.ge(.9) & r.rank_without_best.le(20)].copy();a['role']=role;stronger.append(a)
    pd.concat(stronger).to_csv(out/'stronger_evidence_shortlist.csv',index=False)
    stability=[]
    for role,r in rankings.items():
        t=r[r.eligible].head(10)
        stability.append(dict(role=role,eligible_candidates=int(r.eligible.sum()),top10_in_at_least_80pct_scenarios=int(t.top10_scenario_fraction.ge(.8).sum()),top10_with_positive_rating_lower_bound=int(t.rating_lo95.gt(0).sum()),max_best_match_rank_change=float((t['rank']-t.rank_without_best).abs().max())))
    pd.DataFrame(stability).to_csv(out/'shortlist_stability.csv',index=False)
    for role,r in rankings.items():
        top=r[r.eligible].head(10)
        cols=['rank','player_id','player_name','pressure_matches','pressure_legal','rating','top10_scenario_fraction']
        sections.append(f'## {role} provisional candidates\n\n'+md_table(top[cols]))
        for p in top.itertuples():
            sections.append(f'- **{p.player_name} ({p.player_id})**, rank {int(p.rank)}: {p.pressure_run_adj:.1f} adjusted runs and {p.pressure_wicket_adj:.2f} favorable wicket residuals per 100 legal balls under pressure; run uplift {p.run_uplift:+.1f}. Evidence spans {p.pressure_matches} pressure matches/{p.pressure_legal} legal balls. Rating {p.rating:.2f} [{p.rating_lo95:.2f}, {p.rating_hi95:.2f}]; top ten in {p.top10_scenario_fraction:.0%} of sensitivity scenarios, rank {p.rank_without_best:.0f} after removing the best run-contribution match. Bootstrap ordinary-overlap retention is {p.bootstrap_valid_uplift_fraction:.0%}. {'CAUTION: interval includes zero.' if p.rating_lo95 <= 0 else ''} {'CAUTION: ordinary overlap disappears in more than 10% of resamples.' if p.bootstrap_valid_uplift_fraction < .9 else ''} {'CAUTION: top-ten inclusion is below 80% across scenarios.' if p.top10_scenario_fraction < .8 else ''} These metrics determine position.')
        sections.append(f'\n![{role} adjusted performance](phase2_report_figures/{role}_ordinary_pressure.png)\n\n![{role} uncertainty](phase2_report_figures/{role}_uncertainty.png)\n\n![{role} sensitivity](phase2_report_figures/{role}_sensitivity.png)\n')
    method='''# Phase 2 draft methodology — provisional descriptive shortlist

Phase 1 remains unchanged and provisional. Its exact counts, thresholds and source hashes reproduced in a separate directory. This draft uses retrospective pressure percentiles; it does not claim prospectively tested player rankings or causal psychological effects.

## Attribution and denominators

One event-table row retains each source delivery_id. Batter runs belong to the striker. A striker dismissal requires dismissed_player_id == striker_id; non-striker dismissals are separately preserved and never charged to the striker. Non-striker dismissal avoidance is not included in the striker-opportunity rating: this is a limitation for full batting survival evaluation. Missing or inconsistent dismissal IDs are excluded. Bowler wickets and conceded runs follow Phase 1; uncertain concession events are excluded from both role analyses for a common clean cohort. Wides and no-balls retain their runs and wickets but consume zero legal exposure. Conventional batter balls are separately reported using non-wide recorded deliveries, including Overstep no-balls; this is an explicit scoring convention, unverified against an official balls-faced column (none supplied). Ranking uses legal exposure exclusively. Conventional and legal strike rates are never mixed. Penalty/other uncertainty remains excluded by Phase 1.

## Context expectation

For each calendar quarter, training uses only earlier dates and at least 50 structurally valid prior matches. Coarse contexts are innings × phase × IPL/T20I/other; require ten earlier distinct matches. Fine contexts add wickets (1–3/4–7/8–10) and chase rate (≤8/8–12/12–18/>18; separate first-innings label). Each outcome rate is (historical outcome sum + 200 × parent rate)/(historical legal exposure + 200), first pooling the coarse cell to the global rate, then fine cell to coarse. No player identity, current outcomes, eventual winner, current extras type, or recorded batting order enters the predictor. Position/pace/opponent strength are omitted to avoid sparse or retrospectively assigned roles; remaining differences within phases are an acknowledged confounder. Fold validation compares match-level rate MSE with the earlier global-rate baseline.

Rates predict outcomes per legal opportunity. Expected outcome contribution on a recorded event is its predicted rate times observed legal exposure (zero on an illegal event). Exposure is an outcome denominator, not a predictive covariate. Actual illegal-event runs/wickets remain in the numerator; extra illegal events do not earn additional expected-outcome credit. Batting favorable residuals are actual minus expected runs and expected minus actual striker dismissals. Bowling favorable residuals are expected minus conceded runs and actual minus expected bowler wickets. Aggregate per 100 legal balls. Events share match clusters; they are not independent matches. These rate forecasts are not Bernoulli event probabilities.

## Ordinary comparisons and eligibility

For each player, only innings × phase × competition-type cells with ≥10 ordinary legal balls in ≥2 ordinary matches support the pressure comparison. Pressure rows in unsupported cells are excluded from the matched rating; overlap_fraction records retained pressure exposure. Player ordinary residuals are weighted by that player's high-pressure legal-ball distribution across the supported cells. This prevents phase or format usage alone from creating uplift. Exact demand/wicket overlap is not guaranteed, although historical expectations adjust those dimensions. At least 8 pressure matches/50 pressure legal balls, 5 ordinary matches/50 ordinary legal balls, and 50% retained pressure exposure are required. The complete candidate table also retains excluded players and their sample sizes. Context breakdowns and coverage files expose missingness and role selection.

## Rating and uncertainty

Combine pressure run and favorable wicket residuals after dividing by their standard deviations among eligible candidates. Main score gives 70% weight to runs, 30% to wickets; 75% to absolute pressure performance and 25% to matched ordinary uplift. Multiply by M/(M+8), where M is distinct pressure matches. This heuristic shrinkage and weighting are subjective; no claim of Bayesian posterior validity is made. Positive absolute quality is not guaranteed by eligibility, so candidates' component signs and confidence intervals must be read before submission.

Bootstrap each eligible player's distinct matches 500 times (or configured higher), carrying ALL cell and high/ordinary contributions together. Recompute exposure rates and ordinary mixture weights each time. Replicates losing required ordinary cells have unavailable uplift; report the valid fraction. Confidence intervals are conditional on fitted expectations, pressure choices, matching and fixed shrinkage/scales: they do not include all model-selection uncertainty. Match consistency is the fraction with positive run residual and median match run contribution. Remove the single match with largest positive pressure run total, recompute pressure/ordinary mixture and rank; this diagnoses one-match dominance but does not necessarily identify the best wicket-weighted match. Shrinkage remains fixed in this removal stress test.

## Sensitivity

Evaluate all 18 combinations of top 20/25/30%, raw-leverage scales .05/.10/.15, and stakes included/removed. Stakes removal omits its term without renormalization (percentile ordering is unaffected by constant overall rescaling). Thresholds use the full Phase 1 scored population before role-specific exclusions. Also evaluate twelve run/absolute-weight combinations (including runs only), shrinkage prior 4/8/16, and nine match/ball eligibility alternatives (5/8/10 and 30/50/75). Scaling constants stay fixed to the main eligible cohort. Scenario ranges are not probabilistic rank intervals. Ineligibility counts against top-ten frequency. Context-specific component tables cover each innings, format and phase; they are descriptive slices, not independently reliable rankings.

## Limits and submission status

The first-innings win model has upper-tail overconfidence; legal-ball leverage can saturate and omits illegal outcomes/free hits. Coverage excludes many matches non-randomly. Ordinary comparability is approximate. Player estimation and choice of subjective rating share this development dataset; there is no untouched external validation of the shortlist. Exact top-ten ordering is provisional even if unit tests pass. Do not represent this draft as a validated final ranking. Review sensitivity, intervals and one-match dependence before selecting entries. Optional historical-form/psychological-memory modelling was not added because core uncertainty and exposure need priority.

## Reproduce

`.venv/bin/python build_phase2.py` rebuilds Phase 1 into a separate directory and regenerates Phase 2. `.venv/bin/python -m pytest -q` runs the actual implementation tests. `--skip-reproduction` reuses and verifies the already reproduced Phase 1 manifest. Original CSV hashes are checked before/after. All outputs remain local.

'''
    body=method+'\n## Observed validation and stability\n\n'+md_table(pd.DataFrame(stability))+'\n\nSee phase2_diagnostics.md for outcome-model errors. Wicket expectations did not improve on the simple prior in this run; inspect the runs-only sensitivity before submission.\n\n'+'\n'.join(sections)
    (out/'phase2_methodology_draft.md').write_text(body)
    (out/'phase2_executive_summary.md').write_text('# Phase 2 candidate evidence\n\nProvisional descriptive candidates, pending submission review.\n\n'+'\n'.join(sections))
    model_validation=pd.read_csv(out/'expected_model_validation.csv').groupby('outcome')[['match_rate_mse','prior_match_rate_mse']].mean().reset_index()
    diagnostic=f'''# Diagnostics

Phase 1 reproduction comparisons: {comparisons}.

{md_table(pd.DataFrame(stability))}

{md_table(model_validation)}

The run expectation models improve average fold-level match-rate MSE over their global baselines; wicket/dismissal expectations do not in this run. Wicket adjustment therefore remains unvalidated; the runs-only weight sensitivity is especially relevant. Do not infer that passing chronology tests establishes predictive benefit.

Analysis events: {int(d.analysis_available.sum()):,} of {len(d):,}; unresolved wicket identities: {int(d.dismissal_unresolved.sum())}; non-striker dismissals: {int(d.nonstriker_dismissal.sum())}; uncertain concession events: {int(d.bowler_concession_uncertain.sum())}.

Phase 1 scored leverage saturation (L=1): {d.loc[d.pressure_eligible,'leverage'].eq(1).mean():.2%}. First-innings calibration remains documented in the supplied calibration table and plotted in phase2_report_figures. The highest bin averages about .94 predicted versus .80 observed. No recalibration was applied silently.

Read expected_model_validation.csv for chronological outcome-rate prediction error; ranking_sensitivity.csv for every candidate/scenario (including ineligibility); eligibility_sensitivity.csv for coverage under each sample threshold; phase2_coverage_bias.csv for match, season, phase, format and individual exposure. Name aliases are audited in identity_aliases.csv and joined by stable ID only.

stronger_evidence_shortlist.csv applies an additional conservative screen: top-ten in at least 80% of scenarios, positive lower rating bound, at least 90% valid bootstrap ordinary overlap, and rank at most 20 after removing the best run match. This screen is a review aid, not a statistically validated decision rule.

The shortlist is not approved for automatic submission. Wide intervals, scenario movement, coarse overlap and conditional bootstrap uncertainty must accompany any claimed order. No reputation or external rankings enter the analysis.
'''
    (out/'phase2_diagnostics.md').write_text(diagnostic)
    (out/'START_HERE.md').write_text('''# Start here — Phase 2

Built separate batting and bowling outcome attribution, forward historical contextual expectations, matched ordinary baselines, player/match metrics, shrinkage, match-cluster bootstrap intervals and ranking sensitivities. Phase 1 reproduced exactly in counts, thresholds and original-source hashes. See phase2_run_manifest.json. Existing Phase 1 outputs were preserved.

The analytical pipeline succeeded; exact submission-ready ordering remains unresolved because sensitivity and statistical uncertainty require review. The lists are provisional candidates, not certified final rankings. No unsupported events were given zero pressure. No original CSV was modified or uploaded.

Open phase2_executive_summary.md for the ordered candidates and individual evidence. stronger_evidence_shortlist.csv identifies the smaller subset passing the additional stability and uncertainty screen documented in phase2_diagnostics.md. phase2_batters.csv and phase2_bowlers.csv contain all components, eligibility, sample sizes, intervals and rank sensitivity. phase2_player_match_metrics.csv supports match-level investigation; phase2_player_event_metrics.parquet retains all original keyed events with availability flags. phase2_methodology_draft.md is the submission-document draft with genuine charts and result tables.

Methods: past-only partially pooled rates by innings/phase/competition/wickets/chase demand; own ordinary baseline reweighted to supported pressure contexts; run and wicket components combined with explicit weights, then shrunk by distinct pressure matches. Detailed equations and remaining subjective choices are in the draft. There is no causal psychological interpretation.

Tests actually executed and their results are saved in test_results.txt by the verification command. Source reconstruction runtime checks are in phase1_reproduction/validation_checks.json. A missing test log means tests have not yet been recorded for that build.

Three remaining actions before submission:
1. Review the top candidates' sensitivity ranges and one-match removal; choose a defensible shortlist and explicitly acknowledge interchangeable positions.
2. Inspect attribution, phase/format exposure and weak ordinary overlap for candidates near the cutoff; retain provisional status where evidence is thin.
3. Convert the methodology draft into the required document, verify Unstop IDs/names and enter the final two lists manually. Nothing has been submitted.

Run from the project root:
```
.venv/bin/python build_phase2.py
.venv/bin/python -m pytest -q > outputs/phase2/test_results.txt
```
Bootstrap defaults to 500 replicates with seed 42. The pipeline uses no runtime network or external player ratings. Optional historical form modelling was deferred to protect time for core validation.
''')

if __name__=='__main__':main()
