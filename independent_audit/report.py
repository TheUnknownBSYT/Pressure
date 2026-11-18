"""Evidence summaries for the targeted audit. All numbers come from run CSVs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def reports(out):
    def read(name):return pd.read_csv(out/(name+'.csv'))
    comp=read('reliability_component_audit');pressure=read('pressure_definition_sensitivity');seeds=read('bootstrap_stability')
    formula=read('reliability_formula_sensitivity');wicket=read('wicket_fairness_comparison');wrank=read('wicket_weight_ranking_sensitivity')
    families=read('legacy_scenario_family_audit');appearance=read('candidate_match_appearances');context=read('ordinary_support_and_role_bias')
    manifest=json.loads((out/'run_manifest.json').read_text());pairs=read('adjacent_rank_uncertainty')
    # Descriptive inclusion counts have an explicit, homogeneous denominator.
    p=pressure.assign(top10=pressure.R_rank.le(10)).groupby(['role','player_id']).agg(pressure_variants=('eligible','size'),pressure_eligible_variants=('eligible','sum'),pressure_top10_variants=('top10','sum'),pressure_rank_min=('R_rank','min'),pressure_rank_max=('R_rank','max'))
    s=seeds.assign(top10=seeds.R_rank.le(10)).groupby(['role','player_id']).agg(seed_rank_min=('R_rank','min'),seed_rank_max=('R_rank','max'),seed_eligible_count=('eligible','sum'),seed_valid_min=('bootstrap_valid_fraction','min'),seed_valid_max=('bootstrap_valid_fraction','max'),seed_top10_count=('top10','sum'))
    f=formula.assign(top10=formula['rank'].le(10)).groupby(['role','player_id']).agg(formula_top10_count=('top10','sum'),formula_scenarios=('scenario','size'),formula_rank_min=('rank','min'),formula_rank_max=('rank','max'))
    w=wrank.assign(top10=wrank.R_rank.le(10)).groupby(['role','player_id']).agg(wicket_weight_top10_count=('top10','sum'),wicket_weight_scenarios=('eligible','size'),wicket_weight_rank_min=('R_rank','min'),wicket_weight_rank_max=('R_rank','max'))
    t=comp.merge(p,on=['role','player_id']).merge(s,on=['role','player_id']).merge(f,on=['role','player_id']).merge(w,on=['role','player_id'])
    t['valid_gate_seed_sensitive']=t.seed_valid_min.lt(.9)&t.seed_valid_max.ge(.9)
    t['evidence_tier']=np.select([~t.eligible,t.pressure_eligible_variants.eq(18)&t.pressure_top10_variants.ge(15)&t.seed_eligible_count.eq(3)&t.A_lo95.gt(0)],
        ['Not comparable under selected support rules','Stronger absolute-pressure evidence'],default='Provisional / specification-sensitive')
    t['uplift_assessment']=np.select([t.uplift_lo95.gt(0),t.uplift_hi95.lt(0)],['Positive conditional uplift interval','Negative conditional uplift interval'],default='Uplift interval includes zero or unavailable')
    legacy=[]
    for role,label in [('bat','batters'),('bowl','bowlers')]:
        old=pd.read_csv(f'outputs/phase3/{label}_comparison_all.csv');old['role']=role
        legacy.append(old[['role','player_id','original_rank','revised_rank','scenario_top10_fraction','best_match_drop','combined_positive_match_fraction']])
    legacy=pd.concat(legacy);t=t.merge(legacy,on=['role','player_id'],how='left')
    t['existing_submission_order']=t.revised_rank.where(t.revised_rank.le(10))
    short=pd.concat([read(role+'_short_appearance_sample_sensitivity') for role in ['bat','bowl']],ignore_index=True)
    short=short[['role','player_id','eligible','R_rank','matched_pressure_balls_lost','pressure_matches_lost','overlap_fraction']].rename(columns={c:'six_ball_exclusion_'+c for c in ['eligible','R_rank','matched_pressure_balls_lost','pressure_matches_lost','overlap_fraction']})
    t=t.merge(short,on=['role','player_id'],how='left')
    contender=t.original_rank.le(20)|t.revised_rank.le(20)|t.R_rank.le(15)|t.pressure_rank_min.le(10)
    t=t[contender].copy();t.to_csv(out/'candidate_evidence_tiers.csv',index=False)
    ordinary_failure_diagnostics(out,t)
    # No new top ten: the saved ordered lists remain administrative reference
    # lists. Audit ranks are labelled diagnostic throughout.
    details=[];concentration=[];correlations=[];summary=[]
    for role,label in [('bat','batters'),('bowl','bowlers')]:
        a=t[t.role.eq(role)];b=comp[(comp.role==role)&comp.eligible]
        corr=b[['uncertainty_gap','D']].corr(method='spearman').iloc[0,1]
        correlations.append(dict(role=role,eligible_candidates=len(b),spearman_uncertainty_gap_vs_D=corr))
        q=a[(a.original_rank<=20)|(a.revised_rank<=20)].sort_values('revised_rank',na_position='last')
        details.append(f'## {label.capitalize()}\n')
        dec=read('ranking_change_decomposition_'+label).set_index('player_id')
        for r in q.itertuples():
            def rankstr(x):return 'excluded' if pd.isna(x) else str(int(x))
            d=dec.loc[r.player_id]
            text=(f'- **{r.player_name}** (ID {r.player_id}): Phase 2 #{rankstr(d.A_phase2_rank)} → global-prior 85/15 #{rankstr(d.B_prior85_original_support_rank)} → strong-support performance #{rankstr(d.C_prior85_strong_support_rank)} → saved Phase 3 #{rankstr(d.D_phase3_reliability_rank)}. '
                  f'Corrected diagnostic rank: {rankstr(r.R_rank)}; seed ranks {rankstr(r.seed_rank_min)}–{rankstr(r.seed_rank_max)}. '
                  f'Retained pressure balls {d.old_retained_pressure_balls:.0f} → {d.new_retained_pressure_balls:.0f}; overlap {r.overlap_fraction:.1%}. ')
            if not r.eligible:
                text+=f'Cannot fairly compare under these gates: {r.exclusion_reason}. This is not evidence of poor pressure performance.'
            else:
                text+=(f'Absolute positive-match share {r.f_absolute:.1%}; own-baseline positive-match share {r.f_uplift:.1%}. '
                       f'Absolute quality {r.absolute_quality:.3f}, uplift {r.uplift_quality:.3f}; conditional uplift interval [{r.uplift_lo95:.3f}, {r.uplift_hi95:.3f}]. '
                       f'Pressure definitions: eligible {r.pressure_eligible_variants}/18, top ten {r.pressure_top10_variants}/18; formula top ten {r.formula_top10_count}/{r.formula_scenarios}. '
                       f'{r.brief_matches_lt6:.0f}/{r.pressure_matches:.0f} pressure matches have fewer than six legal balls ({r.brief_balls_lt6:.0f} balls); downweighting these changes f from {r.f_absolute:.3f} to {r.f_absolute_downweight6:.3f}. '
                       f'Tier: {r.evidence_tier}. Entire-pressure-sample six-ball exclusion: eligible={r.six_ball_exclusion_eligible}; loses {r.six_ball_exclusion_matched_pressure_balls_lost:.0f} matched balls and {r.six_ball_exclusion_pressure_matches_lost:.0f} positive-exposure matches. ')
                cx=context[(context.role==role)&context.player_id.eq(r.player_id)&context.setting.eq('matched')&context.high.eq(1)]
                dominant=[]
                for dim in ['competition_type','phase_pre','calendar_year','opposition','venue_id']:
                    gg=cx[cx.dimension.eq(dim)]
                    if len(gg):
                        one=gg.loc[gg.exposure_share.idxmax()]
                        dominant.append(f"{dim}={one['group']} ({one.exposure_share:.0%} of pressure balls; {one.matches} matches)")
                text+=' Largest retained-exposure groups: '+', '.join(dominant)+'. These are concentration diagnostics, not adjusted opposition/venue effects.'
            details.append(text+'\n')
        selected=a[a.revised_rank.le(10)].sort_values('revised_rank')
        robust=a[(a.pressure_top10_variants>=15)&a.pressure_eligible_variants.eq(18)&a.eligible]
        summary.append(dict(role=role,original_top10_names=selected.player_name.tolist(),pressure_robust_names=robust.player_name.tolist(),
             corrected_top10_names=b.nsmallest(10,'R_rank').player_name.tolist(),eligible=len(b),correlation=corr,
             excluded_names=a[(a.original_rank.le(20)|a.revised_rank.le(20))&~a.eligible].player_name.tolist()))
        # Concentration uses retained high-pressure exposure AND absolute
        # residual contribution, so negative/cancelling totals cannot inflate shares.
        c=context[(context.role==role)&context.setting.eq('matched')&context.high.eq(1)&context.player_id.isin(q.player_id)].copy()
        scales_old=pd.read_csv(f'outputs/phase2/phase2_{label}.csv');scales=[scales_old.loc[scales_old.eligible,k].std() for k in ['pressure_run_adj','pressure_wicket_adj']]
        c['signed_quality_numerator']=.85*c.rr/scales[0]+.15*c.wr/scales[1]
        c['absolute_contribution_share']=c.signed_quality_numerator.abs()/c.groupby(['player_id','dimension']).signed_quality_numerator.transform(lambda x:x.abs().sum()).replace(0,np.nan)
        for (pid,dim),g in c.groupby(['player_id','dimension']):
            j=g.exposure_share.idxmax();v=g.loc[j]
            concentration.append(dict(role=role,player_id=pid,player_name=v.player_name,dimension=dim,largest_exposure_group=v['group'],largest_exposure_share=v.exposure_share,
                largest_absolute_contribution_share=g.absolute_contribution_share.max(),groups=g['group'].nunique(),represented_matches_in_largest_group=v.matches,
                small_group_warning=bool(v.matches<5)))
    pd.DataFrame(correlations).to_csv(out/'uncertainty_deletion_overlap.csv',index=False)
    pd.DataFrame(concentration).to_csv(out/'candidate_context_concentration.csv',index=False)
    (out/'candidate_review.md').write_text('# Candidate evidence review\n\nRanks are descriptive decisions, not statistically distinct positions. A missing rank is a comparison-support problem. Counts across model choices are not probabilities.\n\n'+'\n'.join(details))
    # Family mixing changes the meaning even if some flags happen to agree.
    fam=families.pivot(index=['role','player_id'],columns='family',values='top10_fraction').reset_index().merge(legacy,on=['role','player_id'])
    fam['mixed_below_80pct']=fam.scenario_top10_fraction.lt(.8)
    fam['reliability_family_below_80pct']=fam.reliability_formula.lt(.8)
    fam['frequency_flag_disagrees']=fam.mixed_below_80pct.ne(fam.reliability_family_below_80pct)
    fam.to_csv(out/'legacy_fragility_flag_comparison.csv',index=False)
    changed=comp[comp.loo_cohort_change_fraction.gt(0)]
    # Original deletion penalty versus fixed-support exact point calculation.
    dd=comp.merge(legacy[['role','player_id','best_match_drop']],on=['role','player_id'])
    dd['fixed_minus_saved_deletion_penalty']=dd.D-dd.best_match_drop
    comparable_deletions=dd.D.notna()&dd.best_match_drop.notna()
    changed_penalties=int((dd.loc[comparable_deletions,'fixed_minus_saved_deletion_penalty'].abs()>1e-8).sum())
    dd[['role','player_id','player_name','D','best_match_drop','fixed_minus_saved_deletion_penalty','loo_valid_fraction','loo_cohort_change_fraction']].to_csv(out/'deletion_penalty_comparison.csv',index=False)
    issues=f'''# Confirmed code issues versus design choices

Historical source files and Phase 1–3 outputs are unchanged. Corrections are isolated in `independent_audit/core.py`; use the new runner for audit results, not the historical runner.

## Confirmed implementation issues and fixes

1. **Changing deletion estimand.** `phase3_pipeline.audit.reliability` starts from strong 20-ball/three-match support, then calls `summarize` at 10/two after every deletion. Cells can disappear; their pressure events and ordinary baseline disappear together. That is not a fixed-cohort influence calculation. The audit freezes the original cells, deletes the entire match including ordinary outcomes, and recomputes rates. Loss of necessary ordinary exposure is explicitly invalid, not silently rematched. The new diagnostics detect possible cohort changes for {len(changed)} player-role rows. However, **{changed_penalties} of {int(comparable_deletions.sum())} comparable saved deletion penalties changed** at tolerance 1e-8: the problematic deletion need not be the worst deletion. This correction does not explain the leading rank movements. See deletion_penalty_comparison.csv.

2. **Zero pressure exposure resamples.** `summarize.estimate` uses max(hb,1) and zero weights when hb=0, creating finite zero (or illegal-only residual) rates and a neutral ordinary comparison. Such resamples do not estimate performance per legal ball. The audit returns missing rates, counts the failures, and keeps them out of conditional intervals. No fabricated zero enters scoring. The observed frequencies are in bootstrap_stability.csv; a synthetic regression test demonstrates the failure even if a candidate never encounters it in these seeds.

3. **Order-dependent random streams.** The historical single RNG advances over players, so adding an unrelated earlier ID changes another player's draws. This is a reproducibility weakness, not evidence of bias in expectation. The audit hashes seed, role and player ID. A regression test verifies invariance to other candidates. Joint-rank diagnostics instead use shared global match draws.

## Eligibility clarification, not an established attribution defect

Historical match counts include illegal-only appearances while f excludes them. Actual illegal runs/wickets must remain in numerators; erasing them is wrong. The audit counts only positive-legal-exposure matches for exposure-based eligibility and shrinkage, keeps illegal outcomes, and deletes illegal-only matches in influence analysis too. This is an explicit conservative eligibility convention. The count-correction bridge isolates it from stronger matching. Run and wicket event totals reconcile with player/match totals; source ID/category attribution checks pass. Recorded wides include legitimate stumped dismissals. Source consistency is not external scorecard validation.

## Concerns that are statistical/reporting choices

- Reports define Q20 as the **20th bootstrap percentile**; no inspected Phase 3 report calls it a 95% lower bound. Internal `lcb` names are ambiguous; the audit uses Q20 and separate A_lo95/A_hi95. Q20 is not even a calibrated 80% confidence bound under selection and missing comparisons.
- Short appearances receive equal votes in the historical f; this is a subjective design, not an arithmetic bug. The audit reports 3/6/12-ball exclusions from the consistency vote and capped exposure weights. Core quality estimates keep all valid outcomes. Reported lost balls/matches concern the consistency vote only.
- Q20 and D can penalize correlated sampling variation. Their association and score ablations diagnose overlap; neither proves a calibrated optimal coefficient. No new coefficient is adopted.
- Invalid ordinary resamples were dropped explicitly, but conditional quantiles can look optimistic. Missing-mass quantile bounds in bootstrap_stability.csv show sensitivity without inventing replacement outcomes.
- The Phase 3 mixed scenario frequency pools six performance, nine formula and nine gate scenarios. It is neither a pressure-robustness measure nor a probability. {int(fam.frequency_flag_disagrees.sum())} player-role frequency flags disagree with the formula-only frequency at 80%; this count includes non-final players and compares only the frequency part of the flag, not the CI condition. Separate family output is supplied.
- Source inspection confirms Phase 3 never called pressure_variant. The new run executes all 18 combinations of 20/25/30%, stakes on/off and leverage .05/.10/.15 with its corrected ranking engine, new support and eligibility each time. Phase 2 percentages are not reused.

## Remaining implementation scope

Phase 1/2 estimation, chronological calibration, and historical final files are intentionally untouched. Audit fixtures and all existing tests run together; tests establish implementation behavior, not the validity of the modelling assumptions. No new final top-ten files are generated.
'''
    (out/'confirmed_code_issues.md').write_text(issues)
    death=wicket[(wicket.role=='bowl')&(wicket.dimension=='phase_pre')&(wicket.group=='death')].set_index('model')
    brief=appearance.pressure_legal.lt(6)
    baseline_eligible_seeds=seeds.merge(comp[['role','player_id','base_eligible']],on=['role','player_id'])
    zero_eligible_max=baseline_eligible_seeds.loc[baseline_eligible_seeds.base_eligible,'bootstrap_zero_pressure_fraction'].max()
    valid_gate=seeds.groupby(['role','player_id']).agg(valid_min=('bootstrap_valid_fraction','min'),valid_max=('bootstrap_valid_fraction','max'),eligible_seeds=('eligible','sum')).reset_index()
    crossing=valid_gate[valid_gate.valid_min.lt(.9)&valid_gate.valid_max.ge(.9)].merge(comp[['role','player_id','player_name']],on=['role','player_id'])
    crossing.to_csv(out/'validity_gate_seed_crossings.csv',index=False)
    seed_status_changes=int(valid_gate.eligible_seeds.between(1,2).sum())
    limitation='''All intervals condition on fitted run/wicket expectations, selected pressure definitions and retained comparison cells. They omit model-fitting and selection uncertainty. Player-wise bootstrap seeds preserve within-player match dependence; global match draws preserve shared-match dependence for a separate, explicitly linearized rank diagnostic. That diagnostic holds eligibility, f, D and the Q20 offset fixed and is not a nested bootstrap of the full selection algorithm. Failed candidates make rank inclusion only partially identified, so missing-candidate worst/best bounds are reported.

Own-baseline consistency uses each player's pooled ordinary comparison, mixed to each match's pressure contexts; it is descriptive and shares observations with the overall estimate. It is not independent replication or prospective pressure-uplift prediction. A one-over minimum is a convention tested alongside three/twelve balls, not an optimum. Venue and opposition diagnostics are descriptive; sparse groups are not fitted effects. Batting position is a reported-event proxy, not necessarily an exogenous intended role. Bowling usage is previous legal balls in that match, not an inferred bowling-style label. Phase, innings and competition are matched coarsely; wicket/demand expectations adjust more finely. Source availability and exclusions can be informative.

First-innings probability overconfidence remains. The saved forward calibration experiment was not adopted and this audit does not partially adjust its probabilities. All 18 pressure scenarios use internally consistent saved components, not refitted or recalibrated win models. Pressure percentiles and ordinary baselines are retrospective. No causal psychological effect, external validation, unique true top ten, or calibrated scenario-inclusion probability is claimed.'''
    method=f'''# Submission methodology corrections

## Recommendation

Retain the existing ordered lists only as provisional submission reference lists. Do not mechanically replace them with seed-42 corrected diagnostic ranks. The review establishes support exclusions, uncertainty and role-fairness problems; it does not validate a new reliability formula or a unique alternative top ten. Candidate evidence tiers and candidate_review.md explain where the existing order lacks support. Adjacent overlapping conditional difference intervals preclude claims of precise separation; an ordered list can still meet the competition's formatting requirement.

Describe the objective as **strong adjusted pressure performance with descriptive consistency and own-baseline comparison**, rather than proven repeatable pressure improvement. A is 75% absolute and 25% uplift. f in the original score measures positive absolute performance. It cannot establish consistently positive uplift. Show both f_absolute and f_uplift, uplift intervals and sample support.

## Fixes and uncertainty

Use fixed comparison support for deletion influence, mark missing denominators as unavailable, retain illegal-event numerators, and use stable player seeds. Clearly state the positive-exposure match convention. Q20 is a lower performance quantile, not a 95% confidence limit. Keep the original subjective R coefficients only as an audit reference. The simple M/(M+8) performance comparator uses shrinkage alone for sample conservatism and reports consistency separately; it does not stack shrinkage and Q20. The comparator is a sensitivity, not a validated replacement.

Bootstrap: {manifest['bootstrap']:,} whole-match replicates per player for every pressure/wicket scenario, and seeds 42/137/2026 for baseline Monte Carlo stability. All potentially eligible players are reranked, not just incumbents. Gate, formula, pressure, and wicket/weight scenarios have separate files and denominators. Do not pool their inclusion counts.

## Wicket fairness correction

The global prior is not established as superior by overall MSE: paired intervals against contextual and pooled stratified priors include zero. It ignores phase opportunity. For bowlers at the death, global expected-minus-actual error averages {death.loc['global','signed_error_per100']:.2f} wickets per 100 legal balls across match-phase units, compared with {death.loc['stratified','signed_error_per100']:.2f} for the pooled phase/competition prior. Death-phase MSE falls from {death.loc['global','mse']:.6f} to {death.loc['stratified','mse']:.6f}. Pooling uses 200 earlier legal balls toward the earlier global rate; all training precedes the test quarter. Phase-specialist fairness therefore argues against treating the global prior as a privileged primary model. Present the contextual and pooled stratified alternatives alongside it. No automatic wicket-model replacement is certified here, and 70/30, 85/15 and runs-only remain explicit value choices/sensitivities.

## Support and fairness

An exclusion means insufficient ordinary comparison support under the chosen rule, not bad pressure performance. Show A→B→C→D decomposition: original performance, wicket/weight changes on original support, performance on stronger support, then reliability on the exact same eligible population. Rank differences are order-dependent and non-additive; the constant-shrink score difference and gate-only rerank isolate distinct mechanisms. Supported subsets can be narrower and less representative even when their conditional bootstrap becomes more stable.

Use ordinary_support_and_role_bias.csv and candidate_context_concentration.csv to disclose dominance by phase, competition, innings, season, opposition, venue and available role proxies. These summaries diagnose possible confounding; they do not remove it.

## Required limitations

{limitation}
'''
    (out/'submission_methodology_corrections.md').write_text(method)
    sections=[]
    for ss in summary:
        role=ss['role'];a=t[t.role.eq(role)];original=a[a.revised_rank.le(10)].sort_values('revised_rank')
        robust=', '.join(ss['pressure_robust_names']) or 'none under the stated descriptive threshold'
        provisional=', '.join(original[~original.evidence_tier.eq('Stronger absolute-pressure evidence')].player_name) or 'none'
        moved=[]
        for r in original.itertuples():
            if pd.isna(r.R_rank) or abs(r.R_rank-r.revised_rank)>=2:
                moved.append(f'{r.player_name}: saved #{int(r.revised_rank)} → '+('ineligible' if pd.isna(r.R_rank) else f'diagnostic #{int(r.R_rank)}')+f'; seed range {r.seed_rank_min:g}–{r.seed_rank_max:g}')
        sections.append(f'''### {'Batters' if role=='bat' else 'Bowlers'}

- More stable across pressure choices (eligible 18/18 and top ten at least 15/18; descriptive rule): {robust}.
- Existing-list provisional candidates: {provisional}.
- Top-20 contenders failing the default comparison rules: {', '.join(ss['excluded_names']) or 'none'}. Exclusion is not adverse performance evidence.
- Material diagnostic movements among the saved top ten: {'; '.join(moved) or 'none by the two-place-or-exclusion rule'}.
- Eligible baseline candidates: {ss['eligible']}; Spearman association between Q20 uncertainty gap and deletion penalty: {ss['correlation']:.2f}. This correlation does not identify an optimal penalty.
''')
    eligiblecomp=comp[comp.eligible];zeromax=seeds.bootstrap_zero_pressure_fraction.max()
    start=f'''# Independent review — start here

## Three most important confirmed findings

1. **The rating does not establish consistent uplift.** A is predominantly absolute performance and the consistency vote is absolute. The new files separate own-baseline improvement. Across the audited top-20/15 candidate match appearances, {int(brief.sum())}/{len(appearance)} ({brief.mean():.1%}) have fewer than six pressure legal balls, accounting for only {appearance.loc[brief,'pressure_legal'].sum()/appearance.pressure_legal.sum():.1%} of their pressure exposure. Equal voting gives those brief appearances disproportionate influence relative to ball exposure. This does not make exposure weighting uniquely correct.
2. **Support and specification changes explain selection movements.** A→B→C→D decomposition isolates wicket/weight changes, eligibility, changed ordinary contexts and reliability components. Excluded players are not shown to be poor performers. All 18 Phase 3 pressure-definition combinations have now been rebuilt and reranked. Earlier Phase 2 frequencies were not reused.
3. **The global wicket prior conceals phase-specific bias.** Whole-match MSE differences are inconclusive, yet global death-over bowler predictions understate wickets by {abs(death.loc['global','signed_error_per100']):.2f} per 100 balls on average across match-phase units. Phase-aware alternatives substantially reduce that bias and death-phase error. Global prior plus 85/15 is not a statistically optimal method.

## Exactly what changed

Historical pipelines, original CSVs and all earlier outputs are preserved and hash-verified. New audit code freezes ordinary comparison cells for deletion, marks zero-exposure bootstrap draws missing, and makes random draws player-independent. Eligibility now explicitly counts positive-exposure matches, with a bridge to the historical event-match convention. Confirmed mechanisms, observed impact and statistical choices are separated in confirmed_code_issues.md. **None of the saved, comparable deletion penalties changed** in this dataset; this bug mechanism did not cause the leading rank movements. Zero-pressure resamples reach {zeromax:.2%} among very sparse ineligible players, but at most {zero_eligible_max:.2%} among those satisfying the basic sample gates across the three seeds. These edge cases must not be exaggerated into a claim that the prior leading ranks were caused by coding errors.

## Rankings, robustness and evidence tiers

**Do not certify a replacement final top ten from this audit.** Keep the saved ordered lists as provisional reference lists; corrected ranks are diagnostics. Their differences must be considered with seed ranges, pressure eligibility and conditional uncertainty. No new formula is adopted. A stronger evidence tier refers to absolute-pressure evidence, not proven uplift or precise rank. It requires default eligibility, positive conditional A interval, eligibility in 18/18 pressure variants, top-ten inclusion in at least 15/18 and eligibility under all three baseline seeds. Those are transparent descriptive thresholds, not confidence levels.

{''.join(sections)}
Adjacent-rank conditional difference intervals are in adjacent_rank_uncertainty.csv; candidate_evidence_tiers.csv retains existing_submission_order while grouping evidence. Read candidate_review.md for each original/revised top-20 candidate, including all named review concerns. A point-rank change alone is insufficient reason to reorder a close position.

The 90% bootstrap-validity threshold is crossed across seeds by {len(crossing)} player-role rows, but {seed_status_changes} candidates actually change full baseline eligibility across seeds because the other gates also apply. validity_gate_seed_crossings.csv lists them. ordinary_bootstrap_failure_diagnostics.csv identifies thin ordinary cells and their exact cell-specific bootstrap failure chances under fixed-sample multinomial draws; these dependent cell chances must not be added together. Decomposition files retain Phase 2 and Phase 3 valid-comparison fractions so improved conditional support is visible alongside lost pressure exposure.

## What was executed

- All 18 pressure variants: 20/25/30% × stakes on/off × leverage scale .05/.10/.15, reconstructing ordinary/high samples, support, estimates, eligibility and ranks for all candidates.
- Three wicket models × three run weights (70%, 85%, 100%), using the same fixed-support reliability engine; phase and competition predictive diagnostics use earlier-only training.
- {manifest['bootstrap']:,} match-cluster replicates in every scenario; three stable-player seeds for baseline, plus three shared-global-match seeds for separate linearized ranking diagnostics.
- Formula and ordinary-overlap/validity gates tested in separate families; 3/6/12-ball consistency-vote exclusions and capped weights, absolute versus uplift consistency, no-D, no-quantile and shrinkage-only comparators.
- An additional six-ball **whole-pressure-sample exclusion** reruns support, estimates, bootstrap and eligibility, preserves all ordinary events and keeps the original pressure-exposure denominator when measuring coverage loss. Results are in bat_short_appearance_sample_sensitivity.csv and bowl_short_appearance_sample_sensitivity.csv. This is distinct from changing only the consistency vote; it is not adopted as a new eligibility optimum.
- All existing tests plus new regression tests. Pytest exit code: {manifest['pytest_exit_code']}. Exact output: test_results.txt. Event residual/attribution invariants and prior-file hash checks passed.

## Remaining limitations and submission decision

{limitation}

The remaining limitation is statistical identification, not a skipped mandatory pressure run. No refit of Phase 1/2 or prospective validation was attempted. No nested bootstrap of every pressure/model choice was run. The audit supplies evidence for qualifying claims and deferring unsupported reorderings, rather than choosing names by reputation.

## Files and reproduction

Start with candidate_review.md and submission_methodology_corrections.md. Required CSVs: ranking_change_decomposition_batters.csv, ranking_change_decomposition_bowlers.csv, pressure_definition_sensitivity.csv, ordinary_support_and_role_bias.csv, wicket_fairness_comparison.csv, reliability_component_audit.csv, bootstrap_stability.csv and candidate_evidence_tiers.csv. Plots are in figures/. Supporting family, gate, joint-rank, exposure, context-concentration and deletion audits are separate CSVs. run_manifest.json records settings, code hashes and protected-file hashes.

From the project root:

```sh
.venv/bin/python build_independent_audit.py --bootstrap 2000
.venv/bin/python -m pytest -q
```

The runner only writes outputs/independent_audit and refreshes its own audit files. It never invokes the earlier builders. No results were submitted or uploaded.
'''
    (out/'START_HERE.md').write_text(start)
    plots(out,t,comp,pressure,wicket)


def plots(out,t,comp,pressure,wicket):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (out/'figures').mkdir(exist_ok=True)
    for role,label in [('bat','batters'),('bowl','bowlers')]:
        a=t[t.role.eq(role)&t.revised_rank.le(15)].sort_values('revised_rank')
        fig,ax=plt.subplots(figsize=(10,7));y=np.arange(len(a));ax.barh(y-.18,a.f_absolute,height=.35,label='Positive absolute quality');ax.barh(y+.18,a.f_uplift,height=.35,label='Positive own-baseline uplift')
        ax.set_yticks(y,a.player_name);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel('Fraction of positive-exposure pressure matches');ax.set_title(f'{label.capitalize()}: absolute versus own-baseline consistency');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.10),ncol=2);fig.tight_layout();fig.savefig(out/'figures'/f'{label}_consistency.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,7));vals=[]
        for pid in a.player_id:
            v=pressure[(pressure.role==role)&pressure.player_id.eq(pid)].R_rank.dropna().to_numpy();vals.append(v)
        ax.boxplot(vals,vert=False,tick_labels=[f'{r.player_name} ({int(r.pressure_eligible_variants)}/18 eligible)' for r in a.itertuples()],showfliers=True)
        ax.invert_yaxis();ax.set_xlabel('Rank conditional on eligibility; 18 pressure specifications');ax.set_title(f'{label.capitalize()}: pressure-definition rank sensitivity');fig.tight_layout();fig.savefig(out/'figures'/f'{label}_pressure_sensitivity.png',dpi=150);plt.close(fig)
    x=wicket[(wicket.role=='bowl')&(wicket.dimension=='phase_pre')]
    fig,ax=plt.subplots(figsize=(9,5))
    for model,g in x.groupby('model'):
        g=g.set_index('group').loc[['powerplay','middle','death']].reset_index()
        ax.plot(g['group'],g.signed_error_per100,marker='o',label=model)
    ax.axhline(0,color='grey',linewidth=.8);ax.set_xlabel('Innings phase');ax.set_ylabel('Expected minus observed wickets per 100 legal balls\nMean across match-phase units');ax.set_title('Bowler wicket expectations: forward phase-specific bias');ax.legend();fig.tight_layout();fig.savefig(out/'figures'/'wicket_phase_bias.png',dpi=150);plt.close(fig)


def ordinary_failure_diagnostics(out,contenders):
    from independent_audit.core import prepare
    m=pd.read_csv('outputs/phase2/phase2_player_match_metrics.csv');rows=[]
    for role,c in contenders.groupby('role'):
        ids=set(c.loc[c.original_rank.le(20)|c.revised_rank.le(20)|c.R_rank.le(15),'player_id'])
        for pid,a in m[m.role.eq(role)&m.player_id.isin(ids)].groupby('player_id'):
            for balls,matches,positive,setting in [(10,2,False,'original_10_2'),(20,3,True,'audit_20_3')]:
                r,arr,mids,cells,z=prepare(a,balls,matches,positive)
                n=len(mids)
                for j,cell in enumerate(cells):
                    ordinary=arr[:,j,0,0]>0;high=arr[:,j,1,0]>0
                    # Exact event: at least one required high-pressure legal ball
                    # is sampled but none of that cell's ordinary exposure is.
                    probability=(1-ordinary.mean())**n-(1-(ordinary|high).mean())**n
                    rows.append(dict(role=role,player_id=pid,setting=setting,comparison_cell=cell,matched_pool_matches=n,
                        ordinary_positive_exposure_matches=int(ordinary.sum()),ordinary_legal=arr[:,j,0,0].sum(),pressure_legal=arr[:,j,1,0].sum(),
                        pressure_exposure_share=arr[:,j,1,0].sum()/max(r['pressure_legal'],1),cell_missing_comparison_probability=probability))
    pd.DataFrame(rows).merge(contenders[['role','player_id','player_name']],on=['role','player_id'],how='left').to_csv(out/'ordinary_bootstrap_failure_diagnostics.csv',index=False)
