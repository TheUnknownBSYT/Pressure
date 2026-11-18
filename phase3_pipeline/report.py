from pathlib import Path
import numpy as np
import pandas as pd
from pressure_pipeline.report import md_table

def reports(out,results,v,cal,decision,scenarios):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(out/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'figures';folder.mkdir(exist_ok=True)
    sections=[]
    for role,label in [('bat','batters'),('bowl','bowlers')]:
        r=results[role];t=pd.read_csv(out/f'final_{label}_top10.csv')
        sections.append(f'## Recommended ordered {label}\n\n'+md_table(t[['final_order','player_id','player_name','original_rank','pressure_matches','pressure_legal','combined_positive_match_fraction','reliability_score','position_flag']]))
        for p in t.itertuples():
            sections.append(f'- {p.final_order}. **{p.player_name}**: original rank {p.original_rank:.0f}, revised reliability score {p.reliability_score:.3f}; favorable combined contribution in {p.combined_positive_match_fraction:.0%} of matched pressure matches. Conditional unshrunk performance interval [{p.performance_lo95:.2f}, {p.performance_hi95:.2f}], worst one-match removal drop {p.best_match_drop:.2f}, valid ordinary-bootstrap support {p.bootstrap_valid_uplift_fraction:.0%}. {p.position_flag}.')
        z=r[(r.original_rank<=15)|(r.revised_rank<=15)].sort_values('revised_rank')
        sections.append('\n### Original/revised top-15 union and reasons\n\n'+md_table(z[['player_name','original_rank','performance_prior85_rank','revised_rank','rank_improvement','change_reason']],45))
        sections.append('\n### Ordinary-support investigation\n\n'+md_table(r[r.original_rank.le(15)][['player_name','original_bootstrap_valid_uplift_fraction','bootstrap_valid_uplift_fraction','overlap_fraction','ordinary_matches','ordinary_legal','exclusion_reason']],20))
        sr=scenarios[(scenarios.role==role)&scenarios.scenario.str.contains('_runs')]
        piv=sr.pivot(index=['player_id','player_name'],columns='scenario',values='rank')
        piv=piv[piv.le(15).any(axis=1)].reset_index()
        piv.to_csv(out/f'{label}_wicket_top15_changes.csv',index=False)
        sections.append('\n### Six wicket choices: top-15 entrants, exits and reordering\n\n'+md_table(piv,60))
        sections.append('\nThese columns isolate wicket assumptions while preserving Phase 2 performance-first scoring and original eligibility/scales. A change from contextual_0.7 to prior_0.85 reflects both simpler wicket expectations and reduced wicket weight; the intervening columns separate those effects. The final reliability rank additionally changes ordinary support, removes match-count shrinkage, and considers consistency and uncertainty. Its movement must not be attributed solely to wicket weighting.')
        fig,ax=plt.subplots(figsize=(10,max(6,len(z)*.32)))
        y=np.arange(len(z));ax.scatter(z.original_rank,y,label='Original');ax.scatter(z.revised_rank,y,label='Reliability alternative');ax.set_yticks(y,z.player_name);ax.invert_yaxis();ax.set_xlabel('Rank; no revised marker = fails revised eligibility');ax.legend();fig.tight_layout();fig.savefig(folder/f'{label}_rank_changes.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,6),sharey=True);y=np.arange(len(t))
        axes[0].barh(y,t.combined_positive_match_fraction);axes[0].set_yticks(y,t.player_name);axes[0].invert_yaxis();axes[0].set(xlim=(0,1),xlabel='Fraction of favorable pressure matches')
        axes[1].hlines(y,t.performance_lo95,t.performance_hi95);axes[1].scatter(t.performance_lcb,y,label='20th bootstrap percentile');axes[1].axvline(0,color='gray',linestyle='--');axes[1].set_xlabel('Unshrunk performance, conditional 95% interval');axes[1].legend();fig.tight_layout();fig.savefig(folder/f'{label}_reliability.png',dpi=150);plt.close(fig)
        sections.append(f'\n![Rank comparison](figures/{label}_rank_changes.png)\n\n![Consistency and uncertainty](figures/{label}_reliability.png)\n')
    cp=pd.read_parquet(out/'calibration_predictions.parquet')
    cp=cp[cp.calibrated.notna() & cp.match_date_parsed.ge('2026-01-01')].copy()
    cp['w']=1/cp.groupby('match_id').match_id.transform('size')
    rows=[]
    for threshold in [.8,.9]:
        a=cp[cp.win_probability_pre.ge(threshold)]
        rows.append(dict(raw_threshold=threshold,matches=a.match_id.nunique(),events=len(a),raw_mean=np.average(a.win_probability_pre,weights=a.w),calibrated_mean=np.average(a.calibrated,weights=a.w),observed=np.average(a.batting_win_label_outcome,weights=a.w)))
    tail=pd.DataFrame(rows);tail.to_csv(out/'calibration_common_tail.csv',index=False)
    bins=pd.read_csv(out/'calibration_bins.csv');fig,ax=plt.subplots(figsize=(7,5))
    for model,g in bins.groupby('model'):ax.plot(g.predicted,g.observed,'o-',label=model)
    ax.plot([0,1],[0,1],'k--');ax.set(xlabel='Predicted win probability',ylabel='Observed win fraction',title='2026 chronological first-innings calibration');ax.legend();fig.tight_layout();fig.savefig(folder/'calibration_heldout.png',dpi=150);plt.close(fig)
    intro=f'''# Phase 3 submission methodology recommendation

This is a targeted robustness and final-selection pass. Earlier phases and proprietary inputs are preserved and hash-checked. The ordered recommendations implement a transparent decision rule under uncertainty. They do not establish statistically distinct adjacent positions or causal psychological effects.

## Pressure and underlying player comparison

Phase 1 retains 200,999 recorded events across 867 matches, with 113,031 pressure-scored events. Its provisional formula is P=100[.50 L+.30 U C+.20 S], with C=4p(1-p), a chronological innings-specific logistic win model, counterfactual next-legal-ball probability spread for L, explicit urgency and subjective stage stakes. High pressure uses the retrospective global top quarter, cutoff 36.585462. No Phase 1 probabilities or pressure labels are changed in Phase 3. Revised targets with unknown announcement times, ambiguous extras and other quality issues remain excluded.

Phase 2 supplies 110,255 outcome-analysis events. Runs and wicket expectations are trained only on previous calendar quarters, pooling contexts of innings, phase, competition, wickets and chase rate with 200 legal-ball prior exposure. Batting favorable residuals are runs above expectation and expected-minus-observed striker dismissals; bowling favorable residuals are runs prevented and credited wickets above expectation. Ordinary residuals are reweighted to each player's supported high-pressure innings/phase/competition distribution. Illegal events retain actual runs/wickets with zero legal exposure; no extra expected-outcome credit is earned by an illegal event. Names are joined only after estimates using stable IDs. Exact role, opposition, venue and tactical confounding remain unmodelled.

## Wicket attribution and predictive audit

{md_table(v)}

Errors are on the SAME Phase 2 available match population, unlike comparisons that give small folds the same weight as large folds. Each match contributes equally. Difference = contextual minus simple prior error; positive favors the simpler prior. Intervals resample matches 2,000 times. Training of each prior uses only earlier structurally valid, outcome-valid matches, and no evaluated match. These are legal-exposure rate errors, not per-event Bernoulli Brier scores.

Every event's striker dismissal and bowler-credit flag was checked against original source IDs/categories. Non-striker run-outs are not charged to the striker; run-outs are never bowler wickets. Invalid dismissal IDs and uncertain concessions remain excluded. Attribution is algebraically verified against supplied fields, not an independent official scorecard adjudication.

Six alternatives compare 70/30, 85/15 and runs-only scoring under contextual versus simple earlier-history wicket expectations. The recommendation uses 85/15 and historical-prior wickets: run contextual expectations have clearer validation; a smaller wicket component retains wicket-taking/avoidance value while avoiding unsupported fine-context precision. The simpler prior loses phase-specific wicket adjustment, so it is a tradeoff, not a proven superior model. Weights were set before viewing Phase 3 names and are not optimized for famous players. The six-way table records changes in both top-15 lists, including entrants and exits.

## Reliability-first method

Ordinary context cells now require >=20 legal balls in >=3 matches, versus 10/2. Matched summaries and 500 whole-match bootstrap replicates are recomputed. Retain at least 8 pressure matches and 50 pressure legal balls, then require >=75% pressure overlap, >=100 ordinary legal balls across >=8 matches, and >=90% bootstrap replicates with necessary ordinary comparisons. Original eligibility and ranks remain in comparison files. ordinary_support_investigation.csv records before/after sample sizes, retained overlap and bootstrap failure rates for every player. Removing thin cells can improve bootstrap validity but reduce represented pressure situations; no missing comparison is filled with zero.

Let A=.75 times absolute pressure quality + .25 times own ordinary uplift. Each quality uses .85 runs + .15 favorable wickets, divided by their original Phase 2 eligible-cohort standard deviations. A is unshrunk. Let Q20 be the 20th bootstrap percentile of A. Let f be the fraction of matched pressure matches with positive combined absolute quality. Let D=max(0,A-minimum A after removing any one pressure match AND its ordinary observations). All possible pressure-match removals are tested, not just the largest run contribution.

Reliability score = .75 Q20 + .25(2f-1) - .25 D.

The final order sorts this score among supported candidates. There is no additional M/(M+8) shrinkage: bootstrap uncertainty is the estimation penalty and sample counts are eligibility gates. Consistency and deletion robustness describe distribution and influence; they are correlated with uncertainty and must not be treated as independent evidence. Their coefficients are subjective. A short pressure appearance has the same vote in f as a long one, a remaining limitation.

We test Q10/Q20/Q30, consistency weights .10/.25/.40, retained overlaps .50/.75/.90 and bootstrap validity .80/.90/.95. With six performance-first wicket alternatives this gives 24 Phase 3 scenarios per role. Scenario inclusion is a descriptive frequency, not a calibrated probability. Phase 2's 42-scenario pressure/weight results remain separately available as original_top10_scenario_fraction.

Confidence intervals are for A conditional on fitted expectations, pressure choices and ordinary matching, not for the entire reliability ranking. Replicates losing ordinary support are missing; the retained fraction is explicit. Intervals exclude uncertainty in model selection and winner selection across many players. Fragile flags indicate a conditional interval including zero or top-ten inclusion under 80% across the specified Phase 3 scenarios. Adjacent ranks remain uncertain even when a candidate passes that flag.

## Forward first-innings calibration

{md_table(cal)}

A regularized logistic calibrator of logit(raw p) trains each quarter only on earlier OUT-OF-FOLD first-innings predictions/outcomes, after >=100 matches. All matches on a date stay on one side. The scored evaluation uses 93 available 2026 matches. These are held out from each fit but have been inspected during earlier development; they are not a pristine external validation set.

Calibrated-minus-raw Brier difference: {decision['brier_difference']:.6f}; 95% match-bootstrap interval [{decision['bootstrap_lo95']:.6f}, {decision['bootstrap_hi95']:.6f}]. Log loss and fixed-bin expected calibration error improve, but the interval includes no Brier benefit. The pre-specified adoption gate requires >=60 matches, an upper Brier-difference confidence bound below zero, and lower log loss and calibration error. The gate failed, so calibration is NOT adopted. Raw observed and counterfactual probabilities, leverage, pressure classifications and player cohort remain mutually consistent with Phase 1. Calibration research predictions never enter the rankings. If this gate passes on a future build, the runner stops before recommendations until a complete observed/counterfactual rebuild is implemented.

{md_table(tail)}\n\nThe tail comparison uses the SAME states selected by original probability, retaining full-match evaluation weights. At raw p>=.9 only nine matches contribute: mean prediction falls from about .927 to .908, while observed wins are about .687. Upper-tail overconfidence therefore remains substantial despite better overall metrics; sparse correlated states cannot establish precise tail calibration.\n\n![Chronological calibration](figures/calibration_heldout.png)

## Selection and limits

The final lists are defensible decision recommendations under an explicit reliability preference; they are not proof of unique true top tens. The full top-15 comparisons distinguish wicket changes, performance-only ranks and reliability ranks. Fragile positions must retain their caveats in the competition report. No reputation adjustment, causal psychological claim or current-outcome predictive leakage is used.

The ordinary player baselines and high-pressure threshold remain retrospective descriptive quantities. Match bootstrapping does not cure non-random pressure exposure, exclusions, sparse contexts or weak first-innings calibration. The follow-up gives consistency an explicit role while preserving quality and uplift evidence.

Reproduce from the project root with .venv/bin/python build_phase3.py, then .venv/bin/python -m pytest -q > outputs/phase3/test_results.txt. Neither earlier pipeline is invoked. phase3_manifest.json records preservation checks; test_results.txt records actual tests run.

'''
    (out/'modelling_decisions.md').write_text(intro)
    (out/'submission_methodology_recommendation.md').write_text(intro+'\n\n'.join(sections))
    (out/'START_HERE.md').write_text('''# Phase 3 final-selection handoff

Open submission_methodology_recommendation.md for ordered final recommendations, original/revised top-15 unions, six wicket-model/weight comparisons, reasons, uncertainty and limitations. Final CSVs are final_batters_top10.csv and final_bowlers_top10.csv. Fragile positions are explicitly marked.

The recommendation combines 85/15 runs/wickets using simple earlier-history wicket expectations with an explicit reliability score: a lower bootstrap performance quantile, across-match consistency, and an all-match leave-one-out penalty. Ordinary support requirements are stronger. Phase 2's performance-first results remain preserved, and comparable intermediate performance-only ranks are saved.

First-innings calibration was tested strictly forward. It improved point metrics but failed the pre-specified uncertainty gate and was not adopted; no inconsistent partial probability correction enters pressure or rankings. No Phase 1/2 rebuild was performed.

Read *_comparison_top15.csv and *_wicket_top15_changes.csv for ranking changes; ordinary_support_investigation.csv for bootstrap comparison failures and exclusions; ranking_scenarios.csv for sensitivity; wicket_model_validation.csv and calibration_metrics.csv for out-of-sample evidence. Updated plots are in figures/. Source and prior-output hashes are verified in phase3_manifest.json.

Run .venv/bin/python build_phase3.py and .venv/bin/python -m pytest -q > outputs/phase3/test_results.txt. The test log records what actually passed. Statistical validity does not follow from passing implementation tests.

Before submission, verify stable IDs in the official player dropdowns, retain the fragile-position caveats, and format the submission methodology recommendation with the supplied figures. Nothing has been entered or published automatically.
''')
