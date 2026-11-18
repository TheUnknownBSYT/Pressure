"""Build the judges' report and analyst-facing audit from the executed experiments."""
import json
from pathlib import Path
from shutil import copyfile

from final_strategy.ranking_figure import render as draw_intervals

import numpy as np
import pandas as pd

OUT=Path('outputs/final_submission')
E=OUT/'evidence'
SOURCE=Path(__file__).parent
TITLE='SupeRR_Selector_Final_Methodology'


def read(name):
    return pd.read_csv(E/f'{name}.csv')


def p(s): return ['p',s]
def h(s): return ['h',s]
def cap(s): return ['cap',s]
def tab(head,rows,widths): return ['table',head,rows,widths]


def markdown(pages):
    output=[]
    for title,blocks in pages:
        output+=['# '+title,'']
        for kind,*args in blocks:
            if kind=='table':
                head,rows,_=args
                for row in [head,['---']*len(head),*rows]:
                    output.append('| '+' | '.join(str(v).replace('|','/') for v in row)+' |')
            elif kind=='image':output+=['!['+args[1]+']('+args[0]+')']
            else:output+=[('## ' if kind=='h' else '')+args[0]]
            output.append('')
    return '\n'.join(output)


def idea_audit():
    ab=read('response_ablation')
    rows=[]
    def add(concept,avail,impl,test,evidence,decision):
        rows.append(dict(concept=concept,available=avail,implemented=impl,tested=test,
                         evidence_of_benefit=evidence,final_decision=decision))
    add('Contestability versus scoring difficulty','Pre-ball state and results','Static index, calibrated alternatives and six-ball recursion','12 scenario comparisons and out-of-time revision error',
        '22 off six exceeds 36 off six in all 12. Six off six leads under independent outcomes. In 2026 chase revision MSE was 0.000323 for static versus 0.000426 for required rate.','Retain consequence and contestability. Treat exact terminal probabilities cautiously.')
    add('Dynamic weights','Pre-ball resources and prior wickets','Bounded transfer between leverage and urgency weights','Annual chronological loss on observed legal-ball WP revision',
        '2026 overall MSE fell from 0.0003272 to 0.0003234. Chase MSE rose slightly. Fitted interactions barely improved additive MSE.','Show as a sensitivity. Use the interpretable static index for eligibility.')
    groups=[
        ('Role-dependent performance','role','Retain role-aware expectations in both roles.'),
        ('Opponent quality','opposition','Retain as a small peer-benchmark adjustment. Keep competition coverage visible.'),
        ('Consecutive boundaries and sixes','boundaries','Use the trigger for descriptive recovery profiles.'),
        ('Consecutive dots','dots','Report the sequence and individual sample sizes.'),
        ('Batting collapse','collapse','Retain observed loss of resources in the consequence audit. No separate personal bonus.'),
        ('Rapid momentum changes','momentum','Retain as a diagnostic.'),
        ('Recent career trend and strong form','form','Show recent coverage and decay sensitivity.'),
        ('Three consecutive poor performances','slump','Document the lagged proxy and its weak predictive evidence.'),
        ('Selection and opportunity history','opportunity','Use observed appearances as coverage. Reasons for absence are unavailable.'),
        ('Recovery following failure','recovery','Report descriptive profiles.'),
        ('Pooled personal history','all_memory','Do not alter the contribution benchmark using the player’s recent history.'),
        ('Personal history interactions with pressure','interactions','Use for examples and diagnostics only.')]
    for concept,model,decision in groups:
        a=ab[(ab.model==model)&ab['sample'].eq('all')]
        evidence=[]
        for period in a.period.unique():
            parts=[]
            for r in a[a.period.eq(period)].itertuples():
                parts.append(f'{r.role} {r.run_delta:+.5f} [{r.run_lo:+.5f}, {r.run_hi:+.5f}]')
            evidence.append(period+' '+', '.join(parts))
        add(concept,'Observable proxy' if model not in ['role','opposition'] else 'Prior and pre-delivery context',
            'Fitted group '+model,'Added to '+str(a.reference.iloc[0])+'; match-paired change in run MSE (95% interval). Wicket and pressure-subset results in response_ablation.csv.',
            ' / '.join(evidence),decision)
    add('Pressure consistency','Player-match history','Short-appearance weights, continuous contributions, joint match resampling','Repeated-match diagnostics, chronological quality and vote targets',
        'The old 2026 vote-rank correlations reproduced at 0.007 and 0.033. See rating_forward.csv for the revised continuous estimates.','Use lower-tail historical quality subject to positive personal uplift. Publish interval and vote frequency separately.')
    add('Psychological or selection pressure','No direct labels or reasons for absence','No latent mental-state score','No direct validation possible','Fixture history cannot resolve injury, rest, rotation or nonselection.','Requires independent labels and better selection records.')
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'original_ideas_audit.csv',index=False)
    return result


def ranked_page(role,players,all_candidates):
    role_name='batters' if role=='bat' else 'bowlers'
    z=players[players.role==role].sort_values('rank')
    rows=[[str(int(r.rank)),r.player_name,str(r.player_id),f'{r.pressure_matches:.0f} / {r.pressure_legal:.0f}',
           f'{r.absolute:+.2f}',f'{r.uplift:+.2f}',f'{r.rating:+.2f}'] for r in z.itertuples()]
    first=z.iloc[0]
    q1=f"{first.player_name} contributed {first.absolute:.2f} runs above the peer expectation per six legal balls. This player’s ordinary benchmark for the same mix was {first.ordinary:+.2f}. Across weighted pressure appearances, {first.positive_own_share:.0%} beat that personal benchmark."
    low=z.sort_values('recent_pressure_legal').iloc[0]
    zero=z.loc[z.IPL_legal.eq(0),'player_name'].tolist()
    named=all_candidates[all_candidates.role.eq(role)].set_index('player_name')
    contrast=[]
    for name in (['Ravindra Jadeja','Lokesh Rahul','Heinrich Klaasen','Shreyas Iyer'] if role=='bat' else ['Blessing Muzarabani','Jasprit Bumrah','Sandeep Sharma']):
        if name not in named.index:continue
        r=named.loc[name]
        pos=f"rank {int(r['rank'])}" if pd.notna(r['rank']) else str(r.selection_reason)
        contrast.append(f'{name} ({pos})')
    return ('Top ten '+role_name,[
        p('Each rate is a signed run equivalent per six legal balls. Quality compares the player with contextual peers. Uplift compares that quality with the player’s own matched ordinary rate. The rating is the 20th percentile of resampled quality.'),
        tab(['Rank','Player','ID','Matches / balls','Quality','Uplift','Rating'],rows,[.58,1.76,.68,1.16,.90,.82,.90]),
        h('Reading the first selection'),p(q1),
        h('Where the evidence comes from'),p((('Matched pressure evidence comes entirely from T20Is for '+', '.join(zero)+'. ') if zero else 'Every selected player has matched IPL pressure exposure. ')+f"{low.player_name} has {int(low.recent_pressure_legal)} matched pressure balls in the final 365 days. The table pools seasons."),
        cap('Other candidates reviewed include '+ ' · '.join(contrast)+'. Full reasons, intervals and exposure by phase accompany the submission.') ])


def model_tables():
    a=read('response_ablation')
    models=['role','opposition','ability','all_memory','interactions']
    data=a.loc[a['sample'].eq('all')]
    rows=[]
    for m in models:
        z=data[data.model.eq(m)]
        vals=[z.loc[z.role.eq(role)&z.period.str.startswith(year),'run_delta'].iloc[0] for year in ['2023','2026'] for role in ['bat','bowl']]
        rows.append([m.replace('_',' '),z.reference.iloc[0].replace('_',' '),*[f'{v:+.4f}' for v in vals]])
    return rows


def evaluate(players):
    f=read('rating_forward')
    f=f[f.cohort.eq('common_except_recent_only') & f.target.eq('absolute')]
    role_rows=[]
    for year in [2025,2026]:
        for role,label in [('bat','Batters'),('bowl','Bowlers')]:
            t=f[f.year.eq(year)&f.role.eq(role)].set_index('method')
            r=t.loc['lower_bound']
            fmt=lambda x: '—' if pd.isna(x) else f'{x:+.2f}'
            role_rows.append([str(year),label,str(int(r.n)),fmt(r.rho),fmt(t.loc['vote','rho']),f'{r.mse:.2f}',f"{t.loc['peer_zero','mse']:.2f}"])
    t=read('transfer_validation')
    t=t[(t.year==2026)&(t.prior=='T20I')].set_index('role')
    transfer=(f"Among players with earlier T20I pressure evidence and later IPL exposure in 2026, rank correlations were {t.loc['bat','spearman']:+.2f} for {int(t.loc['bat','players'])} batters and {t.loc['bowl','spearman']:+.2f} for {int(t.loc['bowl','players'])} bowlers. A forecast of zero runs above peers had lower error in both. The crossover sample is selective and small. We assign no IPL premium or conversion factor.")
    sens=read('rating_sensitivity')
    def counts(method,col):
        c=sens[sens.method.eq(method)].set_index('role')[col]
        return f"{int(c['bat'])} batters and {int(c['bowl'])} {"bowler" if c['bowl']==1 else "bowlers"}"
    return ('What the forward tests establish',[
        p('Freeze each player rating before January. In the following year, measure quality in the previously supported role cells. Keep players with at least two later matches and 12 pressure balls. The common cohort permits a like-for-like comparison across the principal models.'),
        tab(['Year','Role','Players','Rating ρ','Vote ρ','Rating MSE','Zero MSE'],role_rows,[.58,1.15,.72,.75,.75,1.16,1.16]),
        cap('ρ is Spearman rank correlation with later absolute quality. MSE is in squared run equivalents per six balls. Zero predicts performance at the peer expectation. The conservative rating estimates a lower tail. Using it as a point forecast produced higher error than predicting zero for both 2026 cohorts.'),
        h('The previous vote test'),p('The original exercise instead forecast the fraction of later matches that beat a same-year ordinary benchmark. Its 2026 correlations reproduced at 0.007 for 16 batters and 0.033 for 18 bowlers. The targets and cohorts differ. Of the six January picks in each role with sufficient later coverage, average 2026 pressure quality was −1.42 runs per six for batters and +2.08 for bowlers.'),
        h('Age and portability of the evidence'),p(f"With a 365-day half-life, {counts('decay365','overlap_top10')} retain a top-ten place. Applying the full gates to the last 365 days alone yields {counts('recent365','eligible')}. Under the dynamic pressure index, {counts('dynamic','overlap_top10')} remain in the top ten."),
        p(transfer),
        cap('Calendar 2026 was withheld from the new annual coefficient fits and January player rankings. Earlier project work had inspected that season. A genuinely untouched prospective evaluation requires later fixtures.')]),transfer


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for template, output in [('project_audit.md', 'PROJECT_AUDIT.md'), ('evidence_guide.md', 'EVIDENCE_GUIDE.md')]:
        if (SOURCE/template).is_file():
            copyfile(SOURCE/template, OUT/output)
    players=pd.read_csv(OUT/'component_evidence.csv')
    candidates=read('rating_all_candidates')
    scope=read('rating_scope').set_index('role')
    winners=players.loc[players['rank'].eq(1)].set_index('role')
    checks=json.loads(Path('outputs/run_manifest.json').read_text())
    intro=('T20 performance under pressure',[
        p(f"{winners.loc['bat','player_name']} leads the batting list and {winners.loc['bowl','player_name']} leads the bowling list. Each combined positive value against contextual peers with an improvement over their own ordinary performances. We order qualifying players by a conservative estimate of the size of their contribution across matches."),
        h('What makes a delivery consequential'),p('A delivery matters when its plausible outcomes could move the chance of winning. A team chasing 22 from six balls can still turn the contest. When 36 are needed, many plausible outcomes leave the result effectively settled. We score the situation before the ball, then measure how the player responded.'),
        h('Which cricket enters the decision'),p(f"The supplied files contain {checks['rows']:,} events in {checks['matches_with_deliveries']} matches with deliveries. Reconstruction leaves {checks['standard_eligible_deliveries']:,} structurally eligible events. Earlier-match support allows {checks['scored_deliveries']:,} pressure scores. Our final IPL and T20I comparison supports {int(scope.loc['bat','supported'])} batters and {int(scope.loc['bowl','supported'])} bowlers before requiring positive quality and uplift."),
        h('What the evidence can support'),p('Role and opposition improve expectations for scoring. Recent setbacks and selection history produce much weaker and less stable gains. They help us ask better questions about individual performances. They receive no automatic weight in the final rating.'),
        p('The order summarizes the supplied cricket through 1 November 2026. Small samples, overlapping uncertainty intervals and weak year-to-year repeatability limit what it can say about a 2027 selection. A player can have persuasive historical evidence and very little recent IPL evidence.')])
    pressure=('How much the next ball matters',[
        p('The two logistic regressions predict the batting side’s win probability, p, using separate first-innings and chase states. Every quarter they refit on earlier matches. The first model compares score with a historical trajectory. The chase model measures the requirement against the remaining balls and wickets.'),
        h('Pressure = 100 × [0.50 L + 0.30 U C + 0.20 S]'),
        p('L measures the spread of next-ball win probabilities under earlier innings/phase frequencies for 0–6 runs or a wicket, scaled by 0.10 and capped at one. U combines elapsed time, scoring demand and lost recovery resources in a 45/35/20 split. C = 4p(1−p) is largest in a level contest. S reflects fixture stage, including 0.4 for league matches and 1.0 for a final. These weights are design choices.'),
        ['image','evidence/consequence_scenarios.png','IPL league chase with six legal balls and seven wickets left. All curves use an earlier death-over delivery distribution.'],
        p('The original pressure scores are 12.73, 85.79 and 12.44. In all twelve model/distribution combinations, 22 ranks above 36. The independent six-ball recursion reverses the order of 6 and 22. Only four of 34 comparable chases at 20–24 required won. The fitted 39.3% chance at exactly 22 is especially uncertain. Across the 2026 January-frozen test, chase Brier error was 0.1115 and first-innings error was 0.2254, against a historical prior of 0.2486 in each innings.'),
        cap('The current quarter uses the top-quartile threshold from earlier scored quarters. The first 2,760 scores establish history. Of 110,271 later events in the full feed, 26,895 exceed or equal their static threshold.')])
    dynamic=('How the situation changes',[
        p('The dynamic alternative shifts up to 0.10 of the weight between leverage and contestable urgency. Let c be the last 12 legal balls’ wicket count divided by two, capped at one. Let t be chase progress times contestability, set to zero in the first innings. Then the leverage weight is 0.50 + 0.10(c − t) and the urgency weight is 0.30 − 0.10(c − t). The stage weight stays at 0.20.'),
        h('Two wickets in a Punjab chase'),p('Punjab Kings needed 24 against Chennai Super Kings on 3 April 2026. Each row shows the state immediately before the named batter faced the ball.'),
        tab(['Delivery','Batter','Need','Wkts left','Static','Dynamic'],[
            [r.delivery_id,r.striker_name,f'{int(r.runs_required_pre)} off {int(r.legal_balls_remaining_pre)}',str(int(r.wickets_remaining_pre)),f'{r.pressure_static:.2f}',f'{r.pressure_dynamic:.2f}'] for r in read('consequence_setback_example').itertuples()],[1.18,1.40,1.14,1.00,.90,.96]),
        p('Iyer and Wadhera fell on successive deliveries. Before Stoinis faced Henry, the leverage weight had increased from 46.5% to 54.9% and both indices crossed the historical cutoff. Stoinis hit four. That four enters his response only after his pressure score has been calculated.'),
        h('Was dynamic weighting better?'),p('We trained linear mappings from each index to the next legal ball’s realized absolute win-probability revision. In 93 matches in 2026, dynamic weights reduced mean squared error by 1.17% overall and slightly increased chase error. On the same overall target, additive ridge gave 0.00030042 and adding interactions gave 0.00030018. Required rate alone in chases gave 0.00042557 against 0.00032325 for the static index.'),
        cap('The revision target uses the same win model on both sides of the ball. It tests sensitivity to a change in match state. It supplies no independent label for emotional pressure. We retain static classification and publish the dynamic reranking.')])
    methods=('What a player contributed',[
        h('A comparable delivery'),p('We trained ridge regressions for batter runs and runs charged to the bowler, and logistic regressions for striker dismissal and credited bowler wicket. The peer model uses innings, phase, wickets, scoring demand, role, venue and earlier individual and team opposition records. Within each training block, predictors are standardized using its own training data.'),
        p('Opponent records start from competition, phase and innings expectations. Player residual rates shrink toward zero with 100 prior events and team and venue rates with 500. Before a debut the estimates revert to the group expectation. The peer comparison omits the focal player’s own ability and recent form. An unusually strong player can therefore earn absolute credit.'),
        h('Put runs and wickets on a common scale'),p('A resource curve fitted to earlier first innings predicts future runs from balls and wickets remaining. The estimated cost of losing one wicket after the next legal ball is the difference between two points on that curve. The same opportunity-cost scale is used for chases. The estimate describes associations between states and remaining runs, with incomplete control for batting depth. In 2026 it forecast remaining first-innings runs with an RMSE of 24.04 versus 25.51 for a constant scoring rate.'),
        p('Batting value is actual minus expected runs plus the wicket price times expected minus actual striker dismissals. Bowling value reverses the run difference and rewards credited wickets above expectation. Forecasts are per recorded event. To score per legal ball, expectation is charged on legal events after dividing by the historical probability of a legal delivery. Extras and wickets on illegal events still count.'),
        h('Attribution and missing records'),p('Run-outs accrue no bowler wicket. A nonstriker run-out changes the match state and carries no striker dismissal penalty. Byes and leg-byes are excluded from bowler concessions. Ambiguous extras and unresolved dismissals are excluded from performance scoring. Each role is identified by the delivery IDs for striker or bowler. Revised matches lack target-change timestamps. Super overs, unreconciled innings and unsupported contexts are excluded from the pressure sample. Full-match reconciliation is retrospective quality control. Only the inputs to an eligible ball’s score are available live.')])
    results=('What improves a delivery forecast',[
        p('The table gives changes in run mean squared error after adding predictors. Each match receives equal weight. Negative values mean lower error. Development covers annual forecasts in 2023–25. The 2026 models use coefficients trained before January, with observable match and appearance histories updated through time.'),
        tab(['Model','Compared with','Dev bat','Dev bowl','2026 bat','2026 bowl'],model_tables(),[1.11,1.25,1.08,1.08,1.08,1.13]),
        p('Role terms and earlier opponent quality reduce run error. Ability adds the player’s own historical rate and the static pressure covariate. The memory and interaction rows test additional information after this stronger reference. Their incremental gains vary by period and most match-level intervals include zero. The three-poor-appearance feature reduced high-pressure striker-dismissal Brier error slightly in 2026, an exploratory result that did not repeat in development.'),
        h('A player returning after a poor sequence'),p(across_example()),
        h('Selection opportunities'),p('The opportunity features count recorded appearances in 90 days, time since the previous appearance and participation in the last five observed team fixtures with usable XIs. Unknown fixture coverage is flagged. Injuries, rotation, contract status and reasons for absence are unobserved. The opportunity ablation gives no consistent basis for an individual pressure premium.')])
    gates=('How the two lists are ordered',[
        h('First match the ordinary workload'),p('Use ordinary deliveries from the same player, innings, phase and competition. A comparison cell needs at least 20 ordinary legal balls across three matches. Reweight these ordinary rates to the phase and competition mix the player actually encountered in each pressure match. Subtract the weighted ordinary rate from absolute pressure quality to obtain personal uplift.'),
        h('Then compare sustained contributions'),p('For each pressure appearance compute six times signed contribution divided by legal pressure balls. Weight that appearance by min(1, legal balls / 6). The first six balls build evidence and further balls from that match add no extra match weight. For eligibility require eight matches on each side, 50 pressure balls, 100 ordinary balls, and matched coverage of 75% of the player’s pressure balls.'),
        p('Require positive point estimates for both quality and uplift. Resample whole matches jointly 2,000 times, retaining the same match draws across players. Order eligible players by the 20th percentile of absolute quality. This is an explicit preference for sustained value under uncertainty. It is neither a 95% lower bound nor a calibrated forecast. The uplift gate applies to the point estimate and can admit players whose uplift interval crosses zero.'),
        h('What uncertainty includes'),p('The player evidence gives the 95% resampling ranges for quality and uplift, the share of appearances beating each benchmark, and the lowest quality after removing one pressure appearance. Resampling refits ordinary cell rates within supported cells. Intervals condition on the pressure definition, fitted delivery models and observed player selection.'),
        p('The final lists pool IPL and T20I comparisons, with separate ordinary cells for each. The supplied other-T20 records can inform model training but do not enter the ranked contributions. No nationality, auction status or reputation features enter the score.')])
    validation,transfer=evaluate(players)
    draw_intervals(players,E/'rating_uncertainty.png')
    uncertainty=('How much confidence the ranks deserve',[
        ['image','evidence/rating_uncertainty.png','Points show absolute pressure quality. Lines show the middle 95% of match-cluster resamples. Ranking uses the 20th percentile, so the point order can differ from the published order.'],
        p('None of the ten batting quality intervals excludes zero. Five bowling intervals do. Only two players in each list have a personal uplift interval entirely above zero. Every selected player retains positive absolute point quality after removing their best single pressure appearance. All adjacent 95% intervals for the difference in absolute quality include zero, including tenth versus eleventh. These checks answer different questions about how the evidence survives resampling.'),
        h('Three decisions near the cutoff'),p('Shreyas Iyer averaged +1.97 against peers and +1.96 in his matched ordinary workload. His +0.01 uplift passes the entry rule and loses that distinction without opponent adjustment. Heinrich Klaasen scored nearly the same absolute quality (+1.99), but his own ordinary benchmark was +2.46. Babar Azam had positive absolute quality on the 51 of 77 pressure balls that could be matched. His 66% coverage fell below the 75% requirement.'),
        p('Jimmy Neesham beat his own ordinary benchmark in 30% of weighted appearances. His positive matches carried enough value to qualify on magnitude. Requiring the player to beat their ordinary benchmark in a majority of weighted matches brings George Dockrell into the batting ten and Kuldeep Yadav and Jason Holder into the bowling ten. Requiring both 95% lower endpoints to clear zero would leave no batters and only Bumrah and Abrar Ahmed.')])
    pages=[intro,ranked_page('bat',players,candidates),ranked_page('bowl',players,candidates),pressure,dynamic,methods,results,gates,uncertainty,validation]
    (E/'report_content.json').write_text(json.dumps(pages,indent=2,ensure_ascii=False))
    (OUT/(TITLE+'.md')).write_text(markdown(pages))
    idea_audit()
    support_reports(pages,players,transfer)


def across_example():
    r=read('response_example_across').iloc[0]
    history=read('response_example_across_history')
    prior=history.loc[history.match_date_parsed.lt(r.match_date_parsed)].tail(3)
    return (f"Before {r.player_name} faced delivery {r.delivery_id} on {r.match_date_parsed}, his last three observed scoring performances were below his own historical, context-adjusted level (matches "+', '.join(prior.match_ids.astype(str))+f"). His five-date form proxy was {r.form_trend:+.3f}. The full model estimated {r.expected_runs:.3f} runs and {r.wicket_prob:.2%} dismissal risk. Switching off only the slump flag and its interaction gave {r.flag_off_expected_runs:.3f} and {r.flag_off_wicket_prob:.2%}. He scored zero and survived. This small fitted difference cannot establish the cause of that dot ball.")


def compact_table(headers, records):
    def item(x):
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(map(item,row))+' |' for row in
                      [headers,['---']*len(headers),*records]])


def support_reports(pages,players,transfer):
    a=read('response_ablation')
    ab=a[(a.period=='2026 reserved') & a['sample'].eq('all') & a.model.isin(['role','opposition','all_memory','interactions'])]
    case=read('response_example_short').iloc[-1]
    params=read('response_coefficients')
    pars=params[params.test_start.eq('2026-01-01') & params.model.eq('opposition') & params.feature.isin(['set_batter','opponent_run','opponent_wicket'])]
    head=['Role','Added block','Reference','Δ run MSE [95%]','Δ wicket Brier [95%]']
    rec=[[r.role,r.model,r.reference,f'{r.run_delta:+.5f} [{r.run_lo:+.5f}, {r.run_hi:+.5f}]',f'{r.brier_delta:+.6f} [{r.brier_lo:+.6f}, {r.brier_hi:+.6f}]'] for r in ab.itertuples()]
    alt=read('rating_sensitivity')
    overlap=alt.pivot(index='method',columns='role',values='overlap_top10')
    coef_table=compact_table(['Role','Feature','Runs per training SD','Wicket log-odds per SD'],[[r.role,r.feature,f'{r.run_coefficient:+.4f}',f'{r.wicket_logodds_coefficient:+.4f}'] for r in pars.itertuples()])
    rank_diagnostics=read('rating_forward_shortlists')
    metrics=rank_diagnostics[rank_diagnostics.year==2026].set_index('role')
    shortlist=' '.join(f"Among the {int(metrics.loc[r,'forecasts_observed'])} {label} from the January shortlist who qualified for a later evaluation, mean realized absolute quality was {metrics.loc[r,'mean_future_absolute']:+.2f}." for r,label in [('bat','batters'),('bowl','bowlers')])
    report=f"""# Model comparison and validation

We evaluated three different targets. Predicting who wins a match, predicting a delivery outcome and ranking future player performance answer different questions. Evidence for one target cannot substitute for evidence on the others.

## Win probability and consequence

In the 2026 frozen January test, first-innings/chase Brier scores were 0.22536/0.11147 against a historical outcome prior of 0.24860. Ten-bin calibration errors were 0.11468/0.08064. Global logistic recalibration improved first innings in 2026 but worsened first innings in both development years. Chase results also varied across years. The published model keeps the original logistic form.

For the consequence experiment the target was the magnitude of the realized next legal ball’s change in modelled win probability. Mean squared losses on 2026 matches were 0.00032720 for static, 0.00032337 for dynamic, 0.00030162 for leverage alone, 0.00030042 for additive ridge and 0.00030018 for ridge with interactions. Prediction of this model-derived target naturally favors leverage. Match stage remains a declared tournament-value judgment.

## Delivery outcomes

{compact_table(head,rec)}

Deltas are equal-match-weighted for 93 future matches. These exploratory intervals are paired by match and uncorrected for multiple comparisons. The development comparison, full ablations, high-pressure-only sample and phase/competition breakdowns are in [the ablation table](evidence/response_ablation.csv) and [segment results](evidence/response_segments.csv). Each response fold uses its own start-of-block historical pressure percentile for its high-subset diagnostic. The primary selection uses the finer quarterly thresholds.

{coef_table}

These are conditional associations. The marginal effect of an opponent feature depends on how the historical rates are encoded. The [training log](evidence/response_training.log) and [full parameters](evidence/response_coefficients.csv) give model features, training scales and coefficients for every fit. [Probability calibration](evidence/response_calibration.csv) and [wicket resource fits](evidence/response_wicket_prices.csv) document the other fitted targets.

## Real sequence after consecutive sixes

On {case.match_date_parsed}, {case.player_name} had conceded sixes on his two preceding deliveries in match {int(case.match_id)}. Before {case.delivery_id}, the batting side had {int(case.score_pre)} runs for {10-int(case.wickets_remaining_pre)} wickets. The full model expected {case.expected_runs:.3f} runs and a {case.wicket_prob:.2%} wicket chance. Switching off the boundary flags inside the same fitted model gave {case.flag_off_expected_runs:.3f} and {case.flag_off_wicket_prob:.2%}. He conceded {int(case.team_runs_event)} run. This individual prediction shift coexists with weak group-level evidence that the features improve future forecasts.

Across the 601 follow-ups to two sixes, 29.3% were boundaries and 5.5% yielded a credited wicket. Rashid Khan had the most trigger matches (nine, with 12 follow-up events). His role/opponent-adjusted difference from his other events was +0.035 run equivalents per event after 50-event shrinkage. No bowler had ten trigger matches. Sanju Samson had 126 after-dot events in 52 matches and a shrunk difference of +0.068. Rovman Powell had 28 post-collapse events in 12 matches and +0.076. These examples were chosen for exposure, and the [all-player profiles](evidence/response_setback_players.csv) are descriptive. They do not establish stable individual recovery skills.

## Player ranking

{markdown([pages[-1]])}

{shortlist} Attrition and the small samples preclude a claim about all ten selections.

{compact_table(['Alternative','Batters retained out of ten','Bowlers retained out of ten'],[[k,int(r.bat),int(r.bowl)] for k,r in overlap.iterrows()])}

A stricter majority-above-ordinary rule retains nine batters and eight bowlers. Both 95% lower endpoints clear zero for no batters and only Bumrah and Abrar Ahmed. [Gate sensitivity](evidence/rating_gate_sensitivity.csv) lists the complete alternative selections.

The 85/15 experiment changes the performance units while holding the new ranking rule fixed. The archived vote experiment also changes the vote rule, high-pressure labels and event expectations. [Forward scores](evidence/rating_forward.csv) give candidate counts, errors and correlations against later absolute quality, personal uplift and vote share. [Crossover pairs](evidence/transfer_player_pairs.csv) use at least three earlier matches and 18 legal pressure balls, then two later IPL matches and 12 balls. Forecasts use eight effective matches of shrinkage toward zero. Transfer estimates condition on appearing in both competitions.

## Sources of uncertainty

The same players and fixtures recur across experiments. An isolated favorable ablation or one famous player passing a threshold is weak evidence. Player intervals condition on fitted models, chosen thresholds and the recorded schedule. Tied matches and matches with revised targets or incomplete records are poorly represented. Ordinary comparisons require a visible historical role and can exclude genuine specialists. The model assigns value only to measurable on-ball contributions. A future auction decision would also require availability, price, squad needs and independent evidence from later fixtures.
"""
    (OUT/'MODEL_COMPARISON.md').write_text(report)
    z=players.loc[players['rank'].eq(1)].set_index('role')
    zero=int(players.IPL_legal.eq(0).sum())
    expl=f"""# Defending the SupeRR Selector rankings

I selected **{z.loc['bat','player_name']}** first among batters and **{z.loc['bowl','player_name']}** first among bowlers. I asked two questions about each player. How many runs of value did they deliver above what their role and opponents would lead us to expect? Did they also beat their own ordinary performance in comparable innings, phases and competitions?

**What makes a ball count?** Before it is bowled, I ask how much the plausible next outcomes could change the game. In an IPL league chase with seven wickets left, the original model scores 22 off six at 85.79 and 36 off six at 12.44. The harder chase has less room for one ordinary delivery to change the result. Twelve alternative combinations support that ordering. The size of the difference and which other chase is most consequential depend on the win model.

**What did the computer learn?** Earlier matches trained separate win models for setting and chasing a target. Earlier deliveries trained run and dismissal forecasts using role and opposition. Earlier first innings trained the runs associated with a wicket at different resource levels. The pressure weights, exposure requirements and rule for ranking players were human decisions. Role information helped scoring forecasts in both roles. Opponent adjustment helped especially when estimating what a bowler should concede against the opposing batter.

**Why do the leaders qualify?** {z.loc['bat','player_name']} delivered {z.loc['bat','absolute']:.2f} extra runs per six pressure balls against a personal ordinary benchmark of {z.loc['bat','ordinary']:+.2f}. {z.loc['bowl','player_name']} delivered {z.loc['bowl','absolute']:.2f} runs of bowling value against {z.loc['bowl','ordinary']:+.2f}. Both beat both benchmarks across their pooled sample. We resampled matches and ranked on the lower part of each player’s historical quality distribution. Someone improving from a very low baseline still needs to produce positive value against peers.

### Questions from a selection panel

**Would you bet on these exact positions in 2027?** I would use them as historical evidence for a shortlist. On a common 2026 sample the new rating correlated 0.06 with later batting quality and 0.26 with bowling quality. Predicting the peer average gave less squared error for both roles. The old vote’s correlations were also weak on its own target. An untouched season is the next test.

**How can someone with positive pressure numbers miss the list?** Heinrich Klaasen produced about +1.99 runs per six under pressure but +2.46 in matched ordinary situations. Nitish Reddy improved by about +1.74 from his own ordinary level yet his pressure quality was −0.22. Shreyas Iyer passed with only about +0.01 uplift. The decision boundary changes the answer for that kind of marginal case.

**Is the list consistently good match after match?** We cap the influence of long appearances, downweight very short ones, resample whole matches and report repeat-success shares. We preserve the size of contributions. Jimmy Neesham, for example, beat both benchmarks in only about 30% of weighted appearances. His selection reflects the magnitude of his positive contribution, with considerable uncertainty. Another defensible emphasis on frequency changes the list.

**Can we apply an international performance to IPL cricket?** {zero} of the twenty selections have no matched IPL pressure balls. {transfer} The official challenge covers both competitions. A shortlist built specifically for IPL transfer has a different evidence requirement.

**Did you find pressure memory?** We could observe two sixes, a collapse or a string of poor matches, and sometimes the next-ball estimate moved. The pooled history model did not deliver a clear repeatable gain after role, opponent and existing ability were included. We publish those experiments and give no mental-pressure bonus. Missing XI appearances show opportunity gaps with unknown causes.
"""
    (OUT/'EXPLAIN_MY_STRATEGY.md').write_text(expl)
    print('Published methodology, model comparison, original-ideas audit and defense guide.')


if __name__=='__main__':main()
