"""Create report content from verified selections. The renderer uses this same structure."""
from pathlib import Path
import json
import pandas as pd
from final_strategy.analyse import OUT,E,ROLES

U='https://unstop.com/hackathons/crp-superr-selector-season-2-help-shape-rajasthan-royals-ipl-2027-auction-strategy-rajasthan-royals-1792758'
W='https://www.wisden.com/series/ipl-2027/cricket-news/ipl-2027-retention-list-full-list-retained-released'
PRESS='Pressure is the importance of the next ball to the result, given the runs and resources left, how open the contest is and what the match is worth.'
def p(t):return ['p',t]
def h(t):return ['h',t]
def table(headers,rows,widths):return ['table',headers,rows,widths]
def num(v):return f'{v:.2f}'
def read(s):return pd.read_csv(E/(s+'.csv'))
def report():
    players={r:pd.read_csv(OUT/f'final_{l}_top10.csv') for r,l in ROLES.items()}
    pages=[]
    pages.append(('Reliable T20 performance under pressure',[
        p('Rajasthan Royals   /   SupeRR Selector Season 2   /   Round 1 review draft   /   18 November 2026'),
        h('What counts as pressure'),p(PRESS),
        h('Selection decision'),p('Ravindra Jadeja leads the batters and Blessing Muzarabani leads the bowlers. The ordering rewards players who repeatedly beat their own ordinary performance in comparable situations. To qualify, a player must also deliver positive overall value against the contextual expectation and have enough comparable ordinary balls to support the estimate.'),
        p('The distinction changes a selection decision. Ishan Kishan scored about 48 runs per 100 legal balls above expectation under pressure. Jadeja scored about two below, but preserved his wicket more often than expected. Jadeja’s combination of scoring and survival beat his own ordinary benchmark in 81% of weighted appearances, compared with 65% for Kishan. Kishan is sixth under this repeatability rule.'),
        h('Evidence available'),p('The supplied deliveries span 22 January 2022 to 1 November 2026. Of 200,999 recorded events across 867 matches, 113,031 receive pressure scores and 110,255 also have usable player outcomes and expectations. The analysis includes 27,191 high-pressure events before individual matching. Historical coverage is uneven, especially in the first year.'),
        h('Reading the rankings'),p('Pages 2 and 3 give the order for the two Unstop lists. Scores summarize repeated historical performance. Even a high score can accompany little direct IPL evidence or an uncertain current assessment. The recruitment notes on page 7 use these distinctions to identify what Rajasthan Royals can act on.')]))
    notes={
    'Ravindra Jadeja':'His wicket preservation and a large improvement over ordinary scoring drive first place. His pressure run residual alone is slightly negative.',
    'Nitish Reddy':'All 141 matched pressure balls came in the IPL. Scoring improved from 12.9 runs below expectation per 100 balls in ordinary play to 30.5 above under pressure.',
    'Andre Russell':'Death overs dominate his evidence. His last matched pressure appearance was in July 2025.',
    'Babar Azam':'Scoring improves despite a worse dismissal residual under pressure. His evidence comes from outside the IPL.',
    'Cameron Green':'The middle overs provide his largest share, with 104 matched pressure balls from the IPL.',
    'Ishan Kishan':'His large scoring uplift outweighs an already demanding absolute benchmark. He beats his own baseline less often than the leaders.',
    'George Dockrell':'His largest sample comes at the death. A low ordinary scoring baseline contributes to the size of his uplift.',
    'Azmatullah Omarzai':'His pressure scoring exceeds a strong ordinary level. All 23 observed IPL pressure balls fall outside his supported comparisons.',
    'Shubman Gill':'His pressure scoring is strong in absolute terms and improves on an already positive ordinary residual.',
    'Glenn Phillips':'He edges Shreyas Iyer on repeatability. His 11 IPL pressure balls supply no matched IPL comparison.',
    'Blessing Muzarabani':'Death overs dominate. Runs prevented rise from about one below expectation per 100 ordinary balls to 31 above in pressure situations.',
    'Shaheen Afridi':'Run prevention and wicket-taking both improve against his ordinary baseline. Direct IPL pressure evidence is absent.',
    'Taskin Ahmed':'Across 24 matches, improved run prevention drives his uplift. His ordinary wicket residual was already positive.',
    'Jason Holder':'He contributes the widest match sample among the selected bowlers and 62 matched IPL pressure balls.',
    'Dasun Shanaka':'His middle-over evidence includes a strong pressure wicket residual. Only six observed IPL pressure balls were available.',
    'Jasprit Bumrah':'He prevents runs above expectation in both settings and improves his wicket residual under pressure.',
    'Kuldeep Yadav':'He supplies 249 matched IPL pressure balls. His lighter recent-window estimate sits close to the midpoint.',
    'Mujeeb Ur Rahman':'Strong run prevention earns a place on the minimum eight pressure matches. His recent sample is thin.',
    'Mohammed Siraj':'Run prevention improves markedly from ordinary play. His wicket residual declines, making the component weights consequential.',
    'Alzarri Joseph':'Wicket-taking offsets a slightly negative pressure run residual. Haris Rauf is effectively level at the cutoff.'}
    for r,label in ROLES.items():
        a=players[r];rows=[]
        for x in a.itertuples():
            rows.append([str(x.rank),x.player_name,str(x.player_id),f'{int(x.hm)} / {int(x.hb)}',f'{100*x.score:.1f}',f'{100*x.lo:.0f}–{100*x.hi:.0f}',f'{x.absolute:.2f} / {x.ordinary:.2f}',f'{x.uplift:+.2f}',str(int(x.IPL_balls))])
        pages.append((f'Top ten {label}',[
          p('Ranked by the smoothed share of matched pressure appearances beating the player’s ordinary benchmark. Scores and conditional intervals use a 0–100 scale.'),
          table(['Rank','Player','ID','Matches / balls','Score','95% range','Press / ord¹','Uplift','IPL balls'],rows,[.40,1.45,.50,.73,.47,.62,1.01,.52,.58]),
          p('¹ Press / ord gives adjusted pressure and matched ordinary quality. Uplift is their difference in standardized units. IPL balls count only matched pressure legal balls. Intervals resample whole matches.'),
          h('Selection evidence'),*[p(x.player_name+'  —  '+notes[x.player_name]+f' Matched {x.first[:4]}–{x.last[:4]}. Recent pressure balls before matching: {int(x.recent_pressure_balls_all)}.') for x in a.itertuples()]]))
    pages.append(('From match situation to player rating',[
        h('Score the situation before the delivery'),p('Separate first-innings and chase models estimate the batting side’s win chance from earlier matches. For each state, simulate the next legal ball as zero through six runs or a wicket. L measures the spread in the resulting win chances, weighted by earlier outcome frequencies and scaled by 0.10. U combines time elapsed, scoring demand and lost resources. C = 4p(1−p) is highest in an even contest, where p is the current win chance. S describes match stage.'),
        p('Pressure = 100 × [0.50L + 0.30UC + 0.20S]. The retrospective 75th percentile is 36.585. All four components use information available before the delivery. The weights, scale and cutoff are modelling choices.'),
        p('On 7 October 2022, delivery 17641:2:41 began with 127 needed from 81 balls and nine wickets left. The model gave p = 0.463, L = 0.262, U = 0.524, C = 0.994 and S = 0.400. The score was 36.72, just inside the high-pressure group. The next ball’s actual outcome played no part.'),
        h('Build a fair ordinary comparison'),p('Learn expected runs from earlier quarters, accounting for innings, phase, competition type, wickets and scoring demand. For wickets, pool earlier phase and competition rates toward the global rate with 200 balls of prior exposure. Batters get credit for runs above expectation and dismissals avoided. Bowlers get credit for runs prevented and wickets above expectation. Compute each component per 100 legal balls, standardize by its role-specific spread, and combine 85% runs with 15% wickets.'),
        p('Compare a player with their own ordinary balls in the same innings, phase and competition type. Weight each ordinary cell by its share of the player’s pressure balls in the match. A match earns a success if pressure quality exceeds that benchmark. One match carries at most one vote. A three-ball appearance carries half a vote and six or more balls carry a full vote.'),
        h('Count repeat performances'),p('Score = 100 × (successful weighted votes + 4) / (total weighted votes + 8). Four assumed successes and four failures pull thin samples toward 50. Jadeja has 9.00 successful votes out of 11.17. His score is 100 × 13 / 19.17 = 67.8. Both his pooled pressure quality (+0.25) and his own-baseline uplift (+2.15) are positive, so he qualifies. Sorting the qualifying batters places him first.')]))
    pages.append(('Historical evidence and recent form',[
        p('The final ranking pools historical appearances. A one-year half-life changes four names in each top ten, so the date choice matters. Evidence that those changes improve later rankings is weak.'),
        table(['Source year','Recorded matches','IPL usable events','T20I usable events','Other usable events'],[[str(y),str(int(v.recorded_matches.sum())),*[f"{int(v.loc[v.competition_type.eq(c),'analysable'].sum()):,}" for c in ['IPL','T20I','other_T20']]] for y,v in read('calendar_coverage').groupby('match_date_parsed')],[.85,1.25,1.3,1.35,1.25]),
        h('Forward tests'),p('At the start of 2024, 2025 and 2026, freeze the training sample, learn the pressure threshold and component scales from that history, and evaluate against the following calendar year through the data cutoff. The later outcome is the actual fraction of weighted matches beating a separately measured ordinary benchmark in that test window. Only returning players with supported observations can be evaluated.'),
        p('In 2026, historical and decayed scores could both be tested for 16 batters and 18 bowlers. Their rank correlations with later repeatability were 0.007 and 0.015 for batters, and 0.033 and −0.018 for bowlers. Decay reduced squared error for bowlers but increased it for batters. These estimates offer little basis for predicting a precise future order.'),
        p('A recent-only fit from 2 November 2025 to 1 November 2026 supplies zero qualifying names in either role under the main rules. A lighter diagnostic admits nine batters and six bowlers. Blending those recent scores equally with supported historical scores leaves only two batters and three bowlers who meet both sets of requirements.'),
        p('The lighter recent score did show promise for bowlers. In common forward cohorts of nine in 2025 and eight in 2026, its rank correlations were 0.49 and 0.13, compared with −0.10 and −0.05 for historical scoring. These small, selected groups justify recording recent form as a scouting clue. They cannot support a full twenty-player current-form order.'),
        h('Repeatability and uncertainty'),p('Two thousand shared whole-match resamples recompute the player’s ordinary rates and match votes. Every tested adjacent pair among the first 13 in each role has a conditional score-difference interval spanning zero. Repeating the full ranking under six pressure definitions kept Jadeja and Russell in the batting ten and Holder in the bowling ten in all six. These counts describe specification sensitivity.')]))
    pages.append(('Support and selection decisions',[
        h('Who has enough comparable evidence'),p('An ordinary comparison cell needs 20 legal balls in three matches. A player needs eight pressure matches, 50 pressure legal balls, eight ordinary matches, 100 ordinary legal balls and 75% of pressure exposure represented. At least 90% of resamples must retain the ordinary comparisons they need. These gates protect comparison quality and can disadvantage specialists.'),
        p('Among batters facing at least 50 observed pressure balls, 17 of 60 with at least half those balls at positions 6–11 pass support, compared with 35 of 89 others. Median coverage is 66% versus 78%. Among bowlers delivering at least half their pressure balls at the death, 29 of 77 pass, compared with 35 of 83 others. Relaxing the support rules retains just three of ten batting selections and five of ten bowling selections. This is a material assumption behind the order. These are observed usage proxies. They give clearer evidence of a coverage penalty for lower-order batters than for death bowlers.'),
        h('Why the lists changed'),p('Four names enter each top ten. Jadeja, Green, Dockrell and Phillips enter the batting list. Holder, Shanaka, Siraj and Joseph enter the bowling list. Phase 3 favored absolute quality and used an uncertainty percentile, a positive absolute match vote and a deletion penalty. The final rule counts a match as successful against the player’s own ordinary benchmark. The influence and uncertainty penalties in the audit were strongly correlated. Both are removed from the score.'),
        p('Shreyas Iyer remains strong in absolute terms but ranks 11th on the final rule, just behind Phillips at 54.62 versus 55.00. Klaasen and Mustafizur both have positive absolute quality but negative pooled uplift. Lokesh Rahul (KL Rahul) also has negative uplift. Shahrukh Khan retains only 17 matched batting balls. Jamie Overton retains 49 bowling balls across five matches. Insufficient comparisons cannot establish poor performance.'),
        h('Watchlist for a second look'),p('Liton Das and Pathum Nissanka have positive run-and-wicket quality and positive own-baseline signals but miss the match or overlap gates. Kartik Tyagi has six matched pressure matches and 49 balls. Mayank Markande has seven matches and incomplete ordinary overlap. These are candidates for footage and further data. The watchlist uses evidence coverage as its definition of “under the radar.” Fame, age and market price receive no score.')]))
    pages.append(('Rajasthan Royals recruitment implications',[
        h('Transporting evidence into the IPL'),p('Jadeja’s 179 matched pressure balls all come from the IPL, including 117 at the death. He provides an IPL benchmark for a batter who preserves wickets under pressure. In bowling, Bumrah has 127 matched IPL balls and Kuldeep has 249. Their figures provide domestic reference points for late-over run prevention and middle-over control.'),
        p('Muzarabani’s leading score rests on 96 matched balls in 11 matches, with no matched IPL pressure balls. Shaheen and Taskin likewise lack direct matched IPL pressure evidence here. For these players, compare footage and ball outcomes in the proposed phase, against IPL-calibre opposition, before projecting their international results. Some players have IPL records but insufficient comparable ordinary IPL data. Omarzai illustrates the difference.'),
        h('Auction relevance on 18 November 2026'),p('Wisden’s 17 November retention list places Jadeja with Rajasthan Royals and Green with Kolkata Knight Riders. Jadeja can inform the role RR already possess and Green provides a benchmark for alternatives. Verify the official auction register, retention or trade changes, fitness, availability, role and price before making a purchasing decision. The statistical ordering uses the supplied cricket evidence throughout.'),
        h('Limits of the evidence'),p('Pressure labels concentrate late. In the first innings, 58% of analysable death events qualify versus 4% of powerplay events. In chases the shares are 38% and 3%. First-innings upper-tail probabilities were overconfident. For nine 2026 matches contributing states rated at least 90%, the mean predicted win chance was 0.927 and the weighted observed frequency was 0.687. Forward recalibration improved point estimates but its Brier difference interval included zero.'),
        p('Within-player matching controls innings, competition type and phase. Opposition, venue, lineup, tactical role and selection into pressure exposure can still differ. Conditional intervals also omit uncertainty from fitting the pressure and expectation models. These historical comparisons support differentiated scouting priorities. Causal claims about temperament and exact future rankings exceed the evidence.'),
        h('Sources'),p('Round 1 problem statement, supplied two-page PDF. Unstop, SupeRR Selector Season 2 event listing, checked 18 November 2026. Wisden, IPL 2027 retention list, 17 November 2026. Source links and full calculation settings accompany the teaching guide.')]))
    (E/'report_content.json').write_text(json.dumps(pages,ensure_ascii=False,indent=1))
    md=[]
    for title,blocks in pages:
        md+=['# '+title,'']
        for b in blocks:
            if b[0]=='table':
                md+=['| '+' | '.join(row)+' |' for row in [b[1],['---']*len(b[1]),*b[2]]]
            else:md.append(('## ' if b[0]=='h' else '')+b[1])
            md.append('')
    (OUT/'SupeRR_Selector_Final_Methodology.md').write_text('\n'.join(md))
    teach(players)

def teach(players):
    a=players['bat'].iloc[0];settings=json.loads((E/'settings.json').read_text());sr,sw=settings['scales']['bat']
    text=f"""# How to defend the SupeRR Selector strategy

**The selection question.** When a player repeatedly encounters consequential balls, how often does their contribution beat what that same player usually delivers in comparable cricket? We also require positive pressure quality overall. A steady performer can therefore rank above a brilliant scorer who has a higher ordinary baseline.

## What pressure means

{PRESS} The next delivery can matter greatly when a chase is still winnable. Needing 36 off six is extremely hard, but most plausible next balls leave the game nearly settled. In the saved seven-wicket, league-stage illustrations, needing 6, 22 and 36 off six scores 12.73, 85.79 and 12.44 respectively. These are model examples. The report’s 7 October 2022 example is an actual delivery.

Before the ball we know the score, how many wickets and legal balls remain, and the target if chasing. Two logistic regressions learn a batting win probability from decisive regulation matches completed before each three-month evaluation block. A match has equal total fitting weight regardless of its length. The first innings compares progress to an earlier scoring curve. A chase uses runs required and resources. At least 100 earlier training matches, 25 per outcome class and 10 earlier matches in similar states are needed. These are practical safeguards. Revisions without trustworthy announcement timing, super overs and unreliable states lose coverage. Unscored balls cannot be used as ordinary controls.

The first half of the pressure score measures how far the win chances spread after eight possible next legal outcomes. Earlier frequencies by innings and phase weight those outcomes. Divide that spread by 0.10 and cap it at one. Another 30% comes from urgency multiplied by contestability. Urgency weights elapsed time 45%, scoring demand 35% and resource loss 20%. In a chase demand uses the known target. In the first innings it uses an earlier typical winning first-innings total. Contestability peaks at a 50–50 game. The final 20% comes from stage stakes, with 1 for a final and 0.4 for a league match. These weights are explicit judgements. The main label selects scores at or above the historical 75th percentile. Batters and bowlers face the same pressure label for a given delivery. There is no per-year quota.

**Actual ball.** Before delivery 17641:2:41 the batting side needed 127 off 81 with nine wickets left. Its estimated win chance was 0.46279. Contestability was 4 × 0.46279 × (1 − 0.46279) = 0.99446. The score was 100 × (0.50 × 0.26164 + 0.30 × 0.52408 × 0.99446 + 0.20 × 0.40) = 36.72. That clears the retrospective cutoff of 36.585 by only 0.13. This is a borderline selection. The next delivery’s actual runs did not influence its label.

## What should the player have done

Expected runs are what the data lead us to expect in situations of this kind. Earlier quarters supply rates for the innings, phase and competition type. A finer cell adds wickets remaining and chase scoring demand. Thin cells borrow the equivalent of 200 balls from the broader rate. The player’s identity does not enter this expectation. Wicket opportunities vary strongly by phase. The previous global wicket rate underpredicted death-over bowler wickets by 2.94 per 100 balls across match-phase units. The simpler phase-and-competition wicket estimate reduces the shortfall to 0.54. It pools 200 balls toward the earlier global rate. This fairness gain justifies using the stratified prior even though it has no clear overall match-MSE advantage.

A batter earns positive run credit by scoring more than expected and wicket credit by surviving more often than expected. A bowler earns run credit by conceding fewer than expected and wicket credit by taking more credited wickets than expected. A non-striker run-out is not a striker dismissal. A run-out is not a bowler wicket. Wides and no-balls can contribute actual runs and wickets without consuming a legal ball. Expected amounts accumulate on legal exposure. Exclude ambiguous concession or dismissal assignments. Aggregate first, then express the residuals per 100 legal balls.

Runs and wickets use different units. Divide each by the corresponding Phase 2 eligible-player spread for that role, then weight runs 85% and wickets 15%. For batters the denominators are {sr:.5f} runs and {sw:.5f} wickets per 100. For bowlers they are {settings['scales']['bowl'][0]:.5f} and {settings['scales']['bowl'][1]:.5f}. The weights retain wicket value while giving the better supported run estimates more influence. They are judgement calls. Runs-only, 70/30, contextual and global alternatives are in the method comparison CSVs.

## Whose ordinary performance

Imagine a player’s pressure match contains 70% second-innings IPL middle-over balls and 30% second-innings IPL death balls. Estimate that player’s ordinary adjusted rate separately in those cells. Combine them 70/30 for this match. The mix changes from match to match. This prevents an IPL finisher’s death-over performances from being compared mainly with their ordinary international powerplays. It cannot make two opponents or grounds equally difficult.

If there are fewer than 20 ordinary legal balls or three positive-exposure ordinary matches in a cell, remove the cell from the comparison. Report the pressure balls lost. Require at least 75% retained. The player also needs 50 pressure balls across eight positive-exposure matches and 100 ordinary balls across eight matches. These inherited thresholds trade wider coverage for less fragile comparisons. Relaxing them to 10 balls/two matches per cell, 50% overlap and 50 ordinary balls across five matches increases positive qualifying batters from 27 to 37 and bowlers from 24 to 30. The watchlist is for promising partial evidence. No one receives credit for having little exposure.

## Worked player from components to first place

{a.player_name} has {a.hm:.0f} matched pressure matches, {a.hb:.0f} pressure legal balls, and {a.ob:.0f} comparable ordinary balls across {a.om:.0f} matches. His pressure run residual is {a.run_high:+.2f} and the favorable wicket residual is {a.wicket_high:+.2f}. Pressure quality is 0.85 × ({a.run_high:.2f}/{sr:.2f}) + 0.15 × ({a.wicket_high:.2f}/{sw:.2f}) = {a.absolute:+.2f}. A positive combined value qualifies despite slightly below-expected pressure scoring. His wicket preservation matters.

His matched ordinary run and wicket residuals are {a.run_ord:+.2f} and {a.wicket_ord:+.2f}, giving ordinary quality {a.ordinary:+.2f}. Subtracting it from pressure quality gives uplift {a.uplift:+.2f}. This is a standardized difference. The pressure innings do not yield 2.15 additional runs per ball. The low ordinary run baseline is a major reason to inspect this selection carefully.

For each match compare pressure quality against that match’s own mix of ordinary cells. Give a success one vote for six or more pressure balls, half for three and one-sixth for one. Count a tie as unsuccessful. Jadeja earns {a.success_mass:.2f} successful votes from {a.effective_votes:.2f} in total. His observed success fraction is {a.positive_share:.1%}. Add four successes and four failures as a neutral regularizer. His decision score is 100 × ({a.success_mass:.2f}+4)/({a.effective_votes:.2f}+8) = {100*a.score:.2f}. Four and four reuse the earlier eight-match shrinkage strength and make a perfectly balanced prior explicit. This is not an estimated probability of future success. A 50-point score means balanced weighted outcomes after smoothing. Qualification also requires positive pooled pressure quality and uplift. Rank by score, then larger uplift and finally stable numeric ID for exact ties.

The same match can supply both pressure and ordinary balls. Resample whole matches together across players 2,000 times, refit the matched ordinary rates and recompute votes on the originally supported cells. Mark samples missing if pressure has no exposure or any necessary ordinary cell loses exposure. Require at least 90% valid draws. The reported range is the middle 95% of valid *score* draws. Gates and model fits stay fixed. The paired differences cannot distinguish adjacent ranks among the leading 13 in either role. A close ordering is still needed for the dropdowns.

## Why we changed Phase 3

Phase 1 constructed the situation, and Phase 2 estimated expected contributions and each player's own matched ordinary level. Phase 2’s initial ordering used 70/30 runs and wickets, 75/25 absolute quality and own-baseline uplift, and match-count shrinkage. Phase 3 adopted global historical wicket expectations and 85/15 weighting, retained 75/25, and changed the final ordering to 0.75 × the 20th percentile of performance + 0.25 × (2 × absolute positive-match share − 1) − 0.25 × worst single-match loss. Q20 is a quantile. It is not a 95% confidence bound. Its uncertainty gap and the deletion loss had rank correlations of 0.79 for batters and 0.87 for bowlers. Neither the extra influence penalty nor an absolute consistency vote directly answers how often a player beats their own baseline.

The independent audit froze deletion support, handled zero-pressure bootstrap exposure as missing, and checked random-seed and threshold sensitivity. Freezing support did not change any comparable saved deletion penalty. The final selection changes predominantly because the success criterion changed. There is no additional deletion penalty or Q20 deduction in the final score. The simpler historical approach wins on interpretability and alignment. Predictive superiority has not been established.

## Why historic form is the official answer

We tested all-history, trailing 365 days, exponentially decayed history with a 365.25-day half-life, and an equal blend where both history and recent data qualified. The half-life follows one annual cricket cycle and was tested as a sensitivity. Decay applies to ordinary baselines, pooled performance and match votes. Its four-player turnover in each role could be material to a buying decision. None of the twenty selections clears the main recent-only eligibility rule. Recent exposure is not itself a supported recent ordinary comparison. For example Jadeja has 54 recent pressure legal balls before matching, yet too little recent comparable ordinary evidence.

We also tried a lighter diagnostic with 10 ordinary balls/two matches per cell, four pressure matches/25 balls, three ordinary matches/40 balls, half of pressure exposure retained and 90% valid draws. To assign a recent rank we still require positive absolute quality and uplift. Nine batters and six bowlers qualify. Among the twenty historical selections, only Kuldeep has a supported lighter recent comparison (score about 49). The separate all-player files flag above-midpoint performance in both periods, possible decline, recent improvement, recent-only evidence and insufficient evidence. These are descriptive changes with considerable sampling error.

For forward tests, train before January 1 and evaluate separately on that year. Use only training scores for each pressure cutoff. Fold-specific component spreads use pre-cutoff players with at least 50 high legal balls. The outcome expectations themselves update quarter by quarter using only previously played matches. Evaluate subsequent repeatability only where the later period has at least two matches and 12 pressure balls, two ordinary matches and 20 ordinary balls, supported 10/2 cells and 50% coverage. A noisy later-period estimate is still a noisy target. The historical Phase 3 comparator is a fold-local reconstruction with global wickets, corrected support and 300 draws, so it is an approximation to the saved 500-draw Phase 3 calculation. Its ranking correlation is comparable, its raw score uses a different scale. All model choices have seen these years during development.

**Selection strategy comparison.** Historical Phase 3 rewards all-round absolute pressure quality but adds several correlated decisions. The historical repeatability score better matches the stated target and requires one smoothing rule. Decay uses the same counting rule but emphasizes recent years and alters membership. Recent-only and blended approaches lose too much coverage for a full list. Recent bowler signals deserve monitoring given the small positive later comparisons. None of these designs resolves unrecorded opponent strength, tactics or selection bias. Historic scoring is cheap to reproduce from saved ball outputs, while sensitivity and match resampling account for most of the extra computation.

**Examples of period comparisons.** In the lighter diagnostic, Suryakumar Yadav’s batting score rose from 44.9 to 63.8 across five recent supported matches. Shivam Dube fell from 56.4 to 43.9 across seven. Bowling scores for Roston Chase were above the midpoint in both periods, 58.6 and 66.7, with four recent matches. These are reasons to review performances and selection opportunities. They are too small to establish individual trends. Russell has no observed recent pressure balls in this sample.

**Which assumptions change membership?** Under the final scoring rule, a contextual wicket estimator retained all ten names in each role. The global prior retained all ten batters and nine bowlers. Runs-only retained nine batters and eight bowlers. The relaxed support rule retained just three batters and five bowlers. Removing the shrinkage retained all ten batters and eight bowlers. This is why strong claims about the exact selected cohort would be hard to defend. The strict rules were inherited to support comparability, and neither the stricter nor looser cutoffs were estimated as uniquely optimal. Under the main pressure definition, 1,420 of 1,821 analysable final events (78%) and 17,717 of 72,730 league events (24%) qualify. Stage has a built-in contribution and these samples differ in other ways.

## Five questions from a Royals analyst

1. **Why take Jadeja above Kishan when Kishan is the better pressure run scorer here?** Kishan has the larger pressure run residual, +48.21 versus −2.25 per 100. Jadeja benefits from wicket preservation and much lower ordinary scoring. We chose consistency against ordinary performance as the ordering criterion. For an absolute scoring brief the answer would change.
2. **Are you rewarding a player simply for having a poor ordinary baseline?** A poor baseline can inflate uplift. We require positive absolute pressure quality as well as positive uplift, and disclose the actual ordinary level. Jadeja’s modest +0.25 absolute score makes this a live scouting question.
3. **Can I trust the current No. 10 over No. 11?** The bowling scores are 61.94 for Joseph and 61.90 for Haris Rauf. The matched bootstrap cannot separate them. Their rank order gives a reproducible administrative selection, and the full score range and role evidence belong in the discussion.
4. **Does removing weak samples mainly exclude finishers?** Seventeen of 60 lower-order batting specialists with 50 observed pressure balls passed support, compared with 35 of 89 others. Shahrukh retains only 17 matched balls. The watchlist names players with usable but incomplete signals. It identifies candidates to research, without establishing market undervaluation.
5. **Would you buy the highest-ranked bowler?** Muzarabani’s main evidence is 11 matches, 96 balls, and zero matched IPL pressure balls. Check death-over footage against relevant batting, availability, price and fit. The same exercise for Bumrah has 127 matched IPL balls. Those are different transfer risks even before the auction register is checked.

## File map and reproduction

Run from the project root in this order. The first three modules use the project environment. Report generation reads the resulting CSVs. The Word rendering step uses the document runtime.

    .venv/bin/python -m final_strategy.analyse
    .venv/bin/python -m final_strategy.supplement
    .venv/bin/python -m final_strategy.recent
    .venv/bin/python -m final_strategy.report
    .venv/bin/python -m pytest -q

The exact ordered CSVs carry stable IDs and official-data display names. Under evidence/, all_players includes the whole mapped player universe for each role, including people with no recorded appearance. selection_changes contains the union of former and new top tens with each intermediate rank and selection or exclusion reason. The pressure, method, chronological, usage and interval files preserve alternate results. Negative evidence, missing comparable exposure and no recorded appearances have different interpretations. Missing/untrustworthy coverage can overlap in the reasons a player lacks a rating. The source cannot establish every reason for an absent player.

The six pressure specifications change one of threshold (20/25/30%), stakes (included/excluded), or leverage scale (0.05/0.10/0.15) at a time. Every specification rebuilds pressure and ordinary samples, support, match votes, intervals and ranks for the final rule. The 18 older audit scenarios used a different ranking rule. The present counts use six. None is a probability.

**Ownership and sources.** The PDF is a draft for your review and decision. Unstop allows AI assistance and requires that the submitted insights, picks and recommendations be yours. Compare names and IDs from the two CSVs to the live dropdowns. Account-specific dropdown contents and actual auction registration could not be independently verified. The supplied brief sets out the list and matching requirements. [Official Unstop listing]({U}). [Wisden retention report, 17 November 2026]({W}).
"""
    (OUT/'EXPLAIN_MY_STRATEGY.md').write_text(text)
if __name__=='__main__':report()
