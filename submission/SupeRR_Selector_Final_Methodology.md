# SupeRR Selector | Round 1

## Historical pressure rankings

IPL and T20I, 2022–2026. Scores are conservative estimates of run-equivalent value per six legal balls. Official player IDs identify the dropdown selections.

| Rank | Batter (official ID) | Score | Bowler (official ID) | Score |
|---|---|---|---|---|
| 1 | Ishan Kishan (4117) | +1.63 | Jasprit Bumrah (2933) | +2.41 |
| 2 | Glenn Phillips (4853) | +1.41 | Abrar Ahmed (5979) | +2.30 |
| 3 | Ravindra Jadeja (152) | +0.90 | Shaheen Afridi (6355) | +2.10 |
| 4 | Andre Russell (1318) | +0.68 | Sunil Narine (781) | +1.66 |
| 5 | Cameron Green (5939) | +0.46 | Mujeeb Ur Rahman (6450) | +1.58 |
| 6 | Azmatullah Omarzai (6676) | +0.45 | Axar Patel (2876) | +0.83 |
| 7 | Shreyas Iyer (3647) | +0.45 | Blessing Muzarabani (6523) | +0.69 |
| 8 | Rinku Singh (4708) | +0.42 | Taskin Ahmed (2681) | +0.39 |
| 9 | Virat Kohli (47) | +0.35 | Kuldeep Yadav (3644) | +0.25 |
| 10 | George Dockrell (1101) | -0.12 | Jason Holder (2633) | +0.13 |

## What the order tells us

Ishan Kishan leads batting with 118 matched pressure balls, 100 in the IPL. Bumrah leads bowling with 180 across 19 matches in both competitions. Abrar Ahmed beat his ordinary benchmark in 82% of weighted appearances, all in T20Is. The paired interval for his difference from Bumrah includes zero.

Rinku Singh clears the majority rule at 50.52%. Shreyas Iyer’s uplift is only +0.006 per six. George Dockrell qualifies on positive point quality and uplift but has a negative conservative score. All 20 positions match the approval review.

## How the rankings work

Before each ball, reconstruct the score, wickets, legal balls remaining and chase demand. Separate first-innings and chase models trained on earlier matches estimate win chances under plausible next-ball outcomes. Pressure = 0.50 × leverage + 0.30 × urgency × contestability + 0.20 × recorded stage, using the original scaled components. A ball qualifies at or above the prior-quarters 75th-percentile cutoff. A close chase can carry more pressure than a harder chase whose result is nearly settled.

For each role, 25 pre-delivery inputs describe the match state, role, venue context and earlier run and wicket records of the actual opponent and opposing team. Batting value is runs above expectation plus the run value of dismissals avoided relative to expectation. Bowling value is runs saved relative to expectation plus the run value of wickets above expectation. Wicket costs reflect the scoring resources lost. Run-outs give no bowler wicket credit.

Compare each pressure appearance with the player’s own ordinary rate for the same mix of competition, innings and phase. Each retained comparison cell needs at least 20 ordinary legal balls in three matches. Require eight pressure and eight ordinary matches, 50 pressure and 100 ordinary legal balls, and coverage of at least 75% of the player’s pressure balls.

For match m, let q be six times total pressure value divided by matched legal pressure balls, b its matched ordinary rate, and w = min(1, legal pressure balls / 6). Quality Q = sum(w × q) / sum(w). Uplift = sum(w × (q − b)) / sum(w). Require Q > 0, uplift > 0 and sum(w × indicator[q > b]) / sum(w) > 0.50. A tied appearance contributes no success. Exactly 50% fails.

Rank separately by the 20th percentile of Q over 2,000 seeded whole-match bootstrap draws. Break ties by higher observed Q, then lower official player ID. Short appearances get less influence. Uplift determines eligibility and the conservative absolute score determines order.

## Forward check

Of the majority-rule shortlists formed before 2026, five batters and six bowlers had measurable follow-up. They beat their frozen ordinary benchmarks in 58% and 70% of weighted appearances on average, versus 48% and 60% for the earlier rule. The groups differ and follow-up is sparse. Bowling results were less favorable in 2025. Historical reliability gives limited evidence about future IPL performance.

## RR auction shortlist | 18 November 2026

Screened all 17 eligible batting and 22 bowling entries. RR retained 19 players and has six open places, three overseas, and INR 27.05 crore available. Four released candidates qualify. No official 2027 registration or final auction-list entry was verified for any of them. These are scouting priorities conditional on listing.

| Target and role | Release | Pressure evidence and squad fit |
|---|---|---|
| Matheesha Pathirana · overseas pace | KKR | Bowling #12, +0.02. Won 62% across 19 weighted appearances; 159 IPL and 64 T20I balls. Death-bowling option alongside Archer and Sandeep. |
| Avesh Khan · Indian pace | LSG | Bowling #17, −0.40. 260 IPL and 19 T20I balls in 25 matches. Domestic pace depth after Deshpande’s release preserves an overseas slot. Success narrowly clears 50%. |
| Shimron Hetmyer · overseas finisher | RR | Batting #15, −0.60. 62% success across 30 matches; 196 IPL and 40 T20I balls. A buyback restores a left-handed finishing option. The conservative score is below zero. |
| Blessing Muzarabani · overseas pace | KKR | Bowling #7, +0.69. 81% success in 11 matches; all 94 balls are T20I. An alternative to Pathirana if RR prioritises an additional seam option; IPL transfer is uncertain. |

## Availability across the full ranked pool

Retained by RR: Ravindra Jadeja and Dasun Shanaka. Retained elsewhere: SRH — Kishan, Khaleel Ahmed; GT — Phillips, Gill, Holder, Siraj; KKR — Green, Rinku Singh, Narine, Powell; PBKS — Omarzai, Shreyas Iyer, Prabhsimran Singh; RCB — Kohli, Venkatesh Iyer, Jitesh Sharma, Padikkal, Suyash Sharma; MI — Bumrah; DC — Axar Patel; LSG — Kuldeep Yadav; CSK — Noor Ahmad.

Released: the four players above. Playing availability unverified: Dockrell, Abrar Ahmed, Shaheen Afridi, Mujeeb Ur Rahman, Taskin Ahmed, Maheesh Theekshana, Tanzim Sakib, Alzarri Joseph and Naveen ul Haq. Andre Russell appears in both ranked pools and KKR lists him as a coach.

Sources: [IPL retention advisory and squad slots](https://www.iplt20.com/news/article/tata-ipl-2027-player-retentions-announced), [official released-player list](https://www.iplt20.com/cms/files/jot2qw8s50z1/3gFAJ4z1USxcXYhaZVl65L/2a463a9c02a5bb7fd12fa3fbe91cc299/TATA_IPL_2027-_Release_Player_List.pdf), [KKR staff roster](https://www.kkr.in/players). Squad fits are selection judgments.
