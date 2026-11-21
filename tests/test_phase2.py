from pathlib import Path
import numpy as np
import pandas as pd
from phase2_pipeline.core import attribute,pool_predict,forward_expectations,summarize,rank,pressure_variant,identities

def raw():
    return pd.DataFrame(dict(is_wicket=[True,True,True,False],dismissed_player_id=[2,1,np.nan,np.nan],striker_id=[1]*4,non_striker_id=[2]*4,
        extras_type=['None','Overstep','None','wide'],runs=[1,4,0,0],bowler_runs_event_assumed=[1,5,0,2],bowler_wicket_event=[0]*4,
        wickets_remaining_pre=[8]*4,required_run_rate_pre=[9]*4,innings=[2]*4,phase_pre=['middle']*4,competition_type=['IPL']*4,bowler_concession_uncertain=[False]*4))

def test_attribution_non_striker_and_missing_id():
    d=attribute(raw());assert d.bat_dismissal.tolist()==[0,1,0,0];assert d.nonstriker_dismissal.tolist()==[1,0,0,0];assert d.dismissal_unresolved.tolist()==[False,False,True,False]
    assert d.balls_faced_nonwide.tolist()==[1,1,1,0];assert d.bat_runs.sum()==5;assert d.bowl_wicket.sum()==0

def sample_matches():
    rows=[]
    for m in range(12):
        for high in [0,1]:rows.append(dict(player_id=1,match_id=m,comparison_cell='2|death|IPL',high=high,legal=10,events=11,balls_faced=10,runs=12+high,wickets=0,rr=2+high+(m%3),wr=.2))
    return pd.DataFrame(rows)

def test_whole_match_bootstrap_repeatable_and_eligibility():
    m=sample_matches();a=summarize(m,bootstrap=500);b=summarize(m,bootstrap=500)
    assert a.iloc[0].eligible;assert a.iloc[0].pressure_matches==12
    assert np.allclose(a.iloc[0]['_bootstrap'],b.iloc[0]['_bootstrap'],equal_nan=True)
    # Correlated high/ordinary differences preserved by resampling paired matches.
    v=a.iloc[0]['_bootstrap'];assert np.allclose(v[:,0]-v[:,2],10)
    assert summarize(m[m.match_id<2]).eligible.eq(False).all()

def test_unmatched_ordinary_cells_do_not_inflate_samples():
    m=sample_matches();extra=m[m.high.eq(0)].copy();extra['comparison_cell']='1|powerplay|IPL';extra['legal']=1000
    a=summarize(pd.concat([m,extra]));assert a.iloc[0].ordinary_legal==120

def test_no_overlap_no_eligibility():
    m=sample_matches();m.loc[m.high.eq(0),'comparison_cell']='different';a=summarize(m)
    assert not a.eligible.any();assert a.iloc[0].overlap_fraction==0

def test_pool_ignores_current_outcomes_and_player_id():
    tr=attribute(raw());tr['legal_event']=[1,0,1,0];tr['match_id']=[1,2,3,4]
    a,_=pool_predict(tr,tr,'bat_runs');changed=tr.assign(bat_runs=999,striker_id=1234)
    b,_=pool_predict(tr,changed,'bat_runs');assert np.allclose(a,b)

def test_ranking_is_deterministic_and_ids_stay_distinct():
    a=summarize(sample_matches());b=a.copy();b['player_id']=2;b['pressure_run_adj']+=1
    x,_=rank(pd.concat([a,b]),scales=[10,1]);assert x.player_id.is_unique;assert x.iloc[0].player_id==2
    y,_=rank(pd.concat([a,b]),scales=[10,1]);pd.testing.assert_frame_equal(x,y)

def test_real_forward_predictions_and_pressure_reproduction():
    p=Path('outputs/phase2/phase2_player_event_metrics.parquet')
    if not p.exists():return
    d=pd.read_parquet(p);assert d.delivery_id.is_unique;assert len(d)==200999
    a=d[d.analysis_available];assert (a.expected_train_end<a.match_date_parsed).all()
    high,_=pressure_variant(d);assert high.equals(d.high_pressure_selected)
    assert not d.loc[~d.pressure_eligible,'analysis_available'].any()
    assert not a.bowler_concession_uncertain.any()
    assert a.loc[a.dismissal_type.eq('RunOut'),'bowl_wicket'].eq(0).all()

def test_real_identity_map_unique():
    if not Path("competition_team_player_mapping.csv").exists():
        import pytest
        pytest.skip("Requires the proprietary competition identity mapping")
    names,aliases=identities(Path('.'));assert names.player_id.is_unique
    assert names.player_name.notna().all()

def test_current_match_and_future_outcomes_cannot_change_expectations():
    parts=[]
    for mid in range(1,53):
        a=attribute(raw());a['match_id']=mid;a['legal_event']=[1,0,1,0]
        a['standard_t20_eligible']=True;a['pressure_eligible']=True
        a['match_date_parsed']=pd.Timestamp('2020-01-01' if mid<=50 else '2020-04-01')
        parts.append(a)
    d=pd.concat(parts,ignore_index=True);x,_=forward_expectations(d)
    mutated=d.copy();mutated.loc[mutated.match_id>50,['bat_runs','bowl_runs']]=999
    y,_=forward_expectations(mutated)
    cols=[c for c in x if c.startswith('expected_')]
    pd.testing.assert_frame_equal(x[cols],y[cols]);assert x.loc[x.match_id>50,'expected_bat_runs_per_legal'].notna().any()

def test_generated_rankings_are_eligible_and_intervals_valid():
    import pytest
    for role in ['batters','bowlers']:
        path=Path('outputs/phase2')/f'phase2_{role}.csv'
        if not path.exists():pytest.skip('Run Phase 2 first')
        r=pd.read_csv(path);t=r[r.eligible].head(10)
        assert len(t)==10 and t.player_id.is_unique
        assert t.pressure_matches.ge(8).all() and t.pressure_legal.ge(50).all()
        assert t.ordinary_matches.ge(5).all() and t.ordinary_legal.ge(50).all()
        assert t.overlap_fraction.ge(.5).all()
        assert t.rating.is_monotonic_decreasing
        assert (t.rating_lo95<=t.rating_hi95).all()
        assert t.top10_scenario_fraction.between(0,1).all()
