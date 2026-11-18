import numpy as np
import pandas as pd
from independent_audit.core import estimate,prepare,summarize_fixed,stratified_wickets


def sample(pid=1):
    return pd.DataFrame([dict(player_id=pid,match_id=i,comparison_cell='A',high=h,legal=10,events=10,balls_faced=10,runs=15,wickets=0,rr=2+h+i%3,wr=.1) for i in range(12) for h in [0,1]])


def test_zero_pressure_bootstrap_is_missing_not_neutral():
    s=np.zeros((2,2,3));s[0,0]=[10,2,.1]
    assert np.isnan(estimate(s)).all()


def test_fixed_support_deletion_does_not_discard_low_support_cell():
    # Three ordinary matches initially meet 20/3; deletion leaves 8 balls in
    # two matches. Fixed support retains their measured rate, legacy drops it.
    m=sample();extra=pd.DataFrame([dict(player_id=1,match_id=i,comparison_cell='B',high=h,legal=16 if i==0 and h==0 else 4,events=16,balls_faced=16,runs=0,wickets=0,rr=(-8 if h==0 else 10),wr=0) for i in range(3) for h in [0,1]])
    _,arr,mids,cells,_=prepare(pd.concat([m,extra]));after=arr.sum(axis=0)-arr[mids.index(0)]
    assert len(cells)==2 and after[cells.index('B'),0,0]==8
    assert np.isfinite(estimate(after)).all()
    assert not np.allclose(estimate(after),estimate(after[[cells.index('A')]]))


def test_missing_ordinary_after_deletion_is_explicit():
    a=np.zeros((2,2,3));a[:,1,0]=[5,5];a[0,0]=[10,2,.1]
    result=estimate(a)
    assert np.isfinite(result[:2]).all() and np.isnan(result[2:]).all()


def test_illegal_outcomes_preserved_without_fake_exposure_or_match():
    m=sample();row=m.iloc[0].copy();row['match_id']=99;row['high']=1;row['legal']=0;row['rr']=6;row['wr']=-1
    r,arr,*_=prepare(pd.concat([m,pd.DataFrame([row])]))
    assert r['pressure_matches']==12 and r['pressure_event_matches']==13
    assert r['pressure_legal']==120
    assert np.isclose(estimate(arr.sum(axis=0))[0],100*(m[m.high.eq(1)].rr.sum()+6)/120)


def test_player_seed_invariant_to_unrelated_candidates():
    a,_=summarize_fixed(sample(2),[10,1],'bat',bootstrap=200)
    b,_=summarize_fixed(pd.concat([sample(1),sample(2)]),[10,1],'bat',bootstrap=200)
    assert a.iloc[0].Q20==b[b.player_id.eq(2)].iloc[0].Q20


def test_score_decomposition_and_short_appearance_weight():
    m=sample();m.loc[(m.match_id==0)&m.high.eq(1),['legal','rr']]=[1,-20]
    a,_=summarize_fixed(m,[10,1],'bat',bootstrap=200);r=a.iloc[0]
    assert np.isclose(r.R,r.component_point+r.component_uncertainty+r.component_consistency+r.component_deletion)
    assert r.f_absolute_downweight6>r.f_absolute and r.brief_balls_lt6==1


def test_stratified_prior_is_forward_and_falls_back_to_global():
    d=pd.DataFrame(dict(match_id=[1,2,3],match_date_parsed=pd.to_datetime(['2024-01-01','2024-01-02','2024-04-01']),standard_t20_eligible=True,outcome_valid=True,analysis_available=[False,False,True],legal_event=[10,10,10],bat_dismissal=[1,3,1],bowl_wicket=[1,3,1],phase_pre=['death','middle','unknown'],competition_type='T20I'))
    a,_=stratified_wickets(d);d.loc[2,['bat_dismissal','bowl_wicket']]=100;b,_=stratified_wickets(d)
    assert a.loc[2,'bat_stratified_wicket_rate']==b.loc[2,'bat_stratified_wicket_rate']==.2


def test_vectorized_estimate_preserves_shared_match_covariance():
    _,arr,*_=prepare(sample());draws=estimate(np.stack([arr.sum(axis=0),arr.sum(axis=0)*2]))
    np.testing.assert_allclose(draws[0],draws[1]);assert np.isclose(draws[0,0]-draws[0,2],10)


def test_illegal_only_influential_match_is_deleted_too():
    m=sample();r=m.iloc[0].copy();r['match_id']=99;r['high']=1;r['legal']=0;r['rr']=1000;r['wr']=0
    s,_=summarize_fixed(pd.concat([m,pd.DataFrame([r])]),[10,1],'bat',bootstrap=200)
    # The zero-exposure appearance must not add an eligibility match, but its
    # real runs make it the most influential match and cannot escape deletion.
    assert s.iloc[0].pressure_matches==12 and s.iloc[0].D>50


def test_generated_audit_has_complete_separate_pressure_family():
    from pathlib import Path
    p=Path('outputs/independent_audit/pressure_definition_sensitivity.csv')
    if not p.exists():
        import pytest
        pytest.skip('Audit has not run yet')
    a=pd.read_csv(p)
    for role,g in a.groupby('role'):
        specs=g[['threshold_top_percent','stakes','leverage_scale']].drop_duplicates()
        assert len(specs)==18 and set(specs.threshold_top_percent)=={20,25,30}
        assert g.replicates.ge(2000).all()
        assert g.groupby('player_id').size().eq(18).all()
        assert g.loc[g.eligible,'R_rank'].notna().all()
        assert g.loc[~g.eligible,'R_rank'].isna().all()


def test_runs_only_is_invariant_to_wicket_estimator():
    from pathlib import Path
    p=Path('outputs/independent_audit/wicket_weight_ranking_sensitivity.csv')
    if not p.exists():
        import pytest
        pytest.skip('Audit has not run yet')
    a=pd.read_csv(p);a=a[a.run_weight.eq(1)]
    for (_,pid),g in a.groupby(['role','player_id']):
        v=g.R.to_numpy()
        np.testing.assert_allclose(v,np.repeat(v[:1],len(v)),equal_nan=True)


def test_short_match_exclusion_keeps_ordinary_and_does_not_relabel():
    from build_independent_audit import exclude_brief_pressure_matches
    m=sample();m.loc[m.match_id.eq(0)&m.high.eq(1),'legal']=2
    z=exclude_brief_pressure_matches(m)
    assert not (z.match_id.eq(0)&z.high.eq(1)).any()
    pd.testing.assert_frame_equal(z[z.high.eq(0)],m[m.high.eq(0)])
