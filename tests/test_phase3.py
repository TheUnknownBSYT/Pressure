from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from phase3_pipeline.audit import calibration_audit,supported_matches,reliability,prior_wickets
from phase2_pipeline.core import summarize

def sample():
    rows=[]
    for mid in range(12):
        for high in [0,1]:
            rows.append(dict(player_id=1,match_id=mid,comparison_cell='A',high=high,legal=10,events=10,balls_faced=10,runs=10,wickets=0,rr=2+high+(mid%3),wr=.1))
    return pd.DataFrame(rows)

def test_stronger_support_keeps_pressure_denominator():
    m=sample();thin=m.iloc[:4].copy();thin['comparison_cell']='thin'
    z=supported_matches(pd.concat([m,thin]),20,3)
    assert len(z[z.comparison_cell.eq('thin')&z.high.eq(1)])==2
    assert not (z.comparison_cell.eq('thin')&z.high.eq(0)).any()
    a=summarize(z);assert a.iloc[0].pressure_legal_all==140
    assert a.iloc[0].pressure_legal==120

def test_reliability_does_not_apply_second_sample_size_shrinkage():
    m=sample();s=summarize(m,bootstrap=500)
    a=reliability(s,m,[10,1]);s['shrinkage']=.0001;b=reliability(s,m,[10,1])
    assert np.allclose(a.reliability_score,b.reliability_score)
    assert a.iloc[0].best_match_drop>=0
    assert a.iloc[0].combined_positive_match_fraction==1
    assert a.iloc[0].reliability_eligible

def test_supported_uncertainty_is_required():
    m=sample();s=summarize(m,bootstrap=500);s['bootstrap_valid_uplift_fraction']=.7
    a=reliability(s,m,[10,1]);assert not a.reliability_eligible.any()
    assert a.revised_rank.isna().all()

def test_calibration_uses_only_earlier_out_of_fold_labels():
    a=pd.DataFrame(dict(match_id=range(130),innings=1,training_label_available=True,
        win_probability_pre=np.tile([.2,.8],65),batting_win_label_outcome=np.tile([0.,1.],65),
        match_date_parsed=[pd.Timestamp('2024-01-01')]*110+[pd.Timestamp('2026-01-01')]*20))
    x,*_=calibration_audit(a);a.loc[a.match_id>=110,'batting_win_label_outcome']=1-a.loc[a.match_id>=110,'batting_win_label_outcome']
    y,*_=calibration_audit(a)
    pd.testing.assert_series_equal(x.calibrated,y.calibrated)
    z=x[x.calibrated.notna()];assert (z.calibration_train_end<z.match_date_parsed).all()

def test_final_recommendations_and_prior_output_preservation():
    root=Path('outputs/phase3')
    if not (root/'phase3_manifest.json').exists():pytest.skip('Run Phase 3 first')
    from pressure_pipeline.data import sha256
    m=json.loads((root/'phase3_manifest.json').read_text())
    assert all(m['baseline_rating_reproduced'].values())
    for path,value in m['protected_file_hashes'].items():assert sha256(path)==value
    for role in ['batters','bowlers']:
        a=pd.read_csv(root/f'final_{role}_top10.csv')
        assert len(a)==10 and a.player_id.is_unique
        assert a.final_order.tolist()==list(range(1,11))
        assert a.reliability_score.is_monotonic_decreasing
        assert a.reliability_eligible.all()
        assert a.overlap_fraction.ge(.75).all()
        assert a.bootstrap_valid_uplift_fraction.ge(.9).all()
    decision=json.loads((root/'calibration_decision.json').read_text())
    assert decision['adopt'] is False and decision['bootstrap_hi95']>0

def test_prior_wicket_expectation_excludes_current_match():
    a=pd.DataFrame(dict(match_id=[1,2,3],match_date_parsed=pd.to_datetime(['2024-01-01','2024-01-02','2024-04-01']),
        standard_t20_eligible=True,outcome_valid=True,analysis_available=[False,False,True],
        legal_event=[10,10,10],bat_dismissal=[1,1,1],bowl_wicket=[1,1,1],
        expected_bat_dismissal_per_legal=.1,expected_bowl_wicket_per_legal=.1))
    x,_=prior_wickets(a);a.loc[2,['bat_dismissal','bowl_wicket']]=9;y,_=prior_wickets(a)
    assert x.loc[2,'bat_prior_wicket_rate']==y.loc[2,'bat_prior_wicket_rate']==.1
