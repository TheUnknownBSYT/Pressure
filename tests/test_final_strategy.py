import numpy as np
import pandas as pd
from final_strategy.analyse import sample

def fixture():
    rows=[]
    for i in range(12):
        for high in [0,1]:
            rows.append(dict(player_id=1,comparison_cell='1|middle|IPL',match_id=i,high=high,
                             legal=3 if high and i==0 else 12,rr=(3 if i!=1 else -3) if high else 0,wr=0,date=pd.Timestamp('2024-01-01')+pd.Timedelta(days=i)))
    return pd.DataFrame(rows)

def test_score_is_repeated_within_player_uplift_with_short_vote():
    s,_=sample(fixture(),[1,1],'bat',pd.Timestamp('2025-01-01'),weight=1,draws=500)
    r=s.loc[1]
    assert r.eligible and np.isclose(r.success_mass,10.5) and np.isclose(r.effective_votes,11.5)
    assert np.isclose(r.score,14.5/19.5)
    assert r.lo<=r.score<=r.hi

def test_pressure_good_but_ordinary_better_fails():
    m=fixture();m.loc[m.high.eq(0),'rr']=50
    r=sample(m,[1,1],'bat',pd.Timestamp('2025-01-01'),weight=1)[0].loc[1]
    assert r.absolute>0 and r.uplift<0 and not r.eligible

def test_final_lists_and_joint_intervals():
    from pathlib import Path
    p=Path('outputs/historical_vote_submission')
    if not (p/'final_batters_top10.csv').exists():
        import pytest
        pytest.skip('Requires final strategy output')
    for role,label in [('bat','batters'),('bowl','bowlers')]:
        top=pd.read_csv(p/f'final_{label}_top10.csv')
        universe=pd.read_csv(p/'evidence'/f'{role}_all_players.csv')
        assert top.player_id.tolist()==universe[universe.eligible.eq(True)].sort_values(['score','uplift','player_id'],ascending=[False,False,True]).player_id.head(10).tolist()
        assert top['rank'].tolist()==list(range(1,11))
        assert (top.uplift>0).all() and (top.absolute>0).all()
        assert np.allclose(top.score,(4+top.success_mass)/(8+top.effective_votes))
