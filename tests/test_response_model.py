import numpy as np
import pandas as pd

from final_strategy.response_model import DATE, date_prior, entry_position, prior_style, resource_basis


def test_same_day_fixtures_do_not_supply_an_opponent_rating():
    days = pd.to_datetime(['2025-01-01','2025-01-01','2025-01-03'])
    source = pd.DataFrame({DATE:days,'id':[1,1,1],'r':[5,8,2]})
    earlier = date_prior(source,['id'],['r'])
    assert earlier.hist_r.tolist()==[0,13]


def test_entry_order_cannot_change_as_future_players_arrive():
    innings = pd.DataFrame({'match_id':3,'innings':1,'innings_ball_num':[1,2,3,4],
                            'striker_id':[4,8,4,15],'non_striker_id':[8,4,12,12]})
    np.testing.assert_array_equal(entry_position(innings),[1.5,1.5,1.5,4])
    np.testing.assert_array_equal(entry_position(innings.iloc[:2]),entry_position(innings).iloc[:2])


def test_current_date_style_not_available_at_debut():
    d=pd.DataFrame({'id':1,DATE:pd.to_datetime(['2025-01-01','2025-01-01','2025-01-02']),
                    'style':['Pace','Pace','Spin']})
    assert prior_style(d,'id','style').tolist()==['unknown','unknown','Pace']


def test_no_future_scoring_value_without_balls_or_wickets():
    grid=resource_basis(np.array([0,6,6,6]),np.array([10,0,2,3]))
    assert (grid[:2]==0).all()
    assert np.all(grid[3]>=grid[2])
