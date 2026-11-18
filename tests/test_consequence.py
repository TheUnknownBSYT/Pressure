import numpy as np
import pandas as pd

from final_strategy.consequence import dynamic, tag


def test_temporal_threshold_ignores_test_quarter_and_current_outcome(tmp_path, monkeypatch):
    import final_strategy.consequence as consequence

    monkeypatch.setattr(consequence, 'E', tmp_path)
    frame = pd.DataFrame({
        'delivery_id': ['1:2:1', '2:2:1', '3:2:1', '4:2:1'],
        'match_id': [1, 2, 3, 4],
        'match_date_parsed': pd.to_datetime(['2025-09-01', '2025-09-06', '2025-10-01', '2025-11-01']),
        'model_fold_start': pd.to_datetime(['2025-07-01']*2 + ['2025-10-01']*2),
        'model_train_end_date': pd.to_datetime(['2025-06-30']*2 + ['2025-09-30']*2),
        'pressure_score_v1': [20., 40., 36., 95.],
        'innings': 2, 'legal_balls_remaining_pre': 12, 'leverage': .3, 'urgency': .9,
        'contestability': .8, 'recent_wickets_12_pre': [0,1,2,3], 'match_stakes': .4,
        'team_runs_event': [1,0,6,2], 'wicket_event': 0,
    })
    tag(frame)
    assert frame.high_static.iloc[:2].isna().all()
    assert frame.high_static.iloc[2:].all()  # 75th percentile of [20,40] = 35.
    assert np.allclose(frame.weight_leverage + frame.weight_urgency_contest + .2, 1)
    changed = frame.assign(team_runs_event=10, wicket_event=1)
    dynamic(changed)
    np.testing.assert_allclose(changed.pressure_dynamic, frame.pressure_dynamic)
