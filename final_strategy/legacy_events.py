"""Rebuild the original event expectations needed to reproduce the vote benchmark."""
from pathlib import Path
import pandas as pd
from phase2_pipeline.core import attribute, forward_expectations


def main():
    dest = Path('outputs/phase2')
    dest.mkdir(parents=True, exist_ok=True)
    original = pd.read_parquet('outputs/pressure_scored.parquet')
    events, scores = forward_expectations(attribute(original))
    assert events.delivery_id.is_unique
    eligible = events.loc[events.analysis_available]
    assert eligible.expected_train_end.lt(eligible.match_date_parsed).all()
    events.to_parquet(dest / 'phase2_player_event_metrics.parquet', index=False)
    scores.to_csv(dest / 'expected_model_validation.csv', index=False)
    print(f'Legacy contextual expectations rebuilt for {len(eligible):,} eligible events.')


if __name__ == '__main__':
    main()
