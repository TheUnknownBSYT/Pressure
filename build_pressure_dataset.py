#!/usr/bin/env python3
"""Build offline pre-delivery state, forward-only WP estimates and provisional pressure."""
import argparse
from dataclasses import asdict
from pathlib import Path
import importlib.metadata
import platform
import time
import pandas as pd
from pressure_pipeline.data import load_sources, enrich, sha256, save_json
from pressure_pipeline.state import clean_events, reconstruct, quality_flags
from pressure_pipeline.model import ModelConfig, fit_historical, pressure_flags
from pressure_pipeline.report import verify_invariants, write_reports, plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--manifest-path', type=Path, help='Separate execution record when rebuilding beside an archived manifest')
    parser.add_argument('--top-percent', type=float, default=25)
    parser.add_argument('--min-train-matches', type=int, default=100)
    parser.add_argument('--min-support-matches', type=int, default=10)
    parser.add_argument('--fold-months', type=int, default=3)
    args = parser.parse_args()
    if not 0 < args.top_percent < 100 or min(args.min_train_matches, args.min_support_matches, args.fold_months) < 1:
        parser.error('Invalid percentile or model history settings')
    start = time.monotonic()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = ModelConfig(min_train_matches=args.min_train_matches, min_support_matches=args.min_support_matches, fold_months=args.fold_months)
    print('Auditing source schemas and keys...', flush=True)
    tables, audit = load_sources(args.data_dir)
    print('Reconstructing all delivery events...', flush=True)
    clean = clean_events(enrich(tables))
    clean.to_parquet(out / 'clean_deliveries.parquet', index=False)
    state, innings = reconstruct(clean)
    state, overs, reasons = quality_flags(state, innings)
    innings.to_csv(out / 'innings_summary.csv')
    overs.to_csv(out / 'over_audit.csv', index=False)
    print(f'{len(state):,} events retained; {state.standard_t20_eligible.sum():,} structurally eligible. Fitting chronological models...', flush=True)
    scored, folds, metrics, coef, weights, examples = fit_historical(state, cfg)
    scored, thresholds = pressure_flags(scored, args.top_percent)
    checks = verify_invariants(scored)
    # Include historical expectations and lagged pressure context in the full state.
    state_cols = list(state.columns) + [c for c in scored if c not in state and (c.endswith('_pre') or c.startswith('historical_') or c in ['probability_status', 'model_train_end_date', 'state_support_matches'])]
    scored[state_cols].to_parquet(out / 'match_state.parquet', index=False)
    scored.to_parquet(out / 'pressure_scored.parquet', index=False)
    scored.loc[scored.high_pressure_selected].to_parquet(out / 'high_pressure_deliveries.parquet', index=False)
    compact = ['delivery_id', 'match_id', 'match_date', 'innings', 'overid', 'phase_pre', 'competition_type', 'stage',
               'score_pre', 'wickets_remaining_pre', 'legal_balls_remaining_pre', 'runs_required_pre', 'required_run_rate_pre',
               'win_probability_pre', 'difficulty_proxy', 'contestability', 'leverage', 'urgency', 'match_stakes', 'pressure_score_v1',
               'state_support_matches', 'model_train_end_date', 'leverage_method']
    available = scored[scored.pressure_eligible]
    # Compact file includes representative random rows and the highest pressure states.
    sample = pd.concat([available.sample(n=min(1000, len(available)), random_state=cfg.seed), available.nlargest(100, 'pressure_score_v1')]).drop_duplicates('delivery_id')
    sample[compact].sort_values(['match_date','match_id','innings','overid']).to_csv(out / 'pressure_components.csv', index=False)
    for name, frame in [('model_folds', folds), ('model_evaluation', metrics), ('model_coefficients', coef), ('counterfactual_outcome_weights', weights), ('illustrative_situations', examples)]:
        frame.to_csv(out / f'{name}.csv', index=False)
    print('Writing independent audits, methodology and diagnostic plots...', flush=True)
    summary, audits = write_reports(args.data_dir, out, tables, audit, scored, overs, reasons, checks, thresholds, cfg, metrics, examples)
    methodology = Path(__file__).parent / 'pressure_pipeline' / 'methodology.md'
    if methodology.is_file():
        (out / 'pressure_methodology_v1.md').write_text(methodology.read_text())
    plots(scored, examples, out)
    for name, record in audit.items():
        assert sha256(args.data_dir / f'{name}.csv') == record['sha256'], f'Original file changed: {name}'
    manifest = {'rows': len(scored), 'matches_with_deliveries': scored.match_id.nunique(),
        'standard_eligible_deliveries': int(scored.standard_t20_eligible.sum()), 'scored_deliveries': int(scored.pressure_eligible.sum()),
        'high_pressure_deliveries': int(scored.high_pressure_selected.sum()), 'selected_top_percent': args.top_percent,
        'thresholds': thresholds, 'model_config': asdict(cfg), 'python': platform.python_version(),
        'versions': {lib: importlib.metadata.version(lib) for lib in ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'pyarrow']},
        'source_sha256': {n: a['sha256'] for n, a in audit.items()}, 'source_hashes_unchanged': True,
        'elapsed_seconds': round(time.monotonic() - start, 2)}
    save_json(args.manifest_path or out / 'run_manifest.json', manifest)
    print(f"Done: {len(scored):,} processed; {len(available):,} scored; {manifest['high_pressure_deliveries']:,} high pressure. Outputs: {out}", flush=True)


if __name__ == '__main__':
    main()
