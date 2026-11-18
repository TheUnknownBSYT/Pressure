"""Source inspection and validated joins. Original CSV files are read-only."""
from pathlib import Path
import hashlib
import json
import pandas as pd

FILES = ('ball_by_ball', 'matches', 'playing_eleven',
         'competition_team_player_mapping', 'teams', 'competitions')
KEYS = {
    'ball_by_ball': ['match_id', 'innings', 'innings_ball_num'],
    'matches': ['match_id'], 'competitions': ['comp_id'],
    'teams': ['comp_id', 'team_id'],
    'playing_eleven': ['match_id', 'team_id', 'player_id'],
    'competition_team_player_mapping': ['comp_id', 'team_id', 'player_id'],
}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_sources(root):
    tables, audit = {}, {}
    for name in FILES:
        path = Path(root) / f'{name}.csv'
        # Literal 'None' is an extras category, not a missing value.
        t = pd.read_csv(path, keep_default_na=False, na_values=[''])
        tables[name] = t
        audit[name] = {
            'rows': len(t), 'columns': len(t.columns), 'sha256': sha256(path),
            'duplicate_keys': int(t.duplicated(KEYS[name]).sum()),
            'fields': {c: {'dtype': str(t[c].dtype), 'missing': int(t[c].isna().sum()),
                           'distinct': int(t[c].nunique()),
                           'values_or_examples': t[c].dropna().unique()[:40].tolist()}
                       for c in t.columns},
        }
        if t[KEYS[name]].isna().any().any() or (name != 'playing_eleven' and t.duplicated(KEYS[name]).any()):
            raise ValueError(f'{name}: null or duplicate join keys; resolve before reconstruction')
    return tables, audit


def enrich(tables):
    b = tables['ball_by_ball'].copy()
    b['source_row'] = range(len(b))
    b['delivery_id'] = b.match_id.astype(str) + ':' + b.innings.astype(str) + ':' + b.innings_ball_num.astype(str)
    b = b.merge(tables['matches'], on='match_id', how='left', validate='many_to_one', indicator='_match_join')
    b = b.merge(tables['competitions'], on='comp_id', how='left', validate='many_to_one', indicator='_comp_join')
    b['missing_match_metadata'] = (b._match_join != 'both') | (b._comp_join != 'both')
    b = b.drop(columns=['_match_join', '_comp_join'])
    b['match_date_parsed'] = pd.to_datetime(b.match_date, errors='coerce')
    # The roster has no innings key: super-over orders produce repeated players.
    # Preserve unanimous fields and set conflicting fields missing, never multiply rows.
    roster_keys = KEYS['playing_eleven']
    roster_source = tables['playing_eleven'].groupby(roster_keys, as_index=False).agg(
        {c: lambda x: x.iloc[0] if x.nunique(dropna=False) == 1 else pd.NA
         for c in tables['playing_eleven'] if c not in roster_keys})
    for role, team in [('striker', 'batting'), ('non_striker', 'batting'), ('bowler', 'bowling')]:
        roster = roster_source.rename(columns={
            'team_id': f'{team}_team_id', 'player_id': f'{role}_id',
            **{c: f'{role}_roster_{c}' for c in tables['playing_eleven'].columns
               if c not in ['match_id', 'team_id', 'player_id']}})
        b = b.merge(roster, on=['match_id', f'{team}_team_id', f'{role}_id'],
                    how='left', validate='many_to_one', indicator='_roster_join')
        b[f'{role}_missing_roster'] = b._roster_join != 'both'
        b = b.drop(columns='_roster_join')
        mapping = tables['competition_team_player_mapping'][['comp_id', 'team_id', 'player_id']].rename(
            columns={'team_id': f'{team}_team_id', 'player_id': f'{role}_id'})
        b = b.merge(mapping, on=['comp_id', f'{team}_team_id', f'{role}_id'],
                    how='left', validate='many_to_one', indicator='_mapping_join')
        b[f'{role}_missing_comp_mapping'] = b._mapping_join != 'both'
        b = b.drop(columns='_mapping_join')
    for role in ['batting', 'bowling']:
        t = tables['teams'].rename(columns={'team_id': f'{role}_team_id', 'team_name': f'{role}_team_name'})
        b = b.merge(t, on=['comp_id', f'{role}_team_id'], how='left', validate='many_to_one')
    assert len(b) == len(tables['ball_by_ball'])
    b = b.sort_values(['match_date_parsed', 'match_id', 'innings', 'innings_ball_num'], kind='stable').reset_index(drop=True)
    return b


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + '\n')
