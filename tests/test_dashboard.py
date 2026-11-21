import csv
from pathlib import Path

import pytest

from dashboard.build import load_players, verify_order

ROOT = Path(__file__).resolve().parents[1]


def test_exact_published_order():
    with (ROOT/'submission/round_one_rankings.csv').open() as f:
        players = [dict(role=r['role'], rank=int(r['rank']), player_id=int(r['player_id']),
                        player_name=r['player_name'], eligible=True) for r in csv.DictReader(f)]
    verify_order(players[::-1], ROOT/'submission')
    players[0]['player_id'] += 1
    with pytest.raises(ValueError):
        verify_order(players, ROOT/'submission')


def test_duplicate_identity_rejected(tmp_path):
    fixture = (ROOT/'dashboard/synthetic.csv').read_text()
    path = tmp_path/'duplicate.csv'
    path.write_text(fixture + fixture.splitlines()[1] + '\n')
    with pytest.raises(ValueError, match='duplicate'):
        load_players(path)


def test_local_dashboard_matches_both_submissions():
    path = ROOT/'outputs/round_one_scored.csv'
    if not path.exists():
        pytest.skip('Requires authorized local model output')
    verify_order(load_players(path), ROOT/'submission')
