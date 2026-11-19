"""Build a self-contained local dashboard from the saved player aggregates."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cols = 'player_id player_name role rating absolute ordinary uplift pressure_matches pressure_legal ordinary_matches ordinary_legal pressure_coverage positive_own_share IPL_legal T20I_legal absolute_lo95 absolute_hi95'.split()
with (ROOT/'outputs/final_submission/evidence/rating_all_candidates.csv').open() as f:
    players = [{k: (r[k] if k in ('player_name','role') else float(r[k])) for k in cols}
               for r in csv.DictReader(f) if r['rating'] and math.isfinite(float(r['rating']))]
for role, filename in [('bat','batters'),('bowl','bowlers')]:
    eligible = [p for p in players if p['role']==role and p['pressure_matches']>=8 and p['ordinary_matches']>=8 and p['pressure_legal']>=50 and p['ordinary_legal']>=100 and p['pressure_coverage']>=.75 and p['absolute']>0 and p['uplift']>0 and p['positive_own_share']>.5]
    eligible.sort(key=lambda p:(-p['rating'],-p['absolute'],p['player_id']))
    with (ROOT/f'submission/final_{filename}_top10.csv').open() as f:
        assert [p['player_id'] for p in eligible[:10]] == [int(r['player_id']) for r in csv.DictReader(f)]
template = Path(__file__).with_name('template.html').read_text()
(ROOT/'Pressure Dashboard.html').write_text(template.replace('__PLAYERS__', json.dumps(players,allow_nan=False).replace('<','\\u003c')))
