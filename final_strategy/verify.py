"""Check published rankings in every format and record historical integrity checks."""
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

out=Path('outputs/historical_vote_submission');ev=out/'evidence';base=out/'SupeRR_Selector_Final_Methodology'
doc=ElementTree.fromstring(ZipFile(base.with_suffix('.docx')).read('word/document.xml'))
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
tables=doc.findall('.//w:tbl',ns)
md=base.with_suffix('.md').read_text()
file_hash=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
records={}
for i,role in enumerate(['batters','bowlers']):
    with (out/f'final_{role}_top10.csv').open() as stream:players=list(csv.DictReader(stream))
    rows=tables[i].findall('w:tr',ns)[1:]
    assert len(rows)==len(players)==10
    pdf=subprocess.check_output(['pdftotext','-layout','-f',str(i+2),'-l',str(i+2),str(base.with_suffix('.pdf')),'-'],text=True)
    ids=[]
    for j,(r,line) in enumerate(zip(players,rows),1):
        cells=[''.join(c.itertext()) for c in line.findall('w:tc',ns)]
        key=[str(j),r['player_name'],r['player_id']]
        assert cells[:3]==key and f'| {j} | {key[1]} | {key[2]} |' in md
        assert re.search(r'^\s*'+r'\s+'.join(map(re.escape,key))+r'\s+',pdf,re.M)
        assert cells[4]==f"{100*float(r['score']):.1f}"
        ids.append(key[2])
    assert len(set(ids))==10;records[role]=ids
old=json.loads(Path('outputs/independent_audit/run_manifest.json').read_text())
mismatches={k:[f for f,h in old[k].items() if file_hash(f)!=h] for k in ['protected_file_hashes','artifact_hashes']}
report={'csv_docx_pdf_markdown_order_verified':records,'published_pages':7,'pytest':(ev/'tests.txt').read_text().strip(),'prior_manifest_checks':mismatches,
        'note':'The audit START_HERE file differs from its archived digest. Final computations read event data and numerical CSVs. No repair of historical files was attempted.',
        'scripts':{str(f):file_hash(f) for f in sorted(Path('final_strategy').glob('*.py'))},
        'documents':{f.name:file_hash(f) for f in out.iterdir() if f.is_file()}}
(ev/'final_checks.json').write_text(json.dumps(report,indent=2))
print(records,mismatches)
