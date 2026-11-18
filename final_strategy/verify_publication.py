"""Reconcile the final order against the rule, official IDs, Word, PDF and Markdown."""
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd

FINAL=Path('outputs/final_submission')
E=FINAL/'evidence'
BASE=FINAL/'SupeRR_Selector_Final_Methodology'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def digest(path):
    with path.open('rb') as f:
        h=sha256()
        for buf in iter(lambda:f.read(1048576),b''):
            h.update(buf)
        return h.hexdigest()


def main():
    manifest=json.loads(Path('outputs/run_manifest.json').read_text())
    for name,expected in manifest['source_sha256'].items():
        assert digest(Path(name+'.csv'))==expected,name
    registry=pd.read_csv('competition_team_player_mapping.csv')
    pairs=set(zip(registry.player_id,registry.player_name))
    candidates=pd.read_csv(E/'rating_all_candidates.csv')
    selected=pd.read_csv(FINAL/'component_evidence.csv')
    with ZipFile(BASE.with_suffix('.docx')) as z:
        root=ET.fromstring(z.read('word/document.xml'))
    tables=root.findall('.//w:tbl',NS)
    pages=subprocess.check_output(['pdftotext','-layout',str(BASE.with_suffix('.pdf')),'-'],text=True).split('\f')
    pages=[page for page in pages if page.strip()]
    content=json.loads((E/'report_content.json').read_text())
    assert len(pages)==len(content),(len(pages),len(content))
    md=BASE.with_suffix('.md').read_text()
    release=json.loads((E/'release_audit.json').read_text())
    assert release['raw_delivery_rows_reconciled']==manifest['rows']
    checks={}
    for i,(role,label) in enumerate([('bat','batters'),('bowl','bowlers')]):
        ranked=pd.read_csv(FINAL/f'final_{label}_top10.csv')
        assert ranked['rank'].tolist()==list(range(1,11)) and ranked.player_id.is_unique
        verified=candidates.loc[(candidates.role==role)&candidates.supported&
                                candidates.absolute.gt(0)&candidates.uplift.gt(0)&candidates.positive_own_share.gt(.5)].sort_values(
            ['rating','absolute','player_id'],ascending=[False,False,True]).head(10)
        assert verified.player_id.tolist()==ranked.player_id.tolist()
        subset=selected.loc[selected.role.eq(role)].sort_values('rank')
        assert subset.player_id.tolist()==ranked.player_id.tolist()
        assert np.allclose(subset.absolute-subset.ordinary,subset.uplift)
        assert subset.pressure_coverage.ge(.75).all()
        assert subset.pressure_matches.ge(8).all() and subset.ordinary_matches.ge(8).all()
        assert subset.pressure_legal.ge(50).all() and subset.ordinary_legal.ge(100).all()
        assert subset.absolute.gt(0).all() and subset.uplift.gt(0).all()
        assert np.array_equal(subset.IPL_legal+subset.T20I_legal,subset.pressure_legal)
        assert np.array_equal(subset.middle_legal+subset.death_legal+subset.powerplay_legal,subset.pressure_legal)
        tbl=tables[i].findall('w:tr',NS)[1:]
        assert len(tbl)==10
        for row, wr, metrics in zip(ranked.itertuples(), tbl, subset.itertuples(), strict=True):
            key=[str(row.rank),row.player_name,str(row.player_id),f'{metrics.pressure_matches:.0f} / {metrics.pressure_legal:.0f}',
                 f'{metrics.absolute:+.2f}',f'{metrics.uplift:+.2f}',f'{metrics.rating:+.2f}']
            assert (row.player_id,row.player_name) in pairs
            assert [''.join(c.itertext()) for c in wr.findall('w:tc',NS)]==key
            assert '| '+' | '.join(key)+' |' in md
            assert re.search(r'^\s*'+r'\s+'.join(map(re.escape,key))+r'\s+',pages[i+1],re.M)
        checks[role]=ranked.player_id.tolist()
    pressure=pd.read_parquet(E/'consequence_events.parquet')
    value=pd.read_parquet(E/'response_events.parquet')
    assert pressure.delivery_id.is_unique and value.delivery_id.is_unique
    assert not value[['bat_value','bowl_value','wicket_price']].isna().any().any()
    assert (value.wicket_price>=0).all()
    scenarios=pd.read_csv(E/'consequence_scenarios.csv').pivot(index=['law','model'],columns='runs',values='pressure')
    assert len(scenarios)==12 and (scenarios[22]>scenarios[36]).all()
    assert pd.read_csv(FINAL/'original_ideas_audit.csv').notna().all().all()
    root=Path(__file__).resolve().parents[1]
    scripts=[*sorted(root.glob('build_*.py')),root/'requirements.txt',root/'requirements-documents.txt',
             *sorted((root/'final_strategy').glob('*.md')),root/'pressure_pipeline/methodology.md']
    for folder in ['pressure_pipeline','phase2_pipeline','phase3_pipeline','independent_audit','final_strategy','tests']:
        scripts.extend(sorted((root/folder).glob('*.py')))
    files=[*sorted(FINAL.glob('*.csv')),*sorted(FINAL.glob('*.md')),BASE.with_suffix('.pdf'),BASE.with_suffix('.docx')]
    report=dict(ordered_official_ids=checks,document_pages=len(pages),source_hashes=manifest['source_sha256'], independent_audit=release,
        pressure_events=len(pressure),performance_events=len(value),
        pytest=(E/'logs/tests.log').read_text().strip().splitlines()[-1],
        submission_files={p.name:digest(p) for p in files},
        implementation_hashes={str(p.relative_to(root)):digest(p) for p in scripts if p.is_file()})
    (E/'final_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Verified both official-ID orders against all four formats, eligibility, exposure, source hashes and numerical checks.')


if __name__=='__main__':main()
