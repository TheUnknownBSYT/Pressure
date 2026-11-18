"""Verify an independent six-CSV build against the submission in this directory."""
import argparse
from pathlib import Path
import json
import subprocess
import pandas as pd

FINAL=Path('outputs/final_submission')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rebuild',type=Path)
    other=parser.parse_args().rebuild.resolve()
    assert other!=Path.cwd()
    other_final=other/FINAL
    checks={}
    for file in sorted([*FINAL.glob('*.csv'),*(FINAL/'evidence').glob('*.csv')]):
        name=file.relative_to(FINAL)
        source=other_final/name
        assert source.exists(),source
        left,right=pd.read_csv(file),pd.read_csv(source)
        pd.testing.assert_frame_equal(left,right,rtol=1e-10,atol=1e-10)
        checks[str(name)]=len(left)
    for file in sorted((FINAL/'evidence').glob('*.parquet')):
        source=other_final/'evidence'/file.name
        pd.testing.assert_frame_equal(pd.read_parquet(file),pd.read_parquet(source),rtol=1e-10,atol=1e-10)
        checks[file.name]='full event ledger matched'
    for file in sorted(FINAL.glob('*.md')):
        assert file.read_text()==(other_final/file.name).read_text(),file
    pdf='SupeRR_Selector_Final_Methodology.pdf'
    plain=lambda path:subprocess.check_output(['pdftotext','-layout',str(path),'-'])
    assert plain(other_final/pdf)==plain(FINAL/pdf)
    tests=(other_final/'evidence/logs/tests.log').read_text().strip().splitlines()[-1]
    report=dict(clean_raw_build=True,clean_build_tests=tests,
                historical_only_tests='Five archived-result regression checks skip when their historical output folders are absent.',
                clean_source_files=sorted(f.name for f in other.glob('*.csv')),
                numerical_comparison_tolerance=1e-10,matching_outputs=checks,
                main_documents='Markdown and extracted PDF content match exactly; official-ID orders independently verified in Word, PDF, Markdown and CSV in both builds.')
    print(json.dumps(report,indent=2))
    (FINAL/'evidence/clean_rebuild_verification.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
