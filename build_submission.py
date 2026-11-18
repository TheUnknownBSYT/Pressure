"""Rebuild the Round 1 submission from the six original CSVs, in dependency order."""
import argparse
import hashlib
import json
import shutil
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
FINAL = Path('outputs/final_submission')
STAGES = {
    'pressure': 'build_pressure_dataset',
    'legacy_events': 'final_strategy.legacy_events',
    'legacy_test': 'final_strategy.rating_legacy',
    'consequence': 'final_strategy.consequence',
    'response': 'final_strategy.response_model',
    'ratings': 'final_strategy.ratings',
    'transfer': 'final_strategy.transfer',
    'publication': 'final_strategy.publication',
    'documents': None,
    'verification': None,
}


def run(cmd, log, env, cwd):
    print('Running', *map(str, cmd), flush=True)
    with log.open('w') as out:
        with subprocess.Popen(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, bufsize=1) as proc:
            for line in proc.stdout:
                print(line, end='', flush=True)
                out.write(line)
            if proc.wait():
                raise subprocess.CalledProcessError(proc.returncode, cmd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=ROOT,
                        help='New directory for an isolated raw-data rebuild, or an existing build to resume')
    parser.add_argument('--from-stage',choices=STAGES, default='pressure')
    parser.add_argument('--through',choices=STAGES, default='verification')
    parser.add_argument('--document-python', default=os.environ.get('DOCUMENT_PYTHON',sys.executable))
    parser.add_argument('--soffice', default=os.environ.get('SOFFICE','soffice'))
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(exist_ok=True,parents=True)
    if work != ROOT:
        for source in ROOT.glob('*.csv'):
            target=work/source.name
            if not target.exists():
                target.symlink_to(source)
    logs = work/FINAL/'evidence/logs'
    logs.mkdir(exist_ok=True,parents=True)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1',
               MPLCONFIGDIR=str(logs.parent/'.mplcache'),
               OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    steps = list(STAGES)
    for stage in steps[steps.index(args.from_stage):steps.index(args.through)+1]:
        if stage=='pressure':
            # The Phase 3 archive protects a byte-level Phase 1 manifest including elapsed time.
            # Record timings for fresh runs separately while verifying every deterministic field.
            canonical=work/'outputs/run_manifest.json'
            archive=work/'outputs/phase3/phase3_manifest.json'
            current=logs.parent/'pressure_run_manifest.json'
            cmd=[sys.executable,'-m','build_pressure_dataset','--top-percent','25']
            if archive.exists():
                expected=json.loads(archive.read_text())['protected_file_hashes']['outputs/run_manifest.json']
                if hashlib.sha256(canonical.read_bytes()).hexdigest()!=expected:
                    raise ValueError('Historical Phase 1 manifest changed before rebuild. Use an empty --work-dir.')
                cmd.extend(['--manifest-path',str(current)])
            run(cmd,logs/'pressure.log',env,work)
            if archive.exists():
                prior=json.loads(canonical.read_text())
                fresh=json.loads(current.read_text())
                prior.pop('elapsed_seconds')
                fresh.pop('elapsed_seconds')
                if prior!=fresh:
                    raise ValueError('New run differs from archived configuration or results. Use an empty --work-dir.')
            else:
                shutil.copyfile(canonical,current)
        elif STAGES[stage]:
            run([sys.executable,'-m',STAGES[stage]], logs/f'{stage}.log', env, work)
        elif stage=='documents':
            run([args.document_python,'-m','final_strategy.render'],logs/'word.log',env,work)
            base=work/FINAL/'SupeRR_Selector_Final_Methodology.docx'
            with tempfile.TemporaryDirectory() as profile:
                run([args.soffice,'-env:UserInstallation='+Path(profile).as_uri(),'--headless',
                     '--convert-to','pdf','--outdir',str(base.parent),str(base)],logs/'pdf.log',env,work)
                if not base.with_suffix('.pdf').exists():
                    raise RuntimeError('No PDF was produced')
        else:
            run([sys.executable,'-m','pytest','-q',str(ROOT/'tests')],logs/'tests.log',env,work)
            run([sys.executable,'-m','final_strategy.verify_publication'],logs/'verification.log',env,work)
    print('Completed through',args.through, 'in',work)


if __name__=='__main__':
    main()
