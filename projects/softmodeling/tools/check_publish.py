"""Audit the staged index or every reachable commit before code-only publication.

This is a file/type and history check, not a scientific validation or a general
secret scanner. Review Markdown and source changes before publication as well.
"""
import argparse
from pathlib import PurePosixPath
import subprocess


CONFIGS={'prototype/intact_2mm_baseline.json','prototype/initiation_config.json','prototype/ipc_baseline.json',
         'reproductions/dammass2023/acceptance.json','configs/materials.json','configs/blades.json'}
ROOT_FILES={'.gitignore','.gitattributes','requirements.txt','run_support.py'}
CODE_ROOTS={'prototype','mechanics','reproductions','tests','tools','workbench'}
DOC_ROOTS=CODE_ROOTS|{'docs','references','configs'}
WEB_ASSETS={'workbench/static/index.html','workbench/static/app.js','workbench/static/app.css'}


def allowed(path):
    p=PurePosixPath(path)
    if not p.parts or p.is_absolute() or '..' in p.parts:return False
    if p.name=='video_observations.md':return False
    if path in CONFIGS or path in ROOT_FILES or path in WEB_ASSETS:return True
    if path in {'README.md','AGENTS.md'}:return True
    if p.parts[0] in DOC_ROOTS and p.suffix=='.md':return True
    return p.parts[0] in CODE_ROOTS and p.suffix in {'.py','.sh'}


def git(*args):
    return subprocess.check_output(['git',*args])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref',default='HEAD')
    parser.add_argument('--staged',action='store_true',help='Check the exact proposed index before committing')
    args=parser.parse_args()
    commits=[] if args.staged else git('rev-list',args.ref).decode('ascii').splitlines()
    checked=set();violations=[];total=0
    for commit in (['INDEX'] if args.staged else commits):
        entries=git('ls-files','--stage','-z') if args.staged else git('ls-tree','-rz','--full-tree',commit)
        for entry in entries.split(b'\0'):
            if not entry:continue
            metadata,rawpath=entry.split(b'\t',1)
            if args.staged:
                mode,oid,stage=metadata.decode('ascii').split()
                kind='blob' if stage=='0' else 'unmerged'
            else:
                mode,kind,oid=metadata.decode('ascii').split()
            path=rawpath.decode('utf-8')
            if kind!='blob' or mode not in {'100644','100755'} or not allowed(path):
                violations.append(f'{commit[:12]}: disallowed path/type {path}');continue
            if oid in checked:continue
            checked.add(oid);blob=git('cat-file','blob',oid);total+=len(blob)
            try:blob.decode('utf-8-sig')
            except UnicodeDecodeError:violations.append(f'{commit[:12]}: non-UTF-8 file {path}')
            if b'\0' in blob:violations.append(f'{commit[:12]}: binary content {path}')
    if violations:raise SystemExit('\n'.join(violations))
    scope='staged index' if args.staged else f'{len(commits)} reachable commits'
    print(f'PASS: {scope}; {len(checked)} unique UTF-8 code/document/config/web blobs; {total} bytes. No data/media/PDF paths.')


if __name__=='__main__':main()
