"""Audit every reachable commit before publishing code/documentation only.

This is a file/type and history check, not a scientific validation or a general
secret scanner. Review Markdown and source changes before publication as well.
"""
import argparse
from pathlib import PurePosixPath
import subprocess


CONFIGS={'prototype/initiation_config.json','prototype/ipc_baseline.json','reproductions/dammass2023/acceptance.json'}
ROOT_FILES={'.gitignore','.gitattributes','requirements.txt','run_support.py'}
CODE_ROOTS={'prototype','mechanics','reproductions','tests','tools'}
DOC_ROOTS=CODE_ROOTS|{'docs','references'}


def allowed(path):
    p=PurePosixPath(path)
    if p.name=='video_observations.md':return False
    if path in CONFIGS or path in ROOT_FILES:return True
    if path in {'README.md','AGENTS.md'}:return True
    if p.parts[0] in DOC_ROOTS and p.suffix=='.md':return True
    return p.parts[0] in CODE_ROOTS and p.suffix in {'.py','.sh'}


def git(*args):
    return subprocess.check_output(['git',*args])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref',default='HEAD')
    args=parser.parse_args()
    commits=git('rev-list',args.ref).decode('ascii').splitlines()
    checked=set();violations=[];total=0
    for commit in commits:
        for entry in git('ls-tree','-rz','--full-tree',commit).split(b'\0'):
            if not entry:continue
            metadata,rawpath=entry.split(b'\t',1)
            mode,kind,oid=metadata.decode('ascii').split();path=rawpath.decode('utf-8')
            if kind!='blob' or mode not in {'100644','100755'} or not allowed(path):
                violations.append(f'{commit[:12]}: disallowed path/type {path}');continue
            if oid in checked:continue
            checked.add(oid);blob=git('cat-file','blob',oid);total+=len(blob)
            try:blob.decode('utf-8-sig')
            except UnicodeDecodeError:violations.append(f'{commit[:12]}: non-UTF-8 file {path}')
            if b'\0' in blob:violations.append(f'{commit[:12]}: binary content {path}')
    if violations:raise SystemExit('\n'.join(violations))
    print(f'PASS: {len(commits)} reachable commits; {len(checked)} unique UTF-8 code/document blobs; {total} bytes. No data/media/PDF paths.')


if __name__=='__main__':main()
