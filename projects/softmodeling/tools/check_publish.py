"""Audit the staged index or every reachable commit before code-only publication.

This is a file/type and history check, not a scientific validation or a general
secret scanner. Review Markdown and source changes before publication as well.
"""
import argparse
from pathlib import Path, PurePosixPath
import subprocess


CONFIGS={'prototype/intact_2mm_baseline.json','prototype/initiation_config.json','prototype/ipc_baseline.json',
         'reproductions/dammass2023/acceptance.json','configs/materials.json','configs/blades.json'}
ROOT_FILES={'.gitignore','.gitattributes','requirements.txt','run_support.py'}
# Explicit reviewed research assets; arbitrary JSON/data remains forbidden.
CONFIGS.update({'configs/zero_gap_finite_increment.json', 'configs/fracture_mixed.json', 'configs/fracture_network_regularized.json', 'configs/zero_gap_virtual_cover_budget60_time2.json', 'configs/zero_gap_virtual_chord_cover.json', 'configs/fracture_network.json', 'configs/zero_gap_virtual_geometry.json', 'configs/zero_gap_dynamic_energy.json', 'configs/fracture_labels.example.json', 'configs/zero_gap_virtual_cover_budget60.json', 'configs/independent_fracture_test_plan.json', 'configs/zero_gap_discrete_work.json', 'configs/zero_gap_dynamic_control.json', 'configs/zero_gap_virtual_lagged_newton.json', 'configs/fracture_network_finite_faces.json', 'configs/zero_gap_virtual_constrained.json', 'configs/fracture_demo.json', 'configs/blades.json', 'configs/zero_gap_virtual_precise_ccd.json', 'configs/zero_gap_virtual_continuation.json', 'configs/finite_increment_constrained_control.json', 'configs/materials.json'})
CONFIGS.update({'configs/face_precrack_control.json','configs/face_intact_control.json','configs/node_precrack_control.json'})
ROOT_FILES.update({'virtual_path_geometry.py', 'numerical_runtime.py', 'virtual_friction_work.py', 'release_dynamics.py', 'virtual_path_probe.py', 'stiff_contact.py', 'virtual_constrained_root.py', 'step_control.py', 'face_cohesive_quadrature.py', 'endpoint_friction.py'})
ROOT_FILES.add('face_cut.py')  # Reviewed numerical interface, never generated state.
ROOT=Path(__file__).resolve().parents[1]
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
    return subprocess.check_output(['git',*args],cwd=ROOT)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref',default='HEAD')
    parser.add_argument('--staged',action='store_true',help='Check the exact proposed index before committing')
    args=parser.parse_args()
    commits=[] if args.staged else git('rev-list',args.ref).decode('ascii').splitlines()
    prefix=git('rev-parse','--show-prefix').decode().strip().rstrip('/')
    checked=set();violations=[];total=0
    for commit in (['INDEX'] if args.staged else commits):
        # Inspect this subproject's historical tree in an integration monorepo.
        tree=commit+':'+prefix if prefix else commit
        entries=git('ls-files','--stage','-z') if args.staged else git('ls-tree','-rz','--full-tree',tree)
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
