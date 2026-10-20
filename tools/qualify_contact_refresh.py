"""One bounded, fresh C57 shallow path; no production queue or implicit resume."""
import argparse
import hashlib
from pathlib import Path
import shutil

from tools import run_angle_path_study as study
from tools import check_native_mesh as runner
from prototype.contact_refresh import ContactRefreshMixin


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--depth',type=float,default=.1)
    ap.add_argument('--increment',type=float,default=.0025)
    ap.add_argument('--seconds',type=float,default=600.)
    ap.add_argument('--model-only',action='store_true')
    args=ap.parse_args()
    if not (0<args.depth<=.1 and .000625<=args.increment<=.005 and 0<args.seconds<=600):
        raise ValueError('Only a bounded independent shallow qualification is authorized by this entry')
    args.phase='run';args.angle=57.;args.mode='oblique';args.radius=.5;args.order=3
    args.level=0;args.threads=2;args.resume=None
    def factory(angle,level,radius):
        p=study.make_problem(angle,'oblique',radius=radius,order=3,compact_assembly=True)
        p.__class__=type('QualifiedContactRefresh',(ContactRefreshMixin,type(p)),{})
        p.contact_inner_merit=not args.model_only
        return p
    def freeze(out):
        hashes=study.snapshot(out)
        name='tools/qualify_contact_refresh.py'
        shutil.copy2(name,out/'source'/name)
        hashes[name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
        return hashes
    result=runner.run(args,problem_factory=factory,geometry_checker=study.geometry_audit,
        source_snapshot=freeze,problem_signature=study.signature,
        diagnostic_metadata=dict(name='contact-refresh-shallow-v1',angle=57,mode='oblique',
            radius=.5,order=3,normal_speed=.1,independent_from_rest=True,
            solver='contact-refresh',inner_limit=8,inner_merit=not args.model_only))
    if result['steps']:study.export_case(args.output,result)


if __name__=='__main__':main()
