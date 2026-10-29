"""Exact-rational EV sign-bracket audit of saved virtual-path crossing witnesses."""
import argparse,hashlib,json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from run_support import write_json


def polynomial(start,end,ids):
    a=[[F(float(z)) for z in row] for row in start[ids]]
    b=[[F(float(z)) for z in row] for row in end[ids]]
    sub=lambda x,y:[q-r for q,r in zip(x,y)]
    cross=lambda x,y:x[0]*y[1]-x[1]*y[0]
    e=sub(a[1],a[0]);w=sub(a[2],a[0]);de=sub(sub(b[1],b[0]),e);dw=sub(sub(b[2],b[0]),w)
    return [cross(e,w),cross(e,dw)+cross(de,w),cross(de,dw)],(e,w,de,dw)


def verify_witness(start,end,ids,bracket):
    coef,(e,w,de,dw)=polynomial(start,end,ids);lo,hi=map(lambda x:F(float(x)),bracket)
    evaluate=lambda s:coef[0]+s*(coef[1]+s*coef[2])
    signs=[evaluate(lo),evaluate(hi)]
    E=[sorted((x+lo*d,x+hi*d)) for x,d in zip(e,de)];W=[sorted((x+lo*d,x+hi*d)) for x,d in zip(w,dw)]
    def prod(x,y):return min(a*b for a in x for b in y)
    first=sum(prod(x,y) for x,y in zip(E,W))
    second=sum(prod(x,[x[0]-y[1],x[1]-y[0]]) for x,y in zip(E,W))
    passed=bool(signs[0]*signs[1]<0 and first>0 and second>0)
    return dict(passed=passed,vertices_global=ids,bracket=bracket,
        exact_oriented_area_twice_mm2=[str(x) for x in signs],
        exact_interior_dot_lower_bounds_mm2=[str(first),str(second)],
        exact_polynomial_coefficients=[str(x) for x in coef],
        meaning='Sign change and positive interior dot bounds prove an interior EV crossing of the exact affine chord between binary64 endpoints; no FE acceptance granted.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous audit')
    path=args.directory/'validation.json';raw=json.loads(path.read_text())
    fields=args.directory/'fields.npz'
    with np.load(fields,allow_pickle=False) as z:start=z['before'];end=z['newton_trial']
    checks=[verify_witness(start,end,w['vertices_global'],w['bracket']) for w in raw['full_newton']['crossing_witnesses']]
    report=dict(schema='softcut-exact-chord-crossing-audit/1',loading_started=False,
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (path,fields)},
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=checks,
        crossing_proved=bool(checks and all(q['passed'] for q in checks)),
        no_global_nonexistence_claim=True)
    write_json(args.output,report);print(json.dumps(dict(crossing_proved=report['crossing_proved'],checks=len(checks))))


if __name__=='__main__':main()
