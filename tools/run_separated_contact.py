"""Independent, bounded AL path ONLY in the certified no-self-contact domain.

Default q/.1 is a mechanism control. --lab-speed .2 selects the experimental
post-contact vertical speed while depth/increment arguments remain normal q.
The separated policy stops at nonadjacent near contact; it is not a fold model.
"""
import argparse
from pathlib import Path
import numpy as np
from tools.run_al_integration import factory
from tools import run_augmented_blade as base
from prototype.separated_self_contact import enable_separated_self_contact


def make(args,fast=True):
    p=factory(args.angle,coarse=args.coarse,spacing=args.spacing,order=args.order,
        contact_basis=args.contact_basis,surface_spacing=args.surface_spacing,fast=fast)
    enable_separated_self_contact(p)
    if args.lab_speed is not None:p.loading_rate=args.lab_speed*np.cos(np.deg2rad(args.angle))
    if fast:p.direction_policy='adaptive'
    return p


def run(args):
    made=[]
    def candidate(*a,**kw):
        p=make(args,True);made.append(p);return p
    def auditor(*a,**kw):return make(args,False)
    try:return base.run(args,problem_factory=candidate,audit_factory=auditor)
    finally:
        for p in made:p.close_compute()


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--depth',type=float,default=.1);a.add_argument('--increment',type=float,default=.005)
    a.add_argument('--angle',type=float,choices=(0.,33.,57.),default=57.)
    a.add_argument('--lab-speed',type=float);a.add_argument('--seconds',type=float,default=600.)
    a.add_argument('--audit-seconds',type=float,default=300.);a.add_argument('--order',type=int,choices=(5,8),default=5)
    a.add_argument('--spacing',type=float,default=.000625);a.add_argument('--surface-spacing',type=float,default=.0025)
    a.add_argument('--contact-basis',choices=('P0','P1'),default='P1');a.add_argument('--blas-threads',type=int,default=1)
    a.add_argument('--coarse',action='store_true');a.add_argument('--resume',type=Path)
    v=a.parse_args()
    if not (0<v.depth<=.3 and 0<v.increment<=.005 and 0<v.seconds<=1800 and 0<v.audit_seconds<=600):
        a.error('Bounded q<=.3 only; increment<=.005; solve<=1800s; audit<=600s')
    if v.lab_speed is not None and v.lab_speed!=.2:a.error('Only explicit experimental .2 mm/s currently qualified for study')
    v.self_contact_policy='separated-only-v1'
    run(v)
