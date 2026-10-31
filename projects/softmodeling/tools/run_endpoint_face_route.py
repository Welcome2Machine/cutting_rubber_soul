"""Own-initial-state face route with instantaneous endpoint event localization.

This explicit alternative preserves the mean-reaction route as a numerical
control. The shared loader remains unchanged while that control is running.
Only independently constructed FaceCutProblem objects are upgraded; no history
is transferred from an archived or concurrently running problem.
"""
import json
from pathlib import Path
import numpy as np
from face_cut import FaceCutProblem
from . import run_fracture_route as base
from .face_events import check_connected_endpoint
from .frozen_scene import digest

BASE_PROBLEM=base.problem
BASE_SOURCES=base.sources
BASE_FINISH=base.finish


class EndpointFaceCutProblem(FaceCutProblem):
    def acceptance(self,x,depth,iterations,outer_iterations):
        endpoint=({'endpoint_strength_check':'not applicable: pure precrack forbids activation'} if getattr(self,'pure_precrack',False) else check_connected_endpoint(self,x))
        record=super().acceptance(x,depth,iterations,outer_iterations)
        record.update(endpoint)
        return record


def problem(config):
    if config['interface']!='face':
        raise ValueError('Endpoint event route is explicitly a face model')
    p=BASE_PROBLEM(config)
    p.__class__=EndpointFaceCutProblem
    return p


def sources():
    hashes=BASE_SOURCES()
    root=Path(__file__).resolve().parents[1]
    for name in ('tools/face_events.py','tools/run_endpoint_face_route.py'):
        hashes[name]=digest(root/name)
    return hashes


def native(value):
    """Serialize NumPy scalars without changing numeric precision or NaN rules."""
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [native(v) for v in value]
    return value


def finish(out,result):
    # A physical cutback can return numpy.float64, and its depth comparison a
    # numpy.bool_. Convert at the report boundary, never in the force solver.
    result.update(native(result))
    result['event_rule']='instantaneous_relaxed_endpoint_guard_and_original_algorithmic_handoff'
    BASE_FINISH(out,result)
    rows=result['steps']
    if rows:
        summary=result['summary']
        # Reservoir and K are state values, not additive work increments.
        for key in ('mechanical_reservoir_n_mm','kinetic_energy_n_mm'):
            summary['ledger_n_mm'].pop(key,None)
        summary['terminal_kinetic_energy_n_mm']=rows[-1]['energy']['kinetic_energy_n_mm']
        summary['maximum_kinetic_energy_n_mm']=max(r['energy']['kinetic_energy_n_mm'] for r in rows)
        base.write_json(Path(out)/'result.json',result)
        (Path(out)/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Endpoint face route</title><h1>Endpoint event control</h1><p>Precrack, completion, entry and convergence are separate qualifications.</p><pre>'+json.dumps(summary,indent=2)+'</pre><a href="result.json">Evidence</a>')


def install():
    """Process-local explicit driver selection; no shared FE objects/workers."""
    base.problem=problem
    base.sources=sources
    base.finish=finish


def main():
    install()
    base.main()


if __name__=='__main__':main()
