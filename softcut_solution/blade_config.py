"""Explicit microscope-profile to workbench configuration bridge; no FE loading."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def workbench_config_module():
    # This module only validates JSON and builds argument vectors, never solves.
    path = ROOT / 'projects/softmodeling/workbench/config.py'
    spec = importlib.util.spec_from_file_location('blade_workbench_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_config(catalog, profile_id, base, *, tip_radius_mm, radius_source,
                 confirm_view_plane=False, accept_symmetric_approximation=False):
    if catalog['schema'] != 'softcut-blade-shape-catalog/1':
        raise ValueError('Unsupported blade parameter catalog')
    profiles = [p for p in catalog['profiles'] if p['identity']['id'] == profile_id]
    if len(profiles) != 1:
        raise ValueError('Expected exactly one matching blade profile')
    profile = profiles[0]
    if not (profile['qualification']['side_view_confirmed'] or confirm_view_plane):
        raise ValueError('Confirm the section is normal to the cutting edge first')
    if not accept_symmetric_approximation:
        raise ValueError('The current FE geometry requires an explicit symmetric approximation')
    if not isinstance(radius_source, str) or not radius_source.strip():
        raise ValueError('Provide the independent source or assumption for tip radius')
    if tip_radius_mm is None:
        raise ValueError('Microscopy screenshots do not resolve tip radius')
    c = deepcopy(base)
    measured = profile['apparent_profile']
    c['name'] = profile['identity']['label'] + ' / conditional geometry'
    c['blade'] = dict(id=profile_id, label=profile['identity']['label'], shape='symmetric_wedge',
        thickness_mm=measured['thickness_mm'], included_angle_deg=measured['included_angle_deg'],
        source_note=(f"Microscopy {catalog['source_sha256']}; slide {profile['identity']['slide']}. "
                     'User-confirmed normal section; symmetric macro-wedge approximation. '
                     f'Tip radius independently supplied: {radius_source}. '
                     'Microscopic edge and physical prediction remain unqualified.'))
    c['geometry']['tip_radius_mm'] = tip_radius_mm
    return workbench_config_module().normalize_config(c)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--base-config', type=Path, required=True)
    parser.add_argument('--tip-radius-mm', type=float, required=True)
    parser.add_argument('--radius-source', required=True)
    parser.add_argument('--confirm-view-plane', action='store_true')
    parser.add_argument('--accept-symmetric-approximation', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output; previous simulation configurations are preserved')
    config = build_config(
        json.loads(args.catalog.read_text(encoding='utf-8-sig')), args.profile,
        json.loads(args.base_config.read_text(encoding='utf-8-sig')),
        tip_radius_mm=args.tip_radius_mm, radius_source=args.radius_source,
        confirm_view_plane=args.confirm_view_plane,
        accept_symmetric_approximation=args.accept_symmetric_approximation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Validated configuration saved: {args.output}; FE loading not started')


if __name__ == '__main__':
    main()
