"""Compare accepted IPC results to an intact 3 mm discretization reference.

This is a reported-state comparison, not an equilibrium reassembly, experimental
validation, or a mesh-convergence proof. No interpolation or extrapolation.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

DEPTH_TOL = 1e-10
FORCE_ATOL = .2
FORCE_RTOL = .05
TIP_ATOL = .02
PHYSICAL = ('case', 'config', 'dhat_mm', 'barrier_stiffness_mpa', 'ipc_version',
            'ipc_binary_sha256', 'force_tolerance_n')
MODEL = ('tip_radius_mm', 'nose_facets', 'fracture', 'material', 'friction_smoothing_mm')
CONFIG = ('length_mm', 'height_mm', 'width_mm', 'inclination_deg', 'drop_s_mm',
          'blade_thickness_mm', 'blade_tip_angle_deg', 'shear_modulus_n_mm2',
          'poisson_ratio', 'friction_coefficient')
FIELDS = ('depth_mm', 'reaction_fz_n', 'residual_n', 'min_j',
          'minimum_distance_mm', 'friction_work_n_mm')
POINT_FIELDS = ('depth_mm', 'reference_depth_mm', 'reference_step_index', 'candidate_step_index',
                'reference_fz_n', 'candidate_fz_n', 'fz_difference_n', 'fz_tolerance_n', 'force_pass',
                'reference_tip_status', 'candidate_tip_status', 'tip_status_match',
                'reference_tip_offset_mm', 'candidate_tip_offset_mm', 'tip_difference_mm', 'tip_pass',
                'reference_residual_n', 'candidate_residual_n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def physical_settings(report):
    settings = {key: report[key] for key in PHYSICAL}
    settings['config'] = dict(report['config'])
    # Saved legacy runs predate these explicit fields, with exactly these laws.
    settings['config'].setdefault('ogden_alpha', 2.)
    if number(settings['config']['ogden_alpha'], 'config.ogden_alpha') <= 0:
        raise ValueError('config.ogden_alpha must be positive')
    settings['contact_formulation'] = report.get('contact_formulation', 'improved-max')
    if settings['contact_formulation'] not in ('improved-max', 'offset-geometric'):
        raise ValueError('Unknown contact formulation')
    settings['simplifications'] = {key: report['simplifications'][key] for key in MODEL}
    if settings['simplifications']['fracture'] is not False:
        raise ValueError('An explicitly intact no-fracture model is required')
    for key in CONFIG:
        number(report['config'][key], 'config.'+key)
    tolerance = number(report['force_tolerance_n'], 'force_tolerance_n')
    if not 0 < tolerance <= 1e-4:
        raise ValueError('Force tolerance must be positive and no looser than 1e-4 N')
    for key in ('dhat_mm', 'barrier_stiffness_mpa'):
        if number(report[key], key) <= 0:
            raise ValueError(f'{key} must be positive')
    # Reject nonfinite values in any additional configuration field as well.
    json.dumps(settings, allow_nan=False)
    if not isinstance(report.get('source_sha256'), dict) or not report['source_sha256']:
        raise ValueError('Source fingerprints are required')
    return settings


def nominal_output_grid(report):
    """Declared output anchors; accepted cutback states never replace an anchor."""
    control = report.get('load_control') or {'mode': 'fixed'}
    mode = control.get('mode')
    if mode not in ('fixed', 'adaptive_bisection'):
        raise ValueError('Unknown load-control mode')
    increment = number(report.get('loading_increment_mm'), 'loading_increment_mm')
    target = number(report.get('target_mm'), 'target_mm')
    if increment <= 0 or target <= 0:
        raise ValueError('Loading increment and target must be positive')
    if mode == 'adaptive_bisection':
        minimum = number(control.get('minimum_increment_mm'), 'minimum_increment_mm')
        maximum = number(control.get('maximum_increment_mm'), 'maximum_increment_mm')
        output = number(control.get('output_grid_increment_mm'), 'output_grid_increment_mm')
        if (not 0 < minimum <= maximum or abs(maximum-increment) > DEPTH_TOL
                or abs(output-increment) > DEPTH_TOL):
            raise ValueError('Adaptive load-control bounds/output grid disagree with the nominal increment')
    return [min(i*increment, target) for i in range(math.ceil(target/increment-1e-12)+1)]


def adaptive_grid_audit(report, rows):
    """Audit scheduling metadata without inventing values at unsaved depths."""
    control = report.get('load_control') or {'mode': 'fixed'}
    mode = control.get('mode') if isinstance(control, dict) else None
    if mode == 'fixed':
        return dict(mode=mode, error=None, required_output_depths_mm=None,
                    missing_output_depths_mm=None, output_grid_complete=None)
    try:
        anchors = nominal_output_grid(report)
        increment = number(report['loading_increment_mm'], 'loading_increment_mm')
        if any(b['depth_mm']-a['depth_mm'] > increment+DEPTH_TOL for a, b in zip(rows, rows[1:])):
            raise ValueError('An accepted increment exceeds the declared maximum')
        missing = [depth for depth in anchors
                   if not any(abs(row['depth_mm']-depth) <= DEPTH_TOL for row in rows)]
        error = ('An original output depth was skipped in the accepted prefix'
                 if rows and any(depth < rows[-1]['depth_mm']-DEPTH_TOL for depth in missing) else None)
        return dict(mode=mode, error=error, required_output_depths_mm=anchors,
                    missing_output_depths_mm=missing, output_grid_complete=bool(rows) and not missing)
    except (AttributeError, KeyError, ValueError) as exc:
        return dict(mode=mode, error=str(exc), required_output_depths_mm=None,
                    missing_output_depths_mm=None, output_grid_complete=False)


def accepted_prefix(report):
    """Keep a valid prefix, explicitly recording any invalid reported suffix."""
    rows, issue = [], None
    steps = report.get('steps')
    if not isinstance(steps, list):
        return [], {'error': 'Missing steps list', 'accepted_states': 0, 'reported_states': 0,
                    'loading_grid': adaptive_grid_audit(report, [])}
    for index, step in enumerate(steps):
        try:
            row = {key: number(step.get(key), f'step {index} {key}') for key in FIELDS}
            depth = row['depth_mm']
            if not 0 <= depth <= 3+DEPTH_TOL:
                raise ValueError('Depth must lie in [0,3] mm')
            if ((not rows and abs(depth) > DEPTH_TOL)
                    or (rows and depth-rows[-1]['depth_mm'] <= DEPTH_TOL)):
                raise ValueError('Depths must start at zero and increase distinctly')
            if not 0 <= row['residual_n'] <= report['force_tolerance_n']:
                raise ValueError('Reported residual exceeds its acceptance tolerance')
            if row['min_j'] <= 0 or row['minimum_distance_mm'] <= 0:
                raise ValueError('Reported Jacobian and gap must be positive')
            if row['friction_work_n_mm'] < -1e-12:
                raise ValueError('Reported friction work is negative beyond roundoff')
            row['tip_contact'] = step.get('tip_contact')
            rows.append(row)
        except (ValueError, AttributeError) as exc:
            issue = f'step {index}: {exc}'
            break
    grid = adaptive_grid_audit(report, rows)
    return rows, dict(error=issue or grid['error'], accepted_states=len(rows), reported_states=len(steps),
                      max_residual_n=max((s['residual_n'] for s in rows), default=None),
                      min_j=min((s['min_j'] for s in rows), default=None),
                      minimum_distance_mm=min((s['minimum_distance_mm'] for s in rows), default=None),
                      minimum_friction_work_n_mm=min((s['friction_work_n_mm'] for s in rows), default=None),
                      loading_grid=grid,
                      scope='Reported metadata only; no checkpoint/equilibrium reassembly')


def completed_3mm(report, rows, audit):
    try:
        return bool(report.get('completed') is True and report.get('status') == 'completed'
                    and audit['error'] is None and rows
                    and audit.get('loading_grid', {}).get('output_grid_complete') is not False
                    and abs(number(report.get('target_mm'), 'target_mm')-3) <= DEPTH_TOL
                    and abs(number(report.get('last_depth_mm'), 'last_depth_mm')-3) <= DEPTH_TOL
                    and abs(rows[-1]['depth_mm']-3) <= DEPTH_TOL)
    except ValueError:
        return False


def tip_observation(step, dhat):
    tip = step.get('tip_contact')
    if not isinstance(tip, dict):
        return 'missing', None
    status = tip.get('status', 'missing')
    if status not in ('unique', 'ambiguous', 'no_tip_contact', 'missing', 'error'):
        return 'invalid_observation', None
    if status != 'unique':
        return status, None
    try:
        offset = number(tip.get('reference_offset_mm'), 'tip reference offset')
        gap = number(tip.get('gap_mm'), 'tip gap')
        depth = number(tip.get('depth_mm'), 'tip depth')
        if not 0 <= gap < dhat or abs(depth-step['depth_mm']) > DEPTH_TOL:
            raise ValueError('Unique observation has inconsistent gap/depth')
        return status, offset
    except ValueError:
        return 'invalid_observation', None


def compare(reference, candidate):
    physical = physical_settings(reference)
    if physical != physical_settings(candidate):
        raise ValueError('Physical settings differ; config, contact, blade and acceptance must be identical')
    for key in ('method', 'reference_origin_s_mm', 'reference_distance', 'positive_direction'):
        a = (reference.get('tip_contact_tracking') or {}).get(key)
        b = (candidate.get('tip_contact_tracking') or {}).get(key)
        # Absent tracking remains a missing observation, not fabricated zero.
        # When both runs declare a definition, contradictory definitions fail.
        if a is not None and b is not None and a != b:
            raise ValueError('Tip-contact definition differs: '+key)
    ref, ref_audit = accepted_prefix(reference)
    cur, cur_audit = accepted_prefix(candidate)
    if not completed_3mm(reference, ref, ref_audit):
        raise ValueError('Reference must be completed and accepted through 3 mm')
    complete = completed_3mm(candidate, cur, cur_audit)
    rows, unmatched, used = [], [], set()
    for index, step in enumerate(cur):
        matches = [i for i, value in enumerate(ref) if abs(value['depth_mm']-step['depth_mm']) <= DEPTH_TOL]
        if len(matches) != 1:
            unmatched.append(step['depth_mm'])
            continue
        ri = matches[0]; used.add(ri); baseline = ref[ri]
        rs, ro = tip_observation(baseline, reference['dhat_mm'])
        cs, co = tip_observation(step, candidate['dhat_mm'])
        limit = max(FORCE_ATOL, FORCE_RTOL*abs(baseline['reaction_fz_n']))
        error = step['reaction_fz_n']-baseline['reaction_fz_n']
        tip_error = co-ro if rs == cs == 'unique' else None
        rows.append(dict(depth_mm=step['depth_mm'], reference_depth_mm=baseline['depth_mm'],
            reference_step_index=ri, candidate_step_index=index,
            reference_fz_n=baseline['reaction_fz_n'], candidate_fz_n=step['reaction_fz_n'],
            fz_difference_n=error, fz_tolerance_n=limit, force_pass=abs(error) <= limit,
            reference_tip_status=rs, candidate_tip_status=cs, tip_status_match=rs == cs,
            reference_tip_offset_mm=ro, candidate_tip_offset_mm=co,
            tip_difference_mm=tip_error, tip_pass=abs(tip_error) <= TIP_ATOL if tip_error is not None else None,
            reference_residual_n=baseline['residual_n'], candidate_residual_n=step['residual_n']))
    unique = [row for row in rows if row['tip_difference_mm'] is not None]
    mismatch = sum(not row['tip_status_match'] for row in rows)
    invalid = sum(row['reference_tip_status'] in ('error', 'invalid_observation')
                  or row['candidate_tip_status'] in ('error', 'invalid_observation') for row in rows)
    force_ok = bool(rows) and all(row['force_pass'] for row in rows)
    tip_ok = bool(unique) and all(row['tip_pass'] for row in unique) and mismatch == 0 and invalid == 0
    full_coverage = complete and not unmatched and bool(rows) and abs(rows[-1]['depth_mm']-3) <= DEPTH_TOL
    adaptive = cur_audit['loading_grid']['mode'] == 'adaptive_bisection'
    required_outputs = nominal_output_grid(reference) if adaptive else None
    missing_outputs = ([depth for depth in required_outputs
                        if not any(abs(row['depth_mm']-depth) <= DEPTH_TOL for row in rows)]
                       if adaptive else None)
    qualification_coverage = complete and not missing_outputs if adaptive else full_coverage
    qualified = qualification_coverage and force_ok and tip_ok
    elapsed = {}
    for label, report in (('reference', reference), ('candidate', candidate)):
        value = report.get('elapsed_s')
        elapsed[label+'_elapsed_s'] = (float(value) if isinstance(value, (int, float))
                                      and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None)
    timing_ok = (complete and not reference.get('resume') and not candidate.get('resume')
                 and all(value is not None and value > 0 for value in elapsed.values()))
    elapsed['reference_over_candidate_elapsed_ratio'] = (
        elapsed['reference_elapsed_s']/elapsed['candidate_elapsed_s'] if timing_ok else None)
    elapsed['comparison_scope'] = ('Two completed from-zero runs; resource sharing is uncontrolled'
                                  if timing_ok else 'No elapsed ratio: incomplete, resumed, or missing timing')
    source_names = set(reference['source_sha256']) | set(candidate['source_sha256'])
    return dict(
        accuracy_qualified=bool(qualified), candidate_completed_3mm=complete,
        qualification_meaning='Agreement with a discrete intact reference; not experimental validation or a mesh-convergence proof',
        accuracy_scope=('Exact shared depths, including every reference output depth; unpaired adaptive substeps are not accuracy-assessed'
                        if adaptive else 'Exact shared depths covering every candidate accepted state'),
        material_equivalence='Not established by this tool. Source changes require the separately recorded constitutive/assembly equivalence tests.',
        thresholds=dict(force_absolute_n=FORCE_ATOL, force_relative=FORCE_RTOL,
                        force_rule='max(absolute, relative * abs(reference))', tip_offset_mm=TIP_ATOL),
        matching=dict(method='exact common accepted depth; no interpolation/extrapolation',
                      depth_roundoff_mm=DEPTH_TOL, matched_states=len(rows),
                      matched_depths_mm=[r['depth_mm'] for r in rows],
                      common_first_depth_mm=rows[0]['depth_mm'] if rows else None,
                      common_last_depth_mm=rows[-1]['depth_mm'] if rows else None,
                      all_candidate_depths_matched=bool(cur) and not unmatched,
                      full_candidate_coverage_to_3mm=bool(full_coverage),
                      qualification_grid_coverage_to_3mm=bool(qualification_coverage),
                      required_reference_output_depths_mm=required_outputs,
                      missing_reference_output_depths_mm=missing_outputs,
                      unassessed_adaptive_substep_depths_mm=unmatched if adaptive else [],
                      unmatched_candidate_depths_mm=unmatched,
                      unmatched_reference_depths_mm=[r['depth_mm'] for i, r in enumerate(ref) if i not in used]),
        force=dict(passed=force_ok, failures=sum(not r['force_pass'] for r in rows),
                   maximum_absolute_difference_n=max((abs(r['fz_difference_n']) for r in rows), default=None)),
        tip=dict(passed=tip_ok, both_unique_pairs=len(unique), status_mismatch_count=mismatch,
                 invalid_or_error_pairs=invalid, not_both_unique_pairs=len(rows)-len(unique),
                 unique_failures=sum(not r['tip_pass'] for r in unique),
                 maximum_absolute_difference_mm=max((abs(r['tip_difference_mm']) for r in unique), default=None)),
        reference_audit=ref_audit, candidate_audit=cur_audit,
        candidate_status=candidate.get('status'), candidate_error=candidate.get('error'), elapsed=elapsed,
        physical_settings=physical,
        numerical_settings={label: {key: report.get(key) for key in (
            'mesh_settings', 'discretization', 'loading_increment_mm', 'linear_solver',
            'predictor', 'predictor_settings', 'hessian_mode', 'friction_solver',
            'friction_solver_settings', 'threads', 'resume', 'load_control')}
            for label, report in (('reference', reference), ('candidate', candidate))},
        source_provenance=dict(reference=reference['source_sha256'], candidate=candidate['source_sha256'],
            changed_sources=sorted(name for name in source_names
                                   if reference['source_sha256'].get(name) != candidate['source_sha256'].get(name))),
        tip_definition_provenance={label: report.get('tip_contact_tracking')
                                   for label, report in (('reference', reference), ('candidate', candidate))},
        points=rows,
    )


def pending_comparison(reference):
    """Record absent candidate evidence without implying that it is running."""
    physical = physical_settings(reference)
    rows, audit = accepted_prefix(reference)
    if not completed_3mm(reference, rows, audit):
        raise ValueError('Reference must be completed and accepted through 3 mm')
    return dict(
        accuracy_qualified=False, candidate_completed_3mm=False, candidate_status='pending',
        candidate_error=None, pending_reason='Candidate result.json is not present; process state is unknown',
        qualification_meaning='No candidate evidence; no accuracy, completion or physical-equivalence claim',
        material_equivalence='Not checked; separate constitutive/assembly tests remain required',
        physical_comparison='not_performed_candidate_missing', reference_physical_settings=physical,
        reference_audit=audit, candidate_audit=None,
        matching=dict(method='exact common accepted depth; no interpolation/extrapolation',
                      depth_roundoff_mm=DEPTH_TOL, matched_states=0, matched_depths_mm=[],
                      common_first_depth_mm=None, common_last_depth_mm=None,
                      all_candidate_depths_matched=False, full_candidate_coverage_to_3mm=False,
                      unmatched_candidate_depths_mm=[],
                      unmatched_reference_depths_mm=[row['depth_mm'] for row in rows]),
        force=dict(passed=None, failures=None, maximum_absolute_difference_n=None),
        tip=dict(passed=None, both_unique_pairs=0, status_mismatch_count=None,
                 invalid_or_error_pairs=None, not_both_unique_pairs=0, unique_failures=None,
                 maximum_absolute_difference_mm=None),
        elapsed=dict(reference_elapsed_s=reference.get('elapsed_s'), candidate_elapsed_s=None,
                     reference_over_candidate_elapsed_ratio=None,
                     comparison_scope='No elapsed ratio: candidate result absent'),
        source_provenance=dict(reference=reference['source_sha256'], candidate=None, changed_sources=None),
        points=[],
    )


def write_comparison(reference, candidate, output, *, allow_pending=False):
    reference, candidate, output = Path(reference), Path(candidate), Path(output)
    a, b = reference/'result.json', candidate/'result.json'
    reference_report = json.loads(a.read_text(encoding='utf-8-sig'))
    pending = allow_pending and not b.exists()
    result = (pending_comparison(reference_report) if pending else
              compare(reference_report, json.loads(b.read_text(encoding='utf-8-sig'))))
    result['provenance'] = dict(reference_path=str(a.resolve()), candidate_path=str(b.resolve()),
                                reference_sha256=digest(a), candidate_sha256=None if pending else digest(b),
                                tool_sha256=digest(__file__))
    # Validate JSON before touching existing comparison artifacts.
    payload = json.dumps(result, indent=2, allow_nan=False)+'\n'
    output.mkdir(parents=True, exist_ok=True)
    temporary = output/'accuracy.csv.tmp'
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=POINT_FIELDS)
        writer.writeheader(); writer.writerows(result['points'])
    temporary.replace(output/'accuracy.csv')
    temporary = output/'accuracy.json.tmp'; temporary.write_text(payload, encoding='utf-8')
    temporary.replace(output/'accuracy.json')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--allow-pending', action='store_true',
                        help='Write an unqualified pending report only when candidate result.json is absent')
    args = parser.parse_args(argv)
    try:
        report = write_comparison(args.reference, args.candidate, args.output,
                                  allow_pending=args.allow_pending)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'Accuracy comparison rejected: {exc}\n')
    print(json.dumps({key: report[key] for key in ('accuracy_qualified', 'candidate_completed_3mm')}))


if __name__ == '__main__':
    main()
