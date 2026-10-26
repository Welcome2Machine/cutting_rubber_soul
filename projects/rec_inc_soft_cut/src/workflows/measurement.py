"""Blade-aware measurements and review from complete saved material trajectories.

This runner does not rerun optical tracking or overwrite its source arrays.
Unknown surface identities and occluded forecasts remain explicit in outputs.
"""
from __future__ import annotations

from src.io import write_json as save_json, load_npz, sha as digest

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from src.io import ROOT
NAMES = ('0930l30a1', '0930l60a1', '0930n60a1')
SOURCE = ROOT / 'results/current_material_loading'
OUTPUT = ROOT / 'results/current_contact_measurements'
HIDDEN = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def numeric_or_none(value):
    import numpy as np
    return float(value) if np.isfinite(value) else None


def pack_address(address):
    """NPZ stays pickle-free, including the variable-size supporting ID sets."""
    import numpy as np
    packed = {key: value for key, value in address.items() if key not in ('metadata', 'support_ids')}
    ids = address['support_ids']
    packed['support_id_offsets'] = np.r_[0, np.cumsum([len(row) for row in ids])]
    packed['support_ids_flat'] = np.concatenate(ids).astype(np.int64) if any(map(len, ids)) else np.empty(0, np.int64)
    return packed


def restrict_contact_phase(address, mapping, contact_frame):
    """Withhold precontact identities and the edge entering first contact."""
    import numpy as np
    precontact = np.asarray(mapping) < contact_frame
    address['state'][precontact] = 0
    address['label'][precontact] = 'unknown'
    address['reason'][precontact] = 'before_first_contact'
    for key in ('s_px', 's_um', 'reference_s_px', 'low_px', 'high_px', 'low_um', 'high_um',
                'surface_turn_deg', 'migration_from_initial_contact_um'):
        address[key][precontact] = np.nan
    address['reference_marker_xy'][precontact] = np.nan
    # A backward step at contact still uses a precontact endpoint: remove it
    # without losing the first actual contact's address or migration origin.
    edge_before_contact = np.asarray(mapping) <= contact_frame
    for key in ('address_rate_um_s', 'relative_tangent_step_um', 'relative_tangent_velocity_um_s'):
        address[key][edge_before_contact] = np.nan
    address['motion_step_state'][edge_before_contact] = 0
    return address


def surface_depths(seed, estimator):
    """Signed initial normal distance to the observed initial slope, not blade distance."""
    import numpy as np
    along = estimator.boundary @ estimator.tangent
    order = np.argsort(along)
    station = np.interp(seed @ estimator.tangent, along[order], estimator.addresses[order],
                        left=np.nan, right=np.nan)
    feet = np.array([estimator.surface_point(s) for s in station])
    return np.einsum('ni,i->n', seed-feet, estimator.normal)


class BladeFrames:
    def __init__(self, reader, mapping):
        self.reader, self.mapping = reader, mapping

    def __getitem__(self, index):
        return self.reader(int(self.mapping[index])) == 2


def read_case(name, args):
    import numpy as np
    source = args.source/name
    plan = json.loads((source/'plan.json').read_text(encoding='utf-8'))
    geom_meta = json.loads((source/'geometry.json').read_text(encoding='utf-8'))
    with np.load(source/'tracking/tracks.npz', allow_pickle=False) as stored:
        tracks = {key: stored[key].copy() for key in
                  ('positions', 'supported', 'point_ids', 'source_frame_indices', 'motion_step_valid')}
    mapping = tracks['source_frame_indices']
    if not np.array_equal(mapping, np.arange(len(mapping))) or len(mapping) > plan['full_video_frames']:
        raise ValueError('Measurements require a contiguous saved-frame prefix starting at frame zero')
    if not 0 <= plan['contact_frame'] < plan['reversal_frame'] < len(mapping):
        raise ValueError('Saved trajectories must include first contact through reversal')
    # Older complete-video plans did not carry the explicit scope flag.
    full_video = bool(plan.get('full_video', len(mapping) == plan['full_video_frames']))
    if full_video and len(mapping) != plan['full_video_frames']:
        raise ValueError('Full-video measurements require all source frames')
    plan = dict(plan, full_video=full_video)
    if not full_video:
        # Preparation may retain a short post-reversal image context. Do not
        # accidentally turn that into a withdrawal analysis in a loading run.
        count = plan['reversal_frame'] + 1
        tracks = {key: value if key == 'point_ids' else value[:count]
                  for key, value in tracks.items()}
        mapping = tracks['source_frame_indices']
    query = load_npz(source/'queries.npz')
    if not np.array_equal(tracks['point_ids'], query['point_ids']):
        raise ValueError('Query and track identities disagree')
    geometry = load_npz(source/'geometry.npz')
    geometry = {key: value[mapping] for key, value in geometry.items()
                if key in ('tips_px', 'depth_mm', 'supported')}
    timing = load_npz(source/'frame_mapping.npz')
    timing_count = len(timing['frame_indices'])
    if not np.array_equal(timing['frame_indices'][:len(mapping)], mapping):
        raise ValueError('Physical-time mapping does not align with saved trajectories')
    timing = {key: value[:len(mapping)] if value.ndim and len(value) == timing_count else value
              for key, value in timing.items()}
    scale = 1000*geom_meta['contact_to_reversal_mm']/geom_meta['contact_to_reversal_px']
    return source, plan, geom_meta, tracks, query, geometry, timing, scale


def measure(name, args):
    import cv2
    import numpy as np
    from src.workflows.tracking import label_reader
    from src.analysis.blade_contact_geometry import blade_contact_geometry
    from src.analysis.contact_motion_evidence import analyze_contact_motion, ContactMotionConfig
    from src.analysis.tip_surface_address import TipSurfaceAddressEstimator, analyze_tip_surface_series

    cv2.setNumThreads(2)
    started = time.perf_counter()
    folder = args.out/name
    folder.mkdir(parents=True, exist_ok=True)
    source, plan, geom_meta, tracks, query, geometry, timing, scale = read_case(name, args)
    mapping, p, original_support = (tracks[key] for key in ('source_frame_indices', 'positions', 'supported'))
    tips, tip_good, depth = geometry['tips_px'], geometry['supported'], geometry['depth_mm']
    reader = label_reader(source/'segmentation'/name/'semantic_masks')
    signature_files = [source/'tracking/tracks.npz', source/'queries.npz', source/'geometry.npz',
        source/'frame_mapping.npz', source/'original_surface_reference.npz', Path(__file__),
        ROOT/'src/analysis/blade_contact_geometry.py', ROOT/'src/analysis/contact_motion_evidence.py',
        ROOT/'src/analysis/tip_surface_address.py']
    signatures = {str(path.relative_to(ROOT)): digest(path) for path in signature_files}
    mask_digest = hashlib.sha256()
    frames = []
    for row, source_frame in enumerate(mapping):
        labels = reader(source_frame)
        mask_digest.update(labels.tobytes())
        frames.append(blade_contact_geometry(labels, p[row], tips[row], tip_supported=tip_good[row]))
        if row % 400 == 0:
            print(f'{name}: blade geometry {row}/{len(mapping)}', flush=True)
    contact_geom = {key: np.asarray([row[key] for row in frames]) for key in frames[0]}
    del frames
    signatures['decoded_semantic_masks_sha256'] = mask_digest.hexdigest()
    np.savez_compressed(folder/'blade_contact_geometry.npz', **contact_geom, frame_indices=mapping,
                        point_ids=tracks['point_ids'])
    # All original positions/IDs are retained. This mask governs measurements,
    # not their existence. Semantic blade-edge observations are conservative
    # exclusions; the eroded opaque-core classification is reported separately.
    support = (original_support & contact_geom['in_image'] & contact_geom['material_pixel']
               & ~contact_geom['on_blade'])
    edges = tracks['motion_step_valid'].copy()
    edges[0] = False
    edges[1:] &= support[1:] & support[:-1]
    contact = np.full(support.shape, -1, np.int8)
    visible_gap = support & np.isfinite(contact_geom['signed_blade_gap_px']) & tip_good[:, None]
    contact[visible_gap] = 0
    contact[visible_gap & contact_geom['contact_proximity_candidate']] = 1
    reference = load_npz(source/'original_surface_reference.npz')
    estimator = TipSurfaceAddressEstimator(query['seed_points'], query['point_ids'],
        query['depth_layer'], query['surface_station'], reference, scale,
        initial_material_mask=reader(0) == 1)
    reference_depth = surface_depths(query['seed_points'], estimator)
    cfg = ContactMotionConfig(contact_evidence_kind='geometric_proximity',
        contact_evidence_provenance='Supported material centers at 0..4 native pixels outside semantic blade and within 90 pixels of registered tip; not independently observed mechanical contact')
    boundary_distance = contact_geom['signed_blade_gap_px'].copy()
    boundary_distance[boundary_distance < 0] = np.nan
    print(f'{name}: relative motion and fixed-ID recovery blocks', flush=True)
    motion_summary, motion = analyze_contact_motion(p, support,
        times_s=timing['physical_time_s'], frame_indices=mapping, tip_xy=tips,
        tip_supported=tip_good, depth_mm=depth, contact_frame=plan['contact_frame'],
        reversal_frame=plan['reversal_frame'], reference_surface_center=geom_meta['surface_center'],
        reference_surface_tangent=estimator.tangent, reference_inward_normal=estimator.normal,
        point_ids=tracks['point_ids'], reference_xy=query['seed_points'], reference_depth_px=reference_depth,
        um_per_px=scale, motion_step_valid=edges, observed_contact=contact,
        contact_tangent_xy=contact_geom['down_or_right_tangent_xy'],
        blade_boundary_distance_px=boundary_distance, config=cfg)
    np.savez_compressed(folder/'motion_measurements.npz', **motion)
    save_json(folder/'motion_summary.json', motion_summary)
    print(f'{name}: inverse material address at registered tip', flush=True)
    address = analyze_tip_surface_series(estimator, p, support, tips, timing['physical_time_s'],
        geometry_supported=tip_good, blocked=contact_geom['on_blade'],
        branch_labels=contact_geom['routing_branch'], contact_frame=plan['contact_frame'],
        blade_masks=BladeFrames(reader, mapping))
    restrict_contact_phase(address, mapping, plan['contact_frame'])
    address['metadata']['contact_phase_note'] = 'Precontact addresses withheld. Withdrawal values require current local surface-to-tip support and remain contact candidates.'
    np.savez_compressed(folder/'surface_address.npz', **pack_address(address), frame_indices=mapping,
        physical_time_s=timing['physical_time_s'], depth_mm=depth, tip_xy=tips)
    save_json(folder/'surface_address_metadata.json', address['metadata'])
    # Old node_xy has 31 stations; do not attach it to the extended station
    # axis. The original file stays intact and remains in source provenance.
    extended = {key: reference[key] for key in ('tangent', 'inward_normal', 'contact_s_px')}
    extended.update(boundary_xy=estimator.boundary, surface_s_px=estimator.addresses)
    np.savez_compressed(folder/'initial_surface_extended.npz', **extended)
    np.savez_compressed(folder/'measurement_support.npz', supported=support, motion_step_valid=edges,
                        frame_indices=mapping, point_ids=tracks['point_ids'])
    loading = (mapping >= plan['contact_frame']) & (mapping <= plan['reversal_frame'])
    address_summary = address_coverage(address, depth, loading)
    overlap = contact_geom['on_blade']
    core = contact_geom['in_opaque_core']
    audit = dict(total_point_observations=int(original_support.size), ids_retained=len(tracks['point_ids']),
        positions_changed=0, semantic_blade_overlap=int(overlap.sum()),
        semantic_blade_supported_overlap=int((overlap & original_support).sum()),
        opaque_core_overlap=int(core.sum()), opaque_core_supported_overlap=int((core & original_support).sum()),
        opaque_core_forecasts=int((core & ~original_support).sum()),
        supported_observations_removed_from_measurement=int((original_support & ~support).sum()),
        accepted_measurement_blade_overlap=int((support & overlap).sum()),
        note='Saved forecasts keep their latent coordinates and IDs; no snapping to blade boundary. Display clips all material ink behind semantic blade. Future tracking rejects illegal candidates before winner selection.')
    summary = dict(case=name, frames=len(mapping), points=len(tracks['point_ids']),
        full_video=plan['full_video'], recomputed_optical_tracking=False, all_original_ids_and_positions_preserved=True,
        first_source_frame=int(mapping[0]), final_source_frame=int(mapping[-1]),
        contact_frame=plan['contact_frame'], reversal_frame=plan['reversal_frame'],
        um_per_px=scale, calibration=geom_meta['calibration'],
        scale_source='Approximately 3 mm observed contact-to-reversal blade travel; not tapered-tip width',
        blade_angle_correction='not measured; translation-only',
        blade_audit=audit, surface_address=address_summary,
        motion_group_coverage=motion_summary['groups'],
        recovery_candidates=len(motion_summary['candidate_events']),
        multi_patch_recovery_intervals=motion_summary['multi_patch_relaxation_intervals'],
        first_geometric_relaxation_candidate=motion_summary['first_geometric_relaxation_candidate'],
        physical_fracture_onset=None, physical_truth_validated=False,
        methods=dict(relative_motion=asdict(cfg), tip_address=asdict(estimator.config)),
        source_signatures=signatures, seconds=time.perf_counter()-started)
    save_json(folder/'measurement_summary.json', summary)
    write_measurement_csv(folder/'measurements.csv', mapping, timing['physical_time_s'], depth,
                          plan, motion, address, scale)
    print(f'{name}: measurements complete in {summary["seconds"]:.1f}s; {address_summary}', flush=True)
    return summary


def address_coverage(address, depth, loading):
    import numpy as np
    rows = np.flatnonzero(loading)
    states = address['state']
    bins = []
    for low in np.arange(0, 3, .5):
        selected = loading & (depth >= low) & (depth < low+.5 if low < 2.5 else depth <= 3.01)
        bins.append(dict(depth_mm=[float(low), float(low+.5)], frames=int(selected.sum()),
            direct=int(np.sum(selected & (states == 2))), conditional=int(np.sum(selected & (states == 1))),
            raw_material_inverse=int(np.sum(selected & np.isfinite(address['raw_reference_s_um'])))))
    chosen = rows[states[rows] > 0]
    reasons, counts = np.unique(address['reason'][rows[states[rows] == 0]], return_counts=True)
    return dict(loading_frames=len(rows), direct_frames=int(np.sum(states[rows] == 2)),
        conditional_frames=int(np.sum(states[rows] == 1)), unknown_frames=int(np.sum(states[rows] == 0)),
        available_address_depth_range_mm=[numeric_or_none(np.nanmin(depth[chosen])), numeric_or_none(np.nanmax(depth[chosen]))] if len(chosen) else None,
        depth_bins=bins, unknown_reasons={str(k): int(v) for k, v in zip(reasons, counts)},
        interpretation='Original-surface address migration is distinct from material velocity. Raw inverse interior/high-residual coordinates are diagnostics, never a forced original-surface slip measurement.')


def write_measurement_csv(path, frames, times, depth, plan, motion, address, scale):
    keys = ('s_um', 'low_um', 'high_um', 'surface_turn_deg', 'address_rate_um_s',
            'relative_tangent_velocity_um_s', 'surface_gap_px', 'surface_normal_offset_px',
            'fit_residual_px', 'heldout_residual_px', 'raw_reference_s_um', 'raw_surface_normal_offset_px')
    groups = list(motion['group_names'])
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_frame_for_traceability', 'physical_time_s', 'phase', 'entry_mm',
            'surface_address_state', 'surface_address_reason', 'surface_address_route', *keys,
            'disjoint_relaxing_patch_count', *[f'{g}_{metric}' for g in groups for metric in
                ('relative_x_um_s', 'relative_y_um_s', 'n_velocity', 'candidate_contact_tangent_um_s', 'n_contact')]])
        for i, frame in enumerate(frames):
            phase = 'precontact' if frame < plan['contact_frame'] else 'loading' if frame <= plan['reversal_frame'] else 'withdrawal'
            values = [int(frame), times[i], phase, numeric_or_none(depth[i]), int(address['state'][i]),
                      address['reason'][i], address['route'][i], *[numeric_or_none(address[k][i]) for k in keys],
                      int(motion['disjoint_relaxing_patch_count'][i])]
            for g in range(4):
                values.extend([numeric_or_none(motion['group_x_median_um_s'][i, g]),
                    numeric_or_none(motion['group_y_median_um_s'][i, g]), int(motion['group_y_count'][i, g]),
                    numeric_or_none(motion['group_contact_candidate_local_tangent_median'][i, g]*scale),
                    int(motion['group_contact_candidate_local_tangent_count'][i, g])])
            writer.writerow(values)


def render(name, args):
    import cv2
    from src.workflows.tracking import encode_review_video, label_reader
    from src.analysis.contact_motion_review import render_contact_motion_review
    from src.analysis.material_tracking_review import render_material_review, MaterialReviewConfig
    cv2.setNumThreads(2)
    folder = args.out/name
    source, plan, _, tracks, _, geometry, _, scale = read_case(name, args)
    summary = json.loads((folder/'measurement_summary.json').read_text(encoding='utf-8'))
    support = load_npz(folder/'measurement_support.npz')['supported']
    motion, address = load_npz(folder/'motion_measurements.npz'), load_npz(folder/'surface_address.npz')
    labels = label_reader(source/'segmentation'/name/'semantic_masks')
    blade = lambda frame: labels(frame) == 2
    common = dict(frame_indices=tracks['source_frame_indices'], depth_mm=geometry['depth_mm'],
        tip_xy=geometry['tips_px'], point_ids=tracks['point_ids'], contact_frame=plan['contact_frame'],
        reversal_frame=plan['reversal_frame'], blade_mask_reader=blade, fps=12)
    video = folder/f'{name}_contact_motion.mp4'
    print(f'{name}: rendering registered-tip / original-address / relative-motion review', flush=True)
    result = render_contact_motion_review(source/'normalized_lossless.mkv', tracks['positions'], support,
        video, motion=motion, address=address, surface_reference=load_npz(folder/'initial_surface_extended.npz'),
        um_per_px=scale, overview_path=folder/f'{name}_contact_overview.png', **common)
    result.update(encode_review_video(video))
    save_json(video.with_suffix('.render.json'), result)
    rendered = {}
    if args.analysis_only and (folder/'complete.json').is_file():
        rendered.update(json.loads((folder/'complete.json').read_text(encoding='utf-8')).get('rendering', {}))
    rendered['contact_motion'] = result
    if not args.analysis_only:
        for kind, config in [('material', MaterialReviewConfig(view_mode='full', point_radius=1)),
                             ('tip', MaterialReviewConfig())]:
            output = folder/f'{name}_{kind}_tracking_occlusion.mp4'
            print(f'{name}: rendering {kind} material IDs with blade occlusion', flush=True)
            record = render_material_review(source/'normalized_lossless.mkv', tracks['positions'], output,
                supported=tracks['supported'], config=config,
                overview_path=folder/f'{name}_{kind}_occlusion_overview.png', **common)
            record.update(encode_review_video(output))
            rendered[kind] = record
    summary['rendering'] = rendered
    summary['video'] = str(video)
    summary['completed'] = time.time()
    save_json(folder/'complete.json', summary)
    print(f'{name}: videos complete', flush=True)
    return summary


def write_index(args):
    cards = []
    for name in args.names:
        path = args.out/name/'complete.json'
        if not path.is_file():
            continue
        summary = json.loads(path.read_text(encoding='utf-8'))
        a = summary['surface_address']
        links = [f'<a href="{name}/{name}_contact_motion.mp4">接触运动分析视频</a>',
            f'<a href="{name}/measurements.csv">逐时刻测量 CSV</a>',
            f'<a href="{name}/measurement_summary.json">测量摘要</a>',
            f'<a href="{name}/motion_summary.json">固定材料块与恢复候选</a>',
            f'<a href="{name}/surface_address.npz">原始表面地址 NPZ</a>']
        for kind, title in [('material', '完整材料点阵'), ('tip', '刀尖点阵放大')]:
            file = f'{name}_{kind}_tracking_occlusion.mp4'
            if (args.out/name/file).is_file():
                links.append(f'<a href="{name}/{file}">{title}</a>')
        cards.append(f'<section><h2>{html.escape(name)}</h2>'
            f'<video controls preload="metadata" src="{name}/{name}_contact_motion.mp4"></video>'
            f'<p>{summary["frames"]} 帧，{summary["points"]} 个固定材料 ID；未重算光流。'
            f'加载段原始表面地址：直接支持 {a["direct_frames"]} 帧，条件推断 {a["conditional_frames"]} 帧，'
            f'未知 {a["unknown_frames"]} 帧。</p><p>{" · ".join(links)}</p></section>')
    inspection = ''
    review_file = args.out/'kinematic_validation/review_summary.json'
    if review_file.is_file():
        review = json.loads(review_file.read_text(encoding='utf-8'))
        notes = {
            'kinematic_candidate_with_material_identity_resolution_risk': '材料块存在压扁及像素分辨率风险，不能用作可靠起裂深度。',
            'kinematic_candidate_only_no_new_opening_confirmed': '保留多块几何恢复候选；所抽检图像未确认新裂口。',
            'single_patch_control_no_multi_patch_or_visible_opening_confirmation': '没有多块恢复区间；单块候选对照也未确认新裂口。'}
        items = []
        for case in review.get('cases', []):
            bounds = case.get('event_interval', {}).get('depth_mm_range', [])
            depth_text = f'{bounds[0]:.3f}–{bounds[-1]:.3f} mm' if len(bounds) == 2 else '未提供深度'
            status = case.get('visual_assessment', {}).get('status', '')
            items.append(f'<li>{html.escape(case["case"])}，{depth_text}：{notes.get(status, html.escape(status))}</li>')
        inspection = ('<p>已有局部图像复核（只代表已抽检片段）：</p><ul>' + ''.join(items) +
            '</ul><p><a href="kinematic_validation/review_summary.json">复核记录</a>。'
            '这些结论不排除更早或不可见的起裂；详细检查图位于 kinematic_validation。</p>')
        if (args.out/'kinematic_validation/review.html').is_file():
            inspection += '<p><a href="kinematic_validation/review.html">查看中文复核说明与原图</a></p>'
    document = ('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>刀具—材料接触运动与原始表面地址</title><style>'
        'body{max-width:1500px;margin:30px auto;padding:0 22px;background:#111719;color:#e7eeee;font:17px/1.65 system-ui}'
        'video{width:100%;background:#000}a{color:#79d9e9}section{margin:36px 0}li{margin:8px 0}</style>'
        '<h1>刀具—材料接触运动与原始表面地址</h1>'
        '<p>三组完整视频，包括退刀；横坐标为接触开始为零的刀具竖直压入深度，约 3 mm 接触至回退行程定标。'
        '物理速度使用采集时间，12 fps 仅用于播放。</p><ol>'
        '<li><b>防吸附：</b>刀具内部排除匹配候选；遮挡保留材料 ID 和潜在预测位置，但不在刀具上画材料点，也不把预测当作测量。未把点推到刀边。</li>'
        '<li><b>相对运动：</b>逐个固定 ID 的速度减去刀具平移速度，再分刀具左右、初始表层与次表层汇总。'
        '原始表层为初始表面内 0–12 px，次表层为 12–48 px；仅统计刀尖 60 px 附近。'
        '刀面 4 px 内的图像支持点另有局部刀面切向趋势，几何接近不等于机械接触已证实。</li>'
        '<li><b>切开线索：</b>固定材料块在刀具继续下移时，成对点距向初始形态恢复，作为局部恢复候选；至少两个不共享 ID 的块同时恢复另列区间。'
        '左右差异、shape change 或回弹本身均不是已确认起裂，也不能由位移计算弹性势能。'
        '软材料可能平滑地从压入转为切开，因此没有突变也不能排除切开。'
        '<a href="https://www.nature.com/articles/s41524-025-01869-y">相关实验论文</a>。</li>'
        '<li><b>刀尖接触地址：</b>当前图的环始终位于登记刀尖；原始图上的环表示通过表面法向点列曲线或固定块逆映射估计的初始表面地址。'
        '绿色为当前表层图像支持，粉色为邻近材料支持的条件推断；灰色菱形仅为原始构型的逆映射诊断，可能位于内部、表面外或来自拟合不足，不能当作表面滑动。'
        '包络为扰动敏感性范围，不是统计置信区间。未跨越缺失片段连线或补成连续滑移。</li></ol>'
        '<p><b>重要区别：</b>初始表面地址迁移表示“换成了哪块材料”；同一材料相对刀具的切向运动才是滑动运动量。'
        '局部伸长会让二者数值不同。若刀尖接触已切出的新裂面，原本位于内部的材料没有原始自由表面地址，不能硬算成旧表面滑移。'
        '小残差、轨迹平滑以及保留点数均不保证真实物理身份已验证；深部地址未知区间请结合原视频检查。</p>'
        '<p><a href="method.html">公式、符号与文件字段说明</a></p>'
        + inspection + ''.join(cards) + '</html>')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'index.html').write_text(document, encoding='utf-8')
    (args.out/'method.html').write_text((ROOT/'docs/blade_contact_measurement.html').read_text(encoding='utf-8'), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--names', nargs='+', default=list(NAMES))
    parser.add_argument('--stage', choices=('measure', 'render', 'all'), default='all')
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--analysis-only', action='store_true', help='Skip extra material-only videos')
    args = parser.parse_args(argv)
    from src.video_sources import validate_case_names
    try:
        validate_case_names(args.names)
    except ValueError as error:
        parser.error(str(error))
    args.source, args.out = args.source.resolve(), args.out.resolve()
    if args.source == args.out:
        raise ValueError('Output must differ from the read-only saved-flow source')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        for name in args.names:
            if args.stage in ('measure', 'all'):
                measure(name, args)
            if args.stage in ('render', 'all'):
                render(name, args)
        return
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', OMP_NUM_THREADS='2')
    def worker(name):
        folder = args.out/name
        folder.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, '-u', '-m', 'src.workflows.measurement', '--worker', '--names', name,
               '--source', str(args.source), '--out', str(args.out), '--stage', args.stage]
        if args.analysis_only:
            cmd.append('--analysis-only')
        with (folder/f'{args.stage}.log').open('ab') as log:
            code = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=HIDDEN)
        return dict(case=name, exit_code=code, log=str(folder/f'{args.stage}.log'))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(worker, args.names))
    save_json(args.out/'batch_status.json', dict(stage=args.stage, results=results, completed=time.time()))
    write_index(args)
    if any(row['exit_code'] for row in results):
        raise SystemExit('One or more cases failed; see batch_status.json')


if __name__ == '__main__':
    main()
