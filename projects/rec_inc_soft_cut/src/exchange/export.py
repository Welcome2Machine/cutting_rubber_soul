"""Package existing optical-flow observations for independent FE-side review.

No tracking, geometry fitting, force synchronization or lab registration runs.
See docs/optical_flow_handoff.md for the receiving data contract.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.io import ROOT as REPO, clean as finite, sha as sha256, write_json
from .source import inspect_case
from .observations import export_tracks, export_optional



FRAME_COLUMNS = [
    'case_id', 'frame_id', 'time_s', 'stage', 'depth_mm', 'depth_source',
    'depth_uncertainty_mm', 'blade_tip_x_px', 'blade_tip_y_px',
    'blade_tip_status', 'blade_angle_deg', 'valid_frame', 'invalid_reason',
    'original_avi_frame_id', 'original_avi_header_time_s', 'capture_time_s',
    'legacy_assumed_capture_time_s', 'depth_video_estimate_mm', 'blade_entry_px',
    'geometry_supported', 'geometry_interpolated', 'fracture_state',
    'within_user_analysis_cutoff',
]
USER_CUTOFFS = {
    '0930l30a1': 1., '0930l30a2': 1., '0930l60a1': 1.4, '0930l60a2': 1.4,
    '0930n60a1': 1.5, '0930n60a2': 1.5, '0930l60c1up': .8, '0930l60c1down': 1.1,
}








def write_csv(path, columns, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: finite(v) for k, v in row.items()} for row in rows)


def selected_indices(info, stop_mm=None):
    selected = np.flatnonzero(info['display_mask'] & (info['frames'] <= info['reversal_frame']))
    if stop_mm is not None:
        # Prefix, never selectively delete a depth fluctuation or fabricate an endpoint.
        crossed = np.flatnonzero((info['frames'][selected] >= info['contact_frame']) &
                                (info['depth_estimate_mm'][selected] >= stop_mm))
        if len(crossed):
            selected = selected[:int(crossed[0]) + 1]
    if not len(selected) or info['frames'][selected[0]] != 0:
        raise ValueError('Export must preserve the saved initial reference frame')
    return selected


def export_frames(info, selected, out):
    case = info['case_id']
    cutoff = USER_CUTOFFS.get(case)
    rows = []
    for index in selected:
        frame = int(info['frames'][index])
        tip = info['tips'][index]
        supported = bool(info['geometry_supported'][index])
        tip_finite = bool(np.isfinite(tip).all())
        rows.append(dict(
            case_id=case, frame_id=frame, time_s=info['time_s'][index],
            stage='precontact' if frame < info['contact_frame'] else 'loading',
            depth_mm=None, depth_source='unknown', depth_uncertainty_mm=None,
            blade_tip_x_px=tip[0], blade_tip_y_px=tip[1],
            blade_tip_status=('rigid_inferred' if supported else 'predicted') if tip_finite else 'unavailable',
            blade_angle_deg=None, valid_frame=1, invalid_reason='',
            original_avi_frame_id=int(info['avi_frames'][index]),
            original_avi_header_time_s=info['avi_header_time_s'][index], capture_time_s=None,
            legacy_assumed_capture_time_s=info['legacy_assumed_capture_time_s'][index],
            depth_video_estimate_mm=info['depth_estimate_mm'][index], blade_entry_px=info['entry_px'][index],
            geometry_supported=int(supported), geometry_interpolated=int(info['geometry_interpolated'][index]),
            fracture_state='unknown',
            within_user_analysis_cutoff=int(info['depth_estimate_mm'][index] <= cutoff)
                if cutoff is not None and np.isfinite(info['depth_estimate_mm'][index]) else None,
        ))
    write_csv(out / 'frames.csv', FRAME_COLUMNS, rows)
    events = [
        dict(case_id=case, event='first_contact', frame_start=None, frame_end=None,
             evidence_kind='inferred', estimation_method='algorithmic_proximity', estimated_frame=info['contact_frame'],
             notes='Sustained blade/material image proximity; no independently confirmed mechanical-contact interval.'),
        dict(case_id=case, event='loading_end', frame_start=None, frame_end=None,
             evidence_kind='inferred', estimation_method='registered_blade_reversal', estimated_frame=info['reversal_frame'],
             notes='Estimated full-source reversal, possibly beyond exported prefix; temporal uncertainty not assessed.'),
    ]
    write_csv(out / 'events.csv', list(events[0]), events)


def copy_evidence(info, out, frames):
    evidence = out / 'source_evidence'
    evidence.mkdir(exist_ok=True)
    write_json(evidence / 'source_inspection.json', dict(paths=info['paths'], **info['json_evidence']))
    copied = []
    for key in ('source_metadata', 'alignment', 'geometry', 'geometry_metadata', 'mapping',
                'queries', 'normalization', 'tracking_complete', 'plan'):
        source = Path(info['paths'][key])
        if source.is_file():
            destination = evidence / key / source.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(destination.relative_to(out).as_posix())
    initial_surface = Path(info['paths']['measurements_dir']) / 'initial_surface_extended.npz'
    if initial_surface.is_file():
        shutil.copy2(initial_surface, evidence / initial_surface.name)
        copied.append((evidence / initial_surface.name).relative_to(out).as_posix())
    masks_source = Path(info['paths']['masks'])
    mask_ids = []
    destination = out / 'masks'
    destination.mkdir(exist_ok=True)
    for frame in frames:
        source = masks_source / f'f{int(frame):06d}.png'
        if source.is_file():
            shutil.copy2(source, destination / source.name)
            mask_ids.append(int(frame))
    return dict(files=copied, mask_count=len(mask_ids), mask_frame_ids=mask_ids,
                mask_semantics={'0': 'background', '1': 'material', '2': 'blade', '255': 'ignore'},
                mask_policy='Exact saved semantic masks only; no temporal interpolation or extrapolation. Masks are estimates, not truth.',
                initial_surface_policy='Geometric fitted boundary with tangent/normal only; no persistent material identity.')


def coverage_by_depth(out):
    with (out / 'frames.csv').open(encoding='utf-8', newline='') as stream:
        depths = {int(row['frame_id']): float(row['depth_video_estimate_mm'])
                  if row['depth_video_estimate_mm'] else np.nan for row in csv.DictReader(stream)}
    bins = [('precontact estimate', -np.inf, 0), ('0–0.5 mm estimate', 0, .5),
            ('0.5–1 mm estimate', .5, 1), ('1–2 mm estimate', 1, 2), ('2–3+ mm estimate', 2, np.inf)]
    counts = {(label, roi): [0, 0] for label, _, _ in bins
              for roi in ('near_left', 'near_right', 'far_left', 'far_right')}
    counts.update({('unknown depth', roi): [0, 0]
                   for roi in ('near_left', 'near_right', 'far_left', 'far_right')})
    with (out / 'coverage.csv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            depth = depths[int(row['frame_id'])]
            if not np.isfinite(depth):
                count = counts['unknown depth', row['roi_ref']]
                count[0] += int(row['valid_points'])
                count[1] += int(row['total_points'])
                continue
            for label, lower, upper in bins:
                if lower <= depth < upper:
                    count = counts[label, row['roi_ref']]
                    count[0] += int(row['valid_points'])
                    count[1] += int(row['total_points'])
                    break
    return [dict(depth_bin=label, roi_ref=roi, valid_points=n, total_points=d,
                 valid_fraction=n / d if d else None) for (label, roi), (n, d) in counts.items()]


def quality_report(info, stats, optional, evidence, bins, out):
    frac = lambda v: f'{v:.1%}' if v is not None else 'unavailable'
    roi_lines = '\n'.join(f"| {name} | {r['initial_point_count']} | {frac(r['valid_fraction'])} |" for name, r in stats['roi'].items())
    bin_lines = '\n'.join(f"| {r['depth_bin']} | {r['roi_ref']} | {frac(r['valid_fraction'])} |" for r in bins)
    q = stats['fb_error_px_quantiles']
    missing = '\n'.join(f'- {text}' for text in info['metadata']['unavailable_items'] + optional['missing_items'])
    report = f"""# {info['case_id']} — observational exchange quality report

## 可直接使用的内容 / Delivered observations

保存的加载段 {stats['frame_count']:,} 帧、{stats['initial_point_count']:,} 个初始采样位置、
{stats['exported_rows']:,} 行轨迹全部导出，没有空间降采样，也没有重新计算光流。
有效局部观测 {stats['valid_observations']:,} 行，占 {stats['valid_fraction']:.2%}。
这表示通过已有图像匹配与材料/刀具排除检查，不等于物理真值已验证。

## 1. Reference and contact zero

图像参考帧是源 MP4 的 frame 0；位于算法首触估计 {info['contact_frame']} 之前。
reference_kind=unknown：未独立证明其完全未变形、无初始载荷。
首触由持续的刀尖/材料几何邻近阈值估计，未确认最后分离/首次机械接触区间。
events.csv 因而不给伪精确接触区间；estimated_frame 单独保留算法结果。
阶段 precontact/loading 也只依赖该估计；fracture_state 始终 unknown。

## 2. Coordinates, scale and depth

坐标是未叠加的本地源 MP4 1024×768 像素中心，x 右、y 下；轨迹没有 ROI 偏移。
初始表面拟合角 {info['angle_image_deg']:.6f}° 是图像角，不是已标定实验室倾角。
约 3 mm 接触至回退行程给出条件比例尺 {info['scale_mm_px'] * 1000:.6f} μm/px。
各试次行程依据见 metadata，不能把 assumed 与 user_confirmed 混作同等标定。
未建立固定基底/相机运动/实验室旋转/投影标定，所以核心 depth_mm 和实验室坐标均空。
depth_video_estimate_mm 仅为保存的刀具图像竖直位移乘条件比例尺。
ux_image_est_mm/uy_image_est_mm 也是各连续段参考下的条件图像位移，y 翻转为向上；不是实验室位移。

## 3. Spatial coverage

| Initial sampling region | Initial points | Local support over exported rows |
|---|---:|---:|
{roi_lines}

near=原始密集核心，far=追加稀疏点；left/right 按初始 seed x 与估计首触刀尖 x 比较。
这些是固定采样分组，不代表当前刀具左右接触面。roi_current 留空。
仅覆盖可见橡胶侧面，point_kind=rubber_face；没有把近表面深层点冒充原始上表面材料点。
密点间距不等于独立空间分辨率；匹配窗口重叠，误差可能相关。

| Conditional image-depth bin | Initial region | Local support |
|---|---|---:|
{bin_lines}

## 4. Gaps, predictions and identity

所有初始采样位置在每个已保存导出帧都有一行。无图像支持时 x_px/y_px 留空，
仅 x_pred_px/y_pred_px 保留预测；被语义检查拒绝的原匹配坐标在 x_raw_match_px/y_raw_match_px。
发现 {stats['resumed_segments']:,} 次中断后恢复，导出为新 point_id 与 deformed 参考，
共 {stats['observation_segment_count']:,} 个连续段。parent_point_id 只记录来源，不证明跨段同一材料身份。
original_seed_* 保留追踪器的原始身份假设供审核，不能据此合并未证实的间断。
原轨迹未修改。重新观测段可比较后续增量，不能直接与未变形有限元构型相减。

## 5. Precision checks and remaining validation

反向循环误差（有效观测，非初始帧）：{json.dumps(q, ensure_ascii=False)} px。
它检验当前块回到所选历史参考块的自洽性，不是相邻帧独立真值误差。
confidence 是相关性分数，非概率；frame 0 的人工初始化分数不输出。
独立静止背景误差、标准已知位移试验、亚像素物理准确度与位置不确定度均未评估。
position_uncertainty_px 空值不表示零误差。
原图点位叠加见 previews/contact_sheet.png；数值/源映射检查见根目录 validation.json。

## 6. Blade, masks and contours

刀尖由刀具边缘模板刚体平移推算，几何支持时 rigid_inferred，插值时 predicted；没有逐帧直接尖端观测或转角。
交付 {evidence['mask_count']:,} 张实际保存掩膜（未补帧）。轮廓文件存在时，它是几何检测，非材料身份。
轮廓在无效边、状态改变和缺失点处分段，不跨遮挡拼接闭环。
分割和光流错误可能相关，掩膜不能充当独立真值。

## 7. Force and contact quantities

原力文件共 {optional['counts'].get('force_samples', 0):,} 个传感器采样，保持自身时钟。
V01/V02 为已记录 N 通道但轴向/极性未知；V03 为 V 通道但含义未确认。
没有视频同步映射，不生成 Fz(depth) 或物理速度；历史 200 fps 仅作为未验证假设留存。
model_contact_estimates.csv 存在时，仅为已有条件模型输出；contact_confirmed=0，非观察到的接触点。
敏感性 sigma 不是统计置信区间，零点处的零值是算法原点约束。
没有输出未经证实的 contact_candidates、切向滑速、应变、裂纹、应力或摩擦系数。

## 8. FE use and unresolved fields

data_role=undesignated。建议先核查首触前至条件图像深度约 1 mm 的像素位移，
只在另有完整性证据时与无断裂模型比较；用户分析截止深度不证明此前未起裂。
更长加载段为方便审核一并保留，不能整段默认 intact。
{missing}

导出一致性检查通过也不等于实验坐标已注册或物理精度已达标。
"""
    (out / 'quality_report.md').write_text(report, encoding='utf-8')


def export_case(config, root, stop_mm=None, compress_tracks=True):
    name = config['name']
    out = root / name
    if out.exists():
        raise FileExistsError(f'Refusing to mix existing exchange files: {out}')
    out.mkdir(parents=True)
    print(f'{name}: inspect saved source', flush=True)
    info = inspect_case(REPO, config)
    selected = selected_indices(info, stop_mm)
    frames = info['frames'][selected]
    export_frames(info, selected, out)
    print(f'{name}: export {len(frames)} frames / {len(info["point_ids"])} seeds', flush=True)
    stats = export_tracks(REPO, config, out, frames, 'unknown', info['scale_mm_px'],
                          np.asarray(info['metadata']['reference']['estimated_contact_blade_source_xy_px']),
                          compress=compress_tracks)
    optional = export_optional(REPO, name, out, frames)
    evidence = copy_evidence(info, out, frames)
    previews = export_previews(info, out, frames)
    bins = coverage_by_depth(out)
    metadata = info['metadata']
    metadata['provenance'] = dict(source_paths=info['paths'], export_utc=datetime.now(timezone.utc).isoformat(),
                                  optical_flow_recomputed=False, source_videos_included=False,
                                  source_fingerprints=info['json_evidence']['fingerprints'])
    metadata['export'] = dict(frame_count=len(frames), first_frame=int(frames[0]), last_frame=int(frames[-1]),
                               stop_at_image_depth_mm=stop_mm, point_subsampling=False,
                               user_analysis_cutoff_mm=USER_CUTOFFS.get(name),
                               cutoff_interpretation='External analysis cutoff, not an independently observed fracture event',
                               frame_stage_semantics='Relative to algorithmic proximity contact estimate only',
                               valid_frame_semantics='Saved source frame and mapping available; point/geometry validity remains separate')
    metadata['tracking']['exchange_identity_policy'] = stats['segment_definition']
    metadata['tracking']['exchange_initial_reference_kind'] = 'unknown'
    metadata['tracking']['confidence_definition'] = 'Saved patch correlation; higher is better, nominal range [-1,1], not calibrated probability. Empty for initialization or unsupported match.'
    metadata['tracking']['fb_error_definition'] = 'Reverse-cycle registration distance to selected reference patch center in pixels; reference may be prior observed frame, keyframe or initial.'
    metadata['coverage'] = stats
    metadata['coverage_by_conditional_image_depth'] = bins
    metadata['optional'] = optional
    metadata['evidence'] = evidence
    metadata['previews'] = previews
    metadata['force_sync'].update(available='force' in optional, channel_axes_verified=False,
                                   **{k: optional.get('force', {}).get(k) for k in ('sample_count', 'time_source')})
    metadata['files'].update(tracks='tracks.csv.gz' if compress_tracks else 'tracks.csv',
                             events='events.csv', coverage='coverage.csv', **optional['files'])
    metadata['frame_columns'] = FRAME_COLUMNS
    write_json(out / 'metadata.json', metadata)
    quality_report(info, stats, optional, evidence, bins, out)
    summary = dict(case_id=name, frames=len(frames), initial_points=stats['initial_point_count'],
        track_rows=stats['exported_rows'], valid_rows=stats['valid_observations'],
        valid_fraction=stats['valid_fraction'], new_segments=stats['resumed_segments'],
        estimated_contact_frame=info['contact_frame'], estimated_reversal_frame=info['reversal_frame'],
        last_export_frame=int(frames[-1]), conditional_um_per_px=info['scale_mm_px']*1000,
        image_surface_angle_deg=info['angle_image_deg'],
        stroke_status=config.get('stroke_status', 'unknown'), mask_count=evidence['mask_count'],
        contour_vertices=optional['counts'].get('contour_vertices', 0),
        force_samples=optional['counts'].get('force_samples', 0),
        data_role='undesignated', max_export_image_depth_mm=float(np.nanmax(info['depth_estimate_mm'][selected])))
    print(f'{name}: complete, {stats["exported_rows"]:,} rows; support {stats["valid_fraction"]:.2%}', flush=True)
    return summary


def write_readme(root, summaries):
    rows = '\n'.join(f"| [{r['case_id']}]({r['case_id']}/quality_report.md) | {r['frames']:,} | {r['initial_points']:,} | {r['valid_fraction']:.1%} | {r['stroke_status']} |" for r in summaries)
    body = f"""# Optical-flow exchange for finite-element review

交付日期：2026-10-12。接口依据：[随包交接规范](optical_flow_handoff.md)。
本包整理现有 0930 的 {len(summaries)} 组分析，不重新运行光流，不修改原始 data。
全部文件留在本地 results（Git 忽略目录），未上传实验数据。

## 首先读取 / Start here

1. 阅读 [字段与读取说明](DATA_DICTIONARY.md)，先读 `0930l60a1/metadata.json` 与质量报告。
2. `frames.csv` 是源 MP4 帧索引；轨迹表按帧/固定初始采样顺序逐行导出，实际文件名见 `metadata.files.tracks`（默认 `tracks.csv.gz`，无损 gzip）。
3. 先筛 `valid=1`；从初始构型比较还必须筛 `reference_frame_id=0`。
   新连续段只有变形后参考，不能直接与未变形 FE 位移相比较。
4. 第一轮可筛 `stage == 'precontact'` 或 `depth_video_estimate_mm <= 1`，
   即使部分首触前帧没有刀具深度也保留参考帧。
   这只是近似图像深度，不证明该范围内材料未起裂。
5. 正式毫米比较前由接收方核实实验室方向、基底/相机固定、接触零点、条件比例尺与完整性区间。
   力曲线需确认通道及视频同步后才能关联；不得按行号拼接。

## 完成情况 / Capability

| 项目 | 本次交付 | 限制 |
|---|---|---|
| 源与映射 | 视频SHA、MP4帧、AVI选帧映射、处理链 | AVI原始采集时间未知；历史200fps未验证 |
| 材料运动 P0 | 完整已存点阵、连续段ID、有效/预测/遮挡状态、质量分数 | 图像支持不等于物理精度；断点身份未验证 |
| 刀具 | 保存的刚体推算刀尖坐标/支持状态 | 非逐帧直接尖端测量；旋转未测 |
| 标定与深度 | 约3mm行程条件比例尺、图像深度附加列 | 核心lab/depth字段空；未作固定基底注册 |
| 轮廓/掩膜 | 六组A已有轮廓、各组已有掩膜、初始边界NPZ | 几何检测不等于原表面材料身份；缺段不连接 |
| 力 | 十组原通道与自身500Hz时间 | 轴向/符号/Fz/深度/视频同步未确认 |
| 接触迁移 | 三组A1已有双方法模型估计与支持标记 | 条件输出，非已证实接触点；无P2接触真值 |
| 校验 | 数值表一致性、源指纹、源坐标采样、叠加预览 | 不是物理精度或实验–FE等价性验证 |

## 数据目录

| Trial | Frames | Initial samples | Local observation support | Approx. 3 mm stroke source |
|---|---:|---:|---:|---|
{rows}

合计 {sum(r['track_rows'] for r in summaries):,} 行轨迹。
所有试次 data_role=undesignated，保留独立验证用途，未指定为标定数据。
已有更长加载段一并交付；没有因界面需要重跑C系列或新增结果。

## 文件布局

- `<case>/metadata.json`, `frames.csv`, `tracks.csv.gz`（或旧 `tracks.csv`）, `quality_report.md`：P0。
- `coverage.csv`, `events.csv`, `masks/`, `previews/`：覆盖率、事件估计及原图核查。
- `force_channels.csv`：独立传感器时钟；**不冒名为Fz**。
- `contours.csv`（若有）：只连接同一 frame/contour/component 内按 vertex_order 排列的点。
- `model_contact_estimates.csv`（若有）：已有模型输出，与观测分开。
- `source_evidence/`：现有配置、源证据、初始几何及力文件；完整源视频/大轨迹NPZ不重复装包，保留原路径与SHA。
- `case_index.csv`：试次汇总；`validation.json`：机器校验；`execution_evidence.json`：运行环境与正式资产。
- `SHA256SUMS.json`：包内文件校验。

## 仍需补充才能开展严格定量对照

实际采集帧率/时间戳，力V01/V02轴向与极性、视频同步事件，固定基底/相机注册，
独立比例尺及误差，机械首触区间和可信完整体区间。未知项没有用仿真参数补齐。
用户先前指定的深度截止和显微镜值仅为外部信息，未用于校准本次导出。
"""
    (root / 'README.md').write_text(body, encoding='utf-8')


def write_dictionary(root):
    text = """# Exchange field dictionary / 字段说明

CSV=UTF-8, comma, decimal dot; bool 0/1; missing numeric empty; JSON missing null.
No missing value is zero and no uncertainty was set to zero to indicate unavailable.

## frames.csv

`case_id,frame_id` identify the selected local MP4 decoded frame (zero based).
`time_s=frame_id/12` is that CFR MP4 playback clock, NOT verified acquisition time.
`original_avi_frame_id` is the saved exact upstream selected index; `original_avi_header_time_s`
uses the upstream 30fps header. `legacy_assumed_capture_time_s` preserves the unverified 200fps interpretation.
`capture_time_s` is empty. None of the three clocks is synchronized to the force clock.
`stage` is precontact/loading relative to the proximity estimate, not a fracture classification.
`depth_mm`, `depth_uncertainty_mm`, `blade_angle_deg` are empty; `depth_source=unknown`.
`depth_video_estimate_mm=(tip_y-tip_y_at_estimated_contact)*conditional_image_scale_mm_px`.
`blade_entry_px` is the corresponding signed image-vertical travel.
`blade_tip_x_px/y_px` use source image coordinates. `blade_tip_status` distinguishes
rigid_inferred vs predicted vs unavailable. `geometry_supported/interpolated` are saved support flags.
`valid_frame=1` means a saved source-frame mapping exists, not that every point or geometry is valid.
`fracture_state=unknown`; `within_user_analysis_cutoff` only marks an externally chosen cutoff.

## tracks.csv

`point_id=original_seed_id:sN` denotes a continuous observation segment.
`reference_frame_id`, `reference_kind`, `x_ref_px/y_ref_px` are constant within a segment.
Initial reference_kind is unknown (precontact estimated, undeformed state not independently certified).
After an unsupported interval a resumed point gets a new deformed reference and segment ID.
`parent_point_id` is bookkeeping, not verified reidentification.
`original_seed_point_id,x/y` retain the raw tracker's initial identity hypothesis even across gaps.
They must not silently replace the exchange segment reference.

`x_px/y_px` only contain measurement-supported observations, `valid=1`, `track_status=tracked`.
All coordinates are pixel centers of the unannotated full source MP4, x right, y down.
`point_kind=rubber_face`: visible side-face texture, not verified original top surface.
`raw_tracking_supported` is saved local image support; `measurement_supported` additionally
requires in-image, material semantic region, and not blade.
`track_status`: tracked/predicted/rejected/occluded/out_of_frame.
`invalid_reason`: ambiguity, failed local checks, blade occlusion, outside material or image.
`x_pred_px/y_pred_px` contain forecast-only positions, never valid observations.
`x_raw_match_px/y_raw_match_px` preserve a locally accepted match rejected by semantic guards.

`confidence` is saved correlation, higher better, nominal [-1,1], not calibrated probability.
`fb_error_px` is the backward registration cycle distance to the SELECTED reference;
not necessarily an adjacent-frame check, and not physical position error.
`position_uncertainty_px` is empty (unassessed).
`quality_source` distinguishes seed_initialization and local_image_checks;
initial frame's artificial correlation=1/cycle=0 are omitted.
`selected_reference`: 0 last supported incremental template, 1 historical keyframe, 2 initial template.
`reference_age_steps` counts saved tracking steps, not seconds.
`motion_step_valid` requires saved motion support, present support and no segment restart.
It does not certify physical velocity or remove the need to check both consecutive observations.

`roi_ref`: near_left/near_right/far_left/far_right from the fixed seed grid and initial contact-tip x.
Near means original dense core, far means appended coarse coverage; NOT current contact bands.
`roi_current` empty (no certified current contact classification).
`ux_image_est_mm=(x_px-x_ref_px)*conditional_scale`;
`uy_image_est_mm=-(y_px-y_ref_px)*conditional_scale`. They are conditional image-axis increments.
All `*_lab_mm`, `ux_mm`, `uy_mm` remain empty until independently registered to a fixed lab/base frame.

## coverage.csv

Per-frame/per-initial-region total points, raw_supported_points, valid_points, valid_fraction,
forecast_points, occluded_points, rejected_points, out_of_frame_points,
new_observation_segments, reidentified_points.
Final statuses are mutually exclusive; forecast_points instead means raw support absent and can
overlap occluded/out_of_frame. Do not sum diagnostic counts to infer total points.
reidentified_points is zero because cross-gap identity is not independently established.

## events.csv, optional files, metadata

Event intervals are empty when unconfirmed; `estimated_frame` retains an algorithm estimate.
Force/contour/model column definitions and exact evidence are in metadata.optional.columns,
with method/units/visibility/clock details in metadata.optional.force/contours/contact_models.
Force sensor timestamps retain the original relative clock. Channel V01/V02 are N, V03 V.
Model `depth_video_estimate_mm` is the historical CONDITIONAL IMAGE DEPTH, not frames.csv strict depth_mm.
Model `sensitivity_sigma_um` is algorithm sensitivity, not a statistical confidence interval.
Model `point_id` is empty and `contact_confirmed=0`.
Contour components are open; point_id and reference_s_rel_mm are empty.
Masks: 0 background, 1 material, 2 blade, 255 ignore. No masks were synthesized.

metadata.provenance.source_paths refers to the original local source files, with fingerprints.
metadata.coordinates contains scale provenance and identity pixel transform. No lab transform exists.
metadata.tracking.parameters contains original algorithm parameters/window sizes; no retraining performed.
metadata.coverage includes complete counts and reverse-error diagnostics.

## Minimal streaming example

```python
import json
from pathlib import Path
import pandas as pd
p = Path('0930l60a1')
meta = json.loads((p / 'metadata.json').read_text(encoding='utf-8'))
frames = pd.read_csv(p / 'frames.csv')
early = set(frames.loc[(frames.stage == 'precontact') |
                       (frames.depth_video_estimate_mm <= 1), 'frame_id'])
for chunk in pd.read_csv(p / meta['files']['tracks'], chunksize=100_000):
    observed = chunk[(chunk.valid == 1) & chunk.frame_id.isin(early)]
    from_initial = observed[observed.reference_frame_id == 0].copy()
    from_initial['ux_px'] = from_initial.x_px - from_initial.x_ref_px
    from_initial['uy_px'] = from_initial.y_px - from_initial.y_ref_px
    # Pixel motion can be audited now. Independent lab registration and intact-range
    # evidence are still required before comparing quantitative FE displacements.
```
"""
    (root / 'DATA_DICTIONARY.md').write_text(text, encoding='utf-8')


def existing_summary(config, root, stop_mm):
    """Resume only fully documented cases from this exact source/selection."""
    out = root / config['name']
    metadata = json.loads((out / 'metadata.json').read_text(encoding='utf-8'))
    if not (out / 'quality_report.md').is_file() or metadata['case_id'] != config['name']:
        raise ValueError('Cannot resume incomplete or mismatched case')
    if metadata['export']['stop_at_image_depth_mm'] != stop_mm:
        raise ValueError('Cannot mix pilot and full exports')
    expected_tracks = Path(config['tracks_root']).resolve() / config['name'] / 'tracking/tracks.npz'
    if Path(metadata['provenance']['source_paths']['tracks']).resolve() != expected_tracks:
        raise ValueError('Saved export has different source tracks')
    with (out / 'frames.csv').open(encoding='utf-8', newline='') as stream:
        values = [float(r['depth_video_estimate_mm']) for r in csv.DictReader(stream) if r['depth_video_estimate_mm']]
    stats, optional = metadata['coverage'], metadata['optional']
    return dict(case_id=config['name'], frames=metadata['export']['frame_count'],
        initial_points=stats['initial_point_count'], track_rows=stats['exported_rows'],
        valid_rows=stats['valid_observations'], valid_fraction=stats['valid_fraction'],
        new_segments=stats['resumed_segments'],
        estimated_contact_frame=metadata['reference']['estimated_contact_frame'],
        estimated_reversal_frame=metadata['reference']['estimated_reversal_frame'],
        last_export_frame=metadata['export']['last_frame'],
        conditional_um_per_px=metadata['coordinates']['conditional_image_scale_mm_px'] * 1000,
        image_surface_angle_deg=metadata['reference']['initial_surface_angle_image_deg'],
        stroke_status=config.get('stroke_status', 'unknown'), mask_count=metadata['evidence']['mask_count'],
        contour_vertices=optional['counts'].get('contour_vertices', 0),
        force_samples=optional['counts'].get('force_samples', 0), data_role='undesignated',
        max_export_image_depth_mm=max(values) if values else None)


def _selected_frames(info: dict, frame_ids: np.ndarray) -> list[dict]:
    requested = np.asarray(frame_ids)
    if (requested.ndim != 1 or not len(requested) or requested.dtype.kind not in "iu"
            or np.any(np.diff(requested) <= 0)):
        raise ValueError("Preview frame IDs must be nonempty, integer and strictly increasing")
    source_rows = {int(frame): i for i, frame in enumerate(info["frames"])}
    if any(int(frame) not in source_rows for frame in requested):
        raise ValueError("Preview frame IDs are absent from source inspection")
    labels: dict[int, list[str]] = {}

    def add(frame, label):
        if int(frame) in source_rows and int(frame) in requested:
            labels.setdefault(int(frame), []).append(label)

    add(int(requested[0]), "initial reference" if requested[0] == 0 else "first supplied frame")
    add(int(info["contact_frame"]), "proximity-based contact-zero estimate")
    loading = [int(f) for f in requested if info["contact_frame"] <= f <= info["reversal_frame"]
               and np.isfinite(info["depth_estimate_mm"][source_rows[int(f)]])]
    if loading:
        values = np.asarray([info["depth_estimate_mm"][source_rows[f]] for f in loading])
        for target in (.5, 1.):
            chosen = loading[int(np.argmin(abs(values - target)))]
            add(chosen, f"nearest supplied frame to {target:.1f} mm image-depth estimate")
    add(int(requested[-1]), "last supplied frame")
    return [dict(frame_id=frame, row=source_rows[frame], selection=descriptions)
            for frame, descriptions in sorted(labels.items())]


def _text(image, text, origin, color=(220, 220, 220), scale=.53, thickness=1):
    available = image.shape[1] - origin[0] - 12
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    if size[0] > available:
        scale *= available / size[0]
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise IOError(f"Cannot encode preview: {path}")
    encoded.tofile(str(path))


def export_previews(info: dict, case_out: Path, frame_ids: np.ndarray) -> dict:
    """Draw at most five scientific check frames; no tracking or transforms.

    The image portion is native-size with a header above it. Display coordinates
    only gain that header height; saved source coordinates are never offset by
    a historical ROI. Colors distinguish saved image support from forecasts.
    """
    case_out = Path(case_out)
    directory = case_out / "previews"
    directory.mkdir(parents=True, exist_ok=True)
    selected = _selected_frames(info, frame_ids)
    with np.load(info["paths"]["tracks"], allow_pickle=False) as stored:
        track_frames = stored["source_frame_indices"]
        positions = stored["positions"]
        raw_support = stored["supported"]
        point_ids = stored["point_ids"]
    measurement = Path(info["paths"]["measurements_dir"]) / "measurement_support.npz"
    with np.load(measurement, allow_pickle=False) as stored:
        measurement_frames = stored["frame_indices"]
        support = stored["supported"]
        if not np.array_equal(stored["point_ids"], point_ids):
            raise ValueError("Preview measurement and track IDs disagree")
    tr = {int(frame): i for i, frame in enumerate(track_frames)}
    mr = {int(frame): i for i, frame in enumerate(measurement_frames)}
    if any(sample["frame_id"] not in tr or sample["frame_id"] not in mr for sample in selected):
        raise ValueError("Selected preview frame lacks saved track/measurement evidence")
    cap = cv2.VideoCapture(str(info["paths"]["source_video"]))
    files, thumbnails, records = [], [], []
    green, orange, gray, cyan = (88, 230, 110), (0, 175, 255), (185, 185, 185), (255, 205, 40)
    try:
        if not cap.isOpened():
            raise IOError(f"Cannot open preview source: {info['paths']['source_video']}")
        for sample in selected:
            frame_id, index = sample["frame_id"], sample["row"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, image = cap.read()
            if not ok:
                raise IOError(f"Cannot decode source frame {frame_id}")
            decoded_next = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            if decoded_next != frame_id + 1:
                raise IOError(f"Decoder did not return requested source frame {frame_id}: next={decoded_next}")
            h, w = image.shape[:2]
            p = positions[tr[frame_id]]
            raw, accepted = raw_support[tr[frame_id]], support[mr[frame_id]]
            if np.any(accepted & ~raw):
                raise ValueError("Measurement support cannot promote a forecast to an observation")
            finite = np.isfinite(p).all(1)
            in_image = finite & (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
            predicted = ~raw & in_image
            rejected = raw & ~accepted & in_image
            observed = accepted & in_image
            # Draw excluded positions first and accepted small dots last.
            for position in p[rejected]:
                cv2.drawMarker(image, tuple(np.rint(position).astype(int)), gray,
                               cv2.MARKER_TILTED_CROSS, 4, 1, cv2.LINE_AA)
            for position in p[predicted]:
                cv2.circle(image, tuple(np.rint(position).astype(int)), 2, orange, 1, cv2.LINE_AA)
            for position in p[observed]:
                cv2.circle(image, tuple(np.rint(position).astype(int)), 1, green, -1, cv2.LINE_AA)
            tip = np.asarray(info["tips"][index], float)
            tip_finite = bool(np.isfinite(tip).all())
            tip_status = ("rigid_inferred" if info["geometry_supported"][index] else "predicted") if tip_finite else "unavailable"
            if tip_finite:
                cv2.drawMarker(image, tuple(np.rint(tip).astype(int)), cyan,
                               cv2.MARKER_CROSS, 19, 2, cv2.LINE_AA)
            header, footer = 112, 80
            panel = np.full((h + header + footer, w, 3), (20, 24, 27), np.uint8)
            panel[header:header + h] = image
            depth = float(info["depth_estimate_mm"][index])
            depth_text = f"{depth:+.3f} mm" if np.isfinite(depth) else "unavailable"
            _text(panel, f"{info['case_id']} | Source frame {frame_id} | Saved-track evidence", (14, 25), scale=.66)
            _text(panel, f"Depth: {depth_text} (conditional image estimate) | Blade tip: {tip_status}", (14, 51), cyan)
            _text(panel, "; ".join(sample["selection"]), (14, 75), scale=.47)
            qualifier = "Reference seed initialization; not a measured zero-error state" if frame_id == 0 else "Local image support does not certify cross-gap material identity or physical contact"
            _text(panel, qualifier, (14, 98), scale=.48)
            _text(panel, "Green dot: saved image support + measurement exclusions passed", (14, header + h + 22), green)
            _text(panel, "Orange hollow: forecast, no accepted image match | Gray x: image match rejected by measurement checks", (14, header + h + 45), orange, scale=.48)
            _text(panel, f"Shown: supported {int(observed.sum())} | forecast {int(predicted.sum())} | rejected {int(rejected.sum())} | outside/nonfinite {int((~in_image).sum())}",
                  (14, header + h + 68), scale=.49)
            destination = directory / f"f{frame_id:06d}_check.png"
            _write_png(destination, panel)
            relative = destination.relative_to(case_out).as_posix()
            files.append(relative)
            thumbnail_width = 640
            thumbnails.append(cv2.resize(panel, (thumbnail_width, round(panel.shape[0] * thumbnail_width / w)), interpolation=cv2.INTER_AREA))
            records.append(dict(frame_id=frame_id, selection=sample["selection"], file=relative,
                depth_video_estimate_mm=depth if np.isfinite(depth) else None,
                blade_tip_status=tip_status, observed_display_count=int(observed.sum()),
                forecast_display_count=int(predicted.sum()), rejected_display_count=int(rejected.sum()),
                outside_or_nonfinite_count=int((~in_image).sum()),
                point_positions_transformed=False, source_frame_decoded=True))
    finally:
        cap.release()
    columns = min(3, len(thumbnails))
    cell_height = max(image.shape[0] for image in thumbnails)
    sheet = np.full((int(np.ceil(len(thumbnails) / columns)) * cell_height, columns * 640, 3), (20, 24, 27), np.uint8)
    for i, image in enumerate(thumbnails):
        y, x = (i // columns) * cell_height, (i % columns) * 640
        sheet[y:y + image.shape[0], x:x + image.shape[1]] = image
    sheet_path = directory / "contact_sheet.png"
    _write_png(sheet_path, sheet)
    sheet_relative = sheet_path.relative_to(case_out).as_posix()
    return dict(files=files + [sheet_relative], contact_sheet=sheet_relative, frames=records,
        source_video=str(info["paths"]["source_video"]), point_subsampling=False,
        source_image_photometry="Unannotated source MP4; no new shadow normalization or geometric correction applied to this preview",
        coordinate_space="Saved source-video pixels; native image region has header-height display offset only; no ROI offset added",
        interpretation=["Only saved trajectories and support flags are drawn; no optical-flow computation was run.",
            "Green means saved local image evidence and saved measurement exclusions passed, not ground-truth material identity.",
            "All forecast positions remain orange hollow and are not observations.",
            "Blade apex follows registered template translation (rigid_inferred), or saved prediction; no directly observed apex is asserted.",
            "Depth is the conditional image-derived estimate, not independently calibrated fixed-base laboratory indentation.",
            "At frame 0, dots represent the initialized material seeds, not measured displacements."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=REPO / 'results/20261005_0930_all/manifest.json')
    parser.add_argument('--out', type=Path, default=REPO / 'results/20261012_optical_flow_exchange')
    parser.add_argument('--names', nargs='+')
    parser.add_argument('--stop-mm', type=float, help='Pilot prefix only, based on conditional saved image depth')
    parser.add_argument('--workers', type=int, default=1, choices=(1, 2, 3), help='Independent CSV export processes; no GPU work')
    parser.add_argument('--resume', action='store_true', help='Keep completed cases; reject any incomplete case directory')
    parser.add_argument('--plain-csv', action='store_true', help='Export uncompressed tracks.csv instead of lossless tracks.csv.gz')
    args = parser.parse_args(argv)
    root = args.out.resolve()
    if not root.is_relative_to(REPO):
        raise ValueError('Keep experiment exchange assets inside the local repository ignored output directories')
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError('Choose a new empty export directory; existing datasets are never overwritten')
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    configs = manifest['cases']
    if args.names:
        configs = [c for c in configs if c['name'] in args.names]
        if set(args.names) != {c['name'] for c in configs}:
            raise ValueError('Unknown requested case')
    summaries, pending = [], []
    for config in configs:
        if args.resume and (root / config['name']).exists():
            summaries.append(existing_summary(config, root, args.stop_mm))
            print(f"{config['name']}: reuse completed export", flush=True)
        else:
            pending.append(config)
    if args.workers == 1:
        summaries += [export_case(config, root, args.stop_mm, not args.plain_csv) for config in pending]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(export_case, config, root, args.stop_mm, not args.plain_csv) for config in pending]
            summaries += [job.result() for job in as_completed(jobs)]
    order = {c['name']: i for i, c in enumerate(configs)}
    summaries.sort(key=lambda row: order[row['case_id']])
    write_csv(root / 'case_index.csv', list(summaries[0]), summaries)
    write_readme(root, summaries)
    write_dictionary(root)
    shutil.copy2(REPO / 'docs/optical_flow_handoff.md', root / 'optical_flow_handoff.md')
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True, capture_output=True, check=True).stdout.strip()
    scripts = sorted((REPO / 'src/exchange').glob('*.py')) + [REPO / 'src/io.py']
    evidence = dict(created_utc=datetime.now(timezone.utc).isoformat(), python=sys.executable,
        interpreter_exception='Repository .venv is absent; used existing AcademicForge softcutting0731 project venv, not global Python. No packages installed.',
        spreadsheet_export='Large machine-readable observational CSVs streamed with Python csv/numpy. Artifact runtime loaded, but no documented native CSV export API found; no workbook requested.',
        git_commit=commit, code_sha256={str(p.relative_to(REPO)): sha256(p) for p in scripts},
        manifest_path=str(args.manifest.resolve()), manifest_sha256=sha256(args.manifest),
        optical_flow_recomputed=False, uploaded=False, original_data_modified=False,
        formal_assets=['README.md', 'DATA_DICTIONARY.md', 'case_index.csv', 'optical_flow_handoff.md',
                       'validation.json', 'SHA256SUMS.json'] + [r['case_id'] + '/' for r in summaries],
        summaries=summaries, pilot_depth_mm=args.stop_mm)
    write_json(root / 'execution_evidence.json', evidence)
    print(f'EXPORTED {len(summaries)} cases to {root}; run independent validator before delivery', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
