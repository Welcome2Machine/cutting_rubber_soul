"""Pinned local point-tracking proposals; never upgrades a point to measured status.

Uses the official offline network on exactly the supplied time window. Frames
are RGB uint8, queries are (window index, x, y), and results stay in input pixels.
An offline proposal can use later frames *inside* that window: it is not causal
at each intermediate frame. The caller must validate texture, identity, masks,
and contact independently. No network access, training or automatic download.
"""
from pathlib import Path
import hashlib
import importlib
import json
import sys
import time

import numpy as np


SOURCE_COMMIT = '82e02e8029753ad4ef13cf06be7f4fc5facdda4d'
WEIGHT_SHA256 = '2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834'
TAPNEXT_COMMIT = '730cda1c730877cfedbe01bf87fb1cadb78a565d'
TAPNEXT_SHA256 = 'cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da'


def _sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _verified_assets(asset_dir, package, commit, weight_name, weight_sha):
    assets = Path(asset_dir).resolve()
    provenance = json.loads((assets / 'provenance.json').read_text(encoding='utf-8'))
    if provenance['source_commit'] != commit:
        raise ValueError('unexpected network source revision')
    weight, vendor = assets / weight_name, assets / 'vendor'
    if _sha256(weight) != weight_sha:
        raise ValueError('checkpoint SHA256 does not match the official pinned release')
    actual = {p.relative_to(vendor).as_posix(): _sha256(p) for p in (vendor / package).rglob('*.py')}
    if actual != provenance['source_python_sha256']:
        raise ValueError('vendor source has changed since deployment')
    if 'dependency_python_sha256' in provenance:
        dependencies = assets / 'dependencies'
        actual = {p.relative_to(dependencies).as_posix(): _sha256(p) for p in dependencies.rglob('*.py')}
        if actual != provenance['dependency_python_sha256']:
            raise ValueError('isolated dependency files have changed since deployment')
    previous = sys.modules.get(package)
    if previous is not None and Path(previous.__file__).resolve().parent != vendor / package:
        raise RuntimeError(f'a different {package} package is already imported')
    return assets, provenance, vendor, weight


def _inputs(frames, queries, point_ids, frame_indices, cutoff_frame):
    frames, queries = np.asarray(frames), np.asarray(queries, dtype=np.float32)
    if (frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8
            or min(frames.shape[:3]) < 2):
        raise ValueError('frames must be T,H,W,3 RGB uint8 with T,H,W >= 2')
    if queries.ndim != 2 or queries.shape[1] != 3 or not len(queries) or not np.isfinite(queries).all():
        raise ValueError('queries must be nonempty finite N,3 (window index,x,y)')
    t, h, w = frames.shape[:3]
    if (np.any(queries[:, 0] != np.floor(queries[:, 0])) or np.any(queries < 0)
            or np.any(queries[:, 0] >= t) or np.any(queries[:, 1:] > [w-1, h-1])):
        raise ValueError('queries must lie in the supplied frame window and image')
    ids = np.arange(len(queries)) if point_ids is None else np.asarray(point_ids)
    indices = np.arange(t) if frame_indices is None else np.asarray(frame_indices)
    if ids.shape != (len(queries),) or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids):
        raise ValueError('point_ids must be unique integer identities, one per query')
    if (indices.shape != (t,) or indices.dtype.kind not in 'iu'
            or np.any(indices < 0) or np.any(indices[1:] <= indices[:-1])):
        raise ValueError('frame_indices must be strictly increasing nonnegative integers')
    if cutoff_frame is not None:
        if not np.isfinite(cutoff_frame) or int(cutoff_frame) != cutoff_frame or indices[-1] > cutoff_frame:
            raise ValueError('supplied window exceeds cutoff_frame')
    return frames, queries, ids.copy(), indices.copy()


class CoTrackerCandidates:
    """Local pretrained proposal generator with verified weights and source files.

    ``processing_shape=(384,512)`` and six iterations match the official sparse
    predictor. Reduced shapes are an explicit speed experiment, not equivalent
    benchmark accuracy. Queries are kept in order; no cross-window relabelling.
    """

    def __init__(self, asset_dir='models/point_tracking/cotracker3', device='cuda',
                 *, iterations=6, processing_shape=(384, 512), maximum_frames=64):
        import torch
        shape = tuple(processing_shape)
        if (len(shape) != 2 or any(int(v) != v or v < 64 or v % 32 for v in shape)
                or int(iterations) != iterations or not 1 <= iterations <= 12
                or int(maximum_frames) != maximum_frames or maximum_frames < 2):
            raise ValueError('shape must be multiples of 32 >= 64; iterations 1..12; frame limit >= 2')
        self.assets, self.provenance, vendor, weight = _verified_assets(
            asset_dir, 'cotracker', SOURCE_COMMIT, 'scaled_offline.pth', WEIGHT_SHA256)
        sys.path.insert(0, str(vendor))
        try:
            # Upstream import calls manual_seed; restore host RNG state afterwards.
            with torch.random.fork_rng():
                module = importlib.import_module('cotracker.models.core.cotracker.cotracker3_offline')
                model = module.CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
            state = torch.load(weight, map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
        finally:
            sys.path.remove(str(vendor))
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.iterations, self.processing_shape, self.maximum_frames = int(iterations), tuple(map(int, shape)), int(maximum_frames)

    def track(self, frames, queries, *, point_ids=None, frame_indices=None, cutoff_frame=None):
        """Return positions, predicted visibility/confidence, IDs and provenance.

        ``cutoff_frame`` is an absolute saved-video frame ID; exceeding it raises
        rather than silently trimming. Query frame indices refer to this supplied
        window. Before a query's own frame, output is unavailable (NaN/False).
        """
        import torch
        import torch.nn.functional as functional
        frames, queries, ids, indices = _inputs(frames, queries, point_ids, frame_indices, cutoff_frame)
        if len(frames) > self.maximum_frames:
            raise ValueError('window exceeds maximum_frames; choose explicit shorter windows')
        t, h, w = frames.shape[:3]
        ph, pw = self.processing_shape
        video = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).to(self.device).float()
        video = functional.interpolate(video, (ph, pw), mode='bilinear', align_corners=True)[None]
        query = torch.tensor(queries.copy(), device=self.device)
        scale = query.new_tensor([(pw - 1) / (w - 1), (ph - 1) / (h - 1)])
        query[:, 1:] *= scale
        utils = importlib.import_module('cotracker.models.core.model_utils')
        support = utils.get_points_on_a_grid(6, (ph, pw), device=self.device)[0]
        support = torch.cat((torch.zeros_like(support[:, :1]), support), dim=1)
        all_queries = torch.cat((query, support), dim=0)[None]
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            xy, visible, confidence, _ = self.model(video=video, queries=all_queries, iters=self.iterations)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        positions = (xy[0, :, :len(ids)] / scale).cpu().numpy().copy()
        visibility_score = visible[0, :, :len(ids)].cpu().numpy().copy()
        confidence_score = confidence[0, :, :len(ids)].cpu().numpy().copy()
        anchor = np.zeros((t, len(ids)), bool)
        anchor[queries[:, 0].astype(int), np.arange(len(ids))] = True
        # Official predictor also pins query coordinates; expose that fact, and
        # never force its visibility or confidence to one.
        positions[queries[:, 0].astype(int), np.arange(len(ids))] = queries[:, 1:]
        before = np.arange(t)[:, None] < queries[None, :, 0]
        positions[before] = np.nan
        visibility_score[before] = np.nan
        confidence_score[before] = np.nan
        return dict(positions=positions, visibility=visibility_score > .9,
                    visibility_score=visibility_score, confidence_score=confidence_score,
                    query_anchored=anchor, point_ids=ids, frame_indices=indices,
                    observed=np.zeros((t, len(ids)), bool),
                    metadata=dict(model='CoTracker3 scaled offline', source_commit=SOURCE_COMMIT,
                                  weight_sha256=WEIGHT_SHA256, runtime_seconds=elapsed,
                                  processing_shape=list(self.processing_shape), iterations=self.iterations,
                                  coordinate_mapping='bilinear align_corners=True; (size-1) scaling',
                                  support_grid_points=36, visibility_threshold=.9,
                                  evidence='network proposal only; visibility is not measurement',
                                  cutoff_frame=None if cutoff_frame is None else int(cutoff_frame),
                                  uses_future_within_supplied_window=True))


class TAPNextCandidates:
    """Official TAPNext++ 256 model, executed causally one frame at a time.

    Supports first-frame queries only, matching the released online evaluation.
    Input pixels use integer-centre coordinates; TAP's classification bins use
    edge-origin coordinates with centres at .5. Their conversion is explicit.
    No state is shared between calls, videos or windows.
    """

    def __init__(self, asset_dir='models/point_tracking/tapnextpp', device='cuda',
                 *, maximum_frames=64):
        import torch
        if int(maximum_frames) != maximum_frames or maximum_frames < 2:
            raise ValueError('maximum_frames must be an integer >= 2')
        self.assets, self.provenance, vendor, weight = _verified_assets(
            asset_dir, 'tapnet', TAPNEXT_COMMIT, 'tapnextpp_ckpt.pt', TAPNEXT_SHA256)
        dependencies = self.assets / 'dependencies'
        sys.path[:0] = [str(vendor), str(dependencies)]
        try:
            with torch.random.fork_rng():
                module = importlib.import_module('tapnet.tapnext.tapnext_torch')
                model = module.TAPNext(image_size=(256, 256))
            checkpoint = torch.load(weight, map_location='cpu', weights_only=True, mmap=True)
            state = checkpoint.get('state_dict', checkpoint)
            model.load_state_dict({k.removeprefix('tapnext.'): v for k, v in state.items()}, strict=True)
        finally:
            sys.path.remove(str(vendor))
            sys.path.remove(str(dependencies))
        self.device, self.maximum_frames = torch.device(device), int(maximum_frames)
        self.model = model.eval().to(self.device)

    def track(self, frames, queries, *, point_ids=None, frame_indices=None, cutoff_frame=None):
        import torch
        import torch.nn.functional as functional
        frames, queries, ids, indices = _inputs(frames, queries, point_ids, frame_indices, cutoff_frame)
        if len(frames) > self.maximum_frames:
            raise ValueError('window exceeds maximum_frames; choose explicit shorter windows')
        if np.any(queries[:, 0] != 0):
            raise ValueError('TAPNext++ online adapter requires first-frame queries')
        t, h, w = frames.shape[:3]
        query = torch.tensor(queries[:, [0, 2, 1]].copy(), device=self.device)
        scale = query.new_tensor([256 / h, 256 / w])
        query[:, 1:] = (query[:, 1:] + .5) * scale
        state, positions, visibility = None, [], []
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast(
                self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
            for frame in frames:
                image = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).to(self.device).float()[None]
                image = functional.interpolate(image, (256, 256), mode='bilinear', align_corners=False)
                video = (image / 127.5 - 1).permute(0, 2, 3, 1)[None]
                yx, _, logits, state = self.model(video=video, query_points=query[None] if state is None else None,
                                                 state=state)
                xy = (yx[0, 0].float() / scale - .5).flip(-1)
                positions.append(xy.cpu().numpy())
                visibility.append(logits[0, 0, :, 0].float().sigmoid().cpu().numpy())
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        positions, scores = np.asarray(positions), np.asarray(visibility)
        anchor = np.zeros((t, len(ids)), bool)
        anchor[0] = True
        positions[0] = queries[:, 1:]
        return dict(positions=positions, visibility=scores > .5, visibility_score=scores,
                    confidence_score=np.full_like(scores, np.nan), query_anchored=anchor,
                    point_ids=ids, frame_indices=indices, observed=np.zeros_like(anchor),
                    metadata=dict(model='TAPNext++ 256 online', source_commit=TAPNEXT_COMMIT,
                                  weight_sha256=TAPNEXT_SHA256, runtime_seconds=elapsed,
                                  processing_shape=[256, 256], iterations=None,
                                  coordinate_mapping='align_corners=False; (pixel+.5)*256/size, inverse-.5; t,y,x model queries',
                                  visibility_threshold=.5, confidence_score='not emitted; NaN',
                                  evidence='network proposal only; visibility is not measurement',
                                  cutoff_frame=None if cutoff_frame is None else int(cutoff_frame),
                                  uses_future_within_supplied_window=False))
