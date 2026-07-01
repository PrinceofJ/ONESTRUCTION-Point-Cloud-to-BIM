"""Pure-SAM automatic room segmentation."""

from __future__ import annotations

import numpy as np
import cv2
from skimage.segmentation import watershed as _sk_watershed

from .watershed import _relabel_rooms
from .sam_refine import build_sam_image


class AutoMaskGenerator:
    """Automatic segment-everything interface: generate(image) -> list[dict]."""

    name = 'sam-auto'

    def generate(self, image):  # pragma: no cover - interface
        raise NotImplementedError


class _AutoAdapter(AutoMaskGenerator):
    """Wraps SAM1/SAM2/SAM3 automatic mask generators."""

    def __init__(self, generator, name='sam-auto'):
        self._g = generator
        self.name = name

    def generate(self, image):
        return self._g.generate(np.asarray(image))


def _amg_kwargs(cfg):
    """Table-1 AMG kwargs."""
    return dict(
        points_per_side=int(cfg.sam_points_per_side),
        pred_iou_thresh=float(cfg.sam_pred_iou_thresh),
        stability_score_thresh=float(cfg.sam_stability_score_thresh),
        crop_n_layers=int(cfg.sam_crop_n_layers),
        crop_n_points_downscale_factor=int(cfg.sam_crop_n_points_downscale_factor),
        min_mask_region_area=int(cfg.sam_min_mask_region_area),
    )


def _build_auto_sam1(cfg, device):
    """Original Segment-Anything automatic generator."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry[cfg.sam_arch](checkpoint=cfg.sam_ckpt).to(device)
    return _AutoAdapter(SamAutomaticMaskGenerator(sam, **_amg_kwargs(cfg)), name='sam1-auto')


def _build_auto_sam2(cfg, device):
    """SAM2 automatic generator."""
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    model = build_sam2(cfg.sam_model_cfg, cfg.sam_ckpt, device=device)
    return _AutoAdapter(SAM2AutomaticMaskGenerator(model, **_amg_kwargs(cfg)), name='sam2-auto')


def _build_auto_sam3(cfg, device):
    """SAM3 automatic generator (experimental)."""
    from sam3.build_sam import build_sam3                              # adapt to your build
    from sam3.automatic_mask_generator import SAM3AutomaticMaskGenerator  # adapt to your build
    model = build_sam3(cfg.sam_model_cfg, cfg.sam_ckpt, device=device)
    return _AutoAdapter(SAM3AutomaticMaskGenerator(model, **_amg_kwargs(cfg)), name='sam3-auto')


_AUTO_BUILDERS = {'sam1': _build_auto_sam1, 'sam2': _build_auto_sam2, 'sam3': _build_auto_sam3}


def build_auto_mask_generator(cfg) -> AutoMaskGenerator:
    """Build the automatic AutoMaskGenerator for cfg.sam_backend."""
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backend = getattr(cfg, 'sam_backend', 'sam2')
    builder = _AUTO_BUILDERS.get(backend)
    if builder is None:
        raise ValueError(f"unknown cfg.sam_backend {backend!r}; use 'sam2' | 'sam3' | 'sam1'")
    gen = builder(cfg, device)
    print(f"[sam-auto] using backend '{gen.name}' on {device} "
          f"(points_per_side={cfg.sam_points_per_side})")
    return gen


def _dilate(mask, px):
    if px <= 0:
        return np.asarray(mask, bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(np.asarray(mask, np.uint8), k).astype(bool)


def masks_to_room_labels(masks, scores, walls, coverage, cfg):
    """Convert SAM masks to int32 room labels. Returns (labels, debug)."""
    walls = np.asarray(walls, bool)
    H, W = walls.shape
    cov = None if coverage is None else np.asarray(coverage, bool)
    min_area = int(getattr(cfg, 'sam_min_mask_region_area', 0))
    min_cov = float(getattr(cfg, 'sam_auto_min_coverage_frac', 0.0))

    cand = []                                            # (score, area, idx, mask_on_free)
    n_void = 0
    for i, (m, s) in enumerate(zip(masks, scores)):
        mm = np.asarray(m, bool)
        if mm.shape != walls.shape:
            raise ValueError(f"mask {i} shape {mm.shape} != grid {walls.shape}")
        mm = mm & ~walls                                 # rooms never sit on walls
        area = int(mm.sum())
        if area == 0 or area < min_area:
            continue
        if cov is not None and min_cov > 0.0:
            if float((mm & cov).sum()) / area < min_cov:  # mostly unscanned void / exterior
                n_void += 1
                continue
        cand.append((float(s), area, i, mm))

    # paint best LAST: sort ascending by (score, area, -idx) so the highest-score / largest
    # / lowest-index mask is painted last and wins every overlap. Order-independent.
    cand.sort(key=lambda t: (t[0], t[1], -t[2]))
    out = np.zeros((H, W), np.int32)                     # 0 = exterior / unclaimed
    for rank, (_s, _area, _i, mm) in enumerate(cand, start=1):
        out[mm] = rank                                   # later (better) overwrites earlier
    out[walls] = -1                                      # hard barriers
    out = _relabel_rooms(out)
    return out, dict(n_in=len(masks), n_kept=len(cand), n_void_dropped=n_void)


def classify_rooms_by_area(labels, cfg):
    """Drop rooms below the minimum area threshold."""
    labels = np.asarray(labels)
    min_px = int(round(cfg.sam_auto_min_room_area_m2 / (cfg.pixel_m ** 2)))
    out = labels.copy()
    for r in [int(x) for x in np.unique(labels) if x >= 1]:
        if int((labels == r).sum()) < min_px:
            out[labels == r] = 0                         # too small -> not a room
    return _relabel_rooms(out)


def buffer_room_labels(labels, walls, cfg, buffer_px=None):
    """Expand rooms outward into unclaimed free space, respecting walls."""
    labels = np.asarray(labels).astype(np.int32)
    walls = np.asarray(walls, bool)
    px = cfg.do_buffer_px if buffer_px is None else int(buffer_px)
    ids = [int(x) for x in np.unique(labels) if x >= 1]
    if px <= 0 or not ids:
        return labels.copy()

    rooms = labels >= 1
    claimable = _dilate(rooms, px) & (labels == 0) & ~walls   # caps buffer to px, off walls
    if not claimable.any():
        return labels.copy()
    grow_region = rooms | claimable                            # walls excluded => barriers
    markers = np.where(rooms, labels, 0).astype(np.int32)
    # watershed of a flat image grows each room marker by geodesic distance inside
    # grow_region; ties resolved deterministically by skimage's priority queue.
    ws = _sk_watershed(np.zeros(labels.shape, np.float64), markers, mask=grow_region)
    out = labels.copy()
    claim = claimable & (ws >= 1)
    out[claim] = ws[claim]
    out[walls] = -1
    return out


def reprocess_residual(labels, image, walls, coverage, cfg, generator):
    """Re-run generator on residual free space and merge new rooms in."""
    labels = np.asarray(labels).astype(np.int32)
    walls = np.asarray(walls, bool)
    residual = (labels == 0) & ~walls                    # unclaimed free space (corridors)
    if not residual.any():
        return labels.copy(), dict(ran=False, reason='no residual free space', n_added=0)

    img = np.asarray(image).copy()
    img[~residual] = 0                                   # show SAM only the leftover space
    records = generator.generate(img)
    masks = [np.asarray(r['segmentation'], bool) & residual for r in records]
    scores = [float(r.get('predicted_iou', 1.0)) for r in records]

    new_labels, _ = masks_to_room_labels(masks, scores, walls, coverage, cfg)
    new_labels = classify_rooms_by_area(new_labels, cfg)

    out = labels.copy()
    base = max([int(x) for x in np.unique(labels) if x >= 1] or [0])
    added = 0
    for r in [int(x) for x in np.unique(new_labels) if x >= 1]:
        region = (new_labels == r) & residual & (out == 0)
        if region.any():
            added += 1
            out[region] = base + added                   # fresh id after the current max
    out[walls] = -1
    return _relabel_rooms(out), dict(ran=True, n_added=added)


def segment_rooms_sam_auto(image, walls, coverage, cfg,
                           generator=None, residual_generator=None):
    """Pure-SAM room segmentation. Returns (labels, debug)."""
    walls = np.asarray(walls, bool)
    dbg = {'ran': False, 'backend': None}

    built_here = False
    if generator is None:
        try:
            generator = build_auto_mask_generator(cfg)
            built_here = True
        except Exception as e:  # noqa: BLE001 - any build failure -> clear, no fabrication
            dbg['reason'] = f'no SAM backend: {e}'
            return np.where(walls, -1, 0).astype(np.int32), dbg

    records = generator.generate(np.asarray(image))
    masks = [np.asarray(r['segmentation'], bool) for r in records]
    scores = [float(r.get('predicted_iou', 1.0)) for r in records]

    labels, mdbg = masks_to_room_labels(masks, scores, walls, coverage, cfg)
    labels = classify_rooms_by_area(labels, cfg)
    n_rooms_pass1 = int(labels.max()) if labels.size else 0

    res_dbg = {'ran': False, 'n_added': 0}
    if getattr(cfg, 'sam_reprocess_residual', False):
        rg = residual_generator
        if rg is None:
            if built_here:                               # real run: sparser grid on residual
                import dataclasses
                rg = build_auto_mask_generator(dataclasses.replace(
                    cfg, sam_points_per_side=cfg.sam_residual_points_per_side))
            else:                                        # injected (test): reuse the fake
                rg = generator
        labels, res_dbg = reprocess_residual(labels, image, walls, coverage, cfg, rg)

    if getattr(cfg, 'sam_auto_buffer_rooms', False):
        labels = buffer_room_labels(labels, walls, cfg)

    labels[walls] = -1                                   # walls are sacred
    labels = _relabel_rooms(labels)
    dbg.update(ran=True, backend=getattr(generator, 'name', 'sam-auto'),
               n_masks=len(masks), n_kept=mdbg['n_kept'],
               n_void_dropped=mdbg['n_void_dropped'],
               n_rooms_pass1=n_rooms_pass1, n_rooms_out=int(labels.max()) if labels.size else 0,
               reprocess=res_dbg)
    return labels, dbg
