"""Method 2 — pure-SAM automatic room segmentation (Albadri et al., ISPRS 2025).

This is the project's *third arm*: where the **geometric** method seeds rooms with a
deterministic watershed and **geometric+SAM** refines that watershed with PROMPTED SAM,
this method runs SAM in automatic **"segment everything"** mode directly on the Stage-1
rasters — **no watershed prior, no prompts** — and turns the resulting masks into room
labels. It reproduces the paper's room pipeline (§3.1):

    occupancy image  ->  SamAutomaticMaskGenerator  ->  masks
                     ->  room / not-room by cross-sectional area (A = 1.5 m^2)
                     ->  (optional) outward boundary buffer (do = 1/2 wall thickness)
                     ->  (optional) corridor reprocessing on the residual free space.

Design mirrors ``sam_refine`` exactly: the ONE non-deterministic, GPU-bound step (SAM's
automatic mask generation) is isolated behind a thin :class:`AutoMaskGenerator` adapter,
and EVERYTHING else is a pure function over plain arrays. The orchestrator
(:func:`segment_rooms_sam_auto`) accepts an injected ``generator``, so tests pass a *fake*
generator returning hand-built masks — no torch, no checkpoint, no CUDA.

``AutoMaskGenerator`` is distinct from ``sam_refine.MaskGenerator``: this one takes NO
prompts (``generate(image) -> list[dict]``); that one is a prompted ``set_image`` +
``predict`` segmenter. The two never mix.

Label convention is identical to the watershed everywhere: ``-1`` wall · ``0`` exterior ·
``>=1`` rooms. No room pixel is ever placed on a wall.
"""

from __future__ import annotations

import numpy as np
import cv2
from skimage.segmentation import watershed as _sk_watershed

from .watershed import _relabel_rooms
from .sam_refine import build_sam_image


# ===========================================================================
# model abstraction — the ONLY GPU / non-deterministic piece (mockable)
# ===========================================================================
class AutoMaskGenerator:
    """Automatic 'segment everything' segmenter: ``generate(image) -> list[dict]``.

    Each returned record follows SAM's standard automatic-mask schema, of which this
    pipeline uses ``segmentation`` (bool HxW), ``predicted_iou`` (float) and ``area`` (px).
    Distinct from ``sam_refine.MaskGenerator`` (a *prompted* segmenter) — this one takes
    NO prompts, exactly like the paper's ``SamAutomaticMaskGenerator``.
    """

    name = 'sam-auto'

    def generate(self, image):  # pragma: no cover - interface
        raise NotImplementedError


class _AutoAdapter(AutoMaskGenerator):
    """Wraps any backend's automatic mask generator (SAM1 ``SamAutomaticMaskGenerator`` /
    SAM2 ``SAM2AutomaticMaskGenerator``), both of which expose ``.generate(rgb)`` returning
    the standard list-of-dict record. One adapter therefore covers every backend."""

    def __init__(self, generator, name='sam-auto'):
        self._g = generator
        self.name = name

    def generate(self, image):
        return self._g.generate(np.asarray(image))


def _amg_kwargs(cfg):
    """Map the paper's Table-1 parameters onto the backend's automatic-mask-generator
    kwargs (identical names across SAM1 / SAM2)."""
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
    """SAM2 automatic generator. Verified against github.com/facebookresearch/sam2:
    ``build_sam2(config_file, ckpt_path, device=...)`` + ``SAM2AutomaticMaskGenerator``."""
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    model = build_sam2(cfg.sam_model_cfg, cfg.sam_ckpt, device=device)
    return _AutoAdapter(SAM2AutomaticMaskGenerator(model, **_amg_kwargs(cfg)), name='sam2-auto')


def _build_auto_sam3(cfg, device):
    """TEMPORARY SAM3 adapter — mirrors ``sam_refine._build_sam3``. Targets the same
    automatic-generator shape SAM2 exposes; adapt the two imports to your SAM3 build."""
    from sam3.build_sam import build_sam3                              # adapt to your build
    from sam3.automatic_mask_generator import SAM3AutomaticMaskGenerator  # adapt to your build
    model = build_sam3(cfg.sam_model_cfg, cfg.sam_ckpt, device=device)
    return _AutoAdapter(SAM3AutomaticMaskGenerator(model, **_amg_kwargs(cfg)), name='sam3-auto')


_AUTO_BUILDERS = {'sam1': _build_auto_sam1, 'sam2': _build_auto_sam2, 'sam3': _build_auto_sam3}


def build_auto_mask_generator(cfg) -> AutoMaskGenerator:
    """Factory. ``cfg.sam_backend`` selects 'sam2' (default) | 'sam3' | 'sam1'. Returns an
    automatic ``AutoMaskGenerator`` configured with the paper's Table-1 params. Raises if
    the requested backend cannot be built (no torch / checkpoint)."""
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


# ===========================================================================
# small morphology helpers (kept local so this module stands alone)
# ===========================================================================
def _dilate(mask, px):
    if px <= 0:
        return np.asarray(mask, bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(np.asarray(mask, np.uint8), k).astype(bool)


# ===========================================================================
# deterministic core (fully unit-testable, NO model)
# ===========================================================================
def masks_to_room_labels(masks, scores, walls, coverage, cfg):
    """SAM's mask set -> int32 room labels on the ``walls`` grid.

    ``masks`` / ``scores``: parallel lists of bool HxW masks and their predicted-IoU
    scores (plain arrays, so this is testable with hand-built fixtures). Steps:

      * restrict every mask to free space (``& ~walls``) — no room pixel on a wall;
      * drop masks below ``cfg.sam_min_mask_region_area`` px (raw-mask noise floor);
      * drop masks sitting mostly OFF ``coverage`` (exterior / unscanned void), i.e. the
        big background mask SAM always produces;
      * resolve overlaps deterministically: paint the BEST mask LAST so it wins contested
        pixels. "Best" = higher ``predicted_iou``, ties by larger area, then lower input
        index. The painting order — and therefore the result — is independent of the order
        the generator returned the masks;
      * re-impose ``-1`` on every wall pixel, ``0`` on exterior, and compact ids to 1..k.

    Returns ``(labels, debug)``.
    """
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
    """Paper room / not-room step: relabel-to-0 any region whose cross-sectional area is
    below ``cfg.sam_auto_min_room_area_m2`` (the paper's ``A``). Pure; returns compacted
    labels (walls / exterior untouched)."""
    labels = np.asarray(labels)
    min_px = int(round(cfg.sam_auto_min_room_area_m2 / (cfg.pixel_m ** 2)))
    out = labels.copy()
    for r in [int(x) for x in np.unique(labels) if x >= 1]:
        if int((labels == r).sum()) < min_px:
            out[labels == r] = 0                         # too small -> not a room
    return _relabel_rooms(out)


def buffer_room_labels(labels, walls, cfg, buffer_px=None):
    """Paper boundary buffer (``do`` = 1/2 wall thickness): expand each room outward to
    reclaim adjacent free pixels WITHOUT crossing a wall or bleeding into another room.

    Pure morphological op on the label raster. Only currently-exterior (``0``, non-wall)
    pixels within ``buffer_px`` of a room are claimed; walls stay ``-1``; ties (a pixel
    reachable from two rooms) go to the geodesically nearest room (walls are barriers), so
    the result is order-independent. ``buffer_px`` defaults to ``cfg.do_buffer_px``.
    """
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
    """Paper corridor reprocessing (§4.2): re-run ``generator`` on the residual free space
    (pixels no room claimed) and merge newly-qualifying rooms in.

    The SAM image is masked to the residual so the generator only sees the leftover space;
    new masks are turned into rooms by the same deterministic core + area gate, then
    appended with fresh ids after the current maximum. Deterministic given the generator's
    masks. Returns ``(labels, debug)``.
    """
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


# ===========================================================================
# orchestrator — pure-SAM segmentation (mirrors refine_with_sam)
# ===========================================================================
def segment_rooms_sam_auto(image, walls, coverage, cfg,
                           generator=None, residual_generator=None):
    """Pure-SAM room segmentation. Returns ``(labels, debug)``.

    ``image``: the HxWx3 SAM image (build it with ``scan2bim.build_sam_image``).
    ``walls``: the wall scaffold (wallness raster) — re-imposed as hard ``-1`` barriers.
    ``coverage``: scan-coverage raster (drops the exterior background mask); may be ``None``.

    If ``generator`` is None a real backend is built via ``build_auto_mask_generator(cfg)``;
    on failure (no torch / checkpoint) NO masks are fabricated — an all-exterior label map
    is returned with ``debug['ran'] = False`` (the GPU notebook checks this and raises). The
    deterministic core therefore stays model-free for tests. Pipeline:

        generate -> masks_to_room_labels -> classify_rooms_by_area
                 -> [reprocess_residual]  (if cfg.sam_reprocess_residual)
                 -> [buffer_room_labels]  (if cfg.sam_auto_buffer_rooms)

    Label convention identical to the watershed: ``-1`` wall · ``0`` exterior · ``>=1`` rooms.
    """
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
