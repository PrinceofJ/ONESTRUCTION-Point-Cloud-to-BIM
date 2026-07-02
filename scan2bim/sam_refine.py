"""Pass 2 — grounded, PROMPTED SAM refinement of the watershed proposal.

The watershed (Pass 1) already produces a geometry-correct room prior plus a wall mask
and a distance transform. This stage does NOT run SAM blindly. Instead it:

  1. **Prompts** SAM per watershed room — positive points from the room's eroded interior
     (the reliable interior), the room's bounding box, and optional negative points from
     neighbouring rooms. Every returned mask is therefore labelled *by construction*
     (it belongs to room ``i``); there is no IoU-matching guesswork.
  2. Resolves a **single-pass region-adjacency graph**: nodes are watershed rooms; a
     confident SAM mask that spans an edge votes to MERGE the two rooms, and a confident
     SAM mask that cuts through a room votes to SPLIT it. The whole graph is resolved once
     (union-find -> connected components), so the result is order-independent — unlike the
     old two-pass "merge then split".
  3. Lets SAM override the watershed ONLY where SAM is confident (its predicted IoU) AND
     the geometry is weak (the shared boundary sits on an open distance-transform ridge,
     not on wall pixels). Wall-backed boundaries are never overridden.
  4. **Snaps to the geometric scaffold**: SAM decides TOPOLOGY (how regions group / which
     side of an opening a region is on); the watershed wall mask + DT decide GEOMETRY
     (where the edges are). Output is intersected with free space, ``-1`` is re-imposed on
     every wall pixel, and a split line is placed on the DT ridge (a tiny local watershed
     seeded by the two SAM pieces) rather than on SAM's raw outline.
  5. A cheap **safety rail**: a proposed merge/split is accepted only if it does not worsen
     a sanity score (room sits over scan coverage, does not straddle walls, stays roughly
     compact). A wrong / low-confidence SAM mask cannot destroy a correct watershed result.

Model-agnostic: inference is wrapped behind a ``MaskGenerator`` adapter (now a *prompted*
segmenter: ``set_image`` + ``predict``). The default backend is SAM2; ``cfg.sam_backend``
swaps in SAM3 or SAM1 with no change to the refinement code. ``refine_with_sam`` returns
the watershed labels unchanged whenever SAM is disabled or no real model is available — it
never fabricates masks.

SAM input image (``build_sam_image``): the three rasters Notebook 1 already computed are
stacked as channels — occupancy (free space) / slab wall mask (structure) / coverage (scanned
data). This is a realistic top-down for SAM and needs no point cloud in this stage. The
colourised label map is for human QA only and is never fed to SAM.
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy import ndimage
from skimage.segmentation import watershed as _sk_watershed

from .watershed import _relabel_rooms
from .walls import estimate_wall_thickness_px, resolve_ring_radii_px


class MaskGenerator:
    """Prompted segmenter interface: set_image + predict."""

    name = 'sam'

    def set_image(self, image):  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, point_coords=None, point_labels=None, box=None,
                multimask_output=True):  # pragma: no cover - interface
        raise NotImplementedError


class _PromptPredictorAdapter(MaskGenerator):
    """Wraps SAM1/SAM2/SAM3 image predictors behind a common interface."""

    def __init__(self, predictor, name='sam'):
        self._p = predictor
        self.name = name

    def set_image(self, image):
        self._p.set_image(np.asarray(image))

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
        masks, scores, _ = self._p.predict(
            point_coords=None if point_coords is None else np.asarray(point_coords, dtype=float),
            point_labels=None if point_labels is None else np.asarray(point_labels, dtype=int),
            box=None if box is None else np.asarray(box, dtype=float),
            multimask_output=multimask_output)
        return [np.asarray(m, bool) for m in masks], [float(s) for s in scores]


def _build_sam2(cfg, device):
    """SAM2 prompted predictor."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(cfg.sam_model_cfg, cfg.sam_ckpt, device=device)
    return _PromptPredictorAdapter(SAM2ImagePredictor(model), name='sam2')


def _build_sam1(cfg, device):
    """SAM1 prompted predictor."""
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry[cfg.sam_arch](checkpoint=cfg.sam_ckpt).to(device)
    return _PromptPredictorAdapter(SamPredictor(sam), name='sam1')


def _build_sam3(cfg, device):
    """SAM3 prompted predictor (experimental)."""
    from sam3.build_sam import build_sam3                          # adapt to your build
    from sam3.sam3_image_predictor import SAM3ImagePredictor       # adapt to your build
    model = build_sam3(cfg.sam_model_cfg, cfg.sam_ckpt, device=device)
    return _PromptPredictorAdapter(SAM3ImagePredictor(model), name='sam3')


_BUILDERS = {'sam1': _build_sam1, 'sam2': _build_sam2, 'sam3': _build_sam3}


def build_mask_generator(cfg):
    """Build the prompted MaskGenerator for cfg.sam_backend."""
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backend = getattr(cfg, 'sam_backend', 'sam2')
    builder = _BUILDERS.get(backend)
    if builder is None:
        raise ValueError(f"unknown cfg.sam_backend {backend!r}; use 'sam2' | 'sam3' | 'sam1'")
    gen = builder(cfg, device)
    print(f"[sam] using backend '{gen.name}' on {device}")
    return gen


# ===========================================================================
# SAM input image — stack the already-computed N1 rasters as channels
# ===========================================================================
def build_sam_image(occ, wall_mask=None, coverage=None, mode='stack'):
    """Build the HxWx3 uint8 image fed to SAM from data already computed in Notebook 1.

    Channels (``mode='stack'``): 0 = occupancy free space (255 free / 0 wall) · 1 = slab wall
    mask (structure / barriers) · 2 = coverage (where the scan actually has data). This gives
    SAM a realistic top-down — walls read as boundaries, scanned rooms as filled regions —
    using only N1 artifacts (no point cloud needed here). ``mode='occupancy'`` (or missing
    wall_mask/coverage) falls back to replicating the binary occupancy raster into 3 channels.
    The colourised label map is NEVER used as SAM input.
    """
    occ = np.asarray(occ)
    if occ.ndim == 3:
        occ = occ[..., 0]
    free = (occ >= 128).astype(np.uint8) * 255          # occupancy.png: 255 free, 0 wall
    if mode == 'occupancy' or wall_mask is None or coverage is None:
        return np.stack([free, free, free], -1).astype(np.uint8)
    wn = np.asarray(wall_mask, bool).astype(np.uint8) * 255
    cv = np.asarray(coverage, bool).astype(np.uint8) * 255
    return np.stack([free, wn, cv], -1).astype(np.uint8)


def _erode(mask, px):
    if px <= 0:
        return np.asarray(mask, bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(np.asarray(mask, np.uint8), k).astype(bool)


def _dilate(mask, px):
    if px <= 0:
        return np.asarray(mask, bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(np.asarray(mask, np.uint8), k).astype(bool)


def sample_interior_points(mask, k):
    """Pick up to k interior points deterministically."""
    mask = np.asarray(mask, bool)
    ys, xs = np.where(mask)
    n = len(xs)
    if n == 0:
        return np.zeros((0, 2), float)
    k = max(1, min(int(k), n))
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    py, px = np.unravel_index(int(np.argmax(dt)), dt.shape)
    pts = [(float(px), float(py))]
    if k > 1:
        idx = np.unique(np.linspace(0, n - 1, k - 1).round().astype(int))
        pts.extend((float(xs[i]), float(ys[i])) for i in idx)
    return np.asarray(pts, float)


def room_prompts(labels, r, cfg, erode_px):
    """Build SAM prompts for watershed room r."""
    M = (labels == r)
    interior = _erode(M, erode_px)
    if not interior.any():
        interior = M                                    # tiny room: use the whole mask
    pos = sample_interior_points(interior, cfg.sam_pos_points)
    coords = [pos]
    plabels = [np.ones(len(pos), int)]
    if getattr(cfg, 'sam_use_neg_points', False) and cfg.sam_neg_points > 0:
        ys, xs = np.where(M)
        pad = erode_px + 2
        y0, y1 = max(0, ys.min() - pad), ys.max() + pad + 1
        x0, x1 = max(0, xs.min() - pad), xs.max() + pad + 1
        near = np.zeros_like(M)
        near[y0:y1, x0:x1] = True
        others = (labels >= 1) & (labels != r) & near
        others = _erode(others, max(1, erode_px // 2)) & others
        neg = sample_interior_points(others, cfg.sam_neg_points)
        if len(neg):
            coords.append(neg)
            plabels.append(np.zeros(len(neg), int))
    point_coords = np.concatenate(coords, 0) if any(len(c) for c in coords) else None
    point_labels = np.concatenate(plabels, 0) if point_coords is not None else None
    ys, xs = np.where(M)
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)
    return point_coords, point_labels, box


# geometry evidence on the watershed scaffold (reuses walls + DT)
def region_adjacency(labels, adj_px):
    """Unordered pairs of room ids whose masks come within ``adj_px`` of each other (so
    wall-separated neighbours are detected too, in order to be *rejected* as wall-backed)."""
    ids = [int(r) for r in np.unique(labels) if r >= 1]
    dil = {r: _dilate(labels == r, adj_px) for r in ids}
    pairs = set()
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if (dil[a] & dil[b]).any():
                pairs.add((a, b))
    return pairs


def interface_evidence(labels, walls, dt, a, b, adj_px):
    """Characterise the boundary between rooms ``a`` and ``b``.

    Returns ``wall_frac`` (fraction of the interface zone that is wall pixels) and
    ``ridge_px`` (widest open free connection between them, via the DT). A genuine opening
    has ``wall_frac`` low and ``ridge_px`` high; a wall-backed boundary has ``wall_frac``
    high and no direct free contact (``ridge_px`` ~ 0)."""
    walls = np.asarray(walls, bool)
    Ma, Mb = (labels == a), (labels == b)
    zone = _dilate(Ma, adj_px) & _dilate(Mb, adj_px)
    if not zone.any():
        return dict(wall_frac=1.0, ridge_px=0.0, zone_px=0)
    wall_frac = float((zone & walls).sum()) / float(zone.sum())
    contact = (_dilate(Ma, 1) & Mb) | (_dilate(Mb, 1) & Ma)     # a directly touching b = open
    ridge_px = float(dt[contact].max()) if contact.any() else 0.0
    return dict(wall_frac=wall_frac, ridge_px=ridge_px, zone_px=int(zone.sum()))


def room_sanity_score(mask, coverage, walls):
    """Cheap sanity in [0, 1]: a real room sits over scan coverage, does not straddle walls,
    and is roughly compact. Used by the safety rail to reject damaging merges/splits."""
    mask = np.asarray(mask, bool)
    area = int(mask.sum())
    if area == 0:
        return 0.0
    wall_frac = float((mask & np.asarray(walls, bool)).sum()) / area
    cov = 1.0 if coverage is None else float((mask & np.asarray(coverage, bool)).sum()) / area
    ys, xs = np.where(mask)
    bbox = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    extent = area / float(bbox)                         # compactness vs bounding box
    return cov * extent * (1.0 - wall_frac)


def split_room_on_dt(region, seed_a, seed_b, dt, erode_px=1):
    """Split ``region`` into two parts whose dividing line follows the DT ridge, by running
    a tiny local watershed of ``-dt`` seeded with the two SAM pieces (eroded to cores so the
    cut is decided by geometry, not by SAM's raw outline). Returns ``(part_a, part_b)`` that
    partition ``region`` exactly, or ``None`` if a clean two-way split is not possible."""
    region = np.asarray(region, bool)
    sa = np.asarray(seed_a, bool) & region
    sb = np.asarray(seed_b, bool) & region
    if erode_px > 0:
        ea, eb = _erode(sa, erode_px), _erode(sb, erode_px)
        if ea.any():
            sa = ea
        if eb.any():
            sb = eb
    if not sa.any() or not sb.any():
        return None
    markers = np.zeros(region.shape, np.int32)
    markers[sa] = 1
    markers[sb] = 2
    ws = _sk_watershed(-np.asarray(dt, float), markers, mask=region)
    part_a = (ws == 1)
    part_b = region & ~part_a                           # everything else in the region
    if not part_a.any() or not part_b.any():
        return None
    return part_a, part_b


# single-pass region-adjacency-graph relabel (the testable core, model-free)
def relabel_by_sam(labels, walls, dt, sam_room_masks, sam_room_scores, cfg, coverage=None):
    """Resolve the whole region-adjacency graph ONCE from per-room SAM masks + scores.

    ``sam_room_masks`` / ``sam_room_scores``: ``{room_id: bool HxW}`` / ``{room_id: float}``.
    These are plain arrays, so this core is fully unit-testable with hand-built fixtures and
    needs no model. Returns ``(refined_labels, debug)`` with the watershed's exact shape and
    the ``-1`` wall / ``0`` exterior / ``>=1`` room convention. No room pixel can sit on a wall.
    """
    labels = np.asarray(labels)
    walls = np.asarray(walls, bool)
    sam_room_masks = sam_room_masks or {}
    sam_room_scores = sam_room_scores or {}
    ids = [int(r) for r in np.unique(labels) if r >= 1]

    conf = cfg.sam_conf_thresh
    split_min = cfg.sam_split_min_frac
    merge_cover = cfg.sam_merge_cover_frac
    wall_frac_max = cfg.sam_wall_frac_max
    open_ridge_px = cfg.sam_open_ridge_m / cfg.pixel_m
    margin = cfg.sam_min_sanity_margin
    adj_px = max(2, estimate_wall_thickness_px(walls) + 1)

    dbg = {'merges': [], 'splits': [], 'rejected': []}

    def room_mask(r):
        m = sam_room_masks.get(r)
        return None if m is None else (np.asarray(m, bool) & ~walls)

    def room_score(r):
        return float(sam_room_scores.get(r, 0.0))

    # ---- 1 · per-room SPLIT decision -> fragments ----------------------------------
    fragments = []                                       # list of (room_id, bool mask)
    split_rooms = set()
    for r in ids:
        M = (labels == r)
        area = int(M.sum())
        sam = room_mask(r)
        did_split = False
        if sam is not None and room_score(r) >= conf and area > 0:
            covered = sam & M
            remainder = M & ~sam
            if covered.sum() >= split_min * area and remainder.sum() >= split_min * area:
                parts = split_room_on_dt(M, covered, remainder, dt)
                if parts is not None:
                    p1, p2 = parts
                    before = room_sanity_score(M, coverage, walls)
                    after = 0.5 * (room_sanity_score(p1, coverage, walls) +
                                   room_sanity_score(p2, coverage, walls))
                    if after >= before + margin:         # safety rail
                        fragments.append((r, p1))
                        fragments.append((r, p2))
                        split_rooms.add(r)
                        did_split = True
                        dbg['splits'].append(r)
                    else:
                        dbg['rejected'].append(('split', r))
        if not did_split:
            fragments.append((r, M))

    # ---- 2 · union-find over fragments ---------------------------------------------
    parent = list(range(len(fragments)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # unsplit rooms own exactly one fragment node
    room_node = {r: idx for idx, (r, _) in enumerate(fragments) if r not in split_rooms}

    # ---- 3 · inter-room MERGE votes (a SAM mask spanning a geometry-weak edge) ------
    def cover(mask, room_mask_):
        ra = int(room_mask_.sum())
        return 0.0 if ra == 0 else float((mask & room_mask_).sum()) / ra

    for (a, b) in sorted(region_adjacency(labels, adj_px)):
        if a in split_rooms or b in split_rooms:         # split rooms don't also merge this pass
            continue
        Ma, Mb = (labels == a), (labels == b)
        ma, mb = room_mask(a), room_mask(b)
        span, sconf = False, 0.0
        if ma is not None and cover(ma, Ma) >= merge_cover and cover(ma, Mb) >= merge_cover:
            span, sconf = True, max(sconf, room_score(a))
        if mb is not None and cover(mb, Ma) >= merge_cover and cover(mb, Mb) >= merge_cover:
            span, sconf = True, max(sconf, room_score(b))
        if not span or sconf < conf:
            continue
        ev = interface_evidence(labels, walls, dt, a, b, adj_px)
        weak = (ev['wall_frac'] < wall_frac_max) and (ev['ridge_px'] >= open_ridge_px)
        if not weak:                                     # never override a wall-backed boundary
            dbg['rejected'].append(('merge-geom', a, b))
            continue
        before = 0.5 * (room_sanity_score(Ma, coverage, walls) +
                        room_sanity_score(Mb, coverage, walls))
        after = room_sanity_score(Ma | Mb, coverage, walls)
        if after < before + margin:                      # safety rail
            dbg['rejected'].append(('merge-sanity', a, b))
            continue
        union(room_node[a], room_node[b])
        dbg['merges'].append((a, b))

    # ---- 4 · assemble final labels (snap to scaffold) ------------------------------
    out = np.where(walls, -1, 0).astype(np.int32)        # walls -1, everything else 0
    comp_id = {}
    nxt = 1
    for idx, (_, fmask) in enumerate(fragments):
        root = find(idx)
        if root not in comp_id:
            comp_id[root] = nxt
            nxt += 1
        out[fmask] = comp_id[root]
    out[walls] = -1                                      # hard barriers: no room on a wall pixel
    out = _relabel_rooms(out)
    return out, dbg


def refine_with_sam(geom_labels, occ_gray, walls, footprint, cfg,
                    generator=None, wall_mask=None, coverage=None, dt=None):
    """Refine watershed labels with prompted SAM. Returns ``(labels, debug)``.

    Pass-through (``debug['ran']=False``, no masks fabricated) when ``cfg.use_sam_recall``
    is False or no SAM backend can be built. Otherwise: build the SAM image from the N1
    rasters, prompt SAM once per watershed room, and resolve the region-adjacency graph via
    ``relabel_by_sam``. ``dt`` defaults to the distance transform of free space recomputed
    from ``walls`` (identical to the watershed's own DT), so Notebook 3 needs no new output.
    ``footprint`` is accepted for signature compatibility.
    """
    geom_labels = np.asarray(geom_labels)
    walls = np.asarray(walls, bool)
    dbg = {'ran': False, 'backend': None,
           'n_rooms_in': int(geom_labels.max()) if geom_labels.size else 0}

    if not getattr(cfg, 'use_sam_recall', False):
        dbg['reason'] = 'use_sam_recall is False'
        return geom_labels.copy(), dbg
    if generator is None:
        try:
            generator = build_mask_generator(cfg)
        except Exception as e:  # noqa: BLE001 - any build failure -> safe pass-through
            dbg['reason'] = f'no SAM backend: {e}'
            return geom_labels.copy(), dbg

    if dt is None:
        dt = cv2.distanceTransform((~walls).astype(np.uint8), cv2.DIST_L2, 5)
    image = build_sam_image(occ_gray, wall_mask, coverage,
                            mode=getattr(cfg, 'sam_image_mode', 'stack'))
    generator.set_image(image)

    erode_px, _ = resolve_ring_radii_px(cfg, walls)
    ids = [int(r) for r in np.unique(geom_labels) if r >= 1]
    sam_masks, sam_scores = {}, {}
    for r in ids:
        pc, pl, box = room_prompts(geom_labels, r, cfg, erode_px)
        masks, scores = generator.predict(point_coords=pc, point_labels=pl, box=box,
                                          multimask_output=True)
        if not masks:
            continue
        best = int(np.argmax(scores))
        sam_masks[r] = np.asarray(masks[best], bool) & ~walls
        sam_scores[r] = float(scores[best])

    refined, rdbg = relabel_by_sam(geom_labels, walls, dt, sam_masks, sam_scores, cfg,
                                   coverage=coverage)
    dbg.update(ran=True, backend=getattr(generator, 'name', 'sam'),
               n_sam_rooms=len(sam_masks), n_rooms_out=int(refined.max()),
               n_merges=len(rdbg['merges']), n_splits=len(rdbg['splits']),
               merges=rdbg['merges'], splits=rdbg['splits'], rejected=rdbg['rejected'])
    return refined, dbg
