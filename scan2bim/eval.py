"""Ground truth builder and room-level IoU scoring."""

from __future__ import annotations

import os
import re

import numpy as np

from .raster import point_cells

STRUCTURAL_CLUTTER_CLASSES = ('wall', 'beam', 'column', 'door', 'window', 'clutter')


def annotation_class(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r'^(.*)_\d+$', stem)
    return m.group(1) if m else stem


def _load_xyz(path: str) -> np.ndarray:
    """Load XYZ columns from an S3DIS .txt file."""
    try:
        arr = np.loadtxt(path, usecols=(0, 1, 2), dtype=np.float64)
    except ValueError:
        clean = lambda s: float(re.sub(r'[^0-9eE.+-]', '',
                                       s.decode() if isinstance(s, bytes) else s))
        arr = np.loadtxt(path, usecols=(0, 1, 2), dtype=np.float64,
                         converters={0: clean, 1: clean, 2: clean})
    return arr.reshape(-1, 3) if arr.size else np.empty((0, 3), np.float64)


def load_room_interior_points(room_dir: str,
                              exclude=STRUCTURAL_CLUTTER_CLASSES):
    """Interior points of one S3DIS room, excluding structural/clutter classes."""
    ann = os.path.join(room_dir, 'Annotations')
    if not os.path.isdir(ann):
        whole = os.path.join(room_dir, os.path.basename(room_dir.rstrip(os.sep)) + '.txt')
        if os.path.isfile(whole):
            return _load_xyz(whole), set()
        return np.empty((0, 3), np.float64), set()

    parts, kept = [], set()
    exclude = set(exclude)
    for fn in sorted(os.listdir(ann)):
        if not fn.endswith('.txt'):
            continue
        cls = annotation_class(fn)
        if cls in exclude:
            continue
        p = _load_xyz(os.path.join(ann, fn))
        if len(p):
            parts.append(p)
            kept.add(cls)
    pts = np.concatenate(parts, axis=0) if parts else np.empty((0, 3), np.float64)
    return pts, kept


def build_gt_room_labels(gt_dir: str, transform: dict,
                         exclude=STRUCTURAL_CLUTTER_CLASSES):
    """Rasterise each room's interior points onto the Stage-1 grid."""
    H, W = int(transform['height']), int(transform['width'])
    ax_a, ax_b = int(transform['ax_a']), int(transform['ax_b'])
    gt_labels = np.zeros((H, W), np.int32)

    rooms = sorted(d for d in os.listdir(gt_dir)
                   if os.path.isdir(os.path.join(gt_dir, d)))
    info_rooms = []
    n_pts_total = n_inb_total = 0
    lo = np.array([np.inf, np.inf]); hi = np.array([-np.inf, -np.inf])
    for rid, name in enumerate(rooms, start=1):
        pts, kept = load_room_interior_points(os.path.join(gt_dir, name), exclude=exclude)
        if len(pts):
            row, col, inb = point_cells(pts, transform)
            gt_labels[row[inb], col[inb]] = rid
            ab = pts[:, [ax_a, ax_b]]
            lo = np.minimum(lo, ab.min(0)); hi = np.maximum(hi, ab.max(0))
            n_inb = int(inb.sum())
        else:
            n_inb = 0
        n_pts_total += len(pts); n_inb_total += n_inb
        info_rooms.append(dict(name=name, id=rid, n_pts=int(len(pts)), n_inb=n_inb,
                               frac=float(n_inb / len(pts)) if len(pts) else 0.0,
                               classes=sorted(kept)))
    info = dict(rooms=info_rooms, n_pts_total=int(n_pts_total), n_inb_total=int(n_inb_total),
                ingrid_frac=float(n_inb_total / max(1, n_pts_total)),
                gt_bbox=(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])),
                exclude=tuple(exclude))
    return gt_labels, info


def load_gt_room_points(gt_dir: str, exclude=STRUCTURAL_CLUTTER_CLASSES):
    """All GT interior points + their 1-based room ids, for the point-based paper scorer.

    Shares ``load_room_interior_points`` and the sorted-folder / 1-based id convention with
    ``build_gt_room_labels``, so ``score_rooms_paper`` and the rasterised GT agree on which
    room is which. Rooms with no interior points are skipped (they still consume an id, so
    ids stay aligned with the raster builder). Returns ``(points (N,3) float64,
    room_ids (N,) int32)``."""
    rooms = sorted(d for d in os.listdir(gt_dir)
                   if os.path.isdir(os.path.join(gt_dir, d)))
    pts_parts, id_parts = [], []
    for rid, name in enumerate(rooms, start=1):
        pts, _ = load_room_interior_points(os.path.join(gt_dir, name), exclude=exclude)
        if len(pts):
            pts_parts.append(pts)
            id_parts.append(np.full(len(pts), rid, np.int32))
    if not pts_parts:
        return np.empty((0, 3), np.float64), np.empty((0,), np.int32)
    return np.concatenate(pts_parts, axis=0), np.concatenate(id_parts, axis=0)


# ---------------------------------------------------------------------------
# Paper IoU + p_i/g_i matching (Albadri et al. 2025, Eq. 3/4/6/7)
# ---------------------------------------------------------------------------
def _room_ids(labels) -> np.ndarray:
    return np.array([int(v) for v in np.unique(labels) if v >= 1], dtype=np.int64)


def overlap_stats(pred_labels, gt_labels, valid=None):
    """Per (pred, gt) overlap counts on the shared grid."""
    pred_labels = np.asarray(pred_labels)
    gt_labels = np.asarray(gt_labels)
    if valid is None:
        valid = np.ones(pred_labels.shape, bool)
    else:
        valid = np.asarray(valid, bool)
    pred_ids, gt_ids = _room_ids(pred_labels), _room_ids(gt_labels)
    P, G = len(pred_ids), len(gt_ids)
    inter = np.zeros((P, G), np.int64)
    both = valid & (pred_labels >= 1) & (gt_labels >= 1)
    if both.any() and P and G:
        # ids are sorted (np.unique) -> searchsorted maps a label to its row/col index.
        pr = np.searchsorted(pred_ids, pred_labels[both])
        gr = np.searchsorted(gt_ids, gt_labels[both])
        np.add.at(inter, (pr, gr), 1)
    pred_area = np.array([int((valid & (pred_labels == v)).sum()) for v in pred_ids], np.int64)
    gt_area = np.array([int((valid & (gt_labels == v)).sum()) for v in gt_ids], np.int64)
    return pred_ids, gt_ids, inter, pred_area, gt_area


def score_rooms(pred_labels, gt_labels, wall_mask=None, match_frac=0.5):
    """Room-level IoU scoring with over/under-segmentation counts."""
    pred_labels = np.asarray(pred_labels)
    gt_labels = np.asarray(gt_labels)
    if pred_labels.shape != gt_labels.shape:
        raise ValueError(f"pred {pred_labels.shape} and gt {gt_labels.shape} must share the grid")
    valid = None if wall_mask is None else ~np.asarray(wall_mask, bool)
    pred_ids, gt_ids, inter, pred_area, gt_area = overlap_stats(pred_labels, gt_labels, valid)
    P, G = len(pred_ids), len(gt_ids)

    with np.errstate(divide='ignore', invalid='ignore'):
        union = pred_area[:, None] + gt_area[None, :] - inter
        iou = np.where(union > 0, inter / union, 0.0)                  # Eq. 6
        p = np.where(pred_area[:, None] > 0, inter / pred_area[:, None], 0.0)  # Eq. 3
        g = np.where(gt_area[None, :] > 0, inter / gt_area[None, :], 0.0)      # Eq. 4

    match = (p >= match_frac) | (g >= match_frac)
    over_seg = int((match.sum(axis=0) >= 2).sum()) if P and G else 0   # preds per GT room
    under_seg = int((match.sum(axis=1) >= 2).sum()) if P and G else 0  # GT rooms per pred

    per_room, matched_pairs = [], []
    for j, gid in enumerate(gt_ids):
        if P:
            i = int(np.argmax(iou[:, j]))
            best_iou = float(iou[i, j])
            best_pred = int(pred_ids[i]) if best_iou > 0 else None
        else:
            best_iou, best_pred = 0.0, None
        per_room.append(dict(gt_id=int(gid), pred_id=best_pred, iou=best_iou))
        if best_pred is not None:
            matched_pairs.append(dict(gt_id=int(gid), pred_id=best_pred, iou=best_iou))

    mean_iou = float(np.mean([r['iou'] for r in per_room])) if per_room else 0.0
    return dict(mean_iou=mean_iou, per_room=per_room, matched_pairs=matched_pairs,
                over_seg=over_seg, under_seg=under_seg,
                n_pred_rooms=int(P), n_gt_rooms=int(G), match_frac=float(match_frac))


# ---------------------------------------------------------------------------
# Point-based paper protocol (Albadri et al. 2025, Eq. 2-7) — research-fixes Task 02b
#
# The paper's IoU counts POINTS on the X-Y projection, and the mean is over MATCHED
# ("corresponding") rooms only, matched at p_i/g_i >= 0.75 (Eq. 5). ``score_rooms`` above is
# pixel-based and averages over ALL GT rooms (miss = 0), so its number is NOT comparable to
# the paper. Here each GT interior point is labelled by the predicted room of the cell it
# lands in (``point_cells``); the shared universe is the GT interior points. This removes the
# raster pixel-size confound.
#
# Documented deviation: the paper's P_i is the prediction's OWN retrieved cloud; we use the
# GT-point universe (one shared universe both sides) — cleaner and feasible from a raster
# prediction.
# ---------------------------------------------------------------------------
def point_room_overlap(points, gt_room_ids, pred_labels, transform):
    """Point-count overlap between predicted rooms and GT rooms over the GT-point universe.

    Each point is assigned the predicted room of the cell it lands in (``point_cells``);
    points out of grid or on exterior (0) / wall (-1) cells get predicted room 0 (no room).
    Returns ``(pred_ids, gt_ids, inter, pred_area, gt_area)``: sorted >=1 room ids,
    ``inter[i, j]`` = # points with pred==pred_ids[i] and gt==gt_ids[j], and the ``*_area``
    vectors count each room's points in the universe."""
    points = np.asarray(points, np.float64)
    gt_room_ids = np.asarray(gt_room_ids).astype(np.int64).reshape(-1)
    pred_labels = np.asarray(pred_labels)

    # predicted room per point (0 where out-of-grid / exterior / wall)
    pred_pt = np.zeros(len(points), np.int64)
    if len(points):
        row, col, inb = point_cells(points, transform)
        vals = pred_labels[row[inb], col[inb]]
        pred_pt[inb] = np.where(vals >= 1, vals, 0)

    gt_ids = np.array([int(v) for v in np.unique(gt_room_ids) if v >= 1], np.int64)
    pred_ids = np.array([int(v) for v in np.unique(pred_pt) if v >= 1], np.int64)
    P, G = len(pred_ids), len(gt_ids)
    inter = np.zeros((P, G), np.int64)
    both = (pred_pt >= 1) & (gt_room_ids >= 1)
    if both.any() and P and G:
        # ids are sorted (np.unique) -> searchsorted maps a label to its row/col index.
        pr = np.searchsorted(pred_ids, pred_pt[both])
        gr = np.searchsorted(gt_ids, gt_room_ids[both])
        np.add.at(inter, (pr, gr), 1)
    pred_area = np.array([int((pred_pt == v).sum()) for v in pred_ids], np.int64)
    gt_area = np.array([int((gt_room_ids == v).sum()) for v in gt_ids], np.int64)
    return pred_ids, gt_ids, inter, pred_area, gt_area


def score_rooms_paper(points, gt_room_ids, pred_labels, transform, th=0.75):
    """Albadri et al. 2025 room metric, point-based (Eq. 2-7), matched at ``th`` (default 0.75).

    Matching (Eq. 3/4/5): ``p_ij = |I_ij|/|P_i|``, ``g_ij = |I_ij|/|G_j|``; pred ``i`` and GT
    ``j`` are *corresponding* iff ``p_ij >= th`` OR ``g_ij >= th``. IoU (Eq. 6) is computed per
    pair; each GT room's score is the best IoU among its corresponding predictions. The
    headline mean (Eq. 7) is over **matched GT rooms only** — a completely-missed room lowers
    ``detection_rate``, not the mean (the key difference from ``score_rooms``, which averages
    over all GT rooms with miss = 0).

      ``over_seg``  = # GT rooms with >= 2 corresponding preds (one room split).
      ``under_seg`` = # preds corresponding to >= 2 GT rooms (rooms merged).

    Returns a JSON-serialisable dict: ``mean_iou_matched``, ``detection_rate``, ``over_seg``,
    ``under_seg``, ``n_gt``, ``n_seg``, ``n_matched``, ``per_room`` [{gt_id, pred_id, iou,
    matched}], ``th``."""
    pred_ids, gt_ids, inter, pred_area, gt_area = point_room_overlap(
        points, gt_room_ids, pred_labels, transform)
    P, G = len(pred_ids), len(gt_ids)

    with np.errstate(divide='ignore', invalid='ignore'):
        union = pred_area[:, None] + gt_area[None, :] - inter
        iou = np.where(union > 0, inter / union, 0.0)                          # Eq. 6
        p = np.where(pred_area[:, None] > 0, inter / pred_area[:, None], 0.0)  # Eq. 3
        g = np.where(gt_area[None, :] > 0, inter / gt_area[None, :], 0.0)      # Eq. 4

    match = (p >= th) | (g >= th)                                             # Eq. 5
    over_seg = int((match.sum(axis=0) >= 2).sum()) if P and G else 0          # preds per GT room
    under_seg = int((match.sum(axis=1) >= 2).sum()) if P and G else 0         # GT rooms per pred

    per_room, matched_ious = [], []
    for j, gid in enumerate(gt_ids):
        corr = np.nonzero(match[:, j])[0] if P else np.empty(0, np.int64)
        if len(corr):
            i = int(corr[np.argmax(iou[corr, j])])      # best-IoU corresponding pred
            best_iou, best_pred, is_matched = float(iou[i, j]), int(pred_ids[i]), True
            matched_ious.append(best_iou)
        else:
            best_iou, best_pred, is_matched = 0.0, None, False
        per_room.append(dict(gt_id=int(gid), pred_id=best_pred, iou=best_iou,
                             matched=is_matched))

    n_matched = len(matched_ious)
    return dict(mean_iou_matched=float(np.mean(matched_ious)) if n_matched else 0.0,
                detection_rate=float(n_matched / G) if G else 0.0,
                over_seg=over_seg, under_seg=under_seg,
                n_gt=int(G), n_seg=int(P), n_matched=int(n_matched),
                per_room=per_room, th=float(th))


# ---------------------------------------------------------------------------
# Harmonized comparison — fair filters + shared wall scaffold (research-fixes Task 05)
#
# The three methods drop rooms with different area/void rules and historically used different
# wall masks, so a raw head-to-head partly measures the post-filter, not the segmenter. These
# two pure functions make it apples-to-apples:
#   eval_wall_scaffold    -> the ONE canonical wall scaffold (cleaned slab-occupancy mask),
#                            shared by GT (Task 02), the room metric, and the wall metric (04).
#   harmonize_room_labels -> apply ONE area threshold + ONE void rule + that scaffold to a
#                            method's labels, then relabel 1..k. Method-agnostic, deterministic.
# ---------------------------------------------------------------------------
def eval_wall_scaffold(wall_mask, cfg):
    """The canonical evaluation wall scaffold: the cleaned slab-occupancy wall mask.

    Deterministic and method-agnostic (it comes from Stage-1, not from any segmenter), so GT
    and every prediction exclude the *same* wall pixels. Tiny speckle components are dropped
    with the watershed's own ``min_wall_area_px`` so the scaffold matches the wall set the
    watershed segments against. Returns a bool (H, W) mask."""
    from .walls import clean_wall_mask
    return clean_wall_mask(np.asarray(wall_mask, bool),
                           min_wall_area=int(getattr(cfg, 'min_wall_area_px', 60)))


def harmonize_room_labels(labels, coverage, walls, cfg):
    """Apply ONE shared post-filter to a method's room labels (research-fixes Task 05).

    Makes the three methods comparable by filtering every method's labels identically *before*
    any room/wall metric, removing the confound that each method drops rooms with its own
    area/void rules and wall set. Pure and deterministic — same inputs, same output, regardless
    of which method produced ``labels``.

    Steps (``cfg.eval_profile == 'comparison'``):
      1. Impose the ONE canonical wall scaffold ``walls``: those cells become ``-1``; any cell a
         method marked ``-1`` that is NOT a canonical wall becomes ``0`` (no room) — so the
         ``-1`` set is identical across methods.
      2. Drop rooms below ``cfg.eval_min_room_area_m2`` (one area threshold for all).
      3. Void rejection: drop rooms whose interior coverage ``< cfg.eval_min_coverage_frac``
         (one void rule for all). ``coverage`` may be ``None`` to skip this step.
      4. Relabel the survivors ``1..k`` (exterior ``0`` / wall ``-1`` untouched).

    With ``cfg.eval_profile == 'paper'`` the labels are returned unchanged (each method keeps
    its own standalone filtering — the paper-faithful profile). ``walls`` is the bool scaffold
    from ``eval_wall_scaffold``; ``coverage`` the Stage-1 coverage raster. Returns int32 (H, W)
    with the ``-1`` wall / ``0`` exterior / ``>=1`` room convention."""
    labels = np.asarray(labels).astype(np.int32)
    if getattr(cfg, 'eval_profile', 'comparison') != 'comparison':
        return labels.copy()

    walls = np.asarray(walls, bool)
    out = labels.copy()
    # 1. one shared -1 scaffold: a method-wall that isn't a canonical wall -> 0 (no room)
    out[(out == -1) & ~walls] = 0
    out[walls] = -1

    # 2. one area threshold (in pixels on the shared grid)
    min_area_px = int(round(cfg.eval_min_room_area_m2 / (cfg.pixel_m ** 2)))
    for r in [int(x) for x in np.unique(out) if x >= 1]:
        m = (out == r)
        if int(m.sum()) < min_area_px:
            out[m] = 0

    # 3. one void rule (a real room sits over scanned coverage)
    if coverage is not None:
        cov = np.asarray(coverage, bool)
        thr = float(cfg.eval_min_coverage_frac)
        for r in [int(x) for x in np.unique(out) if x >= 1]:
            m = (out == r)
            frac = float((m & cov).sum() / max(1, int(m.sum())))
            if frac < thr:
                out[m] = 0

    # 4. relabel survivors 1..k (sorted ids -> deterministic)
    relabeled = out.copy()
    for new, r in enumerate([int(x) for x in np.unique(out) if x >= 1], start=1):
        relabeled[out == r] = new
    return relabeled


# ---------------------------------------------------------------------------
# Method wiring — each method reads its OWN stage; never alias two methods (Task 03)
# ---------------------------------------------------------------------------
def load_method_labels(out_root):
    from . import artifacts as A

    def _load(stage, fname):
        try:
            d = A.load_stage_dir(out_root, stage)
        except FileNotFoundError:
            return None
        path = os.path.join(d, fname)
        return A.load_npy(path).astype('int32') if os.path.isfile(path) else None

    return {
        'Geometry':       _load(A.STAGE2, A.ROOM_LABELS_NPY),
        'SAM':            _load(A.STAGE_SAM_AUTO, A.ROOM_LABELS_NPY),
        'Geometry + SAM': _load(A.STAGE4, A.REFINED_LABELS_NPY),
    }
