"""Room-segmentation ground truth + paper-faithful room IoU (research-fixes Task 02).

Everything here is pure and grid-aligned to the Stage-1 ``transform`` so the notebooks stay
thin drivers. Two pieces:

1. **Clean GT builder** (``build_gt_room_labels``). The GT room region is *interior space*:
   each S3DIS room's points MINUS the structural + clutter classes
   (``wall``/``beam``/``column``/``door``/``window``/``clutter`` — see
   ``STRUCTURAL_CLUTTER_CLASSES``). Kept classes (``floor``/``ceiling``/``board``/furniture/…)
   define the indoor area. This matches Albadri et al. (ISPRS 2025) — "the paper segments rooms
   by indoor area" — and, crucially, matches every method's ``-1`` wall convention, so the GT
   never penalises a method for correctly marking a wall as not-room.

2. **Paper IoU + matching** (``score_rooms``). The point-based 2D IoU on the X-Y projection
   (Eq. 6) becomes a *pixel*-based IoU on the shared Stage-1 grid (each occupied cell is one
   projected point). ``p_i``/``g_i`` ratios (Eq. 3/4) pair over-/under-segmented predicted rooms
   to the right GT room; the report is **mean IoU over rooms** (Eq. 7) plus over-/under-seg
   counts (the watershed-vs-SAM failure modes).

The **shared wall scaffold** (Task 04/05): pass the Stage-1 ``wall_mask`` so GT and *every*
prediction exclude the *same* wall pixels — neither side is scored on wall handling.
"""

from __future__ import annotations

import os
import re

import numpy as np

from .raster import point_cells

# ---------------------------------------------------------------------------
# S3DIS classes that are NOT part of the interior room area.
#   structural : wall / beam / column / door / window  (the methods mark these -1)
#   clutter    : clutter                               (not interior floor area)
# Everything else a room folder contains (floor, ceiling, board, bookcase, chair, sofa,
# table, …) IS kept — it sits inside the room and defines the indoor footprint.
# ---------------------------------------------------------------------------
STRUCTURAL_CLUTTER_CLASSES = ('wall', 'beam', 'column', 'door', 'window', 'clutter')


def annotation_class(filename: str) -> str:
    """Map an S3DIS annotation filename to its class prefix.

    ``'clutter_10.txt' -> 'clutter'``, ``'floor_1.txt' -> 'floor'``. S3DIS names every part
    ``<class>_<n>.txt``; the class is everything before the final ``_<n>``."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r'^(.*)_\d+$', stem)
    return m.group(1) if m else stem


def _load_xyz(path: str) -> np.ndarray:
    """Load the leading X Y Z columns of an S3DIS ``.txt`` file as ``(N, 3)`` float64.

    A few S3DIS rows carry a stray control char (e.g. ``'-9.1\\x16'``); on the fast path's
    failure, re-read with a per-field sanitising converter (slower, used only as needed)."""
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
    """Interior points of one S3DIS room: every ``Annotations/<class>_*.txt`` whose class is
    NOT in ``exclude``, concatenated. Returns ``(points (N,3), kept_classes set)``.

    Falls back to the whole-room ``<room>/<room>.txt`` only if there is no ``Annotations/``
    directory (so the builder still runs on a GT model without per-class splits — but then it
    cannot exclude walls/clutter; the returned ``kept_classes`` is empty to flag that)."""
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
    """Rasterise each room's INTERIOR points onto the Stage-1 grid given by ``transform``.

    Rooms get a 1-based id in sorted folder order (matching ``room_labels.npy``'s
    ``>=1 == room`` convention). Pixels no room hits stay ``0`` (exterior); walls are NOT
    marked ``-1`` here — the GT is room membership only, and the shared wall scaffold is
    applied at scoring time.

    Returns ``(gt_labels int32 (H,W), info dict)``. ``info`` carries everything the
    frame-alignment gate and a QA print need:
      ``rooms``           : list of per-room dicts (name, id, n_pts, n_inb, frac, classes)
      ``n_pts_total``     : total GT points across rooms
      ``n_inb_total``     : how many back-projected inside the grid
      ``ingrid_frac``     : n_inb_total / n_pts_total  (the §01 alignment gate input)
      ``gt_bbox``         : (a_min, b_min, a_max, b_max) world bbox over the grid's two axes
      ``exclude``         : the excluded class tuple (for provenance)
    """
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


# ---------------------------------------------------------------------------
# Paper IoU + p_i/g_i matching (Albadri et al. 2025, Eq. 3/4/6/7)
# ---------------------------------------------------------------------------
def _room_ids(labels) -> np.ndarray:
    return np.array([int(v) for v in np.unique(labels) if v >= 1], dtype=np.int64)


def overlap_stats(pred_labels, gt_labels, valid=None):
    """Per (pred, gt) overlap on the shared grid, restricted to ``valid`` cells.

    Returns ``(pred_ids, gt_ids, inter, pred_area, gt_area)`` where ``inter[i, j]`` is the
    number of valid cells with ``pred==pred_ids[i] and gt==gt_ids[j]`` and the ``*_area``
    vectors count each room's valid cells. ``valid`` (bool (H,W)) is the shared wall scaffold's
    complement; ``None`` means all cells count."""
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
    """Score a prediction against the GT with the paper's room IoU + matching.

    ``wall_mask`` (bool (H,W)) is the **shared wall scaffold**: those cells are excluded from
    both sides, so neither GT nor prediction is scored on wall handling (Task 04/05). Pass the
    Stage-1 ``wall_mask`` so every method excludes the *same* pixels.

    Matching (Eq. 3/4): ``p_ij = |I_ij| / |P_i|`` and ``g_ij = |I_ij| / |G_j|``; a (pred, gt)
    pair is a *match candidate* when ``p_ij >= match_frac`` (the prediction is mostly inside that
    GT room) OR ``g_ij >= match_frac`` (that GT room is mostly inside the prediction). From this:
      - **over-seg**  = # GT rooms covered by >= 2 predictions (one room split apart);
      - **under-seg** = # predictions covering >= 2 GT rooms (two rooms merged).

    Mean IoU (Eq. 6/7): each GT room is scored by its best-IoU prediction (0 if none overlaps,
    so a missed room and a merge both drag the mean down); the report is the mean over GT rooms.

    Returns a JSON-serialisable dict:
      ``mean_iou``, ``per_room`` [{gt_id, pred_id, iou}], ``matched_pairs`` (the non-zero ones),
      ``over_seg``, ``under_seg``, ``n_pred_rooms``, ``n_gt_rooms``, ``match_frac``.
    """
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
# Method wiring — each method reads its OWN stage; never alias two methods (Task 03)
# ---------------------------------------------------------------------------
def load_method_labels(out_root):
    """Resolve each segmentation method to its own label map. NEVER alias two methods.

    'Geometry'       <- stage2_watershed/room_labels.npy
    'SAM'            <- stage_sam_auto/room_labels.npy        (pure automatic SAM, Method 2)
    'Geometry + SAM' <- stage4_sam_refined/room_labels_refined.npy  (watershed refined by SAM)

    A method whose stage is absent maps to ``None`` ('not run'), so the caller can skip it
    cleanly. Returns an ordered ``{method_name: labels or None}`` dict."""
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
