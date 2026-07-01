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
