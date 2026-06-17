"""Pass 1 — deterministic distance-transform watershed (section 6), ported verbatim.

No algorithmic change: this is the same high-precision segmenter, only relocated into
the package so Notebook 3 is a thin driver and Notebooks 2/4 can consume its labels.
Label convention: ``-1`` wall · ``0`` exterior · ``>=1`` rooms.

``supervision`` is imported lazily inside ``labels_to_detections`` so the core watershed
runs with no optional deps.
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy import ndimage
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from .walls import clean_wall_mask, _building_footprint


def _relabel_rooms(labels):
    """Compact room ids to 1..k after some were dropped (walls/exterior untouched)."""
    out = labels.copy()
    for new, r in enumerate([int(x) for x in np.unique(labels) if x >= 1], start=1):
        out[labels == r] = new
    return out


def merge_wide_connections(labels, dt, merge_ridge_m, pixel_m):
    """Merge adjacent room basins whose widest shared free connection exceeds
    merge_ridge_m (a genuine doorway is narrower; a spurious split runs through open floor)."""
    ridge_px = merge_ridge_m / pixel_m
    ids = [int(r) for r in np.unique(labels) if r >= 1]
    if len(ids) < 2:
        return labels
    parent = {r: r for r in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    contact = {}

    def add_pairs(la, lb, da, db):
        m = (la >= 1) & (lb >= 1) & (la != lb)
        if not m.any():
            return
        for x, y, d in zip(la[m], lb[m], np.maximum(da[m], db[m])):
            key = (min(int(x), int(y)), max(int(x), int(y)))
            if d > contact.get(key, 0.0):
                contact[key] = float(d)

    add_pairs(labels[:, :-1], labels[:, 1:], dt[:, :-1], dt[:, 1:])
    add_pairs(labels[:-1, :], labels[1:, :], dt[:-1, :], dt[1:, :])
    for (a, b), d in contact.items():
        if d >= ridge_px:
            union(a, b)
    out = labels.copy(); newid = {}; nxt = 1
    for r in ids:
        root = find(r)
        if root not in newid:
            newid[root] = nxt; nxt += 1
    for r in ids:
        out[labels == r] = newid[find(r)]
    return out


def segment_rooms_watershed(wall_mask, pixel_m, marker_h_m=0.30,
                            footprint_close_m=1.0, merge_ridge_m=0.70,
                            min_room_area_m2=2.0, min_wall_area_px=60,
                            door_seal_px=0, coverage=None, min_coverage_frac=0.25,
                            return_aux=False):
    """Deterministic room labels.  -1 wall · 0 exterior · >=1 rooms."""
    walls = clean_wall_mask(wall_mask, min_wall_area=min_wall_area_px, seal_gap=door_seal_px)
    free = ~walls
    dt = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)

    # interior seeds: maxima at least marker_h_m "deep"
    h = max(1.0, marker_h_m / pixel_m)
    markers, n_int = ndimage.label(h_maxima(dt, h))

    # explicit exterior = free space outside the building footprint (leak-immune)
    k = max(1, int(round(footprint_close_m / pixel_m)))
    footprint = _building_footprint(walls, k)
    exterior_seed = free & (~footprint)
    markers = markers.copy(); ext_id = n_int + 1
    if exterior_seed.any():
        markers[exterior_seed] = ext_id

    ws = watershed(-dt, markers, mask=free)                    # walls are barriers

    labels = np.full(wall_mask.shape, -1, np.int32); labels[free] = 0
    min_area_px = int(round(min_room_area_m2 / (pixel_m ** 2)))
    rid = 1
    for b in range(1, n_int + 1):
        m = (ws == b)
        if m.sum() == 0:
            continue
        inside = (m & footprint).sum() / m.sum()
        if m.sum() >= min_area_px and inside >= 0.5:           # big + indoors -> room
            labels[m] = rid; rid += 1
        else:
            labels[m] = 0                                      # speckle / outdoor blob
    labels[ws == ext_id] = 0
    labels = merge_wide_connections(labels, dt, merge_ridge_m, pixel_m)

    # void rejection: a real room sits over scanned data; an unscanned hole does not.
    pre_drop = labels.copy(); room_cov = {}
    if coverage is not None:
        for r in [int(x) for x in np.unique(labels) if x >= 1]:
            interior = (labels == r)
            frac = float((interior & coverage).sum() / max(1, int(interior.sum())))
            room_cov[r] = frac
            if frac < min_coverage_frac:
                labels[interior] = 0                  # empty void -> not a room
        labels = _relabel_rooms(labels)

    if return_aux:
        return labels, dict(walls=walls, dt=dt, footprint=footprint,
                            seeds=markers, ws=ws, exterior=exterior_seed,
                            n_int=n_int, ext_id=ext_id,
                            coverage=coverage, pre_drop=pre_drop, room_coverage=room_cov)
    return labels


def labels_to_detections(labels):
    """sv.Detections straight from the label array -> overlay == export."""
    import supervision as sv
    ids = [int(r) for r in np.unique(labels) if r >= 1]
    if not ids:
        return sv.Detections.empty()
    masks, boxes = [], []
    for r in ids:
        m = (labels == r); ys, xs = np.where(m)
        boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
        masks.append(m)
    return sv.Detections(xyxy=np.array(boxes, float), mask=np.array(masks),
                         confidence=np.ones(len(ids), float))
