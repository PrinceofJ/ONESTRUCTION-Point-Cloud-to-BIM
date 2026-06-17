"""Wall handling.

Three groups of functions:

1. **Wall-mask cleanup / sealing / footprint** (sections 5 of the original) — ported
   verbatim. Used by the watershed stage.
2. **Legacy raster-overlap wall assignment** (``room_footprints`` /
   ``split_rooms_to_clouds`` from section 9) — kept for reference and for the legacy
   debug overlay. **No longer used for export** (see #3). The back-projection core is
   factored out into ``backproject_room_masks`` so it is shared, not duplicated.
3. **NEW boundary-ring wall assignment** (``room_wall_masks_boundary_ring``) — the
   replacement requested in the refactor. For each room it builds a boundary ring
   ``B_i = M_i \\ erode(M_i)``, dilates it by ``r_w``, and keeps the **wallness** pixels
   that fall inside. ``estimate_wall_thickness_px`` is the *existing* distance-transform
   heuristic (originally inlined in ``room_footprints``) reused to auto-size the radii.
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .raster import point_cells


# ===========================================================================
# 1 · wall-mask cleanup / sealing / footprint  (verbatim from the original)
# ===========================================================================
def _fill_holes(mask_bool):
    m = mask_bool.astype(np.uint8); h, w = m.shape
    ff = m.copy(); flood = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood, (0, 0), 1)
    return mask_bool | (ff == 0)


def _building_footprint(walls, close_px):
    """Filled OUTER CONTOUR of the walls = building footprint. Leak-immune: a gap in
    an outer wall does not re-open the interior the way fill-holes-after-close does."""
    w = walls.astype(np.uint8)
    ker = None
    if close_px > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
        w = cv2.dilate(w, ker)                 # heal leaks/hairline gaps first
    cnts, _ = cv2.findContours(w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    f = np.zeros_like(w)
    cv2.drawContours(f, cnts, -1, 1, thickness=cv2.FILLED)
    if ker is not None:
        f = cv2.erode(f, ker)                  # undo the dilation -> hug the walls
    return f.astype(bool)


def _interior_seed(mask_bool):
    """Distance-transform peak -> safe interior point (valid for L-shapes). (x, y)."""
    dist = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(x), int(y)


def clean_wall_mask(wall_mask, min_wall_area=60, seal_gap=0):
    """Drop tiny black components (noise). seal_gap kept for API compatibility but
    defaults to 0 — prefer seal_at_doors()."""
    wm = (np.asarray(wall_mask) > 0).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(wm, connectivity=8)
    keep = np.zeros_like(wm)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_wall_area:
            keep[lbl == i] = 1
    walls = keep.astype(bool)
    if seal_gap > 0:
        walls = bridge_wall_endpoints(walls, max_gap=seal_gap)
    return walls


def seal_at_doors(wall_mask, doors_px, thickness=None):
    """Place wall barriers ONLY at detected doors. doors_px: list of (x0,y0,x1,y1) line
    segments (image px) spanning each door opening."""
    wm = (np.asarray(wall_mask) > 0).astype(np.uint8)
    if thickness is None:
        dist = cv2.distanceTransform(wm, cv2.DIST_L2, 5)
        pos = dist[dist > 0]
        thickness = max(1, int(round(2 * np.median(pos)))) if pos.size else 2
    out = wm.copy()
    for (x0, y0, x1, y1) in doors_px:
        cv2.line(out, (int(x0), int(y0)), (int(x1), int(y1)), 1, int(thickness))
    return out.astype(bool)


# legacy collinear-endpoint bridger (kept; used only if you pass seal_gap > 0)
def _endpoints_and_tangents(skel):
    sk = skel.astype(np.uint8)
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neigh = ndimage.convolve(sk, k, mode="constant")
    eps = np.argwhere((sk == 1) & (neigh == 1))
    ys, xs = np.where(sk); pts = np.column_stack([ys, xs])
    tree = cKDTree(pts); tangents = []
    for (y, x) in eps:
        loc = pts[tree.query_ball_point([y, x], r=6)]
        v = np.array([y, x]) - loc.mean(0); n = np.linalg.norm(v)
        tangents.append(v / n if n else np.zeros(2))
    return eps, (np.array(tangents) if tangents else np.zeros((0, 2)))


def bridge_wall_endpoints(wall_mask, max_gap=12, max_angle_deg=45, thickness=None):
    wm = (np.asarray(wall_mask) > 0).astype(np.uint8)
    if thickness is None:
        dist = cv2.distanceTransform(wm, cv2.DIST_L2, 5)
        thickness = max(1, int(round(2 * np.median(dist[dist > 0]))))
    eps, tang = _endpoints_and_tangents(skeletonize(wm > 0))
    if len(eps) < 2:
        return wm.astype(bool)
    tree = cKDTree(eps); out = wm.copy()
    cos_thr = np.cos(np.deg2rad(max_angle_deg)); used = set()
    cand = sorted((np.linalg.norm(eps[i] - eps[j]), i, j)
                  for i, j in tree.query_pairs(r=max_gap))
    for d, i, j in cand:
        if i in used or j in used:
            continue
        pi, pj = eps[i].astype(float), eps[j].astype(float)
        seg = (pj - pi) / (np.linalg.norm(pj - pi) + 1e-9)
        if np.dot(tang[i], seg) < cos_thr or np.dot(tang[j], -seg) < cos_thr:
            continue
        cv2.line(out, (int(pi[1]), int(pi[0])), (int(pj[1]), int(pj[0])), 1, thickness)
        used.add(i); used.add(j)
    return out.astype(bool)


# ===========================================================================
# 2 · LEGACY raster-overlap wall assignment  (kept for reference / debug only)
# ===========================================================================
def room_footprints(labels, margin=1, thickness=None, walls_only=False):
    """Original behaviour: dilate each room interior and intersect with the occupancy
    wall pixels (labels == -1). Superseded for export by the boundary-ring method but
    retained because the legacy debug overlay visualises it."""
    wall_mask = (labels == -1)
    if thickness is None:
        thickness = estimate_wall_thickness_px(wall_mask)
    k = max(1, thickness + margin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    foot = {}
    for r in np.unique(labels):
        if r < 1:
            continue
        interior = (labels == r)
        grown = cv2.dilate(interior.astype(np.uint8), kernel).astype(bool)
        band = grown & wall_mask
        foot[int(r)] = band if walls_only else (interior | band)
    return foot


def split_rooms_to_clouds(points, labels, transform, colors=None,
                          margin=1, thickness=None, walls_only=False, keep_mask=None):
    """LEGACY export path. Builds legacy footprints then back-projects (shared core)."""
    foot = room_footprints(labels, margin=margin, thickness=thickness, walls_only=walls_only)
    return backproject_room_masks(points, foot, transform, colors=colors, keep_mask=keep_mask)


# ===========================================================================
# 3 · NEW boundary-ring wall assignment  (the refactor's one behavioural change)
# ===========================================================================
def estimate_wall_thickness_px(wall_mask):
    """Median wall thickness in pixels, via the distance transform.

    This is the *existing* heuristic that the original ``room_footprints`` /
    ``seal_at_doors`` used inline (``2 * median(distance-transform over wall pixels)``).
    Factored out here so the new boundary-ring method can reuse it to auto-size its
    erosion / dilation radii when the corresponding config values are left as ``None``.
    """
    wm = (np.asarray(wall_mask) > 0).astype(np.uint8)
    dist = cv2.distanceTransform(wm, cv2.DIST_L2, 5)
    pos = dist[dist > 0]
    return int(round(2 * np.median(pos))) if pos.size else 1


def resolve_ring_radii_px(cfg, wall_mask_for_thickness):
    """Resolve (erode_px, dilate_px) for the boundary-ring method.

    Honours ``cfg.room_erode_m`` and ``cfg.wall_dilate_m`` when set; if either is
    ``None`` it is auto-derived from the estimated wall thickness (the existing
    heuristic): erosion -> half a wall thickness, dilation r_w -> one wall thickness
    plus the export buffer ``do_buffer_px``. Returns ints >= 1.
    """
    t = estimate_wall_thickness_px(wall_mask_for_thickness)
    if getattr(cfg, 'room_erode_m', None) is not None:
        erode_px = int(round(cfg.room_erode_m / cfg.pixel_m))
    else:
        erode_px = t // 2
    if getattr(cfg, 'wall_dilate_m', None) is not None:
        dilate_px = int(round(cfg.wall_dilate_m / cfg.pixel_m))
    else:
        dilate_px = t + cfg.do_buffer_px
    return max(1, erode_px), max(1, dilate_px)


def room_wall_masks_boundary_ring(labels, wallness, cfg,
                                  erode_px=None, dilate_px=None, return_debug=False):
    """NEW per-room wall-pixel assignment.

    For every room id (>= 1) in ``labels``:
        1. M_i  = (labels == i)                      # room mask
        2. I_i  = erode(M_i, erode_px)               # reliable interior
        3. B_i  = M_i \\ I_i                          # boundary ring
        4. B_i' = dilate(B_i, dilate_px)             # reach out by r_w
        5. W_i  = B_i' & wallness                    # keep wallness pixels on the ring

    Returns ``{room_id: bool wall-pixel mask}`` ready for back-projection. The wallness
    raster (preserved unchanged from ``rasterize_wallness``) is the wall source, per spec.
    """
    wallness = np.asarray(wallness, bool)
    re, rd = resolve_ring_radii_px(cfg, wallness)
    erode_px = re if erode_px is None else int(erode_px)
    dilate_px = rd if dilate_px is None else int(dilate_px)
    erode_px = max(1, erode_px); dilate_px = max(1, dilate_px)

    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))

    out, dbg = {}, {}
    for r in [int(x) for x in np.unique(labels) if x >= 1]:
        M = (labels == r)
        I = cv2.erode(M.astype(np.uint8), ek).astype(bool)     # reliable interior
        B = M & ~I                                             # boundary ring B_i = M_i \ I_i
        Bd = cv2.dilate(B.astype(np.uint8), dk).astype(bool)   # dilate by r_w
        W = Bd & wallness                                      # wallness pixels on the ring
        out[r] = W
        if return_debug:
            dbg[r] = dict(M=M, I=I, B=B, Bd=Bd, W=W)
    if return_debug:
        return out, dict(masks=dbg, erode_px=erode_px, dilate_px=dilate_px)
    return out


# ===========================================================================
# shared back-projection core (used by both export paths)
# ===========================================================================
def backproject_room_masks(points, room_masks, transform, colors=None, keep_mask=None):
    """Back-project 3-D points into per-room buckets given image-space room/wall masks.

    ``room_masks``: {room_id: bool HxW mask}. ``keep_mask``: optional per-point bool mask
    (e.g. the floor<->ceiling height band) applied on top of the spatial selection.
    Returns a list of dicts ``{room_id, mask (per-point bool), points[, colors]}``.
    """
    row, col, inb = point_cells(points, transform)
    pts = np.asarray(points); cols = np.asarray(colors) if colors is not None else None
    keep = None if keep_mask is None else np.asarray(keep_mask, bool)
    out = []
    for rid, fmask in sorted(room_masks.items()):
        fmask = np.asarray(fmask, bool)
        sel = np.zeros(len(pts), bool)
        sel[inb] = fmask[row[inb], col[inb]]
        if keep is not None:
            sel &= keep
        e = {'room_id': int(rid), 'mask': sel, 'points': pts[sel]}
        if cols is not None:
            e['colors'] = cols[sel]
        out.append(e)
    return out


def height_band_mask(points, cfg, transform, floor_z=None, ceil_z=None):
    """Per-point bool: floor + wall_floor_margin <= z <= ceil - wall_ceiling_margin.
    Mirrors the export band in the original section 11.6."""
    from .slab import estimate_ceiling
    pts = np.asarray(points, np.float64)
    z = pts[:, transform['up_axis']]
    if floor_z is None or ceil_z is None:
        floor_z, ceil_z = estimate_ceiling(z, return_floor=True)
    band = (z >= floor_z + cfg.wall_floor_margin_m) & (z <= ceil_z - cfg.wall_ceiling_margin_m)
    return band, float(floor_z), float(ceil_z)


def fit_walls_in_room(pc, up_axis=2, normal_tol_deg=10, dist_thresh=0.02,
                      min_inliers=400, max_planes=20, dbscan_eps_mult=8):
    """RANSAC vertical-plane fit per room (verbatim). ``pc`` is an open3d PointCloud."""
    if len(pc.points) < min_inliers:
        return []
    pc.estimate_normals()
    up = np.zeros(3); up[up_axis] = 1.0
    sin_tol = np.sin(np.deg2rad(normal_tol_deg))
    n = np.asarray(pc.normals)
    cand = pc.select_by_index(np.where(np.abs(n @ up) < sin_tol)[0])
    walls, rest = [], cand
    while len(rest.points) >= min_inliers and len(walls) < max_planes:
        model, inl = rest.segment_plane(dist_thresh, 3, 1000)
        if len(inl) < min_inliers:
            break
        nrm = np.array(model[:3]); nrm /= np.linalg.norm(nrm) + 1e-9
        if abs(nrm @ up) < sin_tol:
            seg = rest.select_by_index(inl)
            lbl = np.array(seg.cluster_dbscan(eps=dist_thresh * dbscan_eps_mult,
                                              min_points=max(10, min_inliers // 4)))
            for c in sorted(set(lbl) - {-1}):
                walls.append({'model': model, 'cloud': seg.select_by_index(np.where(lbl == c)[0])})
        rest = rest.select_by_index(inl, invert=True)
    return walls
