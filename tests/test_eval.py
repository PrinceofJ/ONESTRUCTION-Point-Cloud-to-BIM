"""Unit tests for the room-segmentation GT builder + paper IoU (``scan2bim.eval``).

All hand-built  - no pipeline data, no torch, no randomness. Two groups:

  GT builder  : a tiny S3DIS-shaped folder tree (per-class Annotations) on a known grid,
                asserting the structural + clutter classes are dropped from the room area.
  Scoring     : hand-built label maps for the cases the task names  - identical masks score
                IoU 1.0, a known over-seg case yields over_seg=1, an under-seg case yields
                under_seg=1, a missed room drags the mean down, and the shared wall scaffold
                excludes the same pixels from both sides.
"""

import os

import numpy as np
import pytest

from scan2bim import (STRUCTURAL_CLUTTER_CLASSES, annotation_class,
                      load_room_interior_points, build_gt_room_labels, load_gt_room_points,
                      overlap_stats, score_rooms, score_rooms_paper, point_cells,
                      eval_wall_scaffold, harmonize_room_labels, Config)


# --------------------------------------------------------------------------- helpers
def _identity_transform(H, W, pixel_size=1.0):
    """A grid where world (x, y) maps 1:1 to (col, row) cells.

    With ax_a=0 (x->col) and ax_b=1 (y->row via the H-1-b_px flip), a point at world
    (x, y) lands at cell (row=H-1-y, col=x). Used so tests can place GT points at known
    pixels."""
    return dict(a_min=0.0, b_min=0.0, pixel_size=float(pixel_size),
                width=int(W), height=int(H), ax_a=0, ax_b=1, up_axis=2)


def _write_xyz(path, xy, z=0.0):
    """Write 2-D points as an S3DIS-style ``X Y Z R G B`` txt (RGB ignored on read)."""
    xy = np.asarray(xy, float)
    rows = np.column_stack([xy[:, 0], xy[:, 1], np.full(len(xy), z),
                            np.zeros(len(xy)), np.zeros(len(xy)), np.zeros(len(xy))])
    np.savetxt(path, rows, fmt='%.3f')


def _make_room(room_dir, class_points):
    """Create ``room_dir/Annotations/<class>_1.txt`` for each class in ``class_points``."""
    ann = os.path.join(room_dir, 'Annotations')
    os.makedirs(ann, exist_ok=True)
    for cls, pts in class_points.items():
        _write_xyz(os.path.join(ann, f'{cls}_1.txt'), pts)


# =========================================================================== GT builder
def test_annotation_class_strips_index():
    assert annotation_class('clutter_10.txt') == 'clutter'
    assert annotation_class('floor_1.txt') == 'floor'
    assert annotation_class('wall_3.txt') == 'wall'


def test_interior_points_exclude_structural_and_clutter(tmp_path):
    room = tmp_path / 'office_1'
    _make_room(str(room), {
        'floor':   [(0, 0), (1, 1)],          # kept
        'ceiling': [(2, 2)],                   # kept
        'wall':    [(5, 5), (6, 6), (7, 7)],   # excluded (structural)
        'clutter': [(8, 8)],                   # excluded
        'window':  [(9, 9)],                   # excluded
    })
    pts, kept = load_room_interior_points(str(room))
    assert kept == {'floor', 'ceiling'}
    assert len(pts) == 3                       # only floor (2) + ceiling (1)
    # none of the excluded points (x>=5) survived
    assert pts[:, 0].max() <= 2


def test_excluded_classes_constant_is_the_locked_set():
    assert set(STRUCTURAL_CLUTTER_CLASSES) == {
        'wall', 'beam', 'column', 'door', 'window', 'clutter'}


def test_build_gt_room_labels_rasterises_interior_only(tmp_path):
    # Two rooms on a 10x10 grid. Each has interior (floor) points and wall points; only the
    # interior must paint into the label map. room_a -> id 1, room_b -> id 2 (sorted order).
    H = W = 10
    tf = _identity_transform(H, W)
    _make_room(str(tmp_path / 'room_a'), {
        'floor': [(1, 1), (2, 2)],
        'wall':  [(1, 8), (2, 8)],             # excluded: must NOT appear in labels
    })
    _make_room(str(tmp_path / 'room_b'), {
        'floor': [(7, 7), (8, 8)],
        'clutter': [(7, 1)],                   # excluded
    })
    gt, info = build_gt_room_labels(str(tmp_path), tf)

    assert gt.shape == (H, W)
    assert sorted(int(v) for v in np.unique(gt) if v >= 1) == [1, 2]
    # room_a floor point (1,1) -> cell (row=H-1-1=8, col=1)
    assert gt[H - 1 - 1, 1] == 1
    assert gt[H - 1 - 2, 2] == 1
    # room_b floor point (7,7) -> (row=2, col=7)
    assert gt[H - 1 - 7, 7] == 2
    # excluded wall/clutter cells stayed exterior (0)
    assert gt[H - 1 - 8, 1] == 0               # room_a wall (1,8)
    assert gt[H - 1 - 1, 7] == 0               # room_b clutter (7,1)
    # info provenance
    assert info['rooms'][0]['name'] == 'room_a' and info['rooms'][0]['id'] == 1
    assert info['rooms'][0]['classes'] == ['floor']
    assert info['ingrid_frac'] == pytest.approx(1.0)
    assert info['exclude'] == STRUCTURAL_CLUTTER_CLASSES


def test_build_gt_falls_back_to_whole_room_file_without_annotations(tmp_path):
    # No Annotations dir -> use <room>/<room>.txt, kept_classes empty (cannot exclude walls).
    room = tmp_path / 'plain_1'
    room.mkdir()
    _write_xyz(str(room / 'plain_1.txt'), [(3, 3), (4, 4)])
    pts, kept = load_room_interior_points(str(room))
    assert len(pts) == 2 and kept == set()


# =========================================================================== scoring
def _two_room_map(split_col=None, H=10, W=10):
    """Left half = room 1, right half = room 2. If ``split_col`` given, the LEFT room is
    instead split into two rooms (ids 1 and 3) at that column -> an over-segmentation."""
    lab = np.zeros((H, W), np.int32)
    lab[:, :W // 2] = 1
    lab[:, W // 2:] = 2
    if split_col is not None:
        lab[:, :split_col] = 1
        lab[:, split_col:W // 2] = 3
    return lab


def test_identical_masks_score_iou_one():
    gt = _two_room_map()
    r = score_rooms(gt.copy(), gt)
    assert r['mean_iou'] == pytest.approx(1.0)
    assert r['over_seg'] == 0 and r['under_seg'] == 0
    assert r['n_pred_rooms'] == 2 and r['n_gt_rooms'] == 2
    assert len(r['matched_pairs']) == 2


def test_over_segmentation_counted_once_and_lowers_iou():
    gt = _two_room_map()                       # GT: 2 rooms
    pred = _two_room_map(split_col=2)          # pred splits the LEFT GT room into 2 pieces
    r = score_rooms(pred, gt)
    assert r['over_seg'] == 1                   # one GT room covered by >=2 predictions
    assert r['under_seg'] == 0
    # the split GT room can match at most its larger piece -> IoU < 1, so mean drops below 1
    assert r['mean_iou'] < 1.0
    # the intact right room still matches perfectly
    right = [pr for pr in r['per_room'] if pr['gt_id'] == 2][0]
    assert right['iou'] == pytest.approx(1.0)


def test_under_segmentation_counted_once():
    # GT has 2 rooms; the prediction merges them into one -> under-seg.
    gt = _two_room_map()
    pred = np.ones_like(gt)                     # single predicted room covering everything
    r = score_rooms(pred, gt)
    assert r['under_seg'] == 1                   # one prediction covers >=2 GT rooms
    assert r['over_seg'] == 0
    assert r['mean_iou'] < 1.0                   # the merged prediction can't match either room well


def test_missed_room_drags_mean_to_half():
    # pred only contains room 1 (left half); GT has both -> room 2 IoU = 0.
    gt = _two_room_map()
    pred = np.where(gt == 1, 1, 0).astype(np.int32)
    r = score_rooms(pred, gt)
    ious = {pr['gt_id']: pr['iou'] for pr in r['per_room']}
    assert ious[1] == pytest.approx(1.0)
    assert ious[2] == pytest.approx(0.0)
    assert r['mean_iou'] == pytest.approx(0.5)


def test_shared_wall_scaffold_excludes_same_pixels_both_sides():
    # A column of wall pixels the prediction marks but the GT does not. Without the scaffold
    # the GT/pred disagree there; with the scaffold those pixels are removed from both -> IoU 1.
    H = W = 10
    gt = np.ones((H, W), np.int32)             # whole grid is GT room 1
    pred = np.ones((H, W), np.int32)
    wall = np.zeros((H, W), bool)
    wall[:, 5] = True                          # one wall column
    pred[wall] = -1                            # prediction marks the wall as not-room
    # without the scaffold, the wall column is GT-room-1 but pred -1 -> union>inter -> IoU<1
    assert score_rooms(pred, gt)['mean_iou'] < 1.0
    # with the shared scaffold, both sides drop that column -> perfect match
    assert score_rooms(pred, gt, wall_mask=wall)['mean_iou'] == pytest.approx(1.0)


def test_empty_prediction_scores_zero_not_crash():
    gt = _two_room_map()
    pred = np.zeros_like(gt)                    # no predicted rooms
    r = score_rooms(pred, gt)
    assert r['mean_iou'] == 0.0
    assert r['n_pred_rooms'] == 0 and r['n_gt_rooms'] == 2
    assert r['matched_pairs'] == []


def test_shape_mismatch_raises():
    gt = _two_room_map(H=10, W=10)
    pred = _two_room_map(H=8, W=8)
    with pytest.raises(ValueError):
        score_rooms(pred, gt)


def test_overlap_stats_intersection_and_areas():
    gt = _two_room_map()
    pred = _two_room_map()
    pred_ids, gt_ids, inter, pred_area, gt_area = overlap_stats(pred, gt)
    assert list(pred_ids) == [1, 2] and list(gt_ids) == [1, 2]
    # identical maps -> intersection is diagonal, each room is 10x5 = 50 cells
    assert inter[0, 0] == 50 and inter[1, 1] == 50
    assert inter[0, 1] == 0 and inter[1, 0] == 0
    assert list(pred_area) == [50, 50] and list(gt_area) == [50, 50]


# =========================================================================== paper protocol (02b)
def _row_points(xs, y=1):
    """Points on one raster row: with the identity transform, world (x, y) -> cell (H-1-y, x)."""
    xs = np.asarray(xs, float)
    return np.column_stack([xs, np.full(len(xs), float(y)), np.zeros(len(xs))])


def _row_pred(H, W, label_cols, y_row_world=1):
    """A label map with the given {label: [cols]} painted on the row world-y=1 sits on."""
    lab = np.zeros((H, W), np.int32)
    row = H - 1 - y_row_world
    for lab_id, cols in label_cols.items():
        lab[row, list(cols)] = lab_id
    return lab


def test_load_gt_room_points_ids_match_builder(tmp_path):
    _make_room(str(tmp_path / 'room_a'), {'floor': [(1, 1), (2, 2)], 'wall': [(1, 8)]})
    _make_room(str(tmp_path / 'room_b'), {'floor': [(7, 7)], 'clutter': [(7, 1)]})
    pts, ids = load_gt_room_points(str(tmp_path))
    # room_a -> id 1 (2 interior pts), room_b -> id 2 (1 interior pt); excluded classes dropped
    assert len(pts) == 3
    assert sorted(int(v) for v in np.unique(ids)) == [1, 2]
    assert int((ids == 1).sum()) == 2 and int((ids == 2).sum()) == 1


def test_paper_identity_all_matched_iou_one():
    H, W = 3, 20
    tf = _identity_transform(H, W)
    pts = _row_points(list(range(0, 4)) + list(range(6, 10)))
    ids = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    pred = _row_pred(H, W, {1: range(0, 4), 2: range(6, 10)})
    r = score_rooms_paper(pts, ids, pred, tf)
    assert r['mean_iou_matched'] == pytest.approx(1.0)
    assert r['detection_rate'] == pytest.approx(1.0)
    assert r['over_seg'] == 0 and r['under_seg'] == 0
    assert r['n_gt'] == 2 and r['n_seg'] == 2 and r['n_matched'] == 2


def test_paper_missed_room_excluded_from_mean_not_zeroed():
    # The key difference vs score_rooms: a completely-missed GT room lowers detection_rate,
    # it does NOT enter the mean as IoU 0.
    H, W = 3, 20
    tf = _identity_transform(H, W)
    pts = _row_points(list(range(0, 4)) + list(range(6, 10)))
    ids = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    pred = _row_pred(H, W, {1: range(0, 4)})        # room 2 not predicted at all
    r = score_rooms_paper(pts, ids, pred, tf)
    assert r['mean_iou_matched'] == pytest.approx(1.0)   # only the found room counts
    assert r['detection_rate'] == pytest.approx(0.5)     # 1 of 2 GT rooms found
    assert r['n_matched'] == 1
    # contrast: the pixel scorer averages over both GT rooms -> ~0.5
    pix = score_rooms(pred, _row_pred(H, W, {1: range(0, 4), 2: range(6, 10)}))
    assert pix['mean_iou'] < r['mean_iou_matched']


def test_paper_75pct_threshold_bites():
    # One pred straddling two GT rooms partially: p and g both < 0.75 -> no match. Push past
    # 0.75 and it matches. (A pred fully inside a GT room always has p=1, so non-match needs
    # the pred to span two rooms — exactly what the OR rule is designed to catch.)
    H, W = 3, 20
    tf = _identity_transform(H, W)
    pts = _row_points(list(range(0, 10)) + list(range(10, 20)))
    ids = np.array([1] * 10 + [2] * 10)
    # pred1 covers 6 of GT1 (cols 0-5) + 4 of GT2 (cols 10-13): p=g=0.6 for GT1 -> unmatched
    no = score_rooms_paper(pts, ids, _row_pred(H, W, {1: list(range(0, 6)) + list(range(10, 14))}), tf)
    assert no['n_matched'] == 0 and no['detection_rate'] == pytest.approx(0.0)
    # pred1 covers 8 of GT1 (cols 0-7) + 2 of GT2 (cols 10-11): p(GT1)=8/10=0.8 -> matched
    yes = score_rooms_paper(pts, ids, _row_pred(H, W, {1: list(range(0, 8)) + list(range(10, 12))}), tf)
    g1 = [pr for pr in yes['per_room'] if pr['gt_id'] == 1][0]
    assert g1['matched'] is True and yes['n_matched'] >= 1


def test_paper_over_segmentation_counted_once():
    H, W = 3, 20
    tf = _identity_transform(H, W)
    pts = _row_points(range(0, 10))
    ids = np.array([1] * 10)                          # single GT room
    pred = _row_pred(H, W, {1: range(0, 5), 2: range(5, 10)})  # split into two preds inside it
    r = score_rooms_paper(pts, ids, pred, tf)
    assert r['over_seg'] == 1 and r['under_seg'] == 0
    assert r['detection_rate'] == pytest.approx(1.0)  # the room is still found


def test_paper_under_segmentation_counted_once():
    H, W = 3, 20
    tf = _identity_transform(H, W)
    pts = _row_points(range(0, 20))
    ids = np.array([1] * 10 + [2] * 10)              # two GT rooms
    pred = _row_pred(H, W, {1: range(0, 20)})        # one pred swallows both
    r = score_rooms_paper(pts, ids, pred, tf)
    assert r['under_seg'] == 1 and r['over_seg'] == 0
    assert r['n_matched'] == 2                        # both GT rooms correspond to the one pred


# ---------------------------------------------------------------------------
# Harmonized comparison — fair filters + shared scaffold (research-fixes Task 05)
# ---------------------------------------------------------------------------
def _eval_cfg(**kw):
    """Config with pixel_m=1.0 so a room of N px == N m^2 (easy thresholds)."""
    base = dict(pixel_m=1.0, eval_profile='comparison',
                eval_min_room_area_m2=4.0, eval_min_coverage_frac=0.5, min_wall_area_px=2)
    base.update(kw)
    return Config(**base)


def _room_set(labels):
    """The partition as a hashable set of room pixel-sets (ignores label *ids*)."""
    labels = np.asarray(labels)
    return frozenset(frozenset(np.flatnonzero(labels == r).tolist())
                     for r in np.unique(labels) if r >= 1)


def test_eval_wall_scaffold_drops_speckle_keeps_walls():
    cfg = _eval_cfg(min_wall_area_px=5)
    wm = np.zeros((10, 10), bool)
    wm[:, 5] = True            # a real 10-px wall column
    wm[0, 0] = True            # a 1-px speck (below min_wall_area_px=5)
    scaf = eval_wall_scaffold(wm, cfg)
    assert scaf[:, 5].all()    # wall kept
    assert not scaf[0, 0]      # speck dropped
    assert scaf.dtype == bool


def test_harmonize_method_agnostic_same_room_set():
    """Two methods that AGREE on the room interiors but differ in the incidentals harmonization
    normalizes away — the wall scaffold and sub-threshold speckle — must yield the SAME
    harmonized room set (the core Task 05 guarantee). The methods must not disagree on which
    non-wall cell belongs to which room; harmonization unifies the scaffold, it does not
    re-assign a cell one method genuinely carved out."""
    cfg = _eval_cfg()
    walls = np.zeros((10, 10), bool); walls[:, 5] = True          # canonical scaffold = col 5
    cov = np.ones((10, 10), bool)

    # Method A: treats col 5 as wall (-1).
    A = np.zeros((10, 10), np.int32)
    A[:, :5] = 1; A[:, 6:] = 2; A[:, 5] = -1

    # Method B: SAME interiors, but did NOT call col 5 a wall (left room spills onto it) and
    # carries a 1-px speckle room sitting on the scaffold line (gets normalized to -1).
    B = np.zeros((10, 10), np.int32)
    B[:, :6] = 1; B[:, 6:] = 2; B[0, 5] = 3

    hA = harmonize_room_labels(A, cov, walls, cfg)
    hB = harmonize_room_labels(B, cov, walls, cfg)
    assert _room_set(hA) == _room_set(hB)                         # identical room sets
    assert np.array_equal(hA == -1, hB == -1)                    # identical shared -1 scaffold
    assert (hA[:, 5] == -1).all() and (hB[:, 5] == -1).all()     # canonical wall imposed on both
    assert 3 not in np.unique(hB)                                # B's speckle gone (on scaffold)


def test_harmonize_is_deterministic():
    cfg = _eval_cfg()
    walls = np.zeros((8, 8), bool); walls[:, 4] = True
    cov = np.ones((8, 8), bool)
    lab = np.zeros((8, 8), np.int32); lab[:, :4] = 1; lab[:, 5:] = 2; lab[walls] = -1
    first = harmonize_room_labels(lab, cov, walls, cfg)
    for _ in range(3):
        assert np.array_equal(harmonize_room_labels(lab, cov, walls, cfg), first)


def test_harmonize_drops_subthreshold_and_void_rooms():
    cfg = _eval_cfg(eval_min_room_area_m2=4.0, eval_min_coverage_frac=0.5)
    walls = np.zeros((10, 10), bool)
    cov = np.ones((10, 10), bool)
    lab = np.zeros((10, 10), np.int32)
    lab[0:5, 0:5] = 1            # 25-px room -> kept
    lab[0, 9] = 2               # 1-px room  -> below 4 m^2, dropped
    lab[7:10, 7:10] = 3         # 9-px room  -> area-OK but we'll make it void
    cov[7:10, 7:10] = False     # room 3 sits over NO coverage -> void-dropped
    h = harmonize_room_labels(lab, cov, walls, cfg)
    assert sorted(int(x) for x in np.unique(h) if x >= 1) == [1]   # only the big covered room
    # relabelled compactly to 1..k
    assert set(int(x) for x in np.unique(h) if x >= 1) == {1}


def test_harmonize_paper_profile_passes_through_unchanged():
    cfg = _eval_cfg(eval_profile='paper')
    walls = np.zeros((6, 6), bool); walls[:, 3] = True
    cov = np.ones((6, 6), bool)
    lab = np.zeros((6, 6), np.int32); lab[:, 0] = -1; lab[0, 0] = 0; lab[2:4, 4:6] = 7
    out = harmonize_room_labels(lab, cov, walls, cfg)
    assert np.array_equal(out, lab)        # faithful: no filtering, no scaffold, no relabel
