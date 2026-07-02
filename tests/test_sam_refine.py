"""Unit tests for the prompted-SAM graph relabel (``scan2bim.sam_refine``).

All fixtures are tiny HAND-BUILT label arrays + hand-built boolean "SAM masks" — no
pipeline data, no model, no randomness. They exercise the model-free core
(``relabel_by_sam``) plus the no-model pass-through of the orchestrator.

Scenarios (from the brief):
  * a confident mask spanning a no-wall DT ridge MERGES two rooms;
  * a confident SAM split of a wrongly-merged room SPLITS it;
  * a boundary on wall pixels is NEVER overridden;
  * refined output never contains a room pixel on a wall;
  * a low-confidence mask is ignored (the watershed result survives) — safety rail.
"""

import numpy as np
import cv2
import pytest

from scan2bim import Config, relabel_by_sam, refine_with_sam, build_sam_image


# --------------------------------------------------------------------------- helpers
def _frame(h, w):
    """A 1-px wall border so the distance transform is well defined; interior is free."""
    walls = np.zeros((h, w), bool)
    walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = True
    return walls


def _dt(walls):
    return cv2.distanceTransform((~walls).astype(np.uint8), cv2.DIST_L2, 5)


def _n_rooms(labels):
    return len([r for r in np.unique(labels) if r >= 1])


def _cfg(**over):
    # pixel_m=0.1 + sam_open_ridge_m=0.3 -> open-ridge threshold = 3 px, sized for the
    # small synthetic rooms below. Everything else stays at the production default.
    base = dict(pixel_m=0.1, sam_open_ridge_m=0.3)
    base.update(over)
    return Config(**base)


def _two_rooms_open_ridge(h=24, w=48):
    """Two rooms meeting at the vertical midline with NO wall between them (open ridge)."""
    walls = _frame(h, w)
    free = ~walls
    mid = w // 2
    labels = np.where(walls, -1, 0).astype(np.int32)
    left = free.copy(); left[:, mid:] = False
    right = free.copy(); right[:, :mid] = False
    labels[left] = 1
    labels[right] = 2
    return labels, walls, free


def _two_rooms_wall_between(h=24, w=48):
    """Two rooms separated by a genuine interior WALL column."""
    walls = _frame(h, w)
    mid = w // 2
    walls[1:-1, mid] = True
    free = ~walls
    labels = np.where(walls, -1, 0).astype(np.int32)
    left = free.copy(); left[:, mid:] = False
    right = free.copy(); right[:, :mid + 1] = False
    labels[left] = 1
    labels[right] = 2
    return labels, walls, free


# --------------------------------------------------------------------------- merge
def test_confident_mask_over_open_ridge_merges_two_rooms():
    labels, walls, free = _two_rooms_open_ridge()
    span = free.copy()                                  # one confident mask covers both rooms
    out, dbg = relabel_by_sam(labels, walls, _dt(walls),
                              {1: span, 2: span}, {1: 0.95, 2: 0.95}, _cfg())
    assert _n_rooms(out) == 1                            # the open partition is dissolved
    assert dbg['merges']                                # a merge was recorded
    assert (out[walls] == -1).all()                     # walls preserved


# --------------------------------------------------------------------------- split
def test_confident_split_divides_wrongly_merged_room():
    h, w = 24, 48
    walls = _frame(h, w)
    free = ~walls
    labels = np.where(walls, -1, 0).astype(np.int32)
    labels[free] = 1                                    # ONE (wrongly merged) room
    mid = w // 2
    half = free.copy(); half[:, mid:] = False           # SAM confidently sees only the left half
    out, dbg = relabel_by_sam(labels, walls, _dt(walls), {1: half}, {1: 0.95}, _cfg())
    assert _n_rooms(out) == 2                            # the room is split in two
    assert 1 in dbg['splits']
    assert (out[walls] == -1).all()
    assert not ((out >= 1) & walls).any()               # split line never lands on a wall


# --------------------------------------------------------------------------- wall is sacred
def test_boundary_on_wall_is_never_overridden():
    labels, walls, free = _two_rooms_wall_between()
    span = free.copy()                                  # confident mask spanning the wall
    out, dbg = relabel_by_sam(labels, walls, _dt(walls),
                              {1: span, 2: span}, {1: 0.95, 2: 0.95}, _cfg())
    assert _n_rooms(out) == 2                            # NOT merged — the edge is wall-backed
    assert dbg['merges'] == []
    assert (out[walls] == -1).all()                     # the dividing wall stays -1


# --------------------------------------------------------------------------- no room on a wall
def test_refined_output_never_places_a_room_pixel_on_a_wall():
    for builder in (_two_rooms_open_ridge, _two_rooms_wall_between):
        labels, walls, free = builder()
        span = free.copy()
        out, _ = relabel_by_sam(labels, walls, _dt(walls),
                                {1: span, 2: span}, {1: 0.95, 2: 0.95}, _cfg())
        assert out.shape == labels.shape                # same array shape as the watershed
        assert not ((out >= 1) & walls).any()
        assert (out[walls] == -1).all()


# --------------------------------------------------------------------------- safety rail
def test_low_confidence_mask_is_ignored():
    labels, walls, free = _two_rooms_open_ridge()
    span = free.copy()
    out, dbg = relabel_by_sam(labels, walls, _dt(walls),
                              {1: span, 2: span}, {1: 0.50, 2: 0.50}, _cfg())
    assert _n_rooms(out) == 2                            # below confidence -> watershed survives
    assert dbg['merges'] == []


def test_split_rejected_when_below_confidence():
    h, w = 24, 48
    walls = _frame(h, w); free = ~walls
    labels = np.where(walls, -1, 0).astype(np.int32); labels[free] = 1
    half = free.copy(); half[:, w // 2:] = False
    out, dbg = relabel_by_sam(labels, walls, _dt(walls), {1: half}, {1: 0.50}, _cfg())
    assert _n_rooms(out) == 1                            # not confident -> no split
    assert dbg['splits'] == []


# --------------------------------------------------------------------------- orchestrator
def test_pass_through_when_sam_disabled():
    labels, walls, free = _two_rooms_open_ridge()
    occ = np.where(walls, 0, 255).astype(np.uint8)      # occupancy.png convention
    cfg = _cfg(use_sam_recall=False)
    out, dbg = refine_with_sam(labels, occ, walls, free, cfg)
    assert dbg['ran'] is False
    assert np.array_equal(out, labels)                  # watershed returned untouched


def test_pass_through_when_no_backend_builds():
    # use_sam_recall on, but no torch/SAM in this env -> build fails -> safe pass-through.
    labels, walls, free = _two_rooms_open_ridge()
    occ = np.where(walls, 0, 255).astype(np.uint8)
    out, dbg = refine_with_sam(labels, occ, walls, free, _cfg(use_sam_recall=True))
    assert dbg['ran'] is False
    assert 'no SAM backend' in dbg.get('reason', '')
    assert np.array_equal(out, labels)


# --------------------------------------------------------------------------- SAM image
def test_build_sam_image_stacks_three_rasters():
    h, w = 10, 12
    occ = np.full((h, w), 255, np.uint8); occ[:, 0] = 0   # one wall column
    wall_mask = np.zeros((h, w), bool); wall_mask[:, 0] = True
    coverage = np.ones((h, w), bool)
    img = build_sam_image(occ, wall_mask, coverage, mode='stack')
    assert img.shape == (h, w, 3) and img.dtype == np.uint8
    assert (img[:, 0, 0] == 0).all()                    # ch0 = free space (wall col is 0)
    assert (img[:, 0, 1] == 255).all()                  # ch1 = wall mask (wall col is 255)
    assert (img[..., 2] == 255).all()                   # ch2 = coverage (all scanned)


def test_build_sam_image_falls_back_to_occupancy():
    h, w = 8, 8
    occ = np.full((h, w), 255, np.uint8)
    img = build_sam_image(occ, None, None, mode='stack')
    assert img.shape == (h, w, 3)
    assert (img[..., 0] == img[..., 1]).all() and (img[..., 1] == img[..., 2]).all()


if __name__ == '__main__':                              # allow plain `python tests/...`
    raise SystemExit(pytest.main([__file__, '-v']))
