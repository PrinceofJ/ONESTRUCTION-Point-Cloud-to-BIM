"""Unit tests for the pure-SAM automatic room segmentation (``scan2bim.sam_auto``).

Mirrors ``test_sam_refine.py``: tiny HAND-BUILT boolean masks + a FAKE automatic mask
generator — no pipeline data, no torch, no checkpoint, no randomness. They exercise the
model-free deterministic core (``masks_to_room_labels`` / ``classify_rooms_by_area`` /
``buffer_room_labels`` / ``reprocess_residual``) plus the no-model pass-through of the
orchestrator (``segment_rooms_sam_auto``).

These do NOT judge SAM's segmentation quality (that is the real-data ``pq_eval``
comparison). They are contract/regression tests for the deterministic glue that turns
SAM's raw masks into the project's ``-1`` wall / ``0`` exterior / ``>=1`` room convention.

Scenarios (plan §7):
  1. two clean masks -> two rooms (walls stay -1, ids compacted, no room on a wall);
  2. overlapping masks resolved deterministically (higher predicted_iou wins; order-free);
  3. a small mask dropped by area (< A) -> exterior;
  4. a mask over unscanned void dropped via coverage;
  5. the boundary buffer reclaims wall-adjacent pixels without bleeding into a neighbour;
  6. residual reprocessing adds a corridor the first pass missed;
  7. orchestrator returns a clear no-backend flag (no fabricated masks);
  8. the same-grid invariant (shape + label convention).
"""

import numpy as np
import pytest

from scan2bim import (Config, AutoMaskGenerator, segment_rooms_sam_auto,
                      masks_to_room_labels, classify_rooms_by_area,
                      buffer_room_labels)


# --------------------------------------------------------------------------- helpers
def _frame(h, w):
    """A 1-px wall border; interior is free space."""
    walls = np.zeros((h, w), bool)
    walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = True
    return walls


def _cfg(**over):
    # pixel_m=0.1 -> A=1.5 m^2 is 150 px, sized for the small synthetic grids below.
    base = dict(pixel_m=0.1)
    base.update(over)
    return Config(**base)


def _n_rooms(labels):
    return len([r for r in np.unique(labels) if r >= 1])


def _col_grid(h, w):
    """Per-pixel column index, for building vertical-split masks."""
    return np.broadcast_to(np.arange(w), (h, w))


class FakeAutoGen(AutoMaskGenerator):
    """Returns hand-built masks; each positional arg is one ``generate`` call's
    ``[(mask, score), ...]`` (so a 2nd call can return the corridor for reprocessing)."""

    def __init__(self, *calls):
        self._calls = [list(c) for c in calls]
        self._i = 0

    def generate(self, image):
        c = self._calls[min(self._i, len(self._calls) - 1)]
        self._i += 1
        return [dict(segmentation=np.asarray(m, bool), predicted_iou=float(s),
                     stability_score=1.0, area=int(np.asarray(m, bool).sum()))
                for m, s in c]


# --------------------------------------------------------------------------- 1 · clean
def test_two_clean_masks_become_two_rooms():
    h, w = 30, 60
    walls = _frame(h, w)
    free = ~walls
    col = _col_grid(h, w)
    mid = w // 2
    left = free & (col < mid)
    right = free & (col >= mid)
    out, dbg = masks_to_room_labels([left, right], [0.9, 0.9], walls, None, _cfg())
    assert out.shape == walls.shape
    assert _n_rooms(out) == 2                            # two distinct rooms
    assert set(np.unique(out)) == {-1, 1, 2}            # ids compacted to 1..k
    assert (out[walls] == -1).all()                     # walls preserved
    assert not ((out >= 1) & walls).any()               # no room pixel on a wall
    assert dbg['n_kept'] == 2


# --------------------------------------------------------------------------- 2 · overlap
def test_overlapping_masks_resolved_deterministically():
    h, w = 30, 60
    walls = _frame(h, w)
    free = ~walls
    col = _col_grid(h, w)
    a = free & (col < 40)                                # lower score
    b = free & (col >= 20)                               # higher score, wins the overlap
    out, _ = masks_to_room_labels([a, b], [0.90, 0.95], walls, None, _cfg())
    rev, _ = masks_to_room_labels([b, a], [0.95, 0.90], walls, None, _cfg())
    assert np.array_equal(out, rev)                      # independent of input order
    # a pixel in the overlap (col 30) belongs to b, same id as b's exclusive region (col 50)
    assert out[15, 30] == out[15, 50]
    assert out[15, 30] != out[15, 5]                    # ...and not to a's exclusive region


# --------------------------------------------------------------------------- 3 · area drop
def test_small_mask_dropped_by_area():
    h, w = 30, 60
    walls = _frame(h, w)
    small = np.zeros((h, w), bool)
    small[5:15, 5:17] = True                             # 10x12 = 120 px  (> min_mask 100)
    assert int(small.sum()) == 120
    labels, _ = masks_to_room_labels([small], [0.9], walls, None, _cfg())
    assert _n_rooms(labels) == 1                         # survives the px noise floor...
    classified = classify_rooms_by_area(labels, _cfg())  # ...but 120 px < 150 px (A=1.5 m^2)
    assert _n_rooms(classified) == 0                     # -> reclassified as exterior


# --------------------------------------------------------------------------- 4 · void drop
def test_mask_over_unscanned_void_is_dropped():
    h, w = 30, 60
    walls = _frame(h, w)
    free = ~walls
    col = _col_grid(h, w)
    room = free & (col < 30)                             # over scanned data
    void = free & (col >= 30)                            # over an unscanned hole
    coverage = free & (col < 30)                          # only the left half is scanned
    out, dbg = masks_to_room_labels([room, void], [0.9, 0.9], walls, coverage, _cfg())
    assert _n_rooms(out) == 1                            # the void mask is rejected
    assert dbg['n_void_dropped'] == 1
    assert (out[free & (col >= 30)] == 0).all()         # the void stays exterior


# --------------------------------------------------------------------------- 5 · buffer
def test_buffer_reclaims_wall_adjacent_pixels_without_bleeding():
    h, w = 30, 60
    walls = _frame(h, w)
    free = ~walls
    col = _col_grid(h, w)
    labels = np.where(walls, -1, 0).astype(np.int32)
    labels[free & (col >= 5) & (col <= 24)] = 1          # left room
    labels[free & (col >= 35) & (col <= 54)] = 2         # right room, a wide gap apart
    out = buffer_room_labels(labels, walls, _cfg(), buffer_px=3)
    assert _n_rooms(out) == 2
    assert out[14, 26] == 1                              # left room grew right into the gap
    assert out[14, 33] == 2                              # right room grew left into the gap
    assert out[14, 30] == 0                              # ...but the middle is beyond 3 px
    assert out[14, 2] == 1                               # reclaimed wall-adjacent pixels
    assert (out[walls] == -1).all()
    assert not ((out >= 1) & walls).any()               # buffer never lands on a wall


# --------------------------------------------------------------------------- 6 · residual
def test_residual_reprocessing_adds_a_missed_corridor():
    h, w = 30, 60
    walls = _frame(h, w)
    free = ~walls
    col = _col_grid(h, w)
    left = free & (col < 30)
    corridor = free & (col >= 30)
    # 1st generate: only the left room; 2nd generate (on the residual): the corridor.
    gen = FakeAutoGen([(left, 0.9)], [(corridor, 0.9)])
    image = np.zeros((h, w, 3), np.uint8)
    cfg = _cfg(sam_reprocess_residual=True)
    out, dbg = segment_rooms_sam_auto(image, walls, free, cfg, generator=gen)
    assert dbg['ran'] is True
    assert dbg['n_rooms_pass1'] == 1                     # first pass missed the corridor
    assert dbg['reprocess']['ran'] is True and dbg['reprocess']['n_added'] == 1
    assert _n_rooms(out) == 2                            # corridor recovered as a 2nd room
    assert out[15, 45] >= 1                              # the corridor is now a room


# --------------------------------------------------------------------------- orchestrator
def test_orchestrator_two_masks_end_to_end():
    h, w = 30, 60
    walls = _frame(h, w)
    free = ~walls
    col = _col_grid(h, w)
    gen = FakeAutoGen([(free & (col < 30), 0.9), (free & (col >= 30), 0.9)])
    out, dbg = segment_rooms_sam_auto(np.zeros((h, w, 3), np.uint8), walls, free,
                                      _cfg(), generator=gen)
    assert dbg['ran'] is True and dbg['backend'] == 'sam-auto'
    assert _n_rooms(out) == 2
    assert (out[walls] == -1).all()
    assert not ((out >= 1) & walls).any()


# --------------------------------------------------------------------------- 7 · no backend
def test_no_backend_returns_clear_flag_without_fabricating():
    # No generator injected and no torch/SAM in this env -> build fails -> all-exterior,
    # ran=False (the GPU notebook checks this and raises). Nothing is fabricated.
    h, w = 30, 60
    walls = _frame(h, w)
    out, dbg = segment_rooms_sam_auto(np.zeros((h, w, 3), np.uint8), walls, ~walls, _cfg())
    assert dbg['ran'] is False
    assert 'no SAM backend' in dbg.get('reason', '')
    assert out.shape == walls.shape
    assert _n_rooms(out) == 0                            # no rooms invented
    assert (out[walls] == -1).all()


# --------------------------------------------------------------------------- 8 · invariant
def test_same_grid_label_convention_holds():
    h, w = 24, 40
    walls = _frame(h, w)
    free = ~walls
    col = _col_grid(h, w)
    out, _ = masks_to_room_labels([free & (col < 20), free & (col >= 20)], [0.9, 0.9],
                                  walls, None, _cfg())
    assert out.dtype == np.int32
    assert out.shape == (h, w)
    assert set(np.unique(out)).issubset({-1, 0, 1, 2})  # only wall / exterior / rooms


if __name__ == '__main__':                              # allow plain `python tests/...`
    raise SystemExit(pytest.main([__file__, '-v']))
