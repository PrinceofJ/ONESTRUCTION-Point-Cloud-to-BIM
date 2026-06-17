"""A tiny deterministic synthetic point cloud for the end-to-end baseline + guard tests.

A two-room building (8 m x 5 m x 2.5 m) split by an internal wall with a 0.4 m doorway,
plus dense floor and ceiling planes. Self-contained (no dependency on the large clouds in
``data/`` that a colleague may not have checked out) and fully deterministic, so the
pipeline produces the same room count + wall-point totals on every run. The doorway is kept
narrower than ``merge_ridge_m`` (0.70 m) so the watershed keeps the two rooms separate.

Not collected by pytest (leading underscore is not a ``test_*`` module).
"""
from __future__ import annotations

import numpy as np

# building geometry (metres)
W, D, H = 8.0, 5.0, 2.5
DOOR_C, DOOR_W = 2.5, 0.4
DIVIDER_X = 4.0


def synthetic_cloud(seed: int = 0) -> np.ndarray:
    """Return an ``(N, 3)`` float array for the two-room building."""
    rng = np.random.default_rng(seed)

    def plane(z, n):
        return np.column_stack([rng.uniform(0, W, n), rng.uniform(0, D, n), np.full(n, z)])

    def wall(p_xy, n):
        idx = rng.integers(0, len(p_xy), n)
        z = rng.uniform(0, H, n)
        return np.column_stack([p_xy[idx, 0], p_xy[idx, 1], z])

    parts = [plane(0.0, 60_000), plane(H, 60_000)]            # floor + ceiling
    t = np.linspace(0, 1, 400)
    outer = (np.column_stack([t * W, np.zeros_like(t)]),      # 4 outer walls
             np.column_stack([t * W, np.full_like(t, D)]),
             np.column_stack([np.zeros_like(t), t * D]),
             np.column_stack([np.full_like(t, W), t * D]))
    parts += [wall(seg, 20_000) for seg in outer]

    ty = np.linspace(0, D, 400)                               # internal wall with a doorway
    keep = (ty < DOOR_C - DOOR_W / 2) | (ty > DOOR_C + DOOR_W / 2)
    parts.append(wall(np.column_stack([np.full(keep.sum(), DIVIDER_X), ty[keep]]), 20_000))

    return np.concatenate(parts, axis=0)


def write_synthetic_xyz(path: str, seed: int = 0) -> str:
    """Write the synthetic cloud to an ``.xyz`` file (so it loads through the real
    ``scan2bim.load_point_cloud`` / open3d path) and return ``path``."""
    np.savetxt(path, synthetic_cloud(seed), fmt='%.4f')
    return path
