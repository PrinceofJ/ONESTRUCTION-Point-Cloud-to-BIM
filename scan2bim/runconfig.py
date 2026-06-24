"""Single shared loader + cross-stage validation — the one place that turns
``params.yaml`` into a :class:`~scan2bim.config.Config`.

Before this module, every notebook carried its own ~30-line ``_find_project_root`` +
``sys.path`` bootstrap **and** its own ``CFG = Config(...)`` literal, and the copies drifted
(N1 rasterised ``apt_subsampled.ply`` while N2/N3 segmented ``area1.xyz`` — see the refactor
plan §0). Now:

  * ``project_root()`` is the *single* copy of the root-finding logic.
  * ``load_config()`` reads ``params.yaml`` over the ``Config`` defaults, resolves
    ``file_path`` **and** ``out_root`` to absolute paths under the project root, and returns
    the ``CFG`` every notebook uses. ``params.yaml`` is the only file a user edits.
  * ``assert_upstream_config`` / ``assert_points_in_grid`` make it impossible to silently run
    a stage on a cloud/grid that disagrees with the upstream stage it is consuming.

``params.yaml`` is grouped into sections purely for readability (``input:``, ``raster:`` …);
the loader flattens it and matches every leaf key to a ``Config`` field by name, so a section
is just an organisational wrapper. An unknown leaf key is a hard error (catches typos like
``pixel_size`` for ``pixel_m``).
"""

from __future__ import annotations

import os

from .config import Config

# geometry-critical fields that must agree between a stage and the upstream stage it loads.
# (A mismatch here means the two stages saw different clouds or grids — the §0 bug class.)
GEOMETRY_FIELDS = (
    'file_path', 'units_per_meter', 'up_axis', 'voxel_m',
    'pixel_m', 'slab_relative_to', 'slab_lo_m', 'slab_hi_m',
)


def project_root(start=None) -> str:
    """Walk up from ``start`` (or the CWD) to the folder that contains the ``scan2bim``
    package / ``pyproject.toml``. The single copy of the old per-notebook
    ``_find_project_root`` bootstrap."""
    d = os.path.abspath(start or os.getcwd())
    while True:
        if (os.path.isfile(os.path.join(d, 'scan2bim', '__init__.py')) or
                os.path.isfile(os.path.join(d, 'pyproject.toml'))):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.abspath(start or os.getcwd())
        d = parent


def _resolve(root, path):
    """Resolve a possibly-relative config path against the project root."""
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(root, path))


def _collect_overrides(doc, fields, out, prefix=''):
    """Flatten a nested params dict into ``{config_field: value}``; raise on unknown keys."""
    for k, v in doc.items():
        if isinstance(v, dict):
            _collect_overrides(v, fields, out, f'{prefix}{k}.')
        elif k in fields:
            out[k] = v
        else:
            raise KeyError(
                f"params.yaml: unknown key '{prefix}{k}' — not a Config field. "
                f"Check for a typo (e.g. 'pixel_size' should be 'pixel_m').")


def load_config(params='params.yaml', start=None, **overrides) -> Config:
    """Build the pipeline ``CFG`` from ``params.yaml`` over the ``Config`` defaults.

    Steps: find the project root; read ``params.yaml`` (relative to that root) and apply its
    values over ``Config()``; apply any keyword ``overrides`` (these win, for notebook-local
    tweaks); resolve ``file_path`` **and** ``out_root`` to absolute paths under the root.
    A missing ``params.yaml`` is tolerated (pure defaults) so a half-set-up clone still runs.
    """
    root = project_root(start)
    fields = set(Config.__dataclass_fields__)

    params_path = params if os.path.isabs(params) else os.path.join(root, params)
    merged = {}
    if os.path.isfile(params_path):
        import yaml
        with open(params_path) as f:
            doc = yaml.safe_load(f) or {}
        _collect_overrides(doc, fields, merged)
    merged.update(overrides)                      # explicit kwargs win over the file

    cfg = Config(**merged)
    cfg.file_path = _resolve(root, cfg.file_path)
    cfg.out_root = _resolve(root, cfg.out_root)
    return cfg


# ---------------------------------------------------------------------------
# cross-stage validation (refactor plan §3.4 / D5)
# ---------------------------------------------------------------------------
def assert_upstream_config(cfg, upstream_cfg_dict, fields=GEOMETRY_FIELDS):
    """Assert this run's geometry-critical ``cfg`` matches the ``config.json`` an upstream
    stage saved. Raises a clear, field-named error on the first mismatch.

    ``file_path`` is compared by basename so the check survives a fresh clone / different
    machine (different absolute path, same cloud) while still catching a *different cloud*
    (``apt_subsampled.ply`` vs ``area1.xyz`` — the §0 bug). Numeric fields use a tiny float
    tolerance.
    """
    for f in fields:
        if f not in upstream_cfg_dict:
            continue
        have = getattr(cfg, f)
        want = upstream_cfg_dict[f]
        if f == 'file_path':
            # Compare by basename, tolerant of mixed path separators: a config.json written on
            # Windows ('c:\\...\\area1.xyz') is often validated on Linux/Colab, where
            # os.path.basename does NOT split on '\\'. Normalise both to '/' first so the
            # basename is extracted correctly on either platform.
            if (os.path.basename(str(have).replace('\\', '/')) ==
                    os.path.basename(str(want).replace('\\', '/'))):
                continue
        elif isinstance(have, (int, float)) and isinstance(want, (int, float)) \
                and not isinstance(have, bool):
            if abs(float(have) - float(want)) <= 1e-9:
                continue
        elif have == want:
            continue
        raise ValueError(
            f"Config mismatch on '{f}': this run has {have!r} but the upstream stage was "
            f"produced with {want!r}. Re-run the upstream stage after changing params.yaml "
            f"(every stage must see the same cloud + grid).")


def assert_points_in_grid(points, transform, min_frac=0.5):
    """Sanity-check that a reloaded cloud actually lands inside the upstream raster grid.

    Catches the exact §0 failure (one cloud back-projected through another cloud's grid →
    only ~4 % of points in-bounds) even if the provenance fields somehow agree. Returns the
    in-bounds fraction on success.
    """
    from .raster import point_cells
    import numpy as np
    _, _, inb = point_cells(points, transform)
    frac = float(np.mean(inb)) if len(inb) else 0.0
    if frac < min_frac:
        raise ValueError(
            f"Only {frac:.1%} of the reloaded cloud falls inside the upstream grid "
            f"(need >= {min_frac:.0%}). The cloud almost certainly does not match the one "
            f"the upstream stage rasterised — check input.file_path in params.yaml and "
            f"re-run the upstream stage.")
    return frac
