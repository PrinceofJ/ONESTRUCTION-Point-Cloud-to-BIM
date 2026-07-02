"""Config loader and cross-stage validation."""

from __future__ import annotations

import os

from .config import Config

GEOMETRY_FIELDS = (
    'file_path', 'units_per_meter', 'up_axis', 'voxel_m',
    'pixel_m', 'slab_relative_to', 'slab_lo_m', 'slab_hi_m',
)


def project_root(start=None) -> str:
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
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(root, path))


def _collect_overrides(doc, fields, out, prefix=''):
    for k, v in doc.items():
        if isinstance(v, dict):
            _collect_overrides(v, fields, out, f'{prefix}{k}.')
        elif k in fields:
            out[k] = v
        else:
            raise KeyError(
                f"params.yaml: unknown key '{prefix}{k}' - not a Config field. "
                f"Check for a typo (e.g. 'pixel_size' should be 'pixel_m').")


def load_config(params='params.yaml', start=None, **overrides) -> Config:
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
    cfg.gt_dir = _resolve(root, cfg.gt_dir)
    cfg.out_root = _resolve(root, cfg.out_root)
    return cfg


def assert_upstream_config(cfg, upstream_cfg_dict, fields=GEOMETRY_FIELDS):
    for f in fields:
        if f not in upstream_cfg_dict:
            continue
        have = getattr(cfg, f)
        want = upstream_cfg_dict[f]
        if f == 'file_path':
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
    from .raster import point_cells
    import numpy as np
    _, _, inb = point_cells(points, transform)
    frac = float(np.mean(inb)) if len(inb) else 0.0
    if frac < min_frac:
        raise ValueError(
            f"Only {frac:.1%} of the reloaded cloud falls inside the upstream grid "
            f"(need >= {min_frac:.0%}). The cloud almost certainly does not match the one "
            f"the upstream stage rasterised - check input.file_path in params.yaml and "
            f"re-run the upstream stage.")
    return frac
