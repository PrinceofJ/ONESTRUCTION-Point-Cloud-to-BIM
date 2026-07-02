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
    # Normalise separators FIRST: a params.yaml authored on Windows may contain
    # 'data\Area_1'; on POSIX (Colab) the backslash is a literal filename char and the
    # path silently fails to resolve. Forward slashes work on both platforms.
    p = str(path).replace('\\', '/')
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))


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


def load_config(params='params.yaml', start=None, method=None, **overrides) -> Config:
    """Build the effective Config. Precedence (last wins):

        Config defaults  <  params.yaml top-level  <  params.yaml methods.<method>  <  **overrides

    ``method`` selects a per-method override block from the top-level ``methods:``
    mapping in params.yaml (e.g. ``load_config(method='sam_auto')``). Method notebooks
    MUST pass their method name instead of mutating ``CFG`` fields after load — the
    per-method values are declared data in params.yaml, not imperative notebook edits.
    ``**overrides`` are reserved for runtime/environment values only (a Colab checkpoint
    path, an out_root on a mounted drive), never for science parameters.
    """
    root = project_root(start)
    fields = set(Config.__dataclass_fields__)

    params_path = params if os.path.isabs(params) else os.path.join(root, params)
    merged = {}
    method_blocks = {}
    if os.path.isfile(params_path):
        import yaml
        with open(params_path) as f:
            doc = yaml.safe_load(f) or {}
        # Pop 'methods' BEFORE the flat collect: _collect_overrides recurses every
        # nested mapping, so an un-popped methods block would apply EVERY method's
        # overrides globally (and in dict order, so the last method would win).
        method_blocks = doc.pop('methods', None) or {}
        if not isinstance(method_blocks, dict):
            raise TypeError(
                "params.yaml: 'methods' must be a mapping of method name -> "
                "{Config field: value}, e.g. methods: {sam_auto: {sam_image_mode: occupancy}}")
        _collect_overrides(doc, fields, merged)
    if method is not None:
        if method not in method_blocks:
            raise KeyError(
                f"load_config(method={method!r}): params.yaml declares no "
                f"'methods.{method}' block. Declared methods: "
                f"{sorted(method_blocks) or '(none)'}.")
        _collect_overrides(method_blocks[method] or {}, fields, merged,
                           prefix=f'methods.{method}.')
    merged.update(overrides)                      # explicit kwargs win over the file

    cfg = Config(**merged)
    cfg.file_path = _resolve(root, cfg.file_path)
    cfg.gt_dir = _resolve(root, cfg.gt_dir)
    cfg.out_root = _resolve(root, cfg.out_root)
    cfg.method = method            # provenance for config_snapshot (not a dataclass field)
    return cfg


def config_snapshot(cfg) -> str:
    """Loggable resolved-config view: the selected method + every field whose value
    deviates from the Config dataclass default. Drivers print this right after
    load_config() so each run records exactly which knobs were in effect."""
    defaults = Config()
    method = getattr(cfg, 'method', None)
    lines = [f"resolved config (method={method or 'global'}):"]
    for f in sorted(Config.__dataclass_fields__):
        have, base = getattr(cfg, f), getattr(defaults, f)
        if have != base:
            lines.append('  %-30s = %r   (default %r)' % (f, have, base))
    if len(lines) == 1:
        lines.append('  (all fields at Config defaults)')
    return '\n'.join(lines)


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
