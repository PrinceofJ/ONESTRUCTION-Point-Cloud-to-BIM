"""Task 12 — the centralized-config contract.

params.yaml is the ONE editable surface; per-method overrides are declared data under its
`methods:` block; notebooks never mutate science fields on CFG after load. These tests pin
the loader semantics (precedence, rejection, portability) and enforce the notebook invariant
across the whole reorganized tree (independent of tests/test_notebooks.py, which Task 11
retargets separately).
"""
import glob
import json
import os
import re

import pytest

import scan2bim
from scan2bim.config import Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_params(tmp_path, text):
    p = tmp_path / 'params.yaml'
    p.write_text(text, encoding='utf-8')
    return str(p)


# ---------------------------------------------------------------- loader semantics ----

def test_unknown_top_level_key_raises(tmp_path):
    p = _write_params(tmp_path, 'raster:\n  pixel_size: 0.03\n')   # typo for pixel_m
    with pytest.raises(KeyError, match='pixel_size'):
        scan2bim.load_config(params=p, start=ROOT)


def test_unknown_key_inside_method_block_raises(tmp_path):
    p = _write_params(tmp_path, 'methods:\n  sam_auto:\n    sam_imag_mode: occupancy\n')
    with pytest.raises(KeyError, match='methods.sam_auto.sam_imag_mode'):
        scan2bim.load_config(params=p, start=ROOT, method='sam_auto')


def test_methods_must_be_mapping(tmp_path):
    p = _write_params(tmp_path, 'methods: [sam_auto]\n')
    with pytest.raises(TypeError, match='methods'):
        scan2bim.load_config(params=p, start=ROOT)


def test_unknown_method_raises_and_names_declared(tmp_path):
    p = _write_params(tmp_path, 'methods:\n  sam_auto: {sam_image_mode: occupancy}\n')
    with pytest.raises(KeyError, match=r"sam_auto"):
        scan2bim.load_config(params=p, start=ROOT, method='typo_method')


def test_precedence_default_lt_global_lt_method_lt_kwargs(tmp_path):
    p = _write_params(tmp_path, (
        'sam_refinement:\n  sam_image_mode: stack\n'
        'methods:\n  m1: {sam_image_mode: occupancy}\n'))
    assert scan2bim.load_config(params=p, start=ROOT).sam_image_mode == 'stack'
    assert scan2bim.load_config(params=p, start=ROOT, method='m1').sam_image_mode == 'occupancy'
    assert scan2bim.load_config(params=p, start=ROOT, method='m1',
                                sam_image_mode='xyz').sam_image_mode == 'xyz'


def test_methods_block_does_not_leak_into_global(tmp_path):
    # _collect_overrides recurses nested dicts, so an un-popped methods block would apply
    # every method's overrides globally. Pin the pop.
    p = _write_params(tmp_path, 'methods:\n  m1: {sam_image_mode: occupancy}\n')
    assert scan2bim.load_config(params=p, start=ROOT).sam_image_mode == \
        Config().sam_image_mode == 'stack'


def test_method_provenance_and_snapshot(tmp_path):
    p = _write_params(tmp_path, 'methods:\n  m1: {sam_points_per_side: 30}\n')
    cfg = scan2bim.load_config(params=p, start=ROOT, method='m1')
    assert cfg.method == 'm1'
    snap = scan2bim.config_snapshot(cfg)
    assert 'method=m1' in snap and 'sam_points_per_side' in snap and '30' in snap
    assert scan2bim.load_config(params=p, start=ROOT).method is None


def test_windows_backslash_paths_resolve_on_any_os(tmp_path):
    # A params.yaml authored on Windows ('data\Area_1') must resolve identically to the
    # forward-slash form — the loader normalises separators before joining.
    fwd = scan2bim.load_config(start=ROOT, gt_dir='data/Area_1')
    back = scan2bim.load_config(start=ROOT, gt_dir='data\\Area_1')
    assert back.gt_dir == fwd.gt_dir


# ---------------------------------------------------------------- the real params.yaml ----

def test_real_params_yaml_loads_and_declares_the_three_methods():
    cfg = scan2bim.load_config(start=ROOT)               # raises on any unknown key
    assert cfg.sam_image_mode == 'stack'                 # global default input mode
    for m, mode in (('geometric', 'stack'), ('sam_auto', 'occupancy'), ('sam_refine', 'stack')):
        c = scan2bim.load_config(start=ROOT, method=m)
        assert c.sam_image_mode == mode, (m, c.sam_image_mode)
        assert c.method == m


def test_real_params_yaml_keeps_paper_faithful_per_method_thresholds():
    # Intentional (paper-faithful) duplication — Task 05/12. Alarm if someone collapses them.
    cfg = scan2bim.load_config(start=ROOT)
    assert cfg.min_room_area_m2 == 1.0                   # watershed's own A
    assert cfg.sam_auto_min_room_area_m2 == 1.5          # paper's A for SAM-auto
    assert cfg.eval_min_room_area_m2 == 1.0              # harmonized comparison value
    assert cfg.min_coverage_frac == 0.25
    assert cfg.sam_auto_min_coverage_frac == 0.5
    assert cfg.eval_min_coverage_frac == 0.25


# ---------------------------------------------------------------- notebook invariant ----

# Fields a notebook MAY assign on CFG after load: runtime/environment values only.
RUNTIME_OK = {'file_path'}                # Colab cloud copies / CLOUD_OVERRIDE
NB_GLOB = os.path.join(ROOT, 'notebooks', '**', '*.ipynb')


def _notebook_code(path):
    nb = json.load(open(path, encoding='utf-8'))
    return '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')


def test_notebooks_never_mutate_science_config():
    nbs = sorted(glob.glob(NB_GLOB, recursive=True))
    assert nbs, 'no notebooks found — glob broken?'
    offenders = []
    for path in nbs:
        for field in re.findall(r'^\s*CFG\.(\w+)\s*=[^=]', _notebook_code(path), re.M):
            if field not in RUNTIME_OK:
                offenders.append(f'{os.path.relpath(path, ROOT)}: CFG.{field} = ...')
    assert not offenders, (
        'Science parameters must be declared in params.yaml (methods: block for per-method '
        'values), never assigned in a notebook:\n  ' + '\n  '.join(offenders))


def test_method_notebooks_pass_method_to_load_config():
    # The three notebooks that need per-method values must select them via load_config(method=...)
    expect = {
        'notebooks/methods/SAM/notebook_1_sam_auto_segmentation.ipynb': 'sam_auto',
        'notebooks/methods/SAM/notebook_2_wall_assignment.ipynb': 'sam_auto',
        'notebooks/methods/geometric_SAM/notebook_2_sam_refinement.ipynb': 'sam_refine',
    }
    for rel, method in expect.items():
        code = _notebook_code(os.path.join(ROOT, rel))
        assert f"method='{method}'" in code or f'method="{method}"' in code, \
            f'{rel}: expected load_config(method={method!r})'
