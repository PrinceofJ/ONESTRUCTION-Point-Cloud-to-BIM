"""Tests for the shared loader ``scan2bim.runconfig`` (D1/D2/§3.2).

``load_config`` must: read ``params.yaml`` over the ``Config`` defaults, let keyword
overrides win, resolve ``file_path`` **and** ``out_root`` to absolute paths under the project
root, reject unknown keys, and tolerate a missing params file. ``Config`` must round-trip
through ``to_dict``/``from_dict``.
"""

import os
import textwrap

import pytest

import scan2bim
from scan2bim import Config


def _make_project(tmp_path, params_text):
    """A minimal project root: a pyproject.toml marker + a params.yaml."""
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "tmp"\n')
    if params_text is not None:
        (tmp_path / 'params.yaml').write_text(textwrap.dedent(params_text))
    return str(tmp_path)


def test_reads_params_over_defaults(tmp_path):
    root = _make_project(tmp_path, """
        input:
          file_path: data/mine.xyz
          units_per_meter: 1000.0
        raster:
          pixel_m: 0.05
    """)
    cfg = scan2bim.load_config(start=root)
    assert cfg.pixel_m == 0.05
    assert cfg.units_per_meter == 1000.0
    # an omitted field falls back to the Config default
    assert cfg.slab_hi_m == Config().slab_hi_m


def test_resolves_file_path_and_out_root_absolute(tmp_path):
    root = _make_project(tmp_path, """
        input:
          file_path: data/mine.xyz
        output:
          out_root: scan2bim_out
    """)
    cfg = scan2bim.load_config(start=root)
    assert os.path.isabs(cfg.file_path) and os.path.isabs(cfg.out_root)
    assert cfg.file_path == os.path.normpath(os.path.join(root, 'data', 'mine.xyz'))
    assert cfg.out_root == os.path.normpath(os.path.join(root, 'scan2bim_out'))


def test_keyword_overrides_win(tmp_path):
    root = _make_project(tmp_path, "raster:\n  pixel_m: 0.05\n")
    cfg = scan2bim.load_config(start=root, pixel_m=0.07, min_points_per_cell=9)
    assert cfg.pixel_m == 0.07
    assert cfg.min_points_per_cell == 9


def test_unknown_key_raises(tmp_path):
    root = _make_project(tmp_path, "raster:\n  pixel_size: 0.05\n")   # typo for pixel_m
    with pytest.raises(KeyError):
        scan2bim.load_config(start=root)


def test_missing_params_file_uses_defaults(tmp_path):
    root = _make_project(tmp_path, None)             # no params.yaml at all
    cfg = scan2bim.load_config(start=root)
    assert cfg.pixel_m == Config().pixel_m
    assert os.path.isabs(cfg.file_path)              # still resolved under the root


def test_config_round_trips(tmp_path):
    root = _make_project(tmp_path, "raster:\n  pixel_m: 0.04\n")
    cfg = scan2bim.load_config(start=root)
    again = Config.from_dict(cfg.to_dict())
    assert again.to_dict() == cfg.to_dict()


def test_project_root_finds_marker(tmp_path):
    sub = tmp_path / 'a' / 'b'
    sub.mkdir(parents=True)
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "tmp"\n')
    assert scan2bim.project_root(start=str(sub)) == str(tmp_path)
