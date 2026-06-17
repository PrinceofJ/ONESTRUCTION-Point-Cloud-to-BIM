"""Static validation of the four driver notebooks (no execution / no GPU needed):

  * every notebook is valid JSON with the expected stage numbering in its filename;
  * every code cell compiles (syntax check);
  * every ``A.* / scan2bim.* / viz.*`` symbol a cell references actually exists, and every
    ``CFG.*`` attribute is a real ``Config`` field/property.

This is what guards the renumber + the Notebook 4 rewrite: if a symbol is renamed or a stage
constant drifts, these tests fail.
"""

import os
import re
import glob
import json

import pytest

import scan2bim
from scan2bim import Config, artifacts as A, viz

NB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'notebooks')
NOTEBOOKS = sorted(glob.glob(os.path.join(NB_DIR, 'notebook_*.ipynb')))

_CFG_FIELDS = set(Config.__dataclass_fields__) | {
    n for n in dir(Config) if isinstance(getattr(Config, n, None), property)}

_REF = {
    'A': (A, re.compile(r'(?<![\w.])A\.([A-Za-z_]\w*)')),
    'scan2bim': (scan2bim, re.compile(r'(?<![\w.])scan2bim\.([A-Za-z_]\w*)')),
    'viz': (viz, re.compile(r'(?<![\w.])viz\.([A-Za-z_]\w*)')),
}
_CFG_RE = re.compile(r'(?<![\w.])CFG\.([A-Za-z_]\w*)')


def _code_cells(path):
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            src = cell['source']
            yield ''.join(src) if isinstance(src, list) else src


def _nb(name):
    """Resolve a notebook path by basename (so tests don't depend on glob ordering)."""
    for p in NOTEBOOKS:
        if os.path.basename(p) == name:
            return p
    raise AssertionError(f'{name} not found in {[os.path.basename(p) for p in NOTEBOOKS]}')


def test_notebooks_present_and_numbered_in_run_order():
    names = [os.path.basename(p) for p in NOTEBOOKS]
    assert names == [
        'notebook_1_occupancy_raster.ipynb',
        'notebook_2_watershed_segmentation.ipynb',
        'notebook_3_room_masks_and_wall_assignment.ipynb',
        'notebook_4_sam_refinement.ipynb',
        'notebook_5_walls_on_sam_refined.ipynb',
    ], names


@pytest.mark.parametrize('path', NOTEBOOKS, ids=lambda p: os.path.basename(p))
def test_every_code_cell_compiles(path):
    for i, code in enumerate(_code_cells(path)):
        compile(code, f'{os.path.basename(path)}::cell{i}', 'exec')


@pytest.mark.parametrize('path', NOTEBOOKS, ids=lambda p: os.path.basename(p))
def test_referenced_symbols_exist(path):
    code = '\n'.join(_code_cells(path))
    for label, (mod, rx) in _REF.items():
        for name in set(rx.findall(code)):
            assert hasattr(mod, name), f'{os.path.basename(path)}: {label}.{name} does not exist'
    for name in set(_CFG_RE.findall(code)):
        assert name in _CFG_FIELDS, f'{os.path.basename(path)}: CFG.{name} is not a Config field'


def test_notebook4_is_the_colab_gpu_stage():
    code = '\n'.join(_code_cells(_nb('notebook_4_sam_refinement.ipynb')))
    assert 'google.colab' in code and 'drive.mount' in code      # Drive mount
    assert 'cuda.is_available' in code                           # GPU check
    assert 'sam2.1_hiera_large.pt' in code                       # verified SAM 2.1 checkpoint
    assert 'configs/sam2.1/sam2.1_hiera_l.yaml' in code          # verified config name
    # CFG now comes from the unified loader, VALIDATED against the watershed-stage config.json
    assert 'load_config' in code and 'assert_upstream_config' in code
    assert 'A.STAGE2' in code                                    # validates against stage 2
    assert 'A.STAGE4' in code                                    # writes the refined stage


# ---- locks in the refactor (REFACTOR_PLAN §6 acceptance criteria) ----
_NO_LEGACY = ['notebook_1_occupancy_raster.ipynb',
              'notebook_2_watershed_segmentation.ipynb',
              'notebook_3_room_masks_and_wall_assignment.ipynb',
              'notebook_5_walls_on_sam_refined.ipynb']


@pytest.mark.parametrize('name', _NO_LEGACY)
def test_local_notebooks_have_no_inline_config_or_switch(name):
    code = '\n'.join(_code_cells(_nb(name)))
    assert 'Config(' not in code, f'{name}: a `CFG = Config(...)` literal survived'
    assert 'ROOM_MASK_SOURCE' not in code, f'{name}: the N3 variable switch survived'
    assert '_find_project_root' not in code, f'{name}: a duplicated bootstrap survived'
    assert 'load_config' in code, f'{name}: should read CFG via scan2bim.load_config()'


def test_notebook5_assigns_walls_on_sam_refined_masks():
    code = '\n'.join(_code_cells(_nb('notebook_5_walls_on_sam_refined.ipynb')))
    assert 'A.STAGE4' in code and 'A.STAGE5' in code             # reads stage 4 -> writes stage 5
    assert 'A.REFINED_LABELS_NPY' in code                       # the SAM-refined labels
    assert 'room_wall_masks_boundary_ring' in code              # same assignment as N3
    assert 'Notebook 4' in code or 'notebook 4' in code         # fail-loud guidance present
