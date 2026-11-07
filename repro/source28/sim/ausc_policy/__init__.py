"""Configuration-driven auscultation policy experiments (simulation only)."""

__version__ = "0.1.0"

# This executes before every Native/official CPU ausc_policy submodule import.
# Historical archive packages have their own unchanged initializer and are not
# affected. Normal checkouts remain usable without a deployment lock.
from pathlib import Path as _Path
import os as _os
import sys as _sys

_root = _Path(__file__).parents[2]
if (_root/'RUNTIME_DEPENDENCY_LOCK.json').exists() or str(_root) == '/opt/wam-runtime/code/wam-act-recovery-20261106-v27':
    _sys.dont_write_bytecode = True
    _os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    from .runtime_dependency import install as _install
    _install()
