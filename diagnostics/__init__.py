# Diagnostics package for Anti-Collapse experiments.
#
# Re-export all public names from the top-level diagnostics module
# (diagnostics.py at the project root) so that
#   from diagnostics import run_checkpoint_diagnostics
# works whether Python resolves this package or the .py file.

import importlib.util as _ilu
import os as _os
import sys as _sys

_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_mod_path = _os.path.join(_root, "diagnostics.py")

_spec = _ilu.spec_from_file_location("_diagnostics_module", _mod_path)
_mod = _ilu.module_from_spec(_spec)

# Ensure the project root is on sys.path so that diagnostics.py's
# own imports (models, transport, alpha_utils, etc.) resolve.
if _root not in _sys.path:
    _sys.path.insert(0, _root)

_spec.loader.exec_module(_mod)

# Copy all public names into this package's namespace
from types import ModuleType as _MT
for _name in dir(_mod):
    if not _name.startswith("_"):
        _obj = getattr(_mod, _name)
        if not isinstance(_obj, _MT):
            globals()[_name] = _obj

del _ilu, _os, _sys, _root, _mod_path, _spec, _mod, _MT, _name, _obj
