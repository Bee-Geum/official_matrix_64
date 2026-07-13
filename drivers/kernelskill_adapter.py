#!/usr/bin/env python3
import os as _os, sys as _sys
from pathlib import Path as _Path
if _os.environ.get('REST_REMAINING_OFFICIAL') == '1' and any(x in _sys.argv for x in ['--bench-root', '--glob']):
    _script = _Path(__file__).resolve().parent / 'remaining_official_native_driver.py'
    _os.execv(_sys.executable, [_sys.executable, str(_script)] + _sys.argv[1:])

import os as _os, sys as _sys
from pathlib import Path as _Path
if _os.environ.get("REST_OFFICIAL") == "1" and any(x in _sys.argv for x in ["--bench-root", "--glob"]):
    _script = _Path(__file__).resolve().parent / "rest_official_native_driver.py"
    _os.execv(_sys.executable, [_sys.executable, str(_script)] + _sys.argv[1:])

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
cmd = [
    sys.executable,
    str(ROOT / "drivers" / "generic_llm_kernel_driver.py"),
    "--system", "kernelskill",
    "--prefix", "candidate",
]
cmd += sys.argv[1:]
raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))
