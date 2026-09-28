

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import gc
import io
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from gwm.training.evaluate_action_rl import main as evaluate_main


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def main() -> None:
    _emit({"ready": True})
    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        request = json.loads(raw_line)
        if request.get("command") == "close":
            _emit({"closed": True})
            return
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                evaluate_main([str(value) for value in request["args"]])
            response: dict[str, object] = {
                "ok": True,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
            }
        except BaseException:
            response = {
                "ok": False,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "traceback": traceback.format_exc(),
            }
        finally:



            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _emit(response)


if __name__ == "__main__":
    main()
