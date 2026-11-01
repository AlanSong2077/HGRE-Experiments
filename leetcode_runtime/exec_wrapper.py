"""Exec wrapper: records true start/end of the student program."""
import json
import runpy
import sys
import time
import traceback

path = sys.argv[1]
start = time.time()
rc = 0
try:
    runpy.run_path(path, run_name="__main__")
except SystemExit as exc:
    rc = int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
except BaseException:
    traceback.print_exc()
    rc = 1
end = time.time()
print("__WP_TIMING__" + json.dumps({"start": start, "end": end, "rc": rc}), file=sys.stderr, flush=True)
sys.exit(rc)
