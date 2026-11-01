"""Generate deterministic inputs for the controlled local-tool workloads."""
from __future__ import annotations

import csv
import json

from hgre_core.paths import LOCAL_TOOL_DATA

DATA = LOCAL_TOOL_DATA
CSV_ROWS = [10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]
JSONL_ROWS = CSV_ROWS
GZIP_MB = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500]
MATRIX_N = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]


def ensure():
    (DATA / "csv").mkdir(parents=True, exist_ok=True)
    (DATA / "jsonl").mkdir(parents=True, exist_ok=True)
    (DATA / "gzip").mkdir(parents=True, exist_ok=True)
    for rows in CSV_ROWS:
        path = DATA / "csv" / ("rows_%d.csv" % rows)
        if path.exists() and path.stat().st_size > 0:
            continue
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "name", "value"])
            for i in range(rows):
                writer.writerow([i, "row_%d" % (i % 1000), i % 1000])
    for rows in JSONL_ROWS:
        path = DATA / "jsonl" / ("rows_%d.jsonl" % rows)
        if path.exists() and path.stat().st_size > 0:
            continue
        with open(path, "w", encoding="utf-8") as fh:
            for i in range(rows):
                fh.write(json.dumps({"id": i, "value": i % 1000, "keep": i % 7 == 0}) + "\n")
    block = b"0123456789abcdef" * 4096
    for mb in GZIP_MB:
        path = DATA / "gzip" / ("input_%dmb.bin" % mb)
        if path.exists() and path.stat().st_size > 0:
            continue
        target = mb * 1024 * 1024
        with open(path, "wb") as fh:
            written = 0
            while written < target:
                n = min(len(block), target - written)
                fh.write(block[:n])
                written += n
    return {"csv": len(CSV_ROWS), "jsonl": len(JSONL_ROWS), "gzip": len(GZIP_MB), "matrix": len(MATRIX_N)}


if __name__ == "__main__":
    print(ensure())
