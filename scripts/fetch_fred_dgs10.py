#!/usr/bin/env python
import csv
import datetime as dt
import sys
import urllib.request
from pathlib import Path


def _parse_date(val: str) -> dt.date:
    return dt.date.fromisoformat(val.strip())


def fetch_dgs10(
    out_path: Path,
    start: dt.date,
    end: dt.date,
) -> int:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read().decode("utf-8")

    rows = []
    reader = csv.reader(raw.splitlines())
    header = next(reader, None)
    if header is None or len(header) < 2:
        raise RuntimeError("Unexpected FRED CSV format.")

    for row in reader:
        if len(row) < 2:
            continue
        date_str, value_str = row[0].strip(), row[1].strip()
        if not date_str or value_str in ("", "."):
            continue
        try:
            d = _parse_date(date_str)
        except ValueError:
            continue
        if d < start or d >= end:
            continue
        try:
            val = float(value_str)
        except ValueError:
            continue
        rows.append((date_str, f"{val:.6f}"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "DGS10"])
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    start = dt.date(1990, 1, 1)
    end = dt.date(2024, 1, 1)
    out_path = Path("data/FRED_DGS10.csv")

    if len(sys.argv) > 1:
        out_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        start = _parse_date(sys.argv[2])
    if len(sys.argv) > 3:
        end = _parse_date(sys.argv[3])

    n = fetch_dgs10(out_path, start, end)
    print(f"Wrote {n} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
