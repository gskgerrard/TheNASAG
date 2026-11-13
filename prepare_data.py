"""Combine dated local CSV/CVV inputs without inventing missing measurements."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "historical_hazards.csv"
DATE_COLUMNS = ("date", "time", "datetime", "timestamp", "acq_date", "acquisition_date")


def prepare():
    files = sorted(p for p in DATA_DIR.rglob("*")
                   if p.is_file() and p.suffix.lower() in {".csv", ".cvv"}
                   and p.resolve() != OUTPUT_FILE.resolve()
                   and p.name.lower() != "coordinates.csv")
    frames = []
    for path in files:
        try:
            frame = pd.read_csv(path)
            date_col = next((c for c in frame.columns if c.lower() in DATE_COLUMNS), None)
            if date_col is None:
                print(f"Skipped {path.name}: no date column")
                continue
            if date_col != "date":
                frame = frame.rename(columns={date_col: "date"})
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.dropna(subset=["date"])
            if not frame.empty:
                frame["source_file"] = path.name
                frames.append(frame)
                print(f"Loaded {path.name}: {len(frame):,} dated rows")
        except Exception as exc:
            print(f"Skipped {path.name}: {type(exc).__name__}: {exc}")

    if not frames:
        raise ValueError(f"No dated CSV/CVV inputs found in {DATA_DIR}")
    combined = pd.concat(frames, ignore_index=True, sort=False).sort_values("date")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {len(combined):,} rows to {OUTPUT_FILE}")
    return combined


if __name__ == "__main__":
    prepare()
