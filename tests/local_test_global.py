"""
Created on Thu Jan 22 14:56:55 2026

Quick test for gloabl dataset.

Author: Luca Persia (USI/ZHAW)
"""

from __future__ import annotations
import sys
from pathlib import Path

def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    while cur != cur.parent:
        if (cur / "src").exists():
            return cur
        cur = cur.parent
    raise RuntimeError("Could not find repo root (expected a 'src' folder).")

def _ensure_importable_src(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root / "src"))

def run() -> int:
    """
    Build panels using GLOBAL windows and write to outputs/.

    Output files:
      outputs/panel_base_global.csv
      outputs/panel_engineered_global.csv
    """
    root = find_repo_root(Path(__file__).parent)
    _ensure_importable_src(root)

    from pumpdownloader.cli import main

    input_csv = root / "data" / "pump_telegram.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "--input", str(input_csv),
    
        "--window-mode", "dataset_global",
    
        "--out-base-global", str(out_dir / "panel_base_global.csv"),
        "--out-engineered-global", str(out_dir / "panel_engineered_global.csv"),
    
        "--interval", "1h",
        "--quote", "BTC",
        "--days-before", "7",
        "--days-after", "7",
        "--window-days", "3",
        "--exchange-filter", "binance",
        "--timezone", "UTC",
    ]

    return main(argv)


if __name__ == "__main__":
    raise SystemExit(run())
