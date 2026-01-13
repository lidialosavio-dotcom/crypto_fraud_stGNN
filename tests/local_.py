"""
Created on Tue Jan 6 09:21:35 2026

Local runner for building pump-and-dump panels.

This script is meant for quick local execution in a standard repo layout:
  repo_root/
    src/
    data/pump_telegram.csv
    outputs/

It calls the pumpdownloader CLI with explicit output paths and writes all panels.

Author: Lidia Losavio (USI), Luca Persia (USI/ZHAW)
"""

from __future__ import annotations
import sys
from pathlib import Path


# Repo utilities
def find_repo_root(start: Path) -> Path:
    """
    Find the repository root by walking up until a 'src' folder is found.

    Input
    -----
    start : Path
        Starting directory (typically Path(__file__).parent).

    Output
    ------
    Path
        Repository root directory.

    Raises
    ------
    RuntimeError
        If a 'src' folder is not found up the directory tree.
    """
    cur = start.resolve()
    while cur != cur.parent:
        if (cur / "src").exists():
            return cur
        cur = cur.parent
    raise RuntimeError("Could not find repo root (expected a 'src' folder).")


def _ensure_importable_src(repo_root: Path) -> None:
    """
    Make repo_root/src importable by inserting it into sys.path.

    Input
    -----
    repo_root : Path
        Repository root returned by find_repo_root().

    Output
    ------
    None
    """
    sys.path.insert(0, str(repo_root / "src"))


# Runner
def run() -> int:
    """
    Build all panels from data/pump_telegram.csv and write them to outputs/.

    Input
    -----
    Expects:
      repo_root/data/pump_telegram.csv

    Output
    ------
    Writes 8 CSV panels to:
      repo_root/outputs/

    GLOBAL (single global window across all tokens)
      - panel_base.csv
      - panel_engineered.csv
      - single_pump_base.csv
      - single_pump_engineered.csv

    WINDOWED (stacked per-pump chunks, +/- window-days)
      - panel_base_w3.csv
      - panel_engineered_w3.csv
      - single_pump_base_w3.csv
      - single_pump_engineered_w3.csv

    Returns
    -------
    int
        Exit code (0 for success).
    """
    
    root = find_repo_root(Path(__file__).parent)
    _ensure_importable_src(root)

    # Import after sys.path change
    from pumpdownloader.cli import main

    input_csv = root / "data" / "pump_telegram.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "--input", str(input_csv),

        # GLOBAL outputs
        "--out-base", str(out_dir / "panel_base.csv"),
        "--out-engineered", str(out_dir / "panel_engineered.csv"),
        "--out-single-base", str(out_dir / "single_pump_base.csv"),
        "--out-single-engineered", str(out_dir / "single_pump_engineered.csv"),

        # WINDOWED outputs (+/- N days) (we set it to 3)
        "--out-base-w3", str(out_dir / "panel_base_w3.csv"),
        "--out-engineered-w3", str(out_dir / "panel_engineered_w3.csv"),
        "--out-single-base-w3", str(out_dir / "single_pump_base_w3.csv"),
        "--out-single-engineered-w3", str(out_dir / "single_pump_engineered_w3.csv"),

        # Panel config
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
