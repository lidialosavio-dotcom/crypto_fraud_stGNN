from __future__ import annotations

from pathlib import Path
import sys

def find_repo_root(start: Path) -> Path:
    """
    Input
    -----
    start : Path
        Starting directory.

    Output
    ------
    root : Path
        Repository root directory (the one should contain the 'src/').
    """
    cur = start.resolve()
    while cur != cur.parent:
        if (cur / "src").exists():
            return cur
        cur = cur.parent
    raise RuntimeError("Could not find repo root (expected a 'src' folder).")


def run() -> int:
    """
    Local test runner (this assumes standard/imported repo structure).

    Input
    -----
    Uses:
      data/pump_telegram.csv
    Writes:
      outputs/panel.csv

    Output
    ------
    Returns exit code 0 on success.
    """
    root = find_repo_root(Path(__file__).parent)

    # lets make src importable
    sys.path.insert(0, str(root / "src"))

    from pumpdownloader.cli import main  # import after sys.path change

    input_csv = root / "data" / "pump_telegram.csv"
    output_csv = root / "outputs" / "panel.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    return main(["--input", str(input_csv), "--out", str(output_csv)])


if __name__ == "__main__":
    raise SystemExit(run())
