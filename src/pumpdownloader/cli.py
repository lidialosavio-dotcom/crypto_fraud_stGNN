from __future__ import annotations

import argparse
import sys
from pathlib import Path

# It runs as a script by importing the repository.
# If run as a file, Python doesn't know the package; we add project_root/src to sys.path.
if __package__ is None or __package__ == "":
    THIS_FILE = Path(__file__).resolve()
    SRC_DIR = THIS_FILE.parents[1]          
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

from pumpdownloader.panel import PanelConfig, build_panel

def main(argv: list[str] | None = None) -> int:
    """
    Command line interface to build a pump-event panel dataset.

    Input (via CLI flags)
    ---------------------
    --input : str
        Path to pump_telegram.csv.
    --out : str
        Output path for the resulting panel CSV.
    --interval : str
        Kline interval (default '1h').
    --quote : str
        Quote currency used to build Binance REST symbols (default 'BTC').
    --days-before : int
        Days included before each pump timestamp.
    --days-after : int
        Days included after each pump timestamp.
    --exchange-filter : str
        Only keep CSV rows with exchange equal to this value (case-insensitive).
    --timezone : str
        Timezone used to interpret pump timestamps from the CSV (default 'UTC').

    Output
    ------
    Writes a single CSV to --out containing the stacked panel dataset.
    """
    ap = argparse.ArgumentParser(description="Build a panel by downloading Binance klines around pump events.")
    ap.add_argument("--input", required=True, type=Path, help="Path to pump_telegram.csv")
    ap.add_argument("--out", required=True, type=Path, help="Output CSV file path (e.g. outputs/panel.csv)")
    ap.add_argument("--interval", default="1h", help="Binance interval, e.g. 1m, 5m, 1h, 1d")
    ap.add_argument("--quote", default="BTC", help="Quote currency used to build REST symbol (to run the analysis as it is, one should use BTC")
    ap.add_argument("--days-before", type=int, default=3)
    ap.add_argument("--days-after", type=int, default=3)
    ap.add_argument("--exchange-filter", default="binance")
    ap.add_argument("--timezone", default="UTC")
    args = ap.parse_args(argv)

    cfg = PanelConfig(
        interval=args.interval,
        days_before=args.days_before,
        days_after=args.days_after,
        quote=args.quote,
        exchange_filter=args.exchange_filter,
        timezone=args.timezone,
    )

    panel = build_panel(csv_path=args.input, cfg=cfg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out, index=False)
    print("Saved:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
