"""
Created on Fri Dec 12 17:16:45 2025

This command reads a pump_telegram.csv file and builds one or more panels
(base/engineered, multi-pump/single-pump, optionally windowed around pump times).

Author: Luca Persia (USI/ZHAW), Lidia Losavio (USI)
"""

from __future__ import annotations
import argparse
from pathlib import Path
from pumpdownloader.panel import PanelConfig, build_panel, build_panels


### CLI 
def _add_args(p: argparse.ArgumentParser) -> None:
    """
    Register CLI arguments.

    Parameters
    ----------
    p : argparse.ArgumentParser
        Parser to populate.

    Returns
    -------
    None
    """
    
    p.add_argument("--input", required=True, help="Input pump_telegram.csv")

    # Legacy single output
    p.add_argument("--out", default=None, help="(legacy) Write only the engineered multi-pump global panel")

    # Updated multi-output paths
    p.add_argument("--out-base", default=None, help="Write GLOBAL multi-pump base panel CSV")
    p.add_argument("--out-engineered", default=None, help="Write GLOBAL multi-pump engineered panel CSV")
    p.add_argument("--out-single-base", default=None, help="Write GLOBAL single-pump base panel CSV")
    p.add_argument("--out-single-engineered", default=None, help="Write GLOBAL single-pump engineered panel CSV")

    p.add_argument("--out-base-w3", default=None, help="Write WINDOWED(+/-window_days) multi-pump base panel CSV")
    p.add_argument("--out-engineered-w3", default=None, help="Write WINDOWED(+/-window_days) multi-pump engineered panel CSV")
    p.add_argument("--out-single-base-w3", default=None, help="Write WINDOWED(+/-window_days) single-pump base panel CSV")
    p.add_argument("--out-single-engineered-w3", default=None, help="Write WINDOWED(+/-window_days) single-pump engineered panel CSV")

    # Panel configuration
    p.add_argument("--interval", default="1h")
    p.add_argument("--quote", default="BTC")
    p.add_argument("--days-before", type=int, default=7)
    p.add_argument("--days-after", type=int, default=7)
    p.add_argument("--window-days", type=int, default=3)
    p.add_argument("--exchange-filter", default="binance")
    p.add_argument("--timezone", default="UTC")
    p.add_argument("--no-round", action="store_true", help="Do not round pump times to interval")


## I/O helpers to build panels locally
def _ensure_parent_dir(path: Path) -> None:
    """
    Create parent directory for a file path.

    Parameters
    ----------
    path : Path
        Output file path.

    Returns
    -------
    None
    """
    
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_panel_csv(out_path: str | None, outs: dict, key: str) -> None:
    """
    Write a panel to CSV if out_path was provided.

    Parameters
    ----------
    out_path : str | None
        Output CSV path.
    outs : dict
        Dictionary returned by build_panels().
    key : str
        Key inside outs for the desired panel.

    Returns
    -------
    None
    """
    
    if out_path is None:
        return

    outp = Path(out_path)
    _ensure_parent_dir(outp)

    df = outs[key]
    df.to_csv(outp, index=False)
    print(f"[WRITE] {outp}  key={key} rows={len(df):,}")


### Main
def main(argv: list[str] | None = None) -> int:
    """
    CLI main.

    Parameters
    ----------
    argv : list[str] | None
        Optional argument list. If None, argparse uses sys.argv.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    
    ap = argparse.ArgumentParser(prog="pumpdownloader")
    _add_args(ap)
    args = ap.parse_args(argv)

    cfg = PanelConfig(
        interval=args.interval,
        days_before=args.days_before,
        days_after=args.days_after,
        window_days=args.window_days,
        quote=args.quote,
        exchange_filter=args.exchange_filter,
        timezone=args.timezone,
        round_pump_time=(not args.no_round),
    )

    input_csv = Path(args.input)

    # Detect whether the user requested any of the new multi-output flags
    wants_multi_out = any(
        getattr(args, k) is not None
        for k in (
            "out_base", "out_engineered", "out_single_base", "out_single_engineered",
            "out_base_w3", "out_engineered_w3", "out_single_base_w3", "out_single_engineered_w3",
        )
    )

    # Legacy mode: only --out provided
    if (not wants_multi_out) and (args.out is not None):
        out = Path(args.out)
        _ensure_parent_dir(out)

        df = build_panel(input_csv, cfg)
        df.to_csv(out, index=False)
        print(f"[WRITE] {out}  rows={len(df):,}")
        return 0

    # Updated: build all panels according to output provided
    outs = build_panels(input_csv, cfg)

    _write_panel_csv(args.out_base, outs, "panel_base")
    _write_panel_csv(args.out_engineered, outs, "panel_engineered")
    _write_panel_csv(args.out_single_base, outs, "single_pump_base")
    _write_panel_csv(args.out_single_engineered, outs, "single_pump_engineered")

    _write_panel_csv(args.out_base_w3, outs, "panel_base_w")
    _write_panel_csv(args.out_engineered_w3, outs, "panel_engineered_w")
    _write_panel_csv(args.out_single_base_w3, outs, "single_pump_base_w")
    _write_panel_csv(args.out_single_engineered_w3, outs, "single_pump_engineered_w")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

