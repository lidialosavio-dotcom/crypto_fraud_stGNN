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
    p.add_argument(
    "--window-mode",
    choices=["per_token", "dataset_global", "both"],
    default="per_token",
    help="Window definition: per-token, dataset-global, or both.")

    # When window-mode=both, use these
    p.add_argument("--out-base-token", default=None, help="Write per-token base panel CSV")
    p.add_argument("--out-engineered-token", default=None, help="Write per-token engineered panel CSV")
    p.add_argument("--out-base-global", default=None, help="Write dataset-global base panel CSV")
    p.add_argument("--out-engineered-global", default=None, help="Write dataset-global engineered panel CSV")

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
    
    ap = argparse.ArgumentParser(prog="pumpdownloader", allow_abbrev=False)
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

    # Detect whether requests is any of the new multi-output flags
    wants_multi_out = any(
        getattr(args, k) is not None
        for k in (
            "out_base_token",
            "out_engineered_token",
            "out_base_global",
            "out_engineered_global",
        )
    )

    # Legacy mode: only --out provided
    if (not wants_multi_out) and (args.out is not None):
        out = Path(args.out)
        _ensure_parent_dir(out)
        # here the deterministic choice is per_token
        df = build_panel(input_csv, cfg, 
                         window_mode=args.window_mode if args.window_mode != "both" else "per_token")
        df.to_csv(out, index=False)
        print(f"[WRITE] {out}  rows={len(df):,}")
        return 0

    # build panels
    outs = build_panels(input_csv, cfg, window_mode=args.window_mode)
    if args.window_mode == "both":
        # Require at least one of the both-mode outputs
        if not wants_multi_out:
            ap.error("window-mode=both requires at least one of: "
                     "--out-base-token/--out-engineered-token/--out-base-global/--out-engineered-global")
    
        _write_panel_csv(args.out_base_token, outs, "panel_base_token")
        _write_panel_csv(args.out_engineered_token, outs, "panel_engineered_token")
        _write_panel_csv(args.out_base_global, outs, "panel_base_global")
        _write_panel_csv(args.out_engineered_global, outs, "panel_engineered_global")
    
    else:
        # Single mode -> keep a single pair of outputs (token OR global depending on window-mode)
        # If you still have --out-base/--out-engineered CLI, use them here.
        # If not, keep legacy --out or define single-mode outputs.
        if args.out is not None:
            out = Path(args.out)
            _ensure_parent_dir(out)
            df = build_panel(input_csv, cfg, window_mode=args.window_mode)
            df.to_csv(out, index=False)
            print(f"[WRITE] {out} rows={len(df):,}")
        

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

