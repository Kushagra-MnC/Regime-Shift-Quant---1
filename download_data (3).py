#!/usr/bin/env python3
"""
download_data.py  —  Regime-Shift Data Bootstrap
==================================================
Run this script ONCE before opening any notebook.
It downloads all required data from public sources and saves to dataset/.

Usage:
    python download_data.py

Sources:
    - ETF prices        : Yahoo Finance (yfinance)
    - India VIX         : NSE / yfinance
    - G-Sec 10Y yield   : FRED (INDIRLTLT01STM) — free, no API key required
    - USD/INR           : Yahoo Finance

Output layout:
    dataset/
        etf/
            NIFTYBEES.parquet
            JUNIORBEES.parquet
            GOLDBEES.parquet
            LIQUIDBEES.parquet
        signal/
            NSEI.parquet
            USDINR.parquet
            INDIAVIX.parquet
        macro/
            GSEC_10Y.parquet
        metadata.json

If India VIX auto-download fails (yfinance coverage is intermittent),
the script prints exact manual-download instructions for nseindia.com.
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Date range ────────────────────────────────────────────────────────────────
START = "2009-06-01"   # extra warmup before backtest
END   = "2024-12-31"

# ── Output directories ────────────────────────────────────────────────────────
ROOT = Path("dataset")
for sub in ("etf", "signal", "macro"):
    (ROOT / sub).mkdir(parents=True, exist_ok=True)

ETF_TICKERS = {
    "NIFTYBEES":  "NIFTYBEES.NS",
    "JUNIORBEES": "JUNIORBEES.NS",
    "GOLDBEES":   "GOLDBEES.NS",
    "LIQUIDBEES": "LIQUIDBEES.NS",
}
SIGNAL_TICKERS = {
    "NSEI":   "^NSEI",
    "USDINR": "USDINR=X",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def strip_tz(df):
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index = pd.to_datetime(df.index).normalize()
    return df


def download_ticker(ticker: str, start: str, end: str,
                    auto_adjust: bool = True,
                    retries: int = 3) -> pd.Series:
    import yfinance as yf
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.Ticker(ticker).history(
                start=start, end=end,
                auto_adjust=auto_adjust,
                actions=False, repair=True,
            )
            if df.empty:
                raise ValueError("Empty DataFrame returned")
            df = strip_tz(df)
            return df["Close"].rename(ticker)
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                wait = 2 ** attempt
                print(f"      attempt {attempt}/{retries} failed ({exc}) — retry in {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"{ticker}: {last_err}")


def save(series: pd.Series, directory: Path, name: str):
    df = series.rename(name).to_frame()
    df.to_parquet(directory / f"{name}.parquet")
    print(f"    saved  {directory / name}.parquet  "
          f"({len(df)} rows,  {df.index[0].date()} → {df.index[-1].date()})")


# ── ETF prices ────────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"  Regime-Shift  ·  Data Bootstrap")
print(f"  Period: {START} → {END}")
print(f"{'─'*60}\n")

log = {"downloaded_at": datetime.now().isoformat(), "tickers": {}}

print("── ETF prices (NSE, auto_adjust=True) ──────────────────────")
for name, ticker in ETF_TICKERS.items():
    print(f"  {name} ({ticker}) …")
    try:
        s = download_ticker(ticker, START, END, auto_adjust=True)
        save(s, ROOT / "etf", name)
        log["tickers"][name] = {"status": "ok", "rows": len(s)}
    except Exception as exc:
        print(f"    ✗  FAILED: {exc}")
        log["tickers"][name] = {"status": "failed", "error": str(exc)}

# ── Signal assets ─────────────────────────────────────────────────────────────
print("\n── Signal assets ────────────────────────────────────────────")
for name, ticker in SIGNAL_TICKERS.items():
    print(f"  {name} ({ticker}) …")
    try:
        s = download_ticker(ticker, START, END, auto_adjust=False)
        save(s, ROOT / "signal", name)
        log["tickers"][name] = {"status": "ok", "rows": len(s)}
    except Exception as exc:
        print(f"    ✗  FAILED: {exc}")
        log["tickers"][name] = {"status": "failed", "error": str(exc)}
    time.sleep(1)

# ── India VIX (multi-strategy) ────────────────────────────────────────────────
print("\n── India VIX (^NSEIVIX) ─────────────────────────────────────")
vix_ok = False

# Strategy 1: yfinance Ticker
print("  [S1] yfinance Ticker.history …")
try:
    s = download_ticker("^NSEIVIX", START, END, auto_adjust=False)
    if len(s.dropna()) >= 100:
        save(s.rename("INDIAVIX"), ROOT / "signal", "INDIAVIX")
        log["tickers"]["INDIAVIX"] = {"status": "ok", "rows": len(s)}
        vix_ok = True
    else:
        print(f"    only {len(s.dropna())} rows — trying S2")
except Exception as e:
    print(f"    ✗  {e}")

# Strategy 2: manual CSV already saved?
if not vix_ok:
    manual = ROOT / "signal" / "INDIAVIX.csv"
    if manual.exists():
        print(f"  [S2] Loading manual CSV from {manual} …")
        df = pd.read_csv(manual, index_col=0, parse_dates=True)
        if "Date" in df.columns:
            df = df.set_index("Date")
        df = strip_tz(df)
        col = next((c for c in df.columns
                    if "close" in c.lower() or "vix" in c.lower()), df.columns[0])
        s = df[col].astype(float).rename("INDIAVIX")
        save(s, ROOT / "signal", "INDIAVIX")
        log["tickers"]["INDIAVIX"] = {"status": "ok", "rows": len(s), "source": "manual CSV"}
        vix_ok = True

if not vix_ok:
    msg = (
        f"\n{'='*62}\n"
        f"  India VIX could not be downloaded automatically.\n\n"
        f"  MANUAL DOWNLOAD (2 minutes):\n"
        f"  1. Go to: https://www.nseindia.com/products-services/indices-vix\n"
        f"  2. Scroll to 'Historical Data' tab\n"
        f"  3. Set From: 01-06-2009   To: 31-12-2024\n"
        f"  4. Click 'Get Data' → 'Download (.csv)'\n"
        f"  5. Save the file as:  dataset/signal/INDIAVIX.csv\n"
        f"  6. Re-run:  python download_data.py\n"
        f"{'='*62}"
    )
    print(msg)
    log["tickers"]["INDIAVIX"] = {"status": "manual_required"}

# ── G-Sec 10Y yield (FRED) ───────────────────────────────────────────────────
print("\n── G-Sec 10Y yield (FRED: INDIRLTLT01STM) ──────────────────")
gsec_ok = False

# Method 1: pandas_datareader
try:
    import pandas_datareader.data as web
    print("  [M1] pandas_datareader.get_data_fred …")
    df = web.DataReader("INDIRLTLT01STM", "fred", start=START, end=END)
    s  = df["INDIRLTLT01STM"].dropna().rename("GSEC_10Y")
    save(s, ROOT / "macro", "GSEC_10Y")
    log["tickers"]["GSEC_10Y"] = {"status": "ok", "rows": len(s)}
    gsec_ok = True
except Exception as e:
    print(f"    [M1] failed ({e}) — trying M2")

# Method 2: direct FRED CSV (no API key)
if not gsec_ok:
    try:
        import urllib.request
        print("  [M2] FRED direct CSV endpoint …")
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=INDIRLTLT01STM"
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read().decode()
        from io import StringIO
        df = pd.read_csv(StringIO(raw), index_col=0, parse_dates=True)
        df.columns = ["GSEC_10Y"]
        df = df.replace(".", np.nan).astype(float)
        s  = df["GSEC_10Y"].dropna()
        s  = s[(s.index >= START) & (s.index <= END)]
        save(s.rename("GSEC_10Y"), ROOT / "macro", "GSEC_10Y")
        log["tickers"]["GSEC_10Y"] = {"status": "ok", "rows": len(s)}
        gsec_ok = True
    except Exception as e:
        print(f"    [M2] failed ({e})")

if not gsec_ok:
    msg = (
        f"\n  G-Sec yield download failed.\n"
        f"  Manual fix:\n"
        f"    1. https://fred.stlouisfed.org/series/INDIRLTLT01STM\n"
        f"    2. Click Download → Format: CSV\n"
        f"    3. Save as: dataset/macro/GSEC_10Y.csv\n"
        f"  Then re-run this script.\n"
    )
    print(msg)
    log["tickers"]["GSEC_10Y"] = {"status": "manual_required"}

# ── Write metadata ────────────────────────────────────────────────────────────
log["start"] = START
log["end"]   = END
with open(ROOT / "metadata.json", "w") as f:
    json.dump(log, f, indent=2, default=str)

# ── Summary ───────────────────────────────────────────────────────────────────
ok     = [k for k, v in log["tickers"].items() if v.get("status") == "ok"]
manual = [k for k, v in log["tickers"].items() if v.get("status") == "manual_required"]
failed = [k for k, v in log["tickers"].items() if v.get("status") == "failed"]

print(f"\n{'─'*60}")
print("  SUMMARY")
print(f"{'─'*60}")
for name, m in log["tickers"].items():
    icon = "✓" if m.get("status") == "ok" else "⚠" if m.get("status") == "manual_required" else "✗"
    rows = f"  ({m.get('rows','?')} rows)" if m.get("rows") else ""
    print(f"  {icon}  {name:<14}{rows}")

if failed:
    print(f"\n  ✗  {len(failed)} download(s) failed: {failed}")
    sys.exit(1)
elif manual:
    print(f"\n  ⚠  {len(manual)} file(s) need manual download: {manual}")
    print("  See instructions above, then re-run: python download_data.py")
    sys.exit(1)
else:
    print(f"\n  ✓  All {len(ok)} data sources saved to {ROOT.resolve()}")
    print("  Next step: open and run the notebooks in order (Stage 1 → 7)")
    sys.exit(0)
