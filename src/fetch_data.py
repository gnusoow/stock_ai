import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf
import yaml
from datetime import datetime, timedelta

def download_prices(symbols, start, end, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        try:
            df = yf.download(sym, start=start, end=end, auto_adjust=False)
            if df.empty:
                print(f"[WARN] No data for {sym}")
                continue
            df = df.reset_index()
            # 기술 지표용으로 별도 파일명(관례)
            (out_dir / sym).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_dir / sym / "prices_tech.csv", index=False)
            print("saved:", out_dir / sym / "prices_tech.csv")
        except Exception as e:
            print(f"[ERROR] {sym}: {e}")

def download_macro(out_dir: Path):
    """
    ^VIX, ^TNX 종가를 20년치 받아서 data/macro_sample.csv로 저장.
    - yf.Ticker(...).history() 사용으로 단일티커 포맷 고정
    - 숫자형 강제 변환 + 정렬
    """
    import pandas as pd
    import yfinance as yf

    tickers = ["^VIX", "^TNX"]
    frames = []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="20y", auto_adjust=False)
            if hist is None or hist.empty or "Close" not in hist.columns:
                print(f"[WARN] No macro for {t}")
                continue
            s = pd.Series(pd.to_numeric(hist["Close"], errors="coerce"),
                          index=pd.to_datetime(hist.index), name=t)
            frames.append(s)
        except Exception as e:
            print(f"[ERROR] macro {t}: {e}")

    if not frames:
        print("[WARN] No macro frames collected"); return

    out = pd.concat(frames, axis=1).sort_index()
    out.index.name = "Date"
    out = out.reset_index()
    # 최종 숫자형 보증
    for c in ["^VIX", "^TNX"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out.to_csv(out_dir / "macro_sample.csv", index=False)
    print("saved:", out_dir / "macro_sample.csv")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[1] / "configs" / "base.yaml"))
    ap.add_argument("--years", type=int, default=12)  # 12년치 기본
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    symbols = [s.upper() for s in cfg.get("symbols", [])]
    end = datetime.today().date()
    start = end - timedelta(days=args.years*365)

    data_dir = Path(__file__).resolve().parents[1] / "data"
    download_prices(symbols, start, end, data_dir)
    download_macro(data_dir)

if __name__ == "__main__":
    main()