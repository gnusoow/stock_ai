import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

def load_prices(sym_dir: Path):
    f = sym_dir / "prices_tech.csv"
    if not f.exists(): 
        return None
    df = pd.read_csv(f)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
    return df

def load_macro(data_dir: Path):
    f = data_dir / "macro_sample.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    # 간단 지표: 60D z-score
    for c in [c for c in df.columns if c != date_col]:
        roll = df[c].rolling(60, min_periods=20)
        df[c+"_z"] = (df[c]-roll.mean())/(roll.std()+1e-9)
    return df

def make_features(px: pd.DataFrame, macro: pd.DataFrame):
    df = px.copy()
    cols = df.columns.tolist()
    if "Close" not in cols and len(cols) >= 5:
        # yfinance 포맷 다양성 방어: 4번째 열을 Close로 가정
        df = df.rename(columns={cols[3]: "Close"})

    # 날짜/정렬
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    # ✅ 숫자형 강제 (콤마/문자 섞임 방지)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # 지표 계산 전에 결측 제거(최소한 Close는 숫자여야 함)
    df = df.dropna(subset=["Close"])

    from ta.trend import SMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands

    df["SMA20"] = SMAIndicator(df["Close"], 20).sma_indicator()
    df["SMA60"] = SMAIndicator(df["Close"], 60).sma_indicator()
    df["RSI14"] = RSIIndicator(df["Close"], 14).rsi()
    macd = MACD(df["Close"])
    df["MACD_HIST"] = macd.macd_diff()
    bb = BollingerBands(df["Close"], 20, 2)
    df["BB_WIDTH"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["Close"]

    df["ret1"]  = df["Close"].pct_change()
    df["ret5"]  = df["Close"].pct_change(5)
    df["ret20"] = df["Close"].pct_change(20)

    mac = macro.reindex(df.index).ffill() if macro is not None else None
    for c in ["^VIX","^TNX","^VIX_z","^TNX_z"]:
        if mac is not None and c in mac.columns:
            df[c] = pd.to_numeric(mac[c], errors="coerce")

    out = df.dropna().reset_index()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[1] / "configs" / "base.yaml"))
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    symbols = [s.upper() for s in cfg.get("symbols", [])]
    k_look = int(cfg.get("backtest", {}).get("lookahead_days", 5))

    data_dir = Path(__file__).resolve().parents[1] / "data"
    macro = load_macro(data_dir)

    frames=[]
    for sym in symbols:
        px = load_prices(data_dir / sym)
        if px is None: 
            print("skip", sym, ": no price file")
            continue
        feat = make_features(px, macro)
        if feat is None or feat.empty: 
            continue
        feat["Symbol"] = sym
        # target_5d (미래 수익률) -> 수익률은 이후 날짜 Close 사용
        feat = feat.sort_values("Date")
        feat["target_5d"] = feat["Close"].pct_change(k_look).shift(-k_look)
        frames.append(feat)

    if not frames:
        print("No dataset created."); return

    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date","Symbol"]).reset_index(drop=True)

    def label_top30(g):
        thr = np.nanpercentile(g["target_5d"].values, 70)
        return (g["target_5d"] >= thr).astype(int)

    df["label_top30"] = df.groupby("Date", group_keys=False).apply(label_top30)

    out = data_dir / "ml_dataset_cls.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("✅ Saved dataset:", out)
    print(df.head())

if __name__ == "__main__":
    main()