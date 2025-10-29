import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DEFAULTS = ['AAPL','MSFT','GOOGL','AMZN','NVDA','TSLA','META','AVGO','ASML','AMD']

def plot_series(df: pd.DataFrame, xcol: str, ycols: list[str], title: str, out_path: Path):
    plt.figure()
    # 강제 날짜 변환
    if xcol in df.columns:
        try:
            x = pd.to_datetime(df[xcol])
        except Exception:
            x = df[xcol]
    else:
        x = df.index
    plotted = False
    for c in ycols:
        if c in df.columns and getattr(df[c], "notna", lambda: True)().sum() > 0:
            plt.plot(x, df[c], label=c)
            plotted = True
    plt.title(title)
    plt.xlabel("Date")
    if plotted:
        plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=140)
    plt.close()

def ensure_indicators(df: pd.DataFrame) -> pd.DataFrame:
    need = {"SMA20","SMA60","RSI14","MACD","MACD_SIGNAL","MACD_HIST","BB_WIDTH"}
    missing = [c for c in need if c not in df.columns or getattr(df[c], "isna", lambda: True)().all()]
    if missing and "Close" in df.columns:
        from ta.trend import SMAIndicator, MACD as _MACD
        from ta.momentum import RSIIndicator
        from ta.volatility import BollingerBands
        close = df["Close"]
        if "SMA20" in missing: df["SMA20"] = SMAIndicator(close, window=20).sma_indicator()
        if "SMA60" in missing: df["SMA60"] = SMAIndicator(close, window=60).sma_indicator()
        if "RSI14" in missing: df["RSI14"] = RSIIndicator(close, window=14).rsi()
        if {"MACD","MACD_SIGNAL","MACD_HIST"} & set(missing):
            m = _MACD(close)
            df["MACD"] = m.macd()
            df["MACD_SIGNAL"] = m.macd_signal()
            df["MACD_HIST"] = m.macd_diff()
        if {"BB_WIDTH"} & set(missing):
            bb = BollingerBands(close, window=20, window_dev=2)
            df["BB_UPPER"] = bb.bollinger_hband()
            df["BB_LOWER"] = bb.bollinger_lband()
            df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / close
    return df

def normalize_loaded_prices(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    if "Close" not in cols:
        symbol_like = [c for c in cols if c != "Date"]
        if len(symbol_like) >= 5:
            base = symbol_like[:5]
            mapping = {base[0]:"Open", base[1]:"High", base[2]:"Low", base[3]:"Close", base[4]:"Volume"}
            df = df.rename(columns=mapping)
    return df

def run_for_symbol(base_dir: Path, symbol: str):
    sym_dir = base_dir / symbol
    prices_path = sym_dir / "prices_tech.csv"
    news_path = sym_dir / "news_sentiment.csv"
    charts_dir = sym_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    if prices_path.exists():
        df = pd.read_csv(prices_path)
        df = normalize_loaded_prices(df)
        if "Date" not in df.columns:
            df = df.reset_index().rename(columns={"index":"Date"})
        df = ensure_indicators(df)

        plot_series(df, "Date", ["Close","SMA20","SMA60"], f"{symbol} Close & SMA", charts_dir / f"{symbol}_close_sma.png")
        plot_series(df, "Date", ["RSI14"], f"{symbol} RSI14", charts_dir / f"{symbol}_rsi14.png")
        plot_series(df, "Date", ["MACD","MACD_SIGNAL","MACD_HIST"], f"{symbol} MACD", charts_dir / f"{symbol}_macd.png")
        plot_series(df, "Date", ["BB_WIDTH"], f"{symbol} Bollinger Width", charts_dir / f"{symbol}_bbwidth.png")

    if news_path.exists():
        nf = pd.read_csv(news_path)
        if "published" in nf.columns and "compound" in nf.columns:
            try:
                nf["published"] = pd.to_datetime(nf["published"])
                daily = nf.dropna(subset=["published"]).set_index("published").resample("D")["compound"].mean().reset_index()
                daily["Date"] = daily["published"].dt.date.astype(str)
                plot_series(daily, "Date", ["compound"], f"{symbol} News Sentiment (Daily Avg)", charts_dir / f"{symbol}_sentiment.png")
            except Exception:
                pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, help="한 종목만 시각화")
    ap.add_argument("--all", action="store_true", help="DEFAULTS 10개 모두")
    ap.add_argument("--data", type=str, default=str(Path(__file__).resolve().parents[1] / "data"))
    args = ap.parse_args()

    base_dir = Path(args.data)
    if args.all:
        for s in DEFAULTS:
            run_for_symbol(base_dir, s)
        print("✅ charts saved for:", ", ".join(DEFAULTS))
    else:
        sym = args.symbol or "AAPL"
        run_for_symbol(base_dir, sym)
        print(f"✅ charts saved for: {sym}")

if __name__ == "__main__":
    main()
