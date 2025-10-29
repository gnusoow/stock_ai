import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.trend import SMAIndicator, MACD

DEFAULTS = ['AAPL','MSFT','GOOGL','AMZN','NVDA','TSLA','META','AVGO','ASML','AMD']

def load_price(sym_dir: Path):
    f = sym_dir / "prices_tech.csv"
    if not f.exists(): return None
    df = pd.read_csv(f)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    return df

def load_sentiment(sym_dir: Path):
    f = sym_dir / "news_sentiment.csv"
    if not f.exists(): return None
    df = pd.read_csv(f)
    if "published" in df.columns and "compound" in df.columns:
        df["published"] = pd.to_datetime(df["published"], errors="coerce")
        s = df.dropna(subset=["published"]).set_index("published")["compound"].sort_index()
        return s
    return None

def load_macro(data_dir: Path):
    f = data_dir / "macro_sample.csv"
    if not f.exists(): return None
    df = pd.read_csv(f)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    cols = {}
    for target, options in {"^VIX": ["^VIX","VIX"], "^TNX": ["^TNX","TNX"]}.items():
        for opt in options:
            if opt in df.columns:
                cols[target] = opt
                break
    if not cols: return None
    out = df[[cols[c] for c in cols]].copy()
    out.columns = list(cols.keys())
    out = out.apply(pd.to_numeric, errors="coerce").ffill()
    return out

def ensure_indicators(df):
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            df["Close"] = pd.to_numeric(df.iloc[:, 3], errors="coerce")
    if "SMA20" not in df.columns:
        df["SMA20"] = SMAIndicator(df["Close"], 20).sma_indicator()
    if "SMA60" not in df.columns:
        df["SMA60"] = SMAIndicator(df["Close"], 60).sma_indicator()
    if "MACD_HIST" not in df.columns:
        m = MACD(df["Close"])
        df["MACD_HIST"] = m.macd_diff()
    return df

def score_frame(df, sent, macro, wsma, wmacd, wsent, wmacro, sent_win, vix_win, tnx_win, min_hold):
    df = ensure_indicators(df)
    sma_sig = np.where(df["SMA20"] > df["SMA60"], 1, -1)
    macd_sig = np.where(df["MACD_HIST"] > 0, 1, -1)
    sent_daily = sent.reindex(df.index).ffill().fillna(0.0) if sent is not None else pd.Series(0, index=df.index)
    sent_sig = np.where(sent_daily.rolling(sent_win, min_periods=1).mean() > 0, 1, -1)
    mac = macro.reindex(df.index).ffill()
    vix_ma = mac["^VIX"].rolling(vix_win, min_periods=1).mean()
    tnx_ma = mac["^TNX"].rolling(tnx_win, min_periods=1).mean()
    vix_sig = np.where(mac["^VIX"] < vix_ma, 1, -1)
    tnx_sig = np.where(mac["^TNX"] < tnx_ma, 1, -1)
    macro_sig = (vix_sig + tnx_sig)/2
    score = wsma*sma_sig + wmacd*macd_sig + wsent*sent_sig + wmacro*macro_sig
    pos = np.where(score > 0, 1, 0)
    if min_hold > 0:
        p = pd.Series(pos, index=df.index)
        for i in range(1,len(p)):
            if p.iloc[i]==0 and p.iloc[i-1]==1:
                held = 0
                j=i-1
                while j>=0 and p.iloc[j]==1:
                    held += 1; j -= 1
                if held<min_hold:
                    p.iloc[i]=1
        pos = p.values
    ret = df["Close"].pct_change().fillna(0)
    strat = ret * pd.Series(pos, index=df.index).shift(1).fillna(0)
    eq = (1+strat).cumprod()
    rets = eq.pct_change().fillna(0)
    sharpe = rets.mean()/rets.std()*np.sqrt(252) if rets.std()>0 else 0
    return sharpe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default=",".join(DEFAULTS))
    ap.add_argument("--data", type=str, default=str(Path(__file__).resolve().parents[1] / "data"))
    args = ap.parse_args()
    data_dir = Path(args.data)
    macro = load_macro(data_dir)
    if macro is None:
        print("❌ macro data missing.")
        return
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # grid 정의
    WSMA = [0.2, 0.4, 0.6]
    WMACD= [0.2, 0.4]
    WSENT= [0.0, 0.2, 0.4]
    SENT_WIN=[3,5,10]
    VIX_WIN=[30,60]
    TNX_WIN=[30,60]
    MIN_HOLD=[0,3]

    results = []
    for wsma in WSMA:
        for wmacd in WMACD:
            for wsent in WSENT:
                wmacro = 1 - (wsma+wmacd+wsent)
                if wmacro < 0: continue
                for sw in SENT_WIN:
                    for vw in VIX_WIN:
                        for tw in TNX_WIN:
                            for mh in MIN_HOLD:
                                sharpe_list=[]
                                for sym in syms:
                                    df = load_price(data_dir / sym)
                                    sent = load_sentiment(data_dir / sym)
                                    if df is None: continue
                                    try:
                                        sh = score_frame(df, sent, macro, wsma,wmacd,wsent,wmacro,sw,vw,tw,mh)
                                        sharpe_list.append(sh)
                                    except Exception:
                                        continue
                                if sharpe_list:
                                    overall = np.nanmean(sharpe_list)
                                    results.append([wsma,wmacd,wsent,round(wmacro,2),sw,vw,tw,mh,overall])

    res = pd.DataFrame(results, columns=["wSMA","wMACD","wSENT","wMACRO","sent_win","vix_win","tnx_win","min_hold","Sharpe"])
    out_csv = data_dir / "sweep_results.csv"
    res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("✅ Saved:", out_csv)
    print(res.sort_values("Sharpe", ascending=False).head(15).to_string(index=False))

    # heatmap 시각화
    filt = res[(res["wMACD"]==0.4)&(res["vix_win"]==60)&(res["tnx_win"]==60)&(res["min_hold"]==3)&(res["sent_win"]==5)]
    if not filt.empty:
        pivot = filt.pivot_table(index="wSMA", columns="wSENT", values="Sharpe", aggfunc="mean")
        plt.imshow(pivot.values, aspect="auto", cmap="viridis")
        plt.title("Sharpe Heatmap (wSMA vs wSENT)")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.colorbar(label="Sharpe")
        out_png = data_dir / "heatmap_sharpe.png"
        plt.savefig(out_png, bbox_inches="tight", dpi=140)
        plt.close()
        print("✅ Saved heatmap:", out_png)

if __name__ == "__main__":
    main()
