# src/backtest_multifactor.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

DEFAULTS = ['AAPL','MSFT','GOOGL','AMZN','NVDA','TSLA','META','AVGO','ASML','AMD']

def load_price(symbol_dir: Path):
    f = symbol_dir / "prices_tech.csv"
    if not f.exists(): return None
    df = pd.read_csv(f)
    if "Date" not in df.columns:
        df = df.reset_index().rename(columns={"index":"Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date")

def load_sentiment(symbol_dir: Path):
    f = symbol_dir / "news_sentiment.csv"
    if not f.exists(): return None
    df = pd.read_csv(f)
    if "published" in df.columns and "compound" in df.columns:
        df["published"] = pd.to_datetime(df["published"], errors="coerce")
        daily = df.dropna(subset=["published"]).set_index("published").resample("D")["compound"].mean()
        return daily
    return None

def load_macro(data_dir: Path):
    f = data_dir / "macro_sample.csv"
    if not f.exists():
        return None

    df = pd.read_csv(f)

    # 1) 날짜 컬럼 찾기 (여러 케이스 방어)
    date_col = None
    if "Date" in df.columns:
        date_col = "Date"
    else:
        for c in df.columns:
            name = str(c).lower()
            if name.startswith("unnamed") or name in ("index",):
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]

    # 2) 날짜 파싱 및 인덱스 설정
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col)
    df.index.name = "Date"

    # 3) 필요한 컬럼 매핑 (여기서 col_map을 항상 정의)
    wanted = {
        "^VIX": ["^VIX", "VIX", "CBOE_VIX"],
        "^TNX": ["^TNX", "TNX", "US10Y", "UST10Y", "^TNX.X"],
    }
    col_map = {}
    for want, alts in wanted.items():
        for a in alts:
            if a in df.columns:
                col_map[want] = a
                break

    # 4) 둘 다 없으면 종료, 하나라도 있으면 가능한 것만 사용
    if not col_map:
        return None

    keep_cols = [col_map[k] for k in col_map.keys()]
    out = df[keep_cols].copy()
    # 표준 컬럼명으로 교체
    out.columns = list(col_map.keys())

    # 5) 숫자형 강제 변환 + 정렬/전방채움
    out = out.apply(pd.to_numeric, errors="coerce").sort_index().ffill()
    return out



def calc_score(price_df, sent_ser, macro_df):
    # --- 기본 가드 ---
    if price_df is None or len(price_df) == 0 or macro_df is None or len(macro_df) == 0:
        print("⚠️  Skipping symbol: missing price or macro data")
        return pd.DataFrame()

    df = price_df.copy()

    # --- Date 보장: 이미 load_price에서 처리했어도 안전 재확인 ---
    if "Date" in df.columns:
        parsed = pd.to_datetime(df["Date"], errors="coerce")
        if parsed.notna().sum() == 0:
            # 전부 NaT이면 인덱스를 날짜로 시도, 그래도 안되면 가짜 날짜 부여
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                df.index = pd.date_range("2000-01-01", periods=len(df), freq="D")
        else:
            df["Date"] = parsed
            df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    else:
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            df.index = pd.date_range("2000-01-01", periods=len(df), freq="D")

    # --- Close 보장 ---
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            # 심볼.번호 형태일 때 4번째 컬럼을 Close로 추정
            non_date_cols = list(df.columns)
            if len(non_date_cols) >= 4:
                df["Close"] = pd.to_numeric(df[non_date_cols[3]], errors="coerce")
            else:
                print("⚠️  Close not found/inferable")
                return pd.DataFrame()

    # 숫자형 강제 (문자/쉼표 섞임 방지)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    if df["Close"].notna().sum() == 0:
        print("⚠️  Close all NaN after coercion")
        return pd.DataFrame()

    # --- 기술 지표 보장 (없으면 즉석 계산) ---
    from ta.trend import SMAIndicator, MACD
    if "SMA20" not in df.columns:
        df["SMA20"] = SMAIndicator(df["Close"], window=20).sma_indicator()
    if "SMA60" not in df.columns:
        df["SMA60"] = SMAIndicator(df["Close"], window=60).sma_indicator()
    if "MACD_HIST" not in df.columns:
        m = MACD(df["Close"])
        df["MACD_HIST"] = m.macd_diff()

    # 숫자형 강제 + NaN 방어
    for c in ["SMA20","SMA60","MACD_HIST"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- 기술적 요인 ---
    sma_signal  = np.where((df["SMA20"] > df["SMA60"]).fillna(False), 1, -1)
    macd_signal = np.where((df["MACD_HIST"] > 0).fillna(False),        1, -1)

    # --- 감성 요인 ---
    if sent_ser is None:
        sent_aligned = pd.Series(0.0, index=df.index)
    else:
        sent_aligned = sent_ser.reindex(df.index).ffill().fillna(0.0)
    sent_aligned = pd.to_numeric(sent_aligned, errors="coerce").fillna(0.0)
    sent_signal  = np.where(sent_aligned > 0, 1, -1)

    # --- 거시 요인 ---
    macro_aligned = macro_df.reindex(df.index).ffill()
    for c in ["^VIX","^TNX"]:
        if c in macro_aligned.columns:
            macro_aligned[c] = pd.to_numeric(macro_aligned[c], errors="coerce")
    # 60D 평균 (초반 NaN 방어)
    vix_ma = macro_aligned["^VIX"].rolling(60, min_periods=1).mean()
    tnx_ma = macro_aligned["^TNX"].rolling(60, min_periods=1).mean()
    # 비교시 NaN → False
    vix_signal = np.where((macro_aligned["^VIX"] < vix_ma).fillna(False), 1, -1)
    tnx_signal = np.where((macro_aligned["^TNX"] < tnx_ma).fillna(False), 1, -1)

    # --- 종합 점수 & 포지션 ---
    score = 0.25*sma_signal + 0.25*macd_signal + 0.25*sent_signal + 0.25*((vix_signal + tnx_signal)/2)
    df["Score"] = score
    df["Position"] = np.where(df["Score"] > 0, 1, np.where(df["Score"] < 0, 0, np.nan))
    df["Position"] = df["Position"].ffill().fillna(0)

    # --- 전략 수익/에쿼티 ---
    ret = df["Close"].pct_change().fillna(0)
    strat = ret * df["Position"].shift(1)
    df["Equity"] = (1 + strat).cumprod()

    return df



def metrics(eq):
    rets = eq.pct_change().fillna(0)
    tot = eq.iloc[-1]-1
    yrs = max((eq.index[-1]-eq.index[0]).days/365.25,1)
    cagr = (eq.iloc[-1])**(1/yrs)-1
    vol = rets.std()*np.sqrt(252)
    sharpe = rets.mean()/rets.std()*np.sqrt(252)
    dd = (eq/eq.cummax()-1).min()
    return tot, cagr, vol, sharpe, dd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default=",".join(DEFAULTS))
    ap.add_argument("--data", type=str, default=str(Path(__file__).resolve().parents[1] / "data"))
    args = ap.parse_args()

    data_dir = Path(args.data)
    macro_df = load_macro(data_dir)

    summary = []

    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        sym_dir = data_dir / sym
        price_df = load_price(sym_dir)
        sent_ser = load_sentiment(sym_dir)

        if price_df is None:
            print(f"⚠️  {sym}: price data not found, skipping.")
            continue
        if macro_df is None:
            print(f"⚠️  Macro data missing, skipping {sym}.")
            continue

        df = calc_score(price_df, sent_ser, macro_df)
        if df is None or df.empty:
            print(f"⚠️  {sym}: calc_score returned empty, skipping.")
            continue

        out_csv = sym_dir / "bt_multifactor.csv"
        df.reset_index().to_csv(out_csv, index=False, encoding="utf-8-sig")


        eq = df["Equity"]
        tot, cagr, vol, sharpe, dd = metrics(eq)
        summary.append([sym, tot, cagr, vol, sharpe, dd])

        # 차트 저장
        plt.figure()
        plt.plot(df.index, df["Equity"], label="Equity")
        plt.title(f"{sym} MultiFactor Strategy")
        plt.legend()
        plt.savefig(sym_dir / "charts" / f"{sym}_bt_multifactor.png", bbox_inches="tight", dpi=140)
        plt.close()

    if summary:
        cols = ["Symbol","TotalReturn","CAGR","Volatility","Sharpe","MaxDrawdown"]
        df_sum = pd.DataFrame(summary, columns=cols)
        out_sum = data_dir / "bt_multifactor_summary.csv"
        df_sum.to_csv(out_sum, index=False, encoding="utf-8-sig")
        print("✅ Saved:", out_sum)
        print(df_sum.sort_values("Sharpe", ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()
