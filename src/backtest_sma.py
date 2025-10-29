# src/backtest_sma.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DEFAULTS = ['AAPL','MSFT','GOOGL','AMZN','NVDA','TSLA','META','AVGO','ASML','AMD']

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Date/Close 보정 + SMA/RSI 등 없으면 즉석 계산
    if "Date" not in df.columns:
        df = df.reset_index().rename(columns={"index":"Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    if "Close" not in df.columns:
        # yfinance가 심볼.번호로 저장된 경우 복구 시도
        cols = [c for c in df.columns if c != "Date"]
        if len(cols) >= 4:
            df = df.rename(columns={cols[3]:"Close"})
    # SMA 재계산 (지표 없거나 NaN이면)
    return df

def compute_sma(df: pd.DataFrame, fast: int, slow: int):
    close = df["Close"]
    df[f"SMA{fast}"] = close.rolling(fast).mean()
    df[f"SMA{slow}"] = close.rolling(slow).mean()
    return df

def metrics_from_equity(eq: pd.Series, rf: float = 0.0, periods: int = 252) -> dict:
    # eq: 누적자산(=1에서 시작)
    # --- 날짜 인덱스 보정 ---
    if not isinstance(eq.index, pd.DatetimeIndex):
        try:
            eq.index = pd.to_datetime(eq.index)
        except Exception:
            # 그래도 실패하면 임시 날짜 생성
            eq.index = pd.date_range("2000-01-01", periods=len(eq), freq="D")

    rets = eq.pct_change().fillna(0.0)
    total_return = eq.iloc[-1] - 1.0
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (eq.iloc[-1]) ** (1/yrs) - 1 if yrs > 0 else np.nan
    vol = rets.std() * np.sqrt(periods)
    sharpe = (rets.mean() * periods - rf) / (vol + 1e-12)
    dd = (eq / eq.cummax() - 1).min()
    return {
        "TotalReturn": total_return,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": dd
    }

def backtest_long_only(df: pd.DataFrame, fast: int, slow: int, fee_bps: float = 0.0):
    """
    단순 크로스오버:
    - 시그널: SMAfast > SMAslow
    - 시그널을 하루 지연하여 체결(룩어헤드 방지)
    - 비용: 체결 발생일에 fee_bps 한 번 차감(왕복 기준이면 2*bps로 조정)
    """
    df = df.copy()
    df = compute_sma(df, fast, slow)

    sig_today = (df[f"SMA{fast}"] > df[f"SMA{slow}"]).astype(int)
    pos = sig_today.shift(1).fillna(0)  # 다음날부터 보유

    ret = df["Close"].pct_change().fillna(0.0)
    strat = ret * pos  # 보유일에만 수익 반영

    # 거래비용: 포지션 변화가 있을 때만 차감 (bps -> decimal)
    turn = pos.diff().abs().fillna(abs(pos.iloc[0]))

    fee = turn * (fee_bps / 1e4)
    strat_after_fee = strat - fee

    equity = (1 + strat_after_fee).cumprod()
    exposure = pos.mean()  # 평균 포지션(보유 비율)

    # 트레이드 통계(진입/청산)
    trades_in = (pos.diff() == 1).sum()
    trades_out = (pos.diff() == -1).sum()
    trades = max(trades_in, trades_out)

    # 간단 승률: 보유 구간의 일수익 기준
    wins = (strat[strat != 0] > 0).sum()
    winrate = wins / max(len(strat[strat != 0]), 1)

    m = metrics_from_equity(equity)
    m.update({
        "Exposure": exposure,
        "Trades": int(trades),
        "WinRate": winrate
    })
    out = df[["Date","Close",f"SMA{fast}",f"SMA{slow}"]].copy()
    out["Position"] = pos.values
    out["DailyRet"] = ret.values
    out["StratRet"] = strat_after_fee.values
    out["Equity"] = equity.values
    return out, m

def plot_equity(eq: pd.Series, title: str, out_path: Path):
    plt.figure()
    plt.plot(eq.index, eq.values, label="Equity")
    plt.title(title)
    plt.xlabel("Date")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close()

def run_symbol(base_dir: Path, symbol: str, fast: int, slow: int, fee_bps: float):
    sym_dir = base_dir / symbol
    fpath = sym_dir / "prices_tech.csv"
    if not fpath.exists():
        return None

    df = pd.read_csv(fpath)
    df = ensure_cols(df)
    if "Close" not in df.columns or df["Close"].isna().all():
        return None

    # 필요한 기간 정렬
    df = df.sort_values("Date")

    # 지표가 파일에 있더라도 재계산(파라미터가 다를 수 있음)
    res_df, m = backtest_long_only(df, fast, slow, fee_bps=fee_bps)

    # 저장
    out_csv = sym_dir / f"bt_sma_{fast}_{slow}.csv"
    res_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    plot_equity(res_df["Equity"].set_axis(pd.to_datetime(res_df["Date"])),
                f"{symbol} SMA{fast}/{slow} Equity",
                sym_dir / "charts" / f"{symbol}_bt_equity_{fast}_{slow}.png")
    m["Symbol"] = symbol
    return m, res_df

def aggregate_portfolio(bt_results: list[pd.DataFrame]):
    """
    동일가중 포트폴리오:
    - 매일 보유 중인 전략들의 일수익 평균
    - 아무도 보유하지 않으면 0
    """
    if not bt_results:
        return None, {}
    dfs = []
    for df in bt_results:
        d = df[["Date","StratRet","Position"]].copy()
        d["Date"] = pd.to_datetime(d["Date"])
        dfs.append(d.set_index("Date"))
    aligned = pd.concat(dfs, axis=1, keys=range(len(dfs)))
    # 멀티인덱스 정리
    strat_cols = [c for c in aligned.columns if c[1] == "StratRet"]
    pos_cols   = [c for c in aligned.columns if c[1] == "Position"]
    strat = aligned[strat_cols]
    pos   = aligned[pos_cols]

    # 활성 전략 수(분모), 수익은 평균
    active = pos.sum(axis=1).replace(0, np.nan)
    port_ret = strat.mean(axis=1)  # 비활성도 NaN 포함 평균 -> 아래서 보정
    port_ret = port_ret.where(active.notna(), 0.0).fillna(0.0)

    port_eq = (1 + port_ret).cumprod()
    m = metrics_from_equity(port_eq)
    m.update({
        "Exposure": (active.fillna(0) / len(bt_results)).mean()
    })
    port_df = pd.DataFrame({"Date": port_ret.index, "PortRet": port_ret.values, "PortEquity": port_eq.values})
    return port_df, m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default=",".join(DEFAULTS))
    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=60)
    ap.add_argument("--fee_bps", type=float, default=5.0, help="체결당 비용(bps). 5 = 0.05%%")
    ap.add_argument("--data", type=str, default=str(Path(__file__).resolve().parents[1] / "data"))
    args = ap.parse_args()

    base_dir = Path(args.data)
    fast, slow = args.fast, args.slow
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    summary = []
    bt_curves = []

    for s in syms:
        r = run_symbol(base_dir, s, fast, slow, fee_bps=args.fee_bps)
        if r is None:
            continue
        m, curve = r
        summary.append(m)
        bt_curves.append(curve)

    # 요약 저장
    if summary:
        df_sum = pd.DataFrame(summary)[["Symbol","TotalReturn","CAGR","Volatility","Sharpe","MaxDrawdown","Exposure","Trades","WinRate"]]
        out_csv = base_dir / f"bt_summary_sma_{fast}_{slow}.csv"
        df_sum.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print("✅ Saved symbol summary:", out_csv)
        print(df_sum.sort_values("CAGR", ascending=False).to_string(index=False))

    # 포트폴리오
    port, pm = aggregate_portfolio(bt_curves)
    if port is not None:
        out_port = base_dir / f"bt_portfolio_sma_{fast}_{slow}.csv"
        port.to_csv(out_port, index=False, encoding="utf-8-sig")
        print("✅ Saved portfolio series:", out_port)

        # 포트폴리오 차트
        plot_equity(port["PortEquity"].set_axis(pd.to_datetime(port["Date"])),
                    f"Portfolio SMA{fast}/{slow} Equity",
                    base_dir / "portfolio_bt_equity.png")
        print("📈 Portfolio:", pm)

if __name__ == "__main__":
    main()
