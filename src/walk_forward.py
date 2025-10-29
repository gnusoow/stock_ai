
from pathlib import Path
import pandas as pd
import numpy as np
import json
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CFG_FILE = ROOT / "configs" / "base.yaml"

def load_cfg():
    with open(CFG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_dataset():
    f = DATA_DIR / "ml_dataset_cls.csv"
    if not f.exists():
        raise FileNotFoundError("Run make_ml_dataset_cls.py first")
    df = pd.read_csv(f)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date","Symbol"]).reset_index(drop=True)
    return df

def sharpe_daily(returns):
    r = returns.dropna()
    if r.std() == 0 or len(r) == 0:
        return 0.0
    return (r.mean() / r.std()) * np.sqrt(252)

def max_drawdown(eq_curve):
    x = eq_curve.values
    peak = np.maximum.accumulate(x)
    dd = (x - peak) / peak
    return float(dd.min()) if len(dd) else 0.0

def enforce_min_hold(pos_series, min_hold):
    s = pos_series.copy()
    cnt = 0
    for i in range(len(s)):
        if s.iat[i]==1:
            cnt += 1
        else:
            if 0 < cnt < min_hold:
                s.iat[i] = 1; cnt += 1
            else:
                cnt = 0
    return s

def backtest_topk(test_df, preds, k_pct=0.30, fee_bps=5, min_hold=3):
    df = test_df.copy()
    df["pred"] = preds

    # 날짜별 상위 k% 임계치 (transform으로 원래 index 유지)
    thr = df.groupby("Date")["pred"].transform(lambda s: s.quantile(1 - k_pct))
    df["pos"] = (df["pred"] >= thr).astype(int)

    # 다음날 체결
    df["ret1"] = df.groupby("Symbol")["Close"].pct_change().fillna(0.0)
    df["pos_exec"] = df.groupby("Symbol")["pos"].shift(1).fillna(0)

    # 최소 보유 강제 (원래 index 유지)
    if min_hold > 0:
        def _mh(s):
            s = s.astype(int).copy()
            cnt = 0
            for i in range(len(s)):
                if s.iat[i] == 1:
                    cnt += 1
                else:
                    if 0 < cnt < min_hold:
                        s.iat[i] = 1
                        cnt += 1
                    else:
                        cnt = 0
            return s
        df["pos_exec"] = df.groupby("Symbol")["pos_exec"].transform(_mh)

    # 거래비용: 포지션 변화(진입/청산) 절대값
    turn = df.groupby("Symbol")["pos_exec"].transform(lambda s: s.diff().abs()).fillna(df["pos_exec"].abs())
    df["fee"] = turn * (fee_bps / 10000.0)

    # 일별 수익 (벡터화) — 인덱스 정합 보장
    df["gross"] = df["ret1"] * df["pos_exec"]
    daily = df.groupby("Date")[["gross", "fee"]].mean().reset_index()
    daily["ret"] = daily["gross"] - daily["fee"]
    daily["eq"] = (1 + daily["ret"]).cumprod()
    return daily[["Date", "ret", "eq"]]


def month_ends(dates):
    # unique month ends from dataset
    m = pd.Series(1, index=sorted(pd.to_datetime(dates.unique())))
    g = m.groupby([m.index.year, m.index.month])
    ends = [grp.index.max() for _, grp in g]
    return sorted(pd.to_datetime(ends))

def train_model(X, y):
    if HAS_XGB:
        model = XGBClassifier(
            n_estimators=600, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=42, n_jobs=0, tree_method="hist"
        )
        model.fit(X, y)
        return model
    else:
        pipe = make_pipeline(
            StandardScaler(with_mean=False),
            RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
        )
        pipe.fit(X, y)
        return pipe

def main():
    cfg = load_cfg()
    k_pct     = float(cfg["backtest"]["k_top_percent"])
    fee_bps   = int(cfg["backtest"]["fee_bps"])
    min_hold  = int(cfg["backtest"]["min_hold"])

    df = load_dataset()
    feats = [c for c in df.columns if c not in ["Date","Symbol","target_5d","label_top30"]]
    target = "label_top30"

    # 월말 기준 Walk-Forward: train up to month-end t, test on next month (t+1)
    ends = month_ends(df["Date"])
    if len(ends) < 13:
        print("[WARN] Not enough data for walk-forward (need >= 13 month-ends).")
    rows = []
    all_daily = []

    for i in range(len(ends)-1):
        cutoff = ends[i]
        test_end = ends[i+1]
        train = df[df["Date"] <= cutoff]
        test  = df[(df["Date"] > cutoff) & (df["Date"] <= test_end)]
        if train.empty or test.empty:
            continue

        model = train_model(train[feats], train[target])
        proba = model.predict_proba(test[feats])[:,1] if hasattr(model, "predict_proba") else model.predict(test[feats])

        auc  = roc_auc_score(test[target], proba) if test[target].nunique() > 1 else np.nan
        ap   = average_precision_score(test[target], proba) if test[target].nunique() > 1 else np.nan

        daily = backtest_topk(test, proba, k_pct=k_pct, fee_bps=fee_bps, min_hold=min_hold)
        sh = sharpe_daily(daily["ret"])
        mdd = max_drawdown(daily["eq"])
        rows.append({
            "train_upto": cutoff.strftime("%Y-%m-%d"),
            "test_month": test["Date"].dt.to_period("M").iloc[0].strftime("%Y-%m"),
            "days": int(daily.shape[0]),
            "AUC": float(auc) if pd.notna(auc) else None,
            "AP": float(ap) if pd.notna(ap) else None,
            "Sharpe": float(sh),
            "MDD": float(mdd),
            "CAGR_est": float((1+daily["ret"].mean())**252 - 1) if daily.shape[0] else 0.0,
        })
        all_daily.append(daily)

    report = pd.DataFrame(rows)
    out_dir = DATA_DIR / "walk_forward"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "wf_monthly_report.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    # 합산 에쿼티
    if all_daily:
        all_daily_df = pd.concat(all_daily).sort_values("Date")
        all_daily_df = all_daily_df.drop_duplicates("Date")  # 월별 인접 기간 중복 방지
        all_daily_df["eq"] = (1+all_daily_df["ret"]).cumprod()
        all_daily_df.to_csv(out_dir / "wf_equity.csv", index=False, encoding="utf-8-sig")

        plt.figure()
        plt.plot(all_daily_df["Date"], all_daily_df["eq"], label="Walk-Forward Equity")
        plt.legend(); plt.title("Walk-Forward (Monthly) Equity")
        plt.savefig(out_dir / "wf_equity.png", bbox_inches="tight", dpi=140)
        plt.close()

    # 요약 메트릭
    summary = {
        "periods": len(report),
        "Sharpe_avg": float(report["Sharpe"].mean()) if len(report) else 0.0,
        "Sharpe_median": float(report["Sharpe"].median()) if len(report) else 0.0,
        "MDD_min": float(report["MDD"].min()) if len(report) else 0.0,
        "AUC_avg": float(report["AUC"].mean()) if "AUC" in report and report["AUC"].notna().any() else None,
        "AP_avg": float(report["AP"].mean()) if "AP" in report and report["AP"].notna().any() else None,
    }
    with open(out_dir / "wf_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("✅ Walk-Forward saved:",
          report_path,
          out_dir / "wf_equity.png",
          out_dir / "wf_summary.json")

if __name__ == "__main__":
    # 기존 워크포워드 실행
    main()

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # ------------------------------
    # 1) 우리 전략 누적수익 곡선 (wf_equity.csv)
    # ------------------------------
    wf_equity_path = "C:/stock_ai/data/walk_forward/wf_equity.csv"
    if not os.path.exists(wf_equity_path):
        raise FileNotFoundError(f"Not found: {wf_equity_path}. 먼저 walk_forward 실행을 확인하세요.")

    wf = pd.read_csv(wf_equity_path, parse_dates=["Date"])
    wf = wf.set_index("Date").sort_index()
    if "eq" not in wf.columns:
        raise ValueError("wf_equity.csv에 'eq' 컬럼이 없습니다.")
    strat_eq = wf["eq"].dropna()

    # ------------------------------
    # 2) S&P500 (실제 데이터)
    # ------------------------------
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance가 설치되어 있지 않습니다. 'pip install yfinance' 후 다시 실행하세요.") from e

    start_date = strat_eq.index.min().strftime("%Y-%m-%d")
    end_date   = strat_eq.index.max().strftime("%Y-%m-%d")

    # 멀티인덱스 문제 회피: history(auto_adjust=True) 사용
    sp_hist = yf.Ticker("^GSPC").history(start=start_date, end=end_date, auto_adjust=True)
    if sp_hist.empty or "Close" not in sp_hist.columns:
        raise RuntimeError("S&P 500 데이터를 불러오지 못했습니다.")
    sp500 = sp_hist[["Close"]].rename(columns={"Close": "bench_close"})
    sp500["eq"] = sp500["bench_close"] / sp500["bench_close"].iloc[0]
    sp500_eq = sp500["eq"].dropna()

    # ------------------------------
    # 3) Top 10 Equal-Weight 벤치마크
    # ------------------------------
    top10 = ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","AVGO","ASML","AMD"]
    t10 = yf.download(top10, start=start_date, end=end_date, auto_adjust=True, progress=False)

    # 멀티인덱스/단일인덱스 모두 방어
    if isinstance(t10.columns, pd.MultiIndex):
        if "Close" in t10.columns.levels[0]:
            top10_close = t10["Close"].copy()
        elif "Adj Close" in t10.columns.levels[0]:
            top10_close = t10["Adj Close"].copy()
        else:
            # 예상 밖 포맷: 첫 레벨의 첫 항목 사용
            lvl0 = list(t10.columns.levels[0])[0]
            top10_close = t10[lvl0].copy()
    else:
        # 단일 인덱스면 Close 또는 첫 컬럼 사용
        if "Close" in t10.columns:
            top10_close = t10[["Close"]].copy()
        elif "Adj Close" in t10.columns:
            top10_close = t10[["Adj Close"]].copy()
        else:
            first_col = t10.columns[0]
            top10_close = t10[[first_col]].copy()

    # 열 방향: 티커들이 열로 오도록 보장
    if top10_close.shape[1] == 1 and set(top10).intersection(top10_close.columns) == set():
        # 단일 열만 있을 때는 equal-weight 계산이 불가 → 그대로 eq 계산
        top10_ret = top10_close.pct_change().mean(axis=1).fillna(0.0)
    else:
        top10_ret = top10_close.pct_change().mean(axis=1).fillna(0.0)
    top10_eq = (1 + top10_ret).cumprod().rename("eq")

    # ------------------------------
    # 4) 세 곡선 비교 그래프 저장
    # ------------------------------
    out_curves = "C:/stock_ai/data/walk_forward/wf_vs_sp500_top10.png"
    plt.figure(figsize=(14,6))
    plt.plot(strat_eq.index, strat_eq.values, label="Our Strategy", linewidth=2)
    plt.plot(sp500_eq.index, sp500_eq.values, "--", label="S&P 500 (Real)", linewidth=2)
    plt.plot(top10_eq.index, top10_eq.values, label="Top 10 Equal Weight", linewidth=2)
    plt.axhline(1.0, ls="--", color="gray", label="Start")
    plt.title("Strategy vs S&P 500 vs Top 10 Equal Weight")
    plt.xlabel("Date"); plt.ylabel("Cumulative Return")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_curves, dpi=150)
    plt.close()
    print(f"🖼️ Saved curves: {out_curves}")

    # ------------------------------
    # 5) 성과 지표(CAGR / Sharpe / MDD)
    # ------------------------------
    def compute_metrics(eq_series, risk_free=0.02):
        eq_series = eq_series.dropna()
        daily_ret = eq_series.pct_change().dropna()

        # 기간(years)
        years = max((eq_series.index[-1] - eq_series.index[0]).days / 365, 1e-9)

        # CAGR
        total_return = float(eq_series.iloc[-1] / eq_series.iloc[0])
        cagr = total_return ** (1/years) - 1

        # Sharpe (연율화)
        excess = daily_ret - risk_free / 252
        sharpe = (np.sqrt(252) * excess.mean() / (excess.std() + 1e-12))

        # MDD
        roll_max = eq_series.cummax()
        dd = (eq_series / roll_max - 1).fillna(0.0)
        mdd = dd.min()

        return {"CAGR": float(cagr), "Sharpe": float(sharpe), "MDD": float(mdd)}

    metrics_strategy = compute_metrics(strat_eq)
    metrics_sp500    = compute_metrics(sp500_eq)
    metrics_top10    = compute_metrics(top10_eq)

    metrics_df = pd.DataFrame(
        [metrics_strategy, metrics_sp500, metrics_top10],
        index=["Our Strategy","S&P 500","Top 10 Equal Weight"]
    ).round(4)

    out_csv = "C:/stock_ai/data/walk_forward/performance_summary.csv"
    metrics_df.to_csv(out_csv, encoding="utf-8-sig")
    print(f"✅ Saved metrics: {out_csv}")

    # ------------------------------
    # 6) 성과표 시각화 (bar chart)
    # ------------------------------
    disp = metrics_df.copy()
    disp["CAGR(%)"] = (disp["CAGR"] * 100).round(2)
    disp["MDD(%)"]  = (disp["MDD"]  * 100).round(2)  # 음수 유지
    disp = disp[["CAGR(%)","Sharpe","MDD(%)"]]

    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(disp.index))
    w = 0.25

    ax.bar(x - w, disp["CAGR(%)"], width=w, label="CAGR(%)")
    ax.bar(x,      disp["Sharpe"], width=w, label="Sharpe")
    ax.bar(x + w,  disp["MDD(%)"], width=w, label="MDD(%)")

    ax.set_xticks(x)
    ax.set_xticklabels(disp.index)
    ax.set_title("Performance Summary: Strategy vs Benchmarks")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)

    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", padding=2)

    out_img = "C:/stock_ai/data/walk_forward/perf_summary.png"
    plt.tight_layout()
    plt.savefig(out_img, dpi=150)
    plt.close()
    print(f"🖼️ Saved performance chart: {out_img}")



