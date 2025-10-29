from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import yaml

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
    df = df.sort_values(["Symbol","Date"])
    return df

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



def _enforce_min_hold(pos_series, min_hold):
    s = pos_series.copy()
    cnt = 0
    for i in range(len(s)):
        if s.iat[i]==1:
            cnt += 1
        else:
            if 0 < cnt < min_hold:
                s.iat[i] = 1
                cnt += 1
            else:
                cnt = 0
    return s

def main():
    cfg = load_cfg()
    k_pct     = float(cfg["backtest"]["k_top_percent"])
    fee_bps   = int(cfg["backtest"]["fee_bps"])
    min_hold  = int(cfg["backtest"]["min_hold"])
    test_days = int(cfg["backtest"]["test_window_days"])

    df = load_dataset()
    feats = [c for c in df.columns if c not in ["Date","Symbol","target_5d","label_top30"]]
    target = "label_top30"

    cutoff = df["Date"].max() - pd.Timedelta(days=test_days)
    train = df[df["Date"] <= cutoff].copy()
    test  = df[df["Date"] >  cutoff].copy()

    if HAS_XGB:
        model = XGBClassifier(
            n_estimators=600, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=42, n_jobs=0, tree_method="hist"
        )
        pipe = model
    else:
        pipe = make_pipeline(
            StandardScaler(with_mean=False),
            RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
        )

    pipe.fit(train[feats], train[target])
    proba = pipe.predict_proba(test[feats])[:,1]

    auc  = roc_auc_score(test[target], proba)
    ap   = average_precision_score(test[target], proba)

    daily = backtest_topk(test, proba, k_pct=k_pct, fee_bps=fee_bps, min_hold=min_hold)

    (DATA_DIR / "charts").mkdir(exist_ok=True, parents=True)
    daily.to_csv(DATA_DIR / "ml_cls_backtest_equity.csv", index=False, encoding="utf-8-sig")
    with open(DATA_DIR / "ml_cls_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"AUC": float(auc), "AP": float(ap)}, f, indent=2)

    plt.figure()
    plt.plot(daily["Date"], daily["eq"], label="ML Classifier Strategy")
    plt.legend(); plt.title("ML Classifier Backtest (Last {}D)".format(test_days))
    plt.savefig(DATA_DIR / "charts" / "ml_cls_backtest_equity.png", bbox_inches="tight", dpi=140)
    plt.close()

    # 최신 예측 저장(latest_preds.csv) - 웹/시그널 생성용
    latest = test[["Date","Symbol","Close"]].copy()
    latest["proba"] = proba
    latest.to_csv(DATA_DIR / "latest_preds.csv", index=False, encoding="utf-8-sig")

    print("✅ Saved:",
          DATA_DIR / "ml_cls_backtest_equity.csv",
          DATA_DIR / "ml_cls_metrics.json",
          DATA_DIR / "charts" / "ml_cls_backtest_equity.png",
          DATA_DIR / "latest_preds.csv")

if __name__ == "__main__":
    main()