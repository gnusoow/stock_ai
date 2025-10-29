# C:\stock_ai\src\optimize_params.py
from __future__ import annotations
import warnings, json, yaml
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
CONFIGS = ROOT / "configs"
CONFIGS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# ---- 공통 유틸 ----
def backtest_topk(test_df: pd.DataFrame, preds: np.ndarray, k_pct=0.30, fee_bps=5, min_hold=3):
    df = test_df.copy()
    df["pred"] = preds
    # 날짜별 임계치로 상위 k% 포지션
    thr = df.groupby("Date")["pred"].quantile(1 - k_pct)
    df = df.join(thr.rename("thr"), on="Date")
    df["pos"] = (df["pred"] >= df["thr"]).astype(int)

    # 다음날 체결
    df = df.sort_values(["Symbol","Date"])
    df["ret1"] = df.groupby("Symbol")["Close"].pct_change().fillna(0.0)
    df["pos_exec"] = df.groupby("Symbol")["pos"].shift(1).fillna(0)

    # 최소 보유일 적용
    if min_hold > 0:
        df["pos_exec"] = df.groupby("Symbol")["pos_exec"].apply(
            lambda s: _enforce_min_hold(s, min_hold)
        ).values

    # 거래비용(진입/청산 시점 변화량) → 원본 인덱스에 정렬
    turn = df.groupby("Symbol")["pos_exec"].apply(lambda s: s.diff().abs())
    turn.index = turn.index.droplevel(0)          # (Symbol, idx) -> idx로 평탄화
    turn = turn.reindex(df.index).fillna(0.0)     # df 인덱스에 정렬
    fee_series = turn * (fee_bps / 10000.0)

    # 일별 평균 수익 - 평균 비용
    daily = (
        df.assign(fee=fee_series)
          .groupby("Date")
          .apply(lambda g: (g["ret1"] * g["pos_exec"]).mean() - g["fee"].mean())
          .rename("ret")
          .reset_index()
    )
    daily["eq"] = (1 + daily["ret"]).cumprod()

    # 평균 turnover(진입/청산 빈도)도 리포트(스칼라)
    turnover_mean = float(turn.mean())
    return daily, turnover_mean


def _enforce_min_hold(pos_series, min_hold):
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

def sharpe_turn_penalty(daily_ret: pd.Series, turnover_mean: float, rf=0.02, lam=0.5):
    # 연 Sharpe – λ * Turnover (turnover는 0~1 스케일 가정)
    if len(daily_ret) < 10 or daily_ret.std()==0:
        return -1e9
    excess = daily_ret - rf/252
    sharpe = np.sqrt(252)*excess.mean()/ (excess.std()+1e-12)
    return float(sharpe - lam*turnover_mean)

# ---- 데이터 로드 ----
def load_dataset():
    f = DATA / "ml_dataset_cls.csv"
    if not f.exists():
        raise FileNotFoundError("Run make_ml_dataset_cls.py first")
    df = pd.read_csv(f)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol","Date"])
    return df

def timeseries_splits(dates: pd.Series, n_splits=6, min_train_days=365*2, step_days=30):
    # 월 단위 Walk-Forward 느낌으로 자르는 간단한 제너레이터
    dmin, dmax = dates.min(), dates.max()
    cut = dmin + pd.Timedelta(days=min_train_days)
    for i in range(n_splits):
        train_end = cut + pd.Timedelta(days=i*step_days)
        test_end  = train_end + pd.Timedelta(days=step_days)
        train_mask = (dates <= train_end)
        test_mask  = (dates > train_end) & (dates <= test_end)
        if test_mask.sum() < 5: break
        yield train_mask, test_mask

# ---- 최적화 ----
def main():
    import optuna
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    try:
        from xgboost import XGBClassifier
        HAS_XGB = True
    except Exception:
        HAS_XGB = False

    df = load_dataset()
    feats = [c for c in df.columns if c not in ["Date","Symbol","target_5d","label_top30"]]
    target = "label_top30"

    def objective(trial: optuna.Trial):
        # 모델 선택 & 하이퍼파라미터
        model_name = trial.suggest_categorical("model", ["rf","xgb"] if HAS_XGB else ["rf"])
        if model_name == "rf":
            n_estimators = trial.suggest_int("n_estimators", 200, 800, step=100)
            max_depth    = trial.suggest_int("max_depth", 5, 14)
            clf = make_pipeline(
                StandardScaler(with_mean=False),
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth, random_state=42, n_jobs=-1
                )
            )
        else:
            lr   = trial.suggest_float("learning_rate", 0.02, 0.2, log=True)
            md   = trial.suggest_int("max_depth", 4, 8)
            nest = trial.suggest_int("n_estimators", 300, 900, step=150)
            subs = trial.suggest_float("subsample", 0.6, 1.0)
            col  = trial.suggest_float("colsample_bytree", 0.6, 1.0)
            clf = XGBClassifier(
                n_estimators=nest, max_depth=md, learning_rate=lr,
                subsample=subs, colsample_bytree=col, reg_lambda=1.0,
                random_state=42, tree_method="hist", n_jobs=0
            )

        # 전략 파라미터
        k_pct    = trial.suggest_float("k_pct", 0.1, 0.5)     # 상위 10~50%
        min_hold = trial.suggest_int("min_hold", 1, 7)        # 최소 보유일
        fee_bps  = trial.suggest_int("fee_bps", 2, 15)        # 왕복 비용(bps)
        lam      = trial.suggest_float("lambda_turn", 0.2, 0.8)  # turnover 패널티

        scores=[]
        for tr_mask, te_mask in timeseries_splits(df["Date"], n_splits=10, min_train_days=365*2, step_days=30):
            tr, te = df[tr_mask], df[te_mask]
            if len(te)==0 or len(tr)==0: 
                continue
            clf.fit(tr[feats], tr[target])
            proba = (clf.predict_proba(te[feats])[:,1]) if hasattr(clf, "predict_proba") else clf.decision_function(te[feats])
            daily, tmean = backtest_topk(te, proba, k_pct=k_pct, fee_bps=fee_bps, min_hold=min_hold)
            score = sharpe_turn_penalty(daily["ret"], turnover_mean=tmean, lam=lam)
            scores.append(score)

        if not scores: 
            return -1e9
        return float(np.nanmean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40, show_progress_bar=True)

    best = study.best_params
    print("\n⭐ BEST PARAMS")
    print(json.dumps(best, indent=2, ensure_ascii=False))

    # 저장: best_config.yml
    cfg = {
        "model": best.get("model","rf"),
        "hyperparams": {k:v for k,v in best.items() if k in ["n_estimators","max_depth","learning_rate","n_estimators","subsample","colsample_bytree"]},
        "strategy": {
            "k_pct": round(float(best.get("k_pct",0.30)), 4),
            "min_hold": int(best.get("min_hold",3)),
            "fee_bps": int(best.get("fee_bps",5)),
            "lambda_turn": round(float(best.get("lambda_turn",0.5)), 3),
        }
    }
    (CONFIGS / "best_config.yml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"📝 Saved: {CONFIGS / 'best_config.yml'}")

    # 베스트로 전체 재학습 → 모델 저장
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    try:
        from xgboost import XGBClassifier
        HAS_XGB2 = True
    except Exception:
        HAS_XGB2 = False

    if cfg["model"]=="rf" or not HAS_XGB2:
        clf = make_pipeline(
            StandardScaler(with_mean=False),
            RandomForestClassifier(
                n_estimators=int(best.get("n_estimators",500)),
                max_depth=int(best.get("max_depth",10)),
                random_state=42, n_jobs=-1
            )
        )
    else:
        clf = XGBClassifier(
            n_estimators=int(best.get("n_estimators",600)),
            max_depth=int(best.get("max_depth",5)),
            learning_rate=float(best.get("learning_rate",0.05)),
            subsample=float(best.get("subsample",0.9)),
            colsample_bytree=float(best.get("colsample_bytree",0.9)),
            reg_lambda=1.0, random_state=42, tree_method="hist", n_jobs=0
        )

    clf.fit(df[feats], df[target])
    import joblib
    joblib.dump(clf, MODELS / "ml_classifier.pkl")
    print(f"💾 Saved model: {MODELS / 'ml_classifier.pkl'}")
    print("✅ Optimization done.")
    
if __name__ == "__main__":
    main()
