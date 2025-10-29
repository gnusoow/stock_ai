from pathlib import Path
import pandas as pd
import json

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def _detect_trades(pos_series):
    pos = pos_series.fillna(0).astype(int)
    chg = pos.diff().fillna(pos.iloc[0])
    buys = pos.index[chg==1]
    sells= pos.index[chg==-1]
    return list(buys), list(sells)

def main():
    pred_file = DATA_DIR / "latest_preds.csv"
    if not pred_file.exists():
        print("No latest_preds.csv; run train_ml_classifier.py first.")
        return
    dfp = pd.read_csv(pred_file)
    dfp["Date"] = pd.to_datetime(dfp["Date"])
    dfp = dfp.sort_values(["Date","Symbol"])

    # 날짜별 상위 30% 임계치
    thr = dfp.groupby("Date")["proba"].quantile(0.70)
    dfp = dfp.join(thr.rename("thr"), on="Date")
    dfp["pos"] = (dfp["proba"] >= dfp["thr"]).astype(int)

    for sym, g in dfp.groupby("Symbol"):
        g = g.sort_values("Date").set_index("Date")
        buys, sells = _detect_trades(g["pos"])

        price_file = DATA_DIR / sym / "prices_tech.csv"
        if not price_file.exists():
            continue
        px = pd.read_csv(price_file)
        px["Date"] = pd.to_datetime(px["Date"])
        px = px.set_index("Date").sort_index()
        if "Close" not in px.columns and len(px.columns)>=5:
            px = px.rename(columns={px.columns[3]:"Close"})

        def items(dates):
            out=[]
            for dt in dates:
                if dt in px.index:
                    out.append({"date": dt.strftime("%Y-%m-%d"),
                                "price": float(px.loc[dt,"Close"]),
                                "reason": ["ML_prob_top30"]})
            return out

        payload = {
            "symbol": sym,
            "buys":  items(buys),
            "sells": items(sells),
        }
        out = DATA_DIR / sym / "signals.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print("✅ Saved:", out)

if __name__ == "__main__":
    main()