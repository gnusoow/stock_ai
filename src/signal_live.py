# -*- coding: utf-8 -*-
# Robust LIVE signal generator (probability + risk filters + mutual exclusivity)
from pathlib import Path
import argparse, json, sys, datetime as dt
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DEFAULTS = ['AAPL','MSFT','GOOGL','AMZN','NVDA','TSLA','META','AVGO','ASML','AMD']

# ------------------------------
# IO helpers
# ------------------------------
def load_preds():
    f = DATA / "latest_live_preds.csv"
    if not f.exists():
        print("[ERROR] latest_live_preds.csv not found. Run training or pipeline first.")
        sys.exit(1)
    df = pd.read_csv(f)
    if "Symbol" not in df.columns or "proba" not in df.columns:
        print("[ERROR] latest_live_preds.csv must have Symbol, proba columns")
        sys.exit(1)
    df["Date"] = pd.to_datetime(df.get("Date", pd.Timestamp.today()), errors="coerce").dt.date
    return df.rename(columns={"Symbol":"Symbol", "proba":"proba"})

def load_price_features(symbol):
    """prices_tech.csv에서 Close/SMA/RSI 추출 (경량 RSI 구현)"""
    f = DATA / symbol / "prices_tech.csv"
    if not f.exists():
        return None
    px = pd.read_csv(f)
    date_col = "Date" if "Date" in px.columns else px.columns[0]
    px[date_col] = pd.to_datetime(px[date_col], errors="coerce")
    px = px.dropna(subset=[date_col]).sort_values(date_col)

    cols = px.columns.tolist()
    cmap = {}
    if "Close" not in cols and len(cols) >= 5:
        cmap[cols[3]] = "Close"
    if "Open" not in cols and len(cols) >= 5:
        cmap[cols[0]] = "Open"
    if "High" not in cols and len(cols) >= 5:
        cmap[cols[1]] = "High"
    if "Low" not in cols and len(cols) >= 5:
        cmap[cols[2]] = "Low"
    px = px.rename(columns=cmap)
    if "Close" not in px.columns:
        return None

    close = pd.to_numeric(px["Close"], errors="coerce")
    sma20 = close.rolling(20, min_periods=1).mean()
    sma60 = close.rolling(60, min_periods=1).mean()

    # RSI(14) 간단 계산
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=14).mean()
    rs = gain / (loss + 1e-12)
    rsi14 = 100 - (100 / (1 + rs))

    last_idx = px.index[-1]
    return {
        "date": pd.to_datetime(px.loc[last_idx, date_col]).date(),
        "close": float(close.iloc[-1]),
        "SMA20": float(sma20.iloc[-1]),
        "SMA60": float(sma60.iloc[-1]),
        "RSI14": float(rsi14.iloc[-1]) if np.isfinite(rsi14.iloc[-1]) else np.nan,
    }

def load_vix_z(target_date):
    """macro_sample.csv에서 ^VIX의 60D Z-score. 없으면 0."""
    f = DATA / "macro_sample.csv"
    if not f.exists():
        return 0.0
    df = pd.read_csv(f)
    dcol = "Date" if "Date" in df.columns else df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol).set_index(dcol)

    col_vix = None
    for c in df.columns:
        if c.upper() in ("^VIX", "VIX", "CBOE_VIX"):
            col_vix = c
            break
    if col_vix is None:
        return 0.0

    s = pd.to_numeric(df[col_vix], errors="coerce").ffill()
    roll = s.rolling(60, min_periods=20)
    z = (s - roll.mean()) / (roll.std() + 1e-9)
    td = pd.to_datetime(target_date)
    try:
        v = float(z.loc[:td].iloc[-1])
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0

def load_prev_positions():
    f = DATA / "live_positions.csv"
    if not f.exists():
        return {}
    try:
        df = pd.read_csv(f)
        return {str(r["Symbol"]): int(r["pos"]) for _, r in df.iterrows() if "Symbol" in r and "pos" in r}
    except Exception:
        return {}

def save_positions(pos_map):
    rows = [{"Symbol": k, "pos": int(v)} for k, v in sorted(pos_map.items())]
    pd.DataFrame(rows).to_csv(DATA / "live_positions.csv", index=False, encoding="utf-8-sig")

def append_signal(symbol, action, price, date, reasons):
    path = DATA / symbol / "signals_live.json"
    arr = []
    if path.exists():
        try:
            arr = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            arr = []

    # ✅ 기존 기록 덮어쓰기 전에 reason 초기화
    entry = {
        "date": str(date),
        "action": action,
        "price": float(price),
        "reason": reasons,   # <- 최신 근거로 새로 저장
        "ts": str(pd.Timestamp.now())
    }

    # 같은 날짜 액션이 있으면 교체
    arr = [a for a in arr if a.get("date") != str(date)]
    arr.append(entry)

    path.write_text(json.dumps(arr, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "date": str(date),
        "ts": dt.datetime.now().isoformat(timespec="seconds"),  # 🆕 기록 시각
        "action": action,
        "price": float(price),
        "reason": reason_list,
    }
    f = DATA / symbol / "signals_live.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    arr = []
    if f.exists():
        try:
            arr = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(arr, list):
                arr = []
        except Exception:
            arr = []

    # 🆕 같은 날짜 이벤트는 제거(최신 한 건만 유지)
    arr = [e for e in arr if str(e.get("date")) != payload["date"]]

    arr.append(payload)
    f.write_text(json.dumps(arr, indent=2, ensure_ascii=False), encoding="utf-8")

# ------------------------------
# main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate live signals using probability + risk filters")
    ap.add_argument("--symbols", type=str, default=",".join(DEFAULTS), help="comma separated symbols")
    # 히스테리시스/임계값 (주의: help 문자열에 % 사용 금지)
    ap.add_argument("--buy_proba", type=float, default=0.72, help="buy when proba >= this")
    ap.add_argument("--sell_proba", type=float, default=0.45, help="sell when proba <= this")
    ap.add_argument("--rsi_buy_max", type=float, default=68.0, help="buy only if RSI14 < this")
    ap.add_argument("--rsi_sell_min", type=float, default=74.0, help="sell if RSI14 > this")
    ap.add_argument("--vixz_sell", type=float, default=1.5, help="sell if VIX z-score > this")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    preds = load_preds()
    preds = preds[preds["Symbol"].isin(symbols)].copy()

    # 가격/지표 로드
    feats = []
    for sym in symbols:
        feat = load_price_features(sym)
        if feat is not None:
            feats.append({"Symbol": sym, **feat})
    feats = pd.DataFrame(feats)
    if feats.empty:
        print("[ERROR] no price features available")
        sys.exit(1)

    # 날짜/매크로 동기화
    ref_date = preds["Date"].max()
    feats["VIX_z"] = feats["date"].apply(load_vix_z)

    live = preds.merge(feats, on="Symbol", how="inner")

    # === 코어 조건 분리 ===
    core_buy = (
        (live["proba"] >= args.buy_proba) &
        (live["SMA20"] > live["SMA60"]) &
        (live["RSI14"] < args.rsi_buy_max)
    )
    core_sell = (
        (live["proba"] <= args.sell_proba) |
        (live["SMA20"] < live["SMA60"]) |
        (live["RSI14"] > args.rsi_sell_min) |
        (live["VIX_z"] > args.vixz_sell)
    )

    # === 상호배타 보장 ===
    live["buy_ok"]    = core_buy  & (~core_sell)
    live["sell_risk"] = core_sell & (~core_buy)

    # 포지션 갱신 (우선순위 BUY > HOLD > SELL, 충돌 원천 차단)
    prev_pos = load_prev_positions()
    new_pos  = dict(prev_pos)
    buys, sells = [], []

    for _, r in live.iterrows():
        sym, price, today = r["Symbol"], r["close"], r["date"]
        was = prev_pos.get(sym, 0)
        now = was

        if was == 0 and r["buy_ok"]:
            now = 1
            buys.append(sym)
            append_signal(sym, "buy", price, today,
                          [f"proba>={args.buy_proba}", "SMA20>SMA60", f"RSI14<{args.rsi_buy_max}"])
        elif was == 1 and r["sell_risk"]:
            now = 0
            sells.append(sym)
            reasons = []
            if r["proba"] <= args.sell_proba: reasons.append(f"proba<={args.sell_proba}")
            if r["SMA20"] < r["SMA60"]:        reasons.append("SMA20<SMA60")
            if r["RSI14"] > args.rsi_sell_min: reasons.append(f"RSI14>{args.rsi_sell_min}")
            if r["VIX_z"] > args.vixz_sell:    reasons.append(f"VIX_z>{args.vixz_sell}")
            append_signal(sym, "sell", price, today, reasons)

        new_pos[sym] = now

    # 저장 및 로그
    save_positions(new_pos)
    out = live[["Symbol","date","proba","SMA20","SMA60","RSI14","VIX_z","buy_ok","sell_risk"]].copy()
    out.rename(columns={"date":"Date"}, inplace=True)
    out.to_csv(DATA / "latest_live_eval.csv", index=False, encoding="utf-8-sig")

    print("===== LIVE SIGNALS (probability + risk) =====")
    print("BUY :", ", ".join(sorted(buys)) if buys else "-")
    print("SELL:", ", ".join(sorted(sells)) if sells else "-")
    if not buys and not sells:
        print("No changes today.")
    print("=============================================")

if __name__ == "__main__":
    main()
