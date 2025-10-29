# C:\stock_ai\src\web\app.py
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import os
from streamlit_autorefresh import st_autorefresh


# === 사이징 유틸 (NEW) ===
def position_size(price, capital, weight_pct, lot=1):
    if price is None or price <= 0 or capital <= 0 or weight_pct <= 0:
        return 0, 0.0
    notional = capital * (weight_pct/100.0)
    qty = int((notional // price) // lot * lot)  # lot 단위 반올림 내림
    return qty, float(qty * price)

def build_buy_plan(df_buys, total_capital, mode, fixed_weight_pct, lot):
    """df_buys: columns [종목, 가격, 날짜, 근거]"""
    if df_buys.empty:
        return df_buys.assign(수량=0, 금액=0.0), 0.0, total_capital
    n = len(df_buys)
    if mode == "균등 비중" and n > 0:
        weight_each = 100.0 / n
    else:
        weight_each = fixed_weight_pct
    rows, used = [], 0.0
    for _, r in df_buys.iterrows():
        qty, notional = position_size(r.get("가격", None), total_capital, weight_each, lot)
        used += notional
        rows.append({**r.to_dict(), "비중(%)": round(weight_each,2), "수량": qty, "금액": round(notional,2)})
    plan = pd.DataFrame(rows).sort_values(["금액","종목"], ascending=[False, True])
    remain = max(0.0, float(total_capital - used))
    return plan, float(used), remain


ROOT = Path(__file__).resolve().parents[2]  # ...\src\web -> 프로젝트 루트
DATA = ROOT / "data"

DEFAULTS = ['AAPL','MSFT','GOOGL','AMZN','NVDA','TSLA','META','AVGO','ASML','AMD']

st.set_page_config(page_title="📈 주식 AI 대시보드", layout="wide")

st.title("📊 주식 AI — 시그널 & 성과 대시보드")

# ===== 사이드바 =====
with st.sidebar:
    st.header("설정")
    candidates = sorted({*DEFAULTS, *[p.name for p in DATA.iterdir() if p.is_dir()]})
    symbol = st.selectbox("종목 선택", options=candidates, index=candidates.index("AAPL") if "AAPL" in candidates else 0)
    show_live = st.checkbox("실시간 신호 포함", value=True)

    st.divider()
st.subheader("새로고침")
auto_refresh = st.checkbox("자동 새로고침 켜기", value=True)
interval_sec = st.select_slider("주기(초)", options=[15, 30, 60, 120], value=30)
if auto_refresh:
    st_autorefresh(interval=interval_sec * 1000, key="auto-refresh")

if st.button("⟳ 지금 바로 새로고침"):
    st.rerun()

    # === 주문 설정 (NEW) ===
with st.expander("🛒 주문 설정"):
    base_ccy = st.selectbox("통화", ["USD"], index=0)
    total_capital = st.number_input("투자 가능 금액", min_value=0.0, value=10000.0, step=100.0, help="주문 총액(USD)")
    sizing_mode = st.radio("사이징 방식", ["고정 비중(%)", "균등 비중"], index=0, horizontal=True)
    fixed_weight = st.slider("개별 종목 비중(%)", min_value=1, max_value=50, value=10, step=1, help="고정 비중 모드에서 사용")
    lot_size = st.number_input("최소 주문 수량(주)", min_value=1, value=1, step=1)


# ===== 가격 데이터 =====
sym_dir = DATA / symbol
price_file = sym_dir / "prices_tech.csv"
if not price_file.exists():
    st.error(f"가격 파일이 없습니다: {price_file}")
    st.stop()

px = pd.read_csv(price_file)
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

if not {"Open","High","Low","Close"}.issubset(set(px.columns)):
    st.error("OHLC(시가, 고가, 저가, 종가) 컬럼을 찾을 수 없습니다. prices_tech.csv 스키마 확인 필요.")
    st.stop()

px = px[[date_col,"Open","High","Low","Close"]].copy()
px.rename(columns={date_col:"Date"}, inplace=True)

# ===== 기술 지표 =====
px["SMA20"] = px["Close"].rolling(20, min_periods=1).mean()
px["SMA60"] = px["Close"].rolling(60, min_periods=1).mean()

# ===== 백테스트 시그널 =====
sig_file = sym_dir / "signals.json"
sigs = []
if sig_file.exists():
    try:
        data = json.loads(sig_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            buys = data.get("buys", [])
            sells = data.get("sells", [])
            for b in buys: b["action"] = "buy"; sigs.append(b)
            for s in sells: s["action"] = "sell"; sigs.append(s)
        elif isinstance(data, list):
            sigs.extend(data)
    except Exception:
        pass

# ===== 실시간 시그널 =====
if show_live:
    live_file = sym_dir / "signals_live.json"
    if live_file.exists():
        try:
            live = json.loads(live_file.read_text(encoding="utf-8"))
            if isinstance(live, list):
                for row in live:
                    row["live"] = True
                    sigs.append(row)
        except Exception:
            pass

sig_df = pd.DataFrame(sigs)
if not sig_df.empty and "date" in sig_df.columns:
    sig_df["date"] = pd.to_datetime(sig_df["date"], errors="coerce")
    sig_df = sig_df.dropna(subset=["date"]).sort_values("date")

# ===== 차트 =====
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=px["Date"], open=px["Open"], high=px["High"], low=px["Low"], close=px["Close"],
    name="캔들차트", showlegend=True
))
fig.add_trace(go.Scatter(x=px["Date"], y=px["SMA20"], name="SMA20", mode="lines"))
fig.add_trace(go.Scatter(x=px["Date"], y=px["SMA60"], name="SMA60", mode="lines"))

# 시그널 표시
if not sig_df.empty:
    buys = sig_df[sig_df["action"]=="buy"]
    sells= sig_df[sig_df["action"]=="sell"]

    if len(buys):
        bx = buys.merge(px[["Date","Close"]], left_on="date", right_on="Date", how="left")
        fig.add_trace(go.Scatter(
            x=bx["date"], y=bx["Close"], mode="markers",
            marker=dict(symbol="triangle-up", size=10, color="lime"),
            name="매수 (BUY)"
        ))
    if len(sells):
        sx = sells.merge(px[["Date","Close"]], left_on="date", right_on="Date", how="left")
        fig.add_trace(go.Scatter(
            x=sx["date"], y=sx["Close"], mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="red"),
            name="매도 (SELL)"
        ))

# 최근 1개월 데이터 범위 계산
last_date = px["Date"].max()
first_visible = last_date - pd.Timedelta(days=30)

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=True,   # 드래그 가능한 슬라이더 켜기
    xaxis_range=[first_visible, last_date],  # 기본 표시 구간 = 최근 30일
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    title=f"{symbol} — 가격 및 시그널 차트 (최근 1개월)"
)

st.plotly_chart(fig, use_container_width=True)

# ===== 기준 날짜 선택 (preds가 있으면 거기 최대 날짜, 없으면 오늘) =====
preds_for_dates = None
preds_file = DATA / "latest_live_preds.csv"
if preds_file.exists():
    preds_for_dates = pd.read_csv(preds_file)
    preds_for_dates["Date"] = pd.to_datetime(preds_for_dates["Date"], errors="coerce").dt.date
    date_options = sorted([d for d in preds_for_dates["Date"].dropna().unique()])
    default_idx = len(date_options) - 1 if date_options else 0
    st.markdown("### 📆 기준 날짜 선택")
    date_selected = st.selectbox("날짜", options=date_options, index=default_idx)
else:
    st.markdown("### 📆 기준 날짜 선택")
    date_selected = st.date_input("날짜", value=pd.Timestamp.today().date())

# ===== 라이브 신호 모으기 (오늘 buy/sell) =====
def collect_live_actions(date_selected):
    rows = []
    all_syms = sorted({*DEFAULTS, *[p.name for p in DATA.iterdir() if p.is_dir()]})
    for sym in all_syms:
        f = DATA / sym / "signals_live.json"
        if not f.exists():
            continue
        try:
            arr = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(arr, list):
                continue
            for r in arr:
                dt_parsed = pd.to_datetime(r.get("date"), errors="coerce")
                if pd.isna(dt_parsed) or dt_parsed.date() != date_selected:
                    continue
                rows.append({
                    "종목": sym,
                    "액션": r.get("action",""),
                    "가격": r.get("price", None),
                    "날짜": dt_parsed.strftime("%Y-%m-%d"),
                    "근거": ", ".join(r.get("reason", [])) if isinstance(r.get("reason", []), list) else r.get("reason", ""),
                    "ts": r.get("ts", "")  # 🆕 최신 판단용
                })
        except Exception:
            pass

    if not rows:
        empty = pd.DataFrame(columns=["종목","액션","가격","날짜","근거"])
        return empty.copy(), empty.copy()

    df = pd.DataFrame(rows)

    # 🆕 같은 종목-날짜 중복 제거: ts 최신 1건만 유지(없으면 입력순서 가정)
    if "ts" in df.columns and not df["ts"].isna().all():
        df["_order"] = pd.to_datetime(df["ts"], errors="coerce").fillna(pd.Timestamp(0))
    else:
        df["_order"] = np.arange(len(df))  # fallback

    df = df.sort_values(["종목","_order"]).groupby(["종목"], as_index=False).tail(1)

    # 🆕 BUY 우선순위 강제: 만약 액션이 엇갈리면 BUY를 택함
    # (이미 최신 1건만 남겼지만, 그래도 안전하게)
    df["_buy"] = (df["액션"].str.lower()=="buy").astype(int)
    df = df.sort_values(["_buy","_order"]).groupby(["종목"], as_index=False).tail(1)
    df = df.drop(columns=["_order","_buy","ts"], errors="ignore")

    buys = df[df["액션"]=="buy"].copy()
    sells= df[df["액션"]=="sell"].copy()
    if not buys.empty:  buys = buys.sort_values(["종목"])
    if not sells.empty: sells = sells.sort_values(["종목"])
    return buys, sells


# collect_live_actions 바로 다음 줄에 배치 (탭 생성보다 먼저!)
buys_today, sells_today = collect_live_actions(date_selected)
# SELL 근거 중 'ML_prob_top30' 포함된 행 제거 (보정)
sells_today = sells_today[~sells_today["근거"].str.contains("ML_prob_top30", na=False)]


# === 추가 안전장치: 같은 날짜에 겹치면 BUY 우선, SELL에서 제거 ===
if not buys_today.empty and not sells_today.empty:
    conflict_syms = set(buys_today["종목"]) & set(sells_today["종목"])
    if conflict_syms:
        sells_today = sells_today[~sells_today["종목"].isin(conflict_syms)].copy()


# ===== 탭: Top Signal / 오늘 사야될 종목 / 오늘 팔아야될 종목 / 성과 요약 =====
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Top Signal 종목",
    f"🟢 오늘 사야될 종목 ({len(buys_today)})",
    f"🔴 오늘 팔아야될 종목 ({len(sells_today)})",
    "📈 성과 요약"
])

with tab1:
    st.subheader("오늘의 Top Signal 종목 (상승확률 기준)")
    if preds_file.exists():
        preds = preds_for_dates.copy()
        preds["상승확률(%)"] = (pd.to_numeric(preds["proba"], errors="coerce").round(4) * 100).round(2) if "proba" in preds.columns else np.nan
        preds = preds.rename(columns={"Symbol":"종목", "Date":"날짜", "rank":"순위", "topk":"TopK"})
        view = preds[preds["날짜"] == date_selected].copy()
        if not view.empty:
            # TopK 하이라이트
            st.dataframe(
                view.sort_values("상승확률(%)", ascending=False)
                    .style.apply(lambda s: ["background-color: #eaffea" if (s.name in view.index and view.loc[s.name, "TopK"]==1) else "" for _ in s], axis=1),
                use_container_width=True
            )
            st.download_button(
                "CSV 다운로드 (Top Signal)",
                data=view.sort_values("상승확률(%)", ascending=False).to_csv(index=False, encoding="utf-8-sig"),
                file_name=f"top_signals_{date_selected}.csv",
                mime="text/csv"
            )
        else:
            st.info("해당 날짜의 예측 데이터가 없습니다.")
    else:
        st.info("latest_live_preds.csv 파일이 아직 없습니다. signal_live.py를 먼저 실행하세요.")

with tab2:
    st.subheader("🟢 오늘 사야될 종목 (BUY)")
    if not buys_today.empty:
        # 가격 누락 방지(신호 파일에 가격이 없으면 당일 종가 매핑 시도)
        if "가격" not in buys_today.columns or buys_today["가격"].isna().any():
            try:
                last_close = px.set_index("Date")["Close"]
                buys_today["가격"] = buys_today.apply(
                    lambda r: last_close.loc[pd.to_datetime(r["날짜"])] if pd.to_datetime(r["날짜"]) in last_close.index else np.nan, axis=1
                )
            except Exception:
                pass

        plan, used, remain = build_buy_plan(
            df_buys=buys_today[["종목","가격","날짜","근거"]],
            total_capital=total_capital,
            mode=sizing_mode,
            fixed_weight_pct=fixed_weight,
            lot=lot_size
        )

                # === 요약 카드 (총액/잔액/종목수) ===
        m1, m2, m3 = st.columns(3)
        m1.metric("💰 총 주문 금액", f"${used:,.2f}")
        m2.metric("🏦 잔액", f"${remain:,.2f}")
        m3.metric("📦 매수 종목 수", f"{len(plan):,} 개")

        # (표시는 그대로)
        st.dataframe(plan, use_container_width=True)

        st.download_button(
            "CSV 다운로드 (오늘 매수 주문 계획)",
            data=plan.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"buy_orders_{date_selected}.csv",
            mime="text/csv"
        )

    else:
        st.info("오늘 매수 신호가 없습니다.")

with tab3:
    st.subheader("🔴 오늘 팔아야될 종목 (SELL)")
    if not sells_today.empty:
        # 보유 수량을 모르면 전량 매도로 표기만 제공
        sells_view = sells_today.copy()
        sells_view["수량"] = "ALL"   # 필요시 보유수량 연동해서 채우기
        st.dataframe(sells_view, use_container_width=True)
        st.download_button(
            "CSV 다운로드 (오늘 매도 주문 계획)",
            data=sells_view.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"sell_orders_{date_selected}.csv",
            mime="text/csv"
        )
        st.caption("※ 보유 수량을 연동하면 실제 수량 계산도 자동화할 수 있어요. (원하면 CSV 업로드 입력 추가해줄게요)")
    else:
        st.info("오늘 매도 신호가 없습니다.")

        # === 충돌 안전장치: 같은 날짜에 겹치면 BUY 우선, SELL에서 제거 ===
if not buys_today.empty and not sells_today.empty:
    conflict_syms = set(buys_today["종목"]) & set(sells_today["종목"])
    if conflict_syms:
        sells_today = sells_today[~sells_today["종목"].isin(conflict_syms)].copy()



with tab4:
    st.subheader("전략 성과 요약 (Walk-Forward)")
    perf_csv = DATA / "walk_forward" / "performance_summary.csv"
    perf_img = DATA / "walk_forward" / "perf_summary.png"
    if perf_csv.exists():
        perf = pd.read_csv(perf_csv, index_col=0)
        st.dataframe(perf, use_container_width=True)
    if perf_img.exists():
        st.image(str(perf_img))


st.caption("📂 데이터 경로: " + str(DATA))
st.caption(f"마지막 갱신: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

