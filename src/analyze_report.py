# src/analyze_report.py
import pandas as pd
import numpy as np
import os   # ✅ 추가
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# 가능한 한글 폰트 자동 탐색(Windows 우선: 맑은 고딕)
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\malgun.ttf",            # 맑은 고딕
    r"C:\Windows\Fonts\Malgun.ttf",
    r"C:\Windows\Fonts\NanumGothic.ttf",       # 나눔고딕
]
KOREAN_FONT_NAME = None
for fp in FONT_CANDIDATES:
    try:
        if os.path.exists(fp):
            pdfmetrics.registerFont(TTFont("Korean", fp))
            KOREAN_FONT_NAME = "Korean"
            break
    except Exception:
        pass

from pathlib import Path
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet


DEFAULTS = ['AAPL','MSFT','GOOGL','AMZN','NVDA','TSLA','META','AVGO','ASML','AMD']

def analyze_symbol(base_dir: Path, symbol: str) -> dict:
    sym_dir = base_dir / symbol
    pfile = sym_dir / "prices_tech.csv"
    nfile = sym_dir / "news_sentiment.csv"
    res = {"Symbol": symbol}

    if not pfile.exists():
        res.update({k: None for k in ["Trend","RSI","MACD","Volatility","Sentiment","Summary"]})
        return res

    df = pd.read_csv(pfile)
    if "Date" not in df.columns:
        df = df.reset_index().rename(columns={"index":"Date"})
    if "Close" not in df.columns:
        cols = [c for c in df.columns if c != "Date"]
        if len(cols) >= 5:
            df = df.rename(columns={cols[3]:"Close"})

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    close = df["Close"]
    if "SMA20" not in df.columns: df["SMA20"] = SMAIndicator(close, window=20).sma_indicator()
    if "SMA60" not in df.columns: df["SMA60"] = SMAIndicator(close, window=60).sma_indicator()
    if "RSI14" not in df.columns: df["RSI14"] = RSIIndicator(close, window=14).rsi()
    if "MACD" not in df.columns:
        m = MACD(close)
        df["MACD"] = m.macd()
        df["MACD_SIGNAL"] = m.macd_signal()
    if "BB_WIDTH" not in df.columns:
        bb = BollingerBands(close)
        df["BB_WIDTH"] = (bb.bollinger_hband() - bb.bollinger_lband()) / close

    latest = df.iloc[-1]
    sma20, sma60, close_val = latest["SMA20"], latest["SMA60"], latest["Close"]

    # 1️⃣ 추세
    if close_val > sma20 > sma60:
        trend = "상승"
    elif close_val < sma20 < sma60:
        trend = "하락"
    else:
        trend = "횡보"

    # 2️⃣ RSI
    rsi = latest["RSI14"]
    if rsi > 70: rsi_state = "과매수"
    elif rsi < 30: rsi_state = "과매도"
    else: rsi_state = "중립"

    # 3️⃣ MACD
    macd_val = latest["MACD"]
    macd_sig = latest.get("MACD_SIGNAL", macd_val)
    macd_state = "상승모멘텀" if macd_val > macd_sig else "하락모멘텀"

    # 4️⃣ 변동성
    bbwidth = latest["BB_WIDTH"]
    if bbwidth > 0.25: vol_state = "높음"
    elif bbwidth < 0.1: vol_state = "낮음"
    else: vol_state = "보통"

    # 5️⃣ 뉴스 감정
    sentiment_state = "정보 없음"
    if nfile.exists():
        nf = pd.read_csv(nfile)
        if "compound" in nf.columns:
            val = nf["compound"].mean()
            if val > 0.1: sentiment_state = "긍정"
            elif val < -0.1: sentiment_state = "부정"
            else: sentiment_state = "중립"

    # 6️⃣ 종합 판단
    if trend == "상승" and macd_state == "상승모멘텀" and rsi_state != "과매수":
        summary = "📈 상승세 유지"
    elif trend == "하락" and macd_state == "하락모멘텀":
        summary = "📉 약세 추세"
    elif rsi_state == "과매수":
        summary = "⚠️ 상승 과열"
    elif rsi_state == "과매도":
        summary = "🔁 반등 가능성"
    else:
        summary = "➖ 중립 구간"

    res.update({
        "Trend": trend,
        "RSI": f"{rsi_state} ({rsi:.1f})",
        "MACD": macd_state,
        "Volatility": vol_state,
        "Sentiment": sentiment_state,
        "Summary": summary
    })
    return res


def _strip_emoji(text: str) -> str:
    # ReportLab은 컬러 이모지 미지원 → 간단히 제거/치환
    replacements = {
        "📊": "", "📈": "[상승]", "📉": "[하락]",
        "⚠️": "[주의]", "🔁": "[반등]", "➖": "[중립]"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def export_pdf(df: pd.DataFrame, pdf_path: Path):
    styles = getSampleStyleSheet()
    # 한글 폰트가 등록되었으면 모든 스타일에 적용
    if KOREAN_FONT_NAME:
        for _, style in styles.byName.items():   # ✅ styles.byName.values()도 OK
            style.fontName = KOREAN_FONT_NAME


    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    story = []

    title_text = _strip_emoji("📊 NASDAQ 10종 기술 + 심리 분석 리포트")
    title = Paragraph(title_text, styles["Title"])
    story.append(title)
    story.append(Spacer(1, 12))

    # 이모지 제거한 데이터(요약 컬럼 등에서 이모지 치환)
    df2 = df.copy()
    for c in df2.columns:
        if df2[c].dtype == object:
            df2[c] = df2[c].astype(str).map(_strip_emoji)

    data = [list(df2.columns)] + df2.values.tolist()
    table = Table(data, repeatRows=1)

    # 표 스타일
    table_style = [
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#003366')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,-1), KOREAN_FONT_NAME or 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,0), 11),
        ('FONTSIZE', (0,1), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey]),
    ]
    table.setStyle(TableStyle(table_style))

    story.append(table)
    doc.build(story)



def main():
    base_dir = Path(__file__).resolve().parents[1] / "data"
    results = [analyze_symbol(base_dir, s) for s in DEFAULTS]
    df = pd.DataFrame(results)
    out_csv = base_dir / "analysis_report.csv"
    out_pdf = base_dir / "analysis_report.pdf"

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    export_pdf(df, out_pdf)

    print(f"✅ CSV saved at: {out_csv}")
    print(f"✅ PDF saved at: {out_pdf}")
    print(df)


if __name__ == "__main__":
    main()
