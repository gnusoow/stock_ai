
from pathlib import Path
import subprocess, sys, json
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SRC  = ROOT / "src"

def run_optional_sma_benchmark():
    # Run SMA 20/60 benchmark for portfolio, if data exists
    bt_sma = SRC / "backtest_sma.py"
    if not bt_sma.exists():
        return
    cmd = [sys.executable, str(bt_sma), "--fast", "20", "--slow", "60"]
    print("▶ Running SMA benchmark:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=False)
    except Exception as e:
        print("[WARN] SMA benchmark failed:", e)

def safe_read_json(p: Path, default=None):
    if default is None: default = {}
    if not p.exists(): return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default

def safe_read_csv(p: Path):
    if not p.exists(): return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def build_pdf():
    out_dir = DATA / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "daily_summary.pdf"

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Daily Strategy Summary", styles["Title"]))
    story.append(Spacer(1, 8))

    # 1) ML classifier metrics
    ml_metrics = safe_read_json(DATA / "ml_cls_metrics.json", default={})
    story.append(Paragraph("<b>ML Classifier Metrics</b>", styles["Heading2"]))
    if ml_metrics:
        table = Table([["AUC", "AP"], [ml_metrics.get("AUC","-"), ml_metrics.get("AP","-")]])
        table.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.25,colors.grey)]))
        story.append(table)
    else:
        story.append(Paragraph("No ML metrics found.", styles["BodyText"]))
    story.append(Spacer(1, 8))

    # ML equity chart
    ml_chart = DATA / "charts" / "ml_cls_backtest_equity.png"
    if ml_chart.exists():
        story.append(Paragraph("ML Strategy Equity (last window)", styles["Heading3"]))
        story.append(Image(str(ml_chart), width=480, height=280))
        story.append(Spacer(1, 8))

    # 2) Walk-Forward results
    story.append(Paragraph("<b>Walk-Forward Summary</b>", styles["Heading2"]))
    wf_sum = safe_read_json(DATA / "walk_forward" / "wf_summary.json", default={})
    wf_tbl_data = [["Periods","Sharpe(avg)","Sharpe(med)","Min MDD","AUC(avg)","AP(avg)"],
                   [wf_sum.get("periods","-"),
                    wf_sum.get("Sharpe_avg","-"),
                    wf_sum.get("Sharpe_median","-"),
                    wf_sum.get("MDD_min","-"),
                    wf_sum.get("AUC_avg","-"),
                    wf_sum.get("AP_avg","-")]]
    story.append(Table(wf_tbl_data, style=[('GRID',(0,0),(-1,-1),0.25,colors.grey)]))
    story.append(Spacer(1, 8))

    wf_png = DATA / "walk_forward" / "wf_equity.png"
    if wf_png.exists():
        story.append(Paragraph("Walk-Forward Equity", styles["Heading3"]))
        story.append(Image(str(wf_png), width=480, height=280))
        story.append(Spacer(1, 8))

    # 3) SMA benchmark portfolio chart (if exists)
    story.append(Paragraph("<b>SMA(20/60) Benchmark</b>", styles["Heading2"]))
    port_csv = DATA / "bt_portfolio_sma_20_60.csv"
    if port_csv.exists():
        dfp = pd.read_csv(port_csv)
        # quick plot
        fig_path = DATA / "charts" / "sma_benchmark_portfolio.png"
        plt.figure()
        plt.plot(pd.to_datetime(dfp["Date"]), dfp["PortEquity"], label="SMA 20/60 Portfolio")
        plt.legend(); plt.title("SMA 20/60 Portfolio Equity")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, bbox_inches="tight", dpi=140)
        plt.close()
        story.append(Image(str(fig_path), width=480, height=280))
    else:
        story.append(Paragraph("No SMA portfolio result found (optional).", styles["BodyText"]))

    # 4) Latest signals snapshot note
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Signals</b> (see data/<SYM>/signals.json)", styles["Heading2"]))

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    doc.build(story)
    print("✅ PDF saved:", pdf_path)

def main():
    # Ensure ML + signals are up to date
    print("▶ Updating ML pipeline")
    subprocess.run([sys.executable, str(SRC / "update_daily.py")], check=False)
    # Optional: SMA benchmark for comparison
    run_optional_sma_benchmark()
    # Build consolidated PDF
    build_pdf()

if __name__ == "__main__":
    main()
