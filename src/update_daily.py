# C:\stock_ai\src\update_daily.py
import subprocess, sys, datetime, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"

STEPS = [
    [sys.executable, str(SRC / "fetch_data.py")],
    [sys.executable, str(SRC / "make_ml_dataset_cls.py")],
    [sys.executable, str(SRC / "train_ml_classifier.py")],
    [sys.executable, str(SRC / "generate_signals.py")],
    
# ✅ 실시간 신호 생성 단계 (추가)
    [sys.executable, str(SRC / "signal_live.py"),
     "--symbols", "AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META,AVGO,ASML,AMD",
     "--k_pct", "0.30"],
]

def run(cmd):
    print("▶", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("❌ step failed:", cmd); sys.exit(r.returncode)

def main():
    print("🚀 Daily update start:", datetime.datetime.now())
    for cmd in STEPS:
        run(cmd)
    print("✅ Daily update done:", datetime.datetime.now())

if __name__ == "__main__":
    main()
["python", str(SRC / "signal_live.py"),
 "--symbols", "AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META,AVGO,ASML,AMD",
 "--k_pct", "0.30"],
