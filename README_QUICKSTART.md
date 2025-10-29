# Stock AI Starter (Classification + Signals)

## 0) Create venv & install
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1   # (Windows PowerShell)
pip install -r requirements.txt
```

## 1) Fetch data
```powershell
python .\src\fetch_data.py --years 12
```

## 2) Build classification dataset
```powershell
python .\src\make_ml_dataset_cls.py
```

## 3) Train classifier + backtest
```powershell
python .\src\train_ml_classifier.py
```

Outputs under `data/`:
- `ml_cls_metrics.json` (AUC/AP)
- `ml_cls_backtest_equity.csv` & `charts/ml_cls_backtest_equity.png`
- `latest_preds.csv` (Date, Symbol, proba)

## 4) Generate signals.json per symbol
```powershell
python .\src\generate_signals.py
```

Signals written to `data/<SYM>/signals.json`.

## 5) Daily chain
```powershell
python .\src\update_daily.py
```

## Notes
- Config: `configs/base.yaml` to adjust symbols & backtest settings.
- Macro: ^VIX and ^TNX fetched via yfinance.

## 6) Walk-Forward (월별 재학습) 리포트
```powershell
python .\src\walk_forward.py
```
산출물:
- `data\walk_forward\wf_monthly_report.csv`
- `data\walk_forward\wf_equity.csv` & `wf_equity.png`
- `data\walk_forward\wf_summary.json`


## 6) Walk-Forward (월별 재학습)
```powershell
python .\src\walk_forward.py
```

## 7) (선택) SMA(20/60) 벤치마크
```powershell
python .\src\backtest_sma.py --fast 20 --slow 60
```

## 8) 통합 리포트(PDF) 생성 — 한 방에 업데이트 + PDF 생성
```powershell
python .\src\report_daily.py
```
- ML 파이프라인 업데이트(데이터→학습→시그널) 실행
- (선택) SMA 벤치마크 실행
- `data\reports\daily_summary.pdf` 생성 (ML/Walk-Forward/SMA 요약 포함)

### 기존 보조 스크립트
- `src\visualize.py` : 종목별 차트(SMA/MACD/RSI, 감성) 저장
- `src\analyze_report.py` : 기술적 관점 요약 PDF (원본 기능 유지)
- `src\backtest_multifactor.py`, `src\sweep_multifactor.py` : (이전 세대) 필요 시 참고, 기본 플로우에서는 불필요
