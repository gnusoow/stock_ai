@echo off
setlocal enabledelayedexpansion

REM === 경로 세팅 ===
cd /d C:\stock_ai

REM === 로그 폴더 ===
if not exist C:\stock_ai\logs mkdir C:\stock_ai\logs

REM === 가상환경 활성화 ===
call .\.venv\Scripts\activate.bat

REM === 타임스탬프 만들기(YYYYMMDD_HHMMSS) ===
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set ldt=%%I
set ts=!ldt:~0,8!_!ldt:~8,6!

REM === 업데이트 파이프라인 실행 + 로그 저장 ===
python .\src\update_daily.py  >> C:\stock_ai\logs\update_daily_!ts!.log 2>&1

endlocal
