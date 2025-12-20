@echo off
REM Start Trading Bot with console output AND log file

cd /d "%~dp0"

REM Activate virtual environment
call "venv\Scripts\activate.bat"

REM Clear old log file
if exist bot_output.log del bot_output.log

REM Run bot with output to both console and file using PowerShell
REM The -u flag ensures unbuffered output
powershell -Command "& { $ErrorActionPreference = 'Continue'; python -u live_mt5_eval_runner.py --mode eval 2>&1 | ForEach-Object { $_; $_ | Out-File -FilePath 'bot_output.log' -Append -Encoding utf8 } }"

pause
