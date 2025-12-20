@echo off
REM Start Streamlit Dashboard in headless mode

cd /d "%~dp0"

REM Activate virtual environment
call "venv\Scripts\activate.bat"

REM Run Streamlit in headless mode (no email prompt)
streamlit run dashboard\dashboard.py --server.headless true --server.port 8501

pause
