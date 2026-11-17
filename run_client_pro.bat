@echo off
rem Start ACE-Step-client-pro (client-pro\) against the local API server. Environment: client-pro\.venv (made by setup.bat).
setlocal
cd /d "%~dp0client-pro"
if not exist ".venv\Scripts\activate.bat" (
    echo client-pro\.venv is missing: run setup.bat first.
    exit /b 1
)
call ".venv\Scripts\activate.bat"
if not defined ACESTEP_API_PORT set ACESTEP_API_PORT=8001
if not defined ACE_STEP_API_URL set ACE_STEP_API_URL=http://127.0.0.1:%ACESTEP_API_PORT%
python main.py %*
endlocal
