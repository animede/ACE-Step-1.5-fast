@echo off
rem Start ACE-Step-client-pro (client-pro\) against the local API server. Environment: client-pro\.venv (made by setup.bat).
rem   run_client_pro.bat         -> http://localhost:8888      run_client_pro.bat 8890 -> another port
setlocal
cd /d "%~dp0client-pro"
if not exist ".venv\Scripts\activate.bat" (
    echo client-pro\.venv is missing: run setup.bat first.
    exit /b 1
)
call ".venv\Scripts\activate.bat"
if not defined ACESTEP_API_PORT set ACESTEP_API_PORT=8001
if not defined ACE_STEP_API_URL set ACE_STEP_API_URL=http://127.0.0.1:%ACESTEP_API_PORT%
set "ARGS=%*"
echo %~1| findstr /r "^[0-9][0-9]*$" >nul && set "ARGS=--port %*"
python main.py %ARGS%
endlocal
