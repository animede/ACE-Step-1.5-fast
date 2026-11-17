@echo off
rem Start Easy Music (easy-music\) against the local API server. Environment: easy-music\.venv (made by setup.bat).
rem   run_easy_music.bat         -> http://localhost:8888      run_easy_music.bat 8890 -> another port
setlocal
cd /d "%~dp0easy-music"
if not exist ".venv\Scripts\activate.bat" (
    echo easy-music\.venv is missing: run setup.bat first.
    exit /b 1
)
call ".venv\Scripts\activate.bat"
if not defined ACESTEP_API_PORT set ACESTEP_API_PORT=8001
if not defined ACE_STEP_API_URL set ACE_STEP_API_URL=http://127.0.0.1:%ACESTEP_API_PORT%
set "ARGS=%*"
echo %~1| findstr /r "^[0-9][0-9]*$" >nul && set "ARGS=--port %*"
if not defined EASY_MUSIC_RELOAD set "ARGS=--no-reload %ARGS%"
python main.py %ARGS%
endlocal
