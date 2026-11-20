@echo off
rem ACE-Step-1.5-fast setup for Windows: uv -> .venv (uv.lock) -> GPU check -> checkpoints.
rem   setup.bat              16gb profile set: xl-sft, xl-turbo, 2B turbo, LM 0.6B (+ vae, text encoder)  ~45 GB
rem   setup.bat --large      also LM 4B
rem   setup.bat --12gb       2B only: 2B turbo, 2B base, LM 0.6B, no XL (ACESTEP_GPU_PROFILE=12gb)  ~15 GB
rem   setup.bat --no-models  environment only
rem Not yet verified on Windows (written from the Linux setup.sh; the Python side is the same).
setlocal EnableDelayedExpansion
cd /d "%~dp0"
rem Hugging Face cache without symlinks: without Developer Mode, huggingface_hub's parallel downloads race on the
rem symlink check and abort with WinError 1314 (seen on Windows 10). Disk usage is unchanged.
if not defined HF_HUB_DISABLE_SYMLINKS set HF_HUB_DISABLE_SYMLINKS=1
set WANT_MODELS=1
set LARGE=0
set TWELVE=0
:args
if "%~1"=="" goto run
if /i "%~1"=="--no-models" set WANT_MODELS=0
if /i "%~1"=="--large" set LARGE=1
if /i "%~1"=="--12gb" set TWELVE=1
shift
goto args

:run
echo [1/4] uv
where uv >nul 2>nul
if errorlevel 1 (
    where winget >nul 2>nul
    if not errorlevel 1 (
        echo installing uv with winget...
        winget install --id astral-sh.uv -e --source winget --accept-source-agreements --accept-package-agreements
        set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
    ) else (
        echo winget not found; installing uv with the official PowerShell installer...
        powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
    )
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
uv --version || (
    echo.
    echo ============================================================
    echo  uv was just installed but this window does not see it yet.
    echo  CLOSE this window, open a new PowerShell, and run  .\setup.bat  AGAIN.
    echo  Nothing else has been installed yet - the second run does the real work.
    echo ============================================================
    exit /b 1
)

echo [2/4] Python environment (.venv, from uv.lock; Python 3.11 is required by the flash-attn wheel)
uv sync --frozen --python 3.11 || exit /b 1

echo [3/4] GPU / libraries
uv run --no-sync python scripts\check_env.py

if "%WANT_MODELS%"=="1" (
    echo [4/4] checkpoints (skips what already exists)
    if "%TWELVE%"=="1" (
        uv run --no-sync acestep-download --model acestep-v15-base || exit /b 1
    ) else (
        uv run --no-sync acestep-download --model acestep-v15-xl-sft || exit /b 1
        uv run --no-sync acestep-download --model acestep-v15-xl-turbo || exit /b 1
    )
    uv run --no-sync acestep-download --model acestep-5Hz-lm-0.6B || exit /b 1
    if "%LARGE%"=="1" uv run --no-sync acestep-download --model acestep-5Hz-lm-4B
) else (
    echo [4/4] checkpoints: skipped
)
echo [5/5] Easy Music web app environment (easy-music\.venv)
if not exist "easy-music\.venv" uv venv easy-music\.venv --python 3.11 -q
uv pip install --python easy-music\.venv\Scripts\python.exe -q -r easy-music\requirements.txt
rem GGUF lyrics fallback (Gemma 4 E4B, CPU): prebuilt llama-server.exe from llama.cpp releases, no Build Tools.
rem Pre-fetch here (optional, ~25MB); otherwise it is fetched on the first fallback.
easy-music\.venv\Scripts\python.exe easy-music\scripts\fetch_llama_server.py || echo   llama-server not pre-fetched (offline?): it will be fetched on the first fallback
echo.
echo Done. Start the server with:  run_api_server_fast.bat   (then open http://localhost:8001/ui/)
echo       and the web app with:   run_easy_music.bat        (http://localhost:8888)
endlocal
