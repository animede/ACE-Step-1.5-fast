@echo off
rem ACE-Step-1.5-fast setup for Windows: uv -> .venv (uv.lock) -> GPU check -> checkpoints.
rem   setup.bat              16gb profile set: xl-sft, xl-turbo, 2B turbo, LM 0.6B (+ vae, text encoder)  ~45 GB
rem   setup.bat --large      also LM 4B
rem   setup.bat --no-models  environment only
rem Not yet verified on Windows (written from the Linux setup.sh; the Python side is the same).
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set WANT_MODELS=1
set LARGE=0
:args
if "%~1"=="" goto run
if /i "%~1"=="--no-models" set WANT_MODELS=0
if /i "%~1"=="--large" set LARGE=1
shift
goto args

:run
echo [1/4] uv
where uv >nul 2>nul
if errorlevel 1 (
    where winget >nul 2>nul
    if not errorlevel 1 (
        echo installing uv with winget...
        winget install --id astral-sh.uv -e --accept-source-agreements --accept-package-agreements
        set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
    ) else (
        echo winget not found; installing uv with the official PowerShell installer...
        powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
    )
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
uv --version || (echo uv is not on PATH yet: open a new command prompt and run setup.bat again. & exit /b 1)

echo [2/4] Python environment (.venv, from uv.lock; Python 3.11 is required by the flash-attn wheel)
uv sync --frozen --python 3.11 || exit /b 1

echo [3/4] GPU / libraries
uv run --no-sync python -c "import torch; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'cuda available:', ok); [print(' GPU%d: %s %.1f GB sm_%d%d' % (i, p.name, p.total_memory/2**30, p.major, p.minor)) for i in range(torch.cuda.device_count()) for p in [torch.cuda.get_device_properties(i)]]; import importlib; m=importlib.util.find_spec('flash_attn'); print('flash_attn', 'available' if m else 'not installed (the server will use SDPA)')"

if "%WANT_MODELS%"=="1" (
    echo [4/4] checkpoints (skips what already exists)
    uv run --no-sync acestep-download --model acestep-v15-xl-sft || exit /b 1
    uv run --no-sync acestep-download --model acestep-v15-xl-turbo || exit /b 1
    uv run --no-sync acestep-download --model acestep-5Hz-lm-0.6B || exit /b 1
    if "%LARGE%"=="1" uv run --no-sync acestep-download --model acestep-5Hz-lm-4B
) else (
    echo [4/4] checkpoints: skipped
)
echo [5/5] Easy Music web app environment (easy-music\.venv)
if not exist "easy-music\.venv" uv venv easy-music\.venv --python 3.11 -q
findstr /v /i "llama-cpp-python" easy-music\requirements.txt > easy-music\requirements.base.tmp
uv pip install --python easy-music\.venv\Scripts\python.exe -q -r easy-music\requirements.base.tmp
del easy-music\requirements.base.tmp
rem GGUF lyrics fallback (Gemma 4 E4B, CPU). Built from source: needs Visual Studio Build Tools
rem (C++ workload) + CMake. The prebuilt CPU wheels are too old for Gemma 4. Optional.
uv pip install --python easy-music\.venv\Scripts\python.exe -q "llama-cpp-python>=0.3.34" || echo   llama-cpp-python not installed (needs Visual Studio Build Tools C++ and CMake): the client will use the external LLM only
echo.
echo Done. Start the server with:  run_api_server_fast.bat   (then open http://localhost:8001/ui/)
echo       and the web app with:   run_easy_music.bat        (http://localhost:8888)
endlocal
