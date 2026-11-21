@echo off
rem ACE-Step-1.5-fast API server (Windows). Same defaults as run_api_server_fast.sh:
rem   ACESTEP_GPU_PROFILE=16gb (default): XL-SFT on the GPU, XL-turbo / 2B turbo swapped in from the mmap store
rem   ACESTEP_GPU_PROFILE=large          : every DiT resident, LM 4B, no offload
rem   ACESTEP_GPU_PROFILE=12gb           : 2B only (2B turbo on the GPU, 2B base swapped in), no XL
rem Set any variable before calling this script to override it, e.g.  set ACESTEP_API_PORT=8002
rem Verified on Windows (RTX 4060 Laptop 8GB, see README).
setlocal EnableDelayedExpansion
cd /d "%~dp0"
rem The environment is built by setup.bat (twice on a fresh PC: the first run installs uv and stops, the second builds .venv).
rem Check it here so a half-finished install gives one clear message instead of "uvicorn: command not found".
if not exist ".venv\Scripts\python.exe" (
    echo .venv is missing: run  .\setup.bat  first ^(on a fresh PC it stops after installing uv - open a new window and run it again^).
    exit /b 1
)
.venv\Scripts\python.exe -c "import uvicorn, torch" >nul 2>nul
if errorlevel 1 (
    echo .venv is incomplete ^(uvicorn or torch missing^): run  .\setup.bat  again - it resumes where it stopped.
    exit /b 1
)
rem see setup.bat: Hugging Face cache without symlinks (WinError 1314 on PCs without Developer Mode)
if not defined HF_HUB_DISABLE_SYMLINKS set HF_HUB_DISABLE_SYMLINKS=1
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

rem No profile given: pick it from the card's VRAM so a 12 GB GPU does not silently get the 16gb default
rem (XL-SFT, 50 steps: runs, but takes minutes). 14000 MiB threshold: 12 GB cards report ~12288, 16 GB ~16376.
set "ACESTEP_SMI_GPU="
if defined CUDA_VISIBLE_DEVICES set "ACESTEP_SMI_GPU=-i %CUDA_VISIBLE_DEVICES%"
if not defined ACESTEP_GPU_PROFILE (
    for /f "usebackq tokens=1 delims=, " %%v in (`nvidia-smi %ACESTEP_SMI_GPU% --query-gpu=memory.total --format=csv,noheader,nounits 2^>nul`) do (
        if not defined ACESTEP_VRAM_MIB set ACESTEP_VRAM_MIB=%%v
    )
)
if not defined ACESTEP_GPU_PROFILE if defined ACESTEP_VRAM_MIB (
    if %ACESTEP_VRAM_MIB% LSS 14000 (set ACESTEP_GPU_PROFILE=12gb) else (set ACESTEP_GPU_PROFILE=16gb)
    echo [run_api_server_fast] GPU VRAM %ACESTEP_VRAM_MIB% MiB -^> ACESTEP_GPU_PROFILE=!ACESTEP_GPU_PROFILE! ^(set ACESTEP_GPU_PROFILE to override; large for ^>= 40 GB^)
)
if not defined ACESTEP_GPU_PROFILE set ACESTEP_GPU_PROFILE=16gb
if /i "%ACESTEP_GPU_PROFILE%"=="large" (
    if not defined ACESTEP_DIT_RESIDENCY set ACESTEP_DIT_RESIDENCY=shared
    if not defined ACESTEP_LM_MODEL_PATH set ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-4B
    if not defined ACESTEP_OFFLOAD_TO_CPU set ACESTEP_OFFLOAD_TO_CPU=0
    if not defined ACESTEP_LM_OFFLOAD_TO_CPU set ACESTEP_LM_OFFLOAD_TO_CPU=0
    if not defined ACESTEP_RESIDENT_COMPONENTS set "ACESTEP_RESIDENT_COMPONENTS="
    if not defined ACESTEP_VAE_DECODE_CHUNK_SIZE set ACESTEP_VAE_DECODE_CHUNK_SIZE=1024
)

if /i "%ACESTEP_GPU_PROFILE%"=="12gb" (
    if not defined ACESTEP_CONFIG_PATH set ACESTEP_CONFIG_PATH=acestep-v15-turbo
    if not defined ACESTEP_CONFIG_PATH2 set ACESTEP_CONFIG_PATH2=acestep-v15-base
    rem no third slot: leave ACESTEP_CONFIG_PATH3 undefined (an unset / empty slot is not loaded)
    set ACESTEP_NO_THIRD_SLOT=1
)
if not defined ACESTEP_CONFIG_PATH set ACESTEP_CONFIG_PATH=acestep-v15-xl-sft
if not defined ACESTEP_CONFIG_PATH2 set ACESTEP_CONFIG_PATH2=acestep-v15-xl-turbo
if not defined ACESTEP_CONFIG_PATH3 if not defined ACESTEP_NO_THIRD_SLOT set ACESTEP_CONFIG_PATH3=acestep-v15-turbo
if not defined ACESTEP_DIT_RESIDENCY set ACESTEP_DIT_RESIDENCY=exclusive
if not defined ACESTEP_DIT_RESIDENCY_STORE set ACESTEP_DIT_RESIDENCY_STORE=mmap
if not defined ACESTEP_QUANTIZATION set ACESTEP_QUANTIZATION=none
if not defined ACESTEP_COMPILE_MODEL set ACESTEP_COMPILE_MODEL=0
if not defined ACESTEP_FUSED_RMSNORM set ACESTEP_FUSED_RMSNORM=exact
if not defined ACESTEP_INIT_LLM set ACESTEP_INIT_LLM=1
if not defined ACESTEP_LM_MODEL_PATH set ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B
if not defined ACESTEP_LM_BACKEND set ACESTEP_LM_BACKEND=pt
if not defined ACESTEP_DEVICE set ACESTEP_DEVICE=auto
if not defined ACESTEP_OFFLOAD_TO_CPU set ACESTEP_OFFLOAD_TO_CPU=1
if not defined ACESTEP_RESIDENT_COMPONENTS set ACESTEP_RESIDENT_COMPONENTS=vae
if not defined ACESTEP_LM_OFFLOAD_TO_CPU set ACESTEP_LM_OFFLOAD_TO_CPU=0
if not defined ACESTEP_OFFLOAD_DIT_TO_CPU set ACESTEP_OFFLOAD_DIT_TO_CPU=0
if not defined ACESTEP_VAE_DECODE_CHUNK_SIZE set ACESTEP_VAE_DECODE_CHUNK_SIZE=512
if not defined ACESTEP_NO_INIT set ACESTEP_NO_INIT=false
rem PYTORCH_ALLOC_CONF=expandable_segments:True is not set here: not supported by every Windows torch build.

if not defined ACESTEP_QUEUE_WORKERS set ACESTEP_QUEUE_WORKERS=1
if not defined ACESTEP_API_WORKERS set ACESTEP_API_WORKERS=%ACESTEP_QUEUE_WORKERS%
if %ACESTEP_QUEUE_WORKERS% GTR 1 if "%ACESTEP_RESIDENT_COMPONENTS%"=="vae" set ACESTEP_RESIDENT_COMPONENTS=vae,text_encoder
if not defined ACESTEP_QUEUE_MAXSIZE set ACESTEP_QUEUE_MAXSIZE=200
if not defined ACESTEP_API_HOST set ACESTEP_API_HOST=0.0.0.0
if not defined ACESTEP_API_PORT set ACESTEP_API_PORT=8001
if not defined ACESTEP_API_LOG_LEVEL set ACESTEP_API_LOG_LEVEL=info

echo [run_api_server_fast] Starting the ACE server. Loading models takes a few minutes on the first start:
echo [run_api_server_fast] wait for the line "[API Server] READY" before generating from Easy Music.
python -m uvicorn acestep.api_server:app --host %ACESTEP_API_HOST% --port %ACESTEP_API_PORT% --workers 1 --log-level %ACESTEP_API_LOG_LEVEL%
endlocal
