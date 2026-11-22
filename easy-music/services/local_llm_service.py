"""
Local LLM Service - ビルド済み llama-server によるフォールバック作詞

外部 LLM に接続できない場合に Gemma 4 E4B (GGUF Q4_0) を使用してローカルで歌詞・テーマを
生成する。thinking(reasoning)は常に OFF。

2026-11-17: llama-cpp-python(ソースビルドが必要で、Windows では Visual Studio Build Tools の
C++ と CMake で詰まりやすい)をやめ、llama.cpp の GitHub Releases にある **ビルド済み
`llama-server`**(CPU 版、約 25MB)を子プロセスとして起動し、OpenAI 互換 API で話す方式にした。
コンパイラ不要。Linux x64/arm64・Windows x64/arm64・macOS に対応。

- バイナリ: 環境変数 LLAMA_SERVER=<パス> → tools/llama/<tag>/llama-server[.exe] → PATH → 無ければ
  固定タグ(LLAMA_CPP_TAG、既定 b12475)から自動取得。
- モデル: LOCAL_LLM_GGUF=<パス> → models/gemma-4-E4B-it-Q4_0.gguf(無ければ HF から約 4.4GB を自動取得)。
- 起動: llama-server -m <gguf> --host 127.0.0.1 --port <LOCAL_LLM_PORT、既定 11440> -c 8192 -t <CPU/2>
  --reasoning off。/health が 200 を返すまで待つ。そのポートで既に何かが応答していればそれを使う。
- 子プロセスはプロセス終了時(atexit)に止める。親が強制終了された場合も Windows は Job Object
  (KILL_ON_JOB_CLOSE)、Linux は prctl(PR_SET_PDEATHSIG) で道連れにする。

2026-11-20: GPU 対応。LOCAL_LLM_GPU=auto(既定)で nvidia-smi が見えれば llama.cpp の **CUDA 版**
(Windows: bin-win-cuda-12.4-x64 + cudart、Linux: bin-ubuntu-cuda-12.8-x64 + cudart、合計 0.6〜0.8GB)を取得して
`--fit on`(空き VRAM に入る層数だけ GPU に載せ、残りは CPU)で起動する。E4B Q4_0 は全部載ると約 3GB + KV。
3060 12GB で ACE(12gb プロファイル、待機 約 7GB)と同居できるのはこの自動フィットのおかげで、i3 の CPU で
1 分 20 秒かかっていた歌詞が十数秒になる。ACE に生成を投げる直前に `release_before_ace()` で止めて VRAM を
返し(ACE の拡散ピーク 8.5GB のため)、次の LLM 呼び出しで起こし直す(ページキャッシュから数秒)。
LOCAL_LLM_GPU=off で従来の CPU 版、LOCAL_LLM_GPU=all / <層数> で固定。
"""
import asyncio
import atexit
import io
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from typing import Any, Dict, Optional

logger = logging.getLogger("uvicorn.error")   # Easy Music のログ設定で INFO が画面に出るロガー

# モデル設定
MODEL_REPO = "ggml-org/gemma-4-E4B-it-GGUF"
MODEL_FILE = "gemma-4-E4B-it-Q4_0.gguf"
MODEL_SIZE_HINT = "~4.4GB"
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(APP_DIR, "models")
TOOLS_DIR = os.path.join(APP_DIR, "tools", "llama")
LOGS_DIR = os.path.join(APP_DIR, "logs")
DEFAULT_TAG = "b12475"      # 2026-11-17 に Gemma 4 E4B Q4_0 で動作確認(Linux x64 / Windows x64 の CPU 版)
CUDA_WIN = "12.4"          # Windows 用 CUDA ビルドの CUDA 版数(ドライバ 550 以上。ACE-Step 自体が 12.8 対応ドライバを要求するので満たす)
CUDA_LINUX = "12.8"        # Linux 用
FIT_TARGET_MIB = 1536      # --fit が GPU に残す余白(MiB)。ACE の待機分はすでに確保済みだが、LLM と同時に動く ACE の LM(format_input)の分を残す
ACE_BUSY_MAX_S = 1200      # ACE の生成中フラグの最長(クライアントが status を取りに来なくなったときの保険)
DEFAULT_PORT = 11440
N_CTX = 8192                # システムプロンプト + 歌詞に十分
HEALTH_TIMEOUT_S = 300      # 初回はモデルの読み込みに時間がかかる
REQUEST_TIMEOUT_S = 600     # CPU 推論

# Gemma 4 の思考チャネル / Qwen 系の <think> ブロック(念のため両方除去)
_THOUGHT_RE = re.compile(
    r"<\|channel>thought\n.*?<channel\|>\s*|<think>.*?</think>\s*",
    flags=re.DOTALL,
)

_PARENT_PID = os.getpid()


def _tag() -> str:
    return os.environ.get("LLAMA_CPP_TAG", "").strip() or DEFAULT_TAG


def _port() -> int:
    return int(os.environ.get("LOCAL_LLM_PORT", "").strip() or DEFAULT_PORT)


def _exe_name() -> str:
    return "llama-server.exe" if sys.platform == "win32" else "llama-server"


def gpu_setting() -> str:
    """LOCAL_LLM_GPU: auto(既定) / off / all / <層数>"""
    return (os.environ.get("LOCAL_LLM_GPU", "").strip() or "auto").lower()


GPU_AUTO_MIN_FREE_MIB = 1024     # auto で GPU を使うのに要る空き VRAM。CUDA 版は層が少なくても入力の読み込み(prompt eval)を GPU で
                                 # 行うので、i3 では層が 0 でも CPU 版より速い。文脈と作業領域の分だけ要る


def nvidia_total_mib() -> Optional[int]:
    """最初の GPU の VRAM 総量(MiB)。nvidia-smi が無ければ None。"""
    return _nvidia_query("memory.total")


def nvidia_free_mib() -> Optional[int]:
    """最初の GPU(CUDA_VISIBLE_DEVICES があればその GPU)の空き VRAM(MiB)。nvidia-smi が無ければ None。"""
    return _nvidia_query("memory.free")


def _nvidia_query(field: str) -> Optional[int]:
    exe = shutil.which("nvidia-smi")
    if exe is None and sys.platform == "win32":
        # 新しいドライバは System32、古いドライバは NVIDIA Corporation\NVSMI(PATH に無い)
        for cand in (os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe"),
                     os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")):
            if os.path.exists(cand):
                exe = cand
                break
    if exe is None:
        return None
    cmd = [exe, f"--query-gpu={field}", "--format=csv,noheader,nounits"]
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if vis:
        cmd[1:1] = ["-i", vis]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, **_popen_kwargs_quiet())
        line = out.stdout.strip().splitlines()[0].strip()
        return int(float(line))
    except Exception:  # noqa: BLE001
        return None


def use_gpu() -> bool:
    """CUDA 版を使うか。auto は nvidia-smi が見えて x64 の Windows / Linux のとき。"""
    mode = gpu_setting()
    if mode == "off":
        return False
    x64 = platform.machine().lower() in ("x86_64", "amd64")
    cuda_platform = sys.platform == "win32" or sys.platform.startswith("linux")
    if not (x64 and cuda_platform):
        return False
    if mode == "auto":
        free = nvidia_free_mib()
        if free is None:
            return False
        if free < GPU_AUTO_MIN_FREE_MIB:
            # 空きが小さい(別の GPU アプリが使っている)ときだけ CPU。8GB / 6GB 級でも ACE の待機は約 3.3GB なので普通は載る。
            # ACE の format_input と生成の前には止めるので、同居であふれることはない(2026-11-22 に 4060 8GB / i3 で確認: 歌詞が速くなった)
            logger.info("ローカルLLM: 空き VRAM %d MiB が %d MiB 未満なので CPU で動かします(LOCAL_LLM_GPU=on で GPU に固定)", free, GPU_AUTO_MIN_FREE_MIB)
            return False
        return True
    return True


def release_asset(gpu: Optional[bool] = None) -> Optional[tuple]:
    """(アーカイブ名, アーカイブ内の llama-server の相対パス, [cudart アーカイブ名])。未対応プラットフォームは None。

    gpu=True は CUDA 版(x64 の Windows / Linux のみ)。アーカイブは 1 段のディレクトリに入っていることがあるので
    展開側(_extract_archive)で平らにする。
    """
    tag = _tag()
    machine = platform.machine().lower()
    x64 = machine in ("x86_64", "amd64")
    if gpu is None:
        gpu = use_gpu()
    if gpu and x64:
        if sys.platform.startswith("linux"):
            return (f"llama-{tag}-bin-ubuntu-cuda-{CUDA_LINUX}-x64.tar.gz", "llama-server",
                    f"cudart-llama-{tag}-bin-ubuntu-cuda-{CUDA_LINUX}-x64.tar.gz")
        if sys.platform == "win32":
            return (f"llama-{tag}-bin-win-cuda-{CUDA_WIN}-x64.zip", "llama-server.exe",
                    f"cudart-llama-bin-win-cuda-{CUDA_WIN}-x64.zip")
    machine = platform.machine().lower()
    x64 = machine in ("x86_64", "amd64")
    arm = machine in ("aarch64", "arm64")
    if sys.platform.startswith("linux"):
        if x64:
            return f"llama-{tag}-bin-ubuntu-x64.tar.gz", f"llama-{tag}/llama-server"
        if arm:
            return f"llama-{tag}-bin-ubuntu-arm64.tar.gz", f"llama-{tag}/llama-server"
    elif sys.platform == "win32":
        if x64:
            return f"llama-{tag}-bin-win-cpu-x64.zip", "llama-server.exe"
        if arm:
            return f"llama-{tag}-bin-win-cpu-arm64.zip", "llama-server.exe"
    elif sys.platform == "darwin":
        if arm:
            return f"llama-{tag}-bin-macos-arm64.tar.gz", f"llama-{tag}/llama-server"
        if x64:
            return f"llama-{tag}-bin-macos-x64.tar.gz", f"llama-{tag}/llama-server"
    return None


def _tools_dir(gpu: bool) -> str:
    return os.path.join(TOOLS_DIR, _tag() + ("-cuda" if gpu else ""))


def find_binary(gpu: Optional[bool] = None) -> Optional[str]:
    override = os.environ.get("LLAMA_SERVER", "").strip()
    if override:
        p = os.path.expanduser(override)
        return p if os.path.exists(p) else None
    if gpu is None:
        gpu = use_gpu()
    local = os.path.join(_tools_dir(gpu), _exe_name())
    if os.path.exists(local):
        return local
    if gpu:
        return None  # CPU 版が落ちていても GPU 版は別に取る(PATH の llama-server も CPU 版かもしれないので使わない)
    return shutil.which("llama-server")


def _extract_archive(name: str, data: bytes, dest_dir: str) -> None:
    """zip / tar.gz を dest_dir に展開し、1 段のディレクトリに入っていれば中身を持ち上げる(.so / .dll が exe と同じ場所に要る)。"""
    os.makedirs(dest_dir, exist_ok=True)
    before = set(os.listdir(dest_dir))
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(dest_dir)
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
            try:
                t.extractall(dest_dir, filter="data")
            except TypeError:  # Python < 3.12
                t.extractall(dest_dir)
    added = [x for x in os.listdir(dest_dir) if x not in before]
    if len(added) == 1 and os.path.isdir(os.path.join(dest_dir, added[0])):
        sub = os.path.join(dest_dir, added[0])
        for item in os.listdir(sub):
            target = os.path.join(dest_dir, item)
            if not os.path.exists(target):
                shutil.move(os.path.join(sub, item), target)
        shutil.rmtree(sub, ignore_errors=True)


def _fetch(url: str) -> bytes:
    logger.info("Downloading: %s", url)
    with urllib.request.urlopen(url, timeout=600) as resp:
        return resp.read()


def download_binary(gpu: Optional[bool] = None) -> str:
    """Releases の固定タグから llama-server(GPU なら CUDA 版 + cudart)を取得して tools/llama/<tag>[-cuda]/ に展開する。"""
    if gpu is None:
        gpu = use_gpu()
    asset = release_asset(gpu)
    if asset is None:
        raise RuntimeError(
            f"No prebuilt llama.cpp binary for this platform ({sys.platform}/{platform.machine()}). "
            "Set LLAMA_SERVER=<path> or use the external LLM only."
        )
    name, inner = asset[0], asset[1]
    cudart = asset[2] if len(asset) > 2 else None
    base = f"https://github.com/ggml-org/llama.cpp/releases/download/{_tag()}/"
    dest_dir = _tools_dir(gpu)
    total = 0
    for archive in [name] + ([cudart] if cudart else []):
        data = _fetch(base + archive)
        total += len(data)
        _extract_archive(archive, data, dest_dir)
    exe = os.path.join(dest_dir, os.path.basename(inner))
    if not os.path.exists(exe):
        raise RuntimeError(f"{exe} not found after extraction (archive layout changed?)")
    if sys.platform != "win32":
        os.chmod(exe, os.stat(exe).st_mode | 0o111)
    logger.info("llama-server ready: %s (%s, %.0f MB)", exe, "CUDA" if gpu else "CPU", total / 1e6)
    return exe


def model_path() -> str:
    override = os.environ.get("LOCAL_LLM_GGUF", "").strip()
    return os.path.expanduser(override) if override else os.path.join(MODELS_DIR, MODEL_FILE)


def download_model() -> str:
    path = model_path()
    if os.path.exists(path):
        return path
    if os.environ.get("LOCAL_LLM_GGUF", "").strip():
        raise FileNotFoundError(f"LOCAL_LLM_GGUF not found: {path}")
    logger.info(f"ローカルLLMモデルをダウンロード中: {MODEL_REPO}/{MODEL_FILE} ({MODEL_SIZE_HINT})")
    os.makedirs(MODELS_DIR, exist_ok=True)
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir=MODELS_DIR)
    logger.info(f"ダウンロード完了: {downloaded}")
    return downloaded


def _health_ok(port: int, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _linux_die_with_parent():
    """子プロセス側で実行: 親が死んだら SIGTERM を受ける(prctl より前に親が死んでいた場合も検出)。"""
    import ctypes
    import signal

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG
    except OSError:
        return
    if os.getppid() != _PARENT_PID:
        os._exit(0)


def _popen_kwargs_quiet() -> dict:
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if sys.platform == "win32" else {}


def _popen_kwargs() -> dict:
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    if sys.platform.startswith("linux"):
        return {"preexec_fn": _linux_die_with_parent}
    return {}


def _windows_job_object(proc: subprocess.Popen):
    """Windows: 親の終了で子ごと kill される Job Object に入れる。失敗しても致命的ではない。"""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount",
                                                       "OtherOperationCount", "ReadTransferCount",
                                                       "WriteTransferCount", "OtherTransferCount")]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION), ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise OSError("CreateJobObjectW failed")
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError("SetInformationJobObject failed")
        if not kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle))):
            raise OSError("AssignProcessToJobObject failed")
        return job
    except Exception as e:  # noqa: BLE001
        logger.warning("Job Object not set; llama-server may outlive a force-killed parent: %s", e)
        return None


class LocalLLMService:
    """ビルド済み llama-server を子プロセスで管理するフォールバック LLM"""

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._job = None
        self._lock = threading.Lock()
        self._unavailable_reason = ""
        self._external_on_port = False
        self._gpu = False            # 今の子プロセスが CUDA 版か
        self._starts = 0
        self._ace_jobs = 0           # Easy Music が投げて未完了の ACE タスク数(この間は GPU で起こさない)
        self._ace_last = 0.0
        self._ace_idle = threading.Event()
        self._ace_idle.set()
        atexit.register(self.stop)

    @property
    def is_available(self) -> bool:
        """ローカルLLMが利用可能かどうか(バイナリは起動時に自動取得するので、プラットフォーム対応のみ見る)"""
        if self._unavailable_reason:
            return False
        if release_asset() is None and find_binary() is None:  # noqa: SIM102
            self._unavailable_reason = (
                f"no prebuilt llama-server for {sys.platform}/{platform.machine()} (set LLAMA_SERVER=<path>)"
            )
            logger.info("ローカルLLMは使用不可: %s", self._unavailable_reason)
            return False
        return True

    def _base_url(self) -> str:
        return f"http://127.0.0.1:{_port()}"

    def _running(self) -> bool:
        if self._external_on_port:
            return _health_ok(_port())
        return self._proc is not None and self._proc.poll() is None

    def _ensure_running_locked(self) -> str:
        port = _port()
        if self._running() and _health_ok(port):
            return self._base_url()
        if _health_ok(port):
            logger.info("ポート %d で既に llama-server(互換サーバ)が動いているのでそれを使います", port)
            self._external_on_port = True
            return self._base_url()
        self._external_on_port = False
        gpu = use_gpu()
        try:
            return self._launch(port, gpu)
        except RuntimeError as e:
            if not gpu:
                raise
            # CUDA 版が起動できない(空き VRAM が足りない、cudart が合わない等)ときは CPU 版で続ける
            logger.warning("ローカルLLM: CUDA 版の起動に失敗したので CPU 版で起動し直します: %s", e)
            self.stop()
            return self._launch(port, False)

    def _launch(self, port: int, gpu: bool) -> str:
        exe = find_binary(gpu) or download_binary(gpu)
        gguf = download_model()
        os.makedirs(LOGS_DIR, exist_ok=True)
        log = open(os.path.join(LOGS_DIR, "llama-server.log"), "ab")
        cmd = [exe, "-m", gguf, "--host", "127.0.0.1", "--port", str(port), "-c", str(N_CTX),
               "-t", str(max(4, (os.cpu_count() or 8) // 2)), "--reasoning", "off"]
        if gpu:
            mode = gpu_setting()
            if mode in ("auto", "on"):
                # 空き VRAM に入る層数だけ GPU へ(llama.cpp の --fit)。ACE が待機中でも共存できる
                cmd += ["--fit", "on", "--fit-target", str(FIT_TARGET_MIB)]
            else:
                cmd += ["-ngl", mode, "--fit", "off"]
            free = nvidia_free_mib()
            logger.info("ローカルLLM: CUDA 版、空き VRAM %s MiB、LOCAL_LLM_GPU=%s", free, mode)
        self._gpu = gpu
        self._starts += 1
        logger.info("ローカルLLMを起動: %s", " ".join(cmd))
        self._proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      cwd=os.path.dirname(exe), **_popen_kwargs())
        self._job = _windows_job_object(self._proc)
        t0 = time.time()
        while time.time() - t0 < HEALTH_TIMEOUT_S:
            if self._proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self._proc.returncode} (see logs/llama-server.log)")
            if _health_ok(port):
                logger.info("ローカルLLM 起動完了 (%.1fs, %s, port %d)", time.time() - t0, os.path.basename(gguf), port)
                return self._base_url()
            time.sleep(0.5)
        self.stop()
        raise RuntimeError(f"llama-server did not become healthy within {HEALTH_TIMEOUT_S}s")

    # ---- ACE の生成中は GPU で起こさない -------------------------------------------------
    # 生成タスクを作った直後に背景画像のプロンプト生成などが LLM を呼ぶため、止めた直後に GPU で起き直して
    # ACE の拡散(12GB で約 8.5GB)と VRAM を取り合い OOM になる。ACE が終わるまでここで待たせる。
    def ace_started(self) -> None:
        with self._lock:
            self._ace_jobs += 1
            self._ace_last = time.time()
            self._ace_idle.clear()

    def ace_finished(self) -> None:
        with self._lock:
            self._ace_jobs = max(0, self._ace_jobs - 1)
            if self._ace_jobs == 0:
                self._ace_idle.set()

    def _ace_busy(self) -> bool:
        if self._ace_jobs <= 0:
            return False
        if time.time() - self._ace_last > ACE_BUSY_MAX_S:  # 取りに来ないクライアント: 期限切れで解除
            with self._lock:
                self._ace_jobs = 0
                self._ace_idle.set()
            return False
        return True

    def _wait_for_ace_idle(self) -> None:
        """GPU モードで、子プロセスが居らず ACE が生成中なら、終わるまで起動を待つ(最長 ACE_BUSY_MAX_S)。"""
        if not self._ace_busy() or self._running() or not use_gpu():
            return
        setting = (os.environ.get("LOCAL_LLM_STOP_BEFORE_ACE", "").strip() or "auto").lower()
        if setting in ("0", "off", "false", "no"):
            return
        logger.info("ローカルLLM: ACE が生成中なので GPU での起動を待ちます(未完了 %d 件)", self._ace_jobs)
        t0 = time.time()
        while self._ace_busy():
            self._ace_idle.wait(timeout=1.0)
        logger.info("ローカルLLM: ACE 完了、起動します(%.0fs 待ち)", time.time() - t0)

    def release_before_ace(self) -> bool:
        """ACE に生成を投げる直前に呼ぶ。CUDA 版の子プロセスが居れば止めて VRAM を返す(次の呼び出しで起こし直す)。

        LOCAL_LLM_STOP_BEFORE_ACE=0 で無効、=1 で CPU 版でも止める(RAM 16GB で 4.4GB を返したいとき)。
        戻り値: 止めたかどうか。
        """
        setting = (os.environ.get("LOCAL_LLM_STOP_BEFORE_ACE", "").strip() or "auto").lower()
        if setting in ("0", "off", "false", "no"):
            return False
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return False
            if setting == "auto" and not self._gpu:
                return False
            t0 = time.time()
            self.stop()
            logger.info("ローカルLLMを停止(ACE 生成のため VRAM/RAM を返す、%.1fs)", time.time() - t0)
            return True

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        job, self._job = self._job, None
        if job is not None and sys.platform == "win32":
            import ctypes

            ctypes.windll.kernel32.CloseHandle(job)

    async def chat(
        self,
        user_message: str,
        system_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.8,
    ) -> str:
        """
        ローカル LLM でチャット応答を生成

        llm_service.chat() と同じインターフェース。起動待ちと CPU 推論はブロッキングなので
        スレッドプールで実行する。
        """
        if not self.is_available:
            raise RuntimeError(f"ローカルLLMが利用できません: {self._unavailable_reason}")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_sync, user_message, system_prompt, max_tokens, temperature)

    def _generate_sync(self, user_message: str, system_prompt: str, max_tokens: int, temperature: float) -> str:
        self._wait_for_ace_idle()
        with self._lock:
            try:
                base = self._ensure_running_locked()
            except Exception as e:
                self._unavailable_reason = f"failed to start llama-server: {e}"
                raise
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 64,
            "repeat_penalty": 1.1,
        }
        req = urllib.request.Request(
            f"{base}/v1/chat/completions", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return _THOUGHT_RE.sub("", content).strip()

    def get_status(self) -> Dict[str, Any]:
        """現在の状態を返す(`loaded` は互換のため残す = llama-server が起動済み)"""
        path = model_path()
        exe = find_binary()
        asset = release_asset()
        return {
            "available": self.is_available,
            "reason": self._unavailable_reason or None,
            "loaded": self._running(),
            "running": self._running(),
            "port": _port(),
            "model": MODEL_FILE,
            "model_downloaded": os.path.exists(path),
            "model_path": path,
            "binary": exe,
            "binary_source": f"llama.cpp {_tag()} {asset[0]}" if asset else None,
            "gpu": self._gpu if self._running() else use_gpu(),
            "gpu_setting": gpu_setting(),
            "free_vram_mib": nvidia_free_mib(),
            "starts": self._starts,
            "ace_jobs_in_flight": self._ace_jobs,
        }


# シングルトン
local_llm_service = LocalLLMService()
