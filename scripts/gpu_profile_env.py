"""Pick ACESTEP_GPU_PROFILE (and the 8 GB settings) from the card's VRAM. Called by run_api_server_fast.bat.

Prints one `NAME=VALUE` line per variable to stdout (the .bat turns them into `set`), and human
messages to stderr. Only variables that are not already set are printed, so anything the user set
by hand wins. Kept in Python because cmd's ( ... ) blocks, escaping and code pages are fragile.

  12gb when the total VRAM is below 14000 MiB, else 16gb (large is never picked automatically).
  Below 10000 MiB the verified 8 GB settings are added: LM/VAE offloaded, decode chunk 128, int8.
"""
import os
import shutil
import subprocess
import sys


def vram_mib():
    """Total VRAM of the first visible GPU in MiB, or None. nvidia-smi (PATH, System32, old NVSMI dir), then torch."""
    exe = shutil.which("nvidia-smi")
    if exe is None and sys.platform == "win32":
        for cand in (os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe"),
                     os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")):
            if os.path.exists(cand):
                exe = cand
                break
    if exe:
        cmd = [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        if vis:
            cmd[1:1] = ["-i", vis]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if out.returncode == 0 and out.stdout.strip():
                return int(float(out.stdout.strip().splitlines()[0]))
        except Exception:  # noqa: BLE001
            pass
    try:
        import torch
        return int(torch.cuda.get_device_properties(0).total_memory // 1048576)
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    say = lambda *a: print(*a, file=sys.stderr, flush=True)  # noqa: E731
    env = os.environ
    mib = vram_mib()
    out = {}
    profile = env.get("ACESTEP_GPU_PROFILE", "").strip()
    if mib is None:
        if not profile:
            say("=" * 76)
            say(" WARNING: could not read the GPU VRAM (nvidia-smi and torch both failed). Assuming 16gb.")
            say(" On an 8 GB / 12 GB card set ACESTEP_GPU_PROFILE=12gb before starting (8 GB: see README).")
            say(" 警告: GPU の VRAM を読めませんでした。16gb を仮定します。8GB / 12GB のカードなら")
            say(" 起動前に ACESTEP_GPU_PROFILE=12gb を指定してください(8GB は README の 5 つの設定)。")
            say("=" * 76)
        return
    if not profile:
        profile = "12gb" if mib < 14000 else "16gb"
        out["ACESTEP_GPU_PROFILE"] = profile
        say(f"[run_api_server_fast] GPU VRAM {mib} MiB -> ACESTEP_GPU_PROFILE={profile} "
            "(set ACESTEP_GPU_PROFILE to override; large for >= 40 GB)")
        if mib < 10000:
            # 8 GB class: 12gb alone does not fit (2B turbo 4.8 + LM 1.3 + VAE 0.3 + TE 1.2 GB)
            for k, v in (("ACESTEP_LM_OFFLOAD_TO_CPU", "1"), ("ACESTEP_RESIDENT_COMPONENTS", "none"),
                         ("ACESTEP_VAE_DECODE_CHUNK_SIZE", "128"), ("ACESTEP_QUANTIZATION", "int8_weight_only")):
                if not env.get(k):
                    out[k] = v
            say("[run_api_server_fast] 8 GB class: LM/VAE offloaded, decode chunk 128, int8_weight_only "
                "(VRAM 8GB 向けの設定を自動で入れました)")
    elif mib < 14000 and profile.lower() != "12gb":
        say("=" * 76)
        say(f" WARNING: ACESTEP_GPU_PROFILE={profile} is set, but this GPU has only {mib} MiB.")
        say(" XL models do not fit: expect shared-GPU-memory spill and very slow generation.")
        say(" Clear it (set ACESTEP_GPU_PROFILE= / unset) and start again so 12gb / the 8 GB settings are picked.")
        say(f" 警告: ACESTEP_GPU_PROFILE={profile} が指定されていますが、この GPU は {mib} MiB です。")
        say(" 指定を消してから起動し直してください(自動で 12gb / 8GB 向けの設定が入ります)。")
        say("=" * 76)
    for k, v in out.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
