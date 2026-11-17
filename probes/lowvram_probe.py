#!/usr/bin/env python3
"""ACE-Step 1.5 low-VRAM probe: load one DiT with a given quantization, generate, record VRAM/time."""
import argparse, json, os, sys, time, subprocess
PROJECT_ROOT = "/home/animede/ACE-Step-1.5-fast"
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

p = argparse.ArgumentParser()
p.add_argument("--config-path", default="acestep-v15-xl-sft")
p.add_argument("--quantization", default=None)   # int8_weight_only | w8a8_dynamic | fp8_weight_only | None
p.add_argument("--compile", action="store_true")
p.add_argument("--compile-target", default="model", choices=["model","decoder","layers","forward"])
p.add_argument("--cpu-quant", action="store_true", help="load+quantize on CPU, then move to GPU (low load peak)")
p.add_argument("--duration", type=int, default=60)
p.add_argument("--steps", type=int, default=None)
p.add_argument("--guidance", type=float, default=7.0)
p.add_argument("--seed", type=int, default=12345)
p.add_argument("--runs", type=int, default=1)
p.add_argument("--out", required=True)
p.add_argument("--tag", default="")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

import torch
from loguru import logger
logger.remove(); logger.add(sys.stderr, level="INFO")
from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

def smi_mb():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).decode()
        for line in out.splitlines():
            pid, mem = [x.strip() for x in line.split(",")]
            if int(pid) == os.getpid():
                return int(mem)
    except Exception:
        pass
    return -1

h = AceStepHandler()
t0 = time.time()
if a.cpu_quant:
    # Load with DiT parked on CPU so quantize_ runs on CPU, then move the (small) quantized model to GPU.
    msg, ok = h.initialize_service(project_root=PROJECT_ROOT, config_path=a.config_path, device="cuda",
                                   use_flash_attention=False, compile_model=False,
                                   offload_to_cpu=True, offload_dit_to_cpu=True, quantization=a.quantization)
    assert ok, msg
    h.model = h.model.to("cuda")
    h.silence_latent = h.silence_latent.to("cuda")
    h.offload_to_cpu = False; h.offload_dit_to_cpu = False
    h.vae = h.vae.to("cuda").to(h._get_vae_dtype("cuda")); h.text_encoder = h.text_encoder.to("cuda").to(h.dtype)
    if a.compile:
        if a.compile_target == "model":
            h.model = torch.compile(h.model)
        elif a.compile_target == "decoder":
            h.model.decoder = torch.compile(h.model.decoder)
        elif a.compile_target == "forward":
            h.model.decoder.forward = torch.compile(h.model.decoder.forward)
        else:
            for i, layer in enumerate(h.model.decoder.layers):
                h.model.decoder.layers[i] = torch.compile(layer)
else:
    msg, ok = h.initialize_service(project_root=PROJECT_ROOT, config_path=a.config_path, device="cuda",
                                   use_flash_attention=False, compile_model=a.compile,
                                   offload_to_cpu=False, offload_dit_to_cpu=False, quantization=a.quantization)
    assert ok, msg
torch.cuda.synchronize()
load_s = time.time() - t0
load_peak = torch.cuda.max_memory_allocated() / 2**30
resident = torch.cuda.memory_allocated() / 2**30
print(f"[probe] loaded in {load_s:.1f}s resident={resident:.2f}GB load_peak={load_peak:.2f}GB smi={smi_mb()}MB", flush=True)

ex = json.load(open("examples/text2music/example_05.json"))
is_turbo = "turbo" in a.config_path
steps = a.steps or (8 if is_turbo else 50)
params = GenerationParams(caption=ex["caption"], lyrics=ex["lyrics"], bpm=ex["bpm"], keyscale=ex["keyscale"],
                          vocal_language=ex["language"], timesignature=ex["timesignature"], duration=a.duration,
                          inference_steps=steps, guidance_scale=(1.0 if is_turbo else a.guidance), seed=a.seed, thinking=False)
cfg = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[a.seed], audio_format="wav")

rows = []
for i in range(a.runs):
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    res = generate_music(h, None, params, cfg, save_dir=a.out)
    torch.cuda.synchronize()
    gen_s = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.memory_reserved() / 2**30
    ok = getattr(res, "success", True)
    paths = getattr(res, "audios", None) or getattr(res, "audio_paths", None)
    print(f"[probe] run{i} ok={ok} gen={gen_s:.1f}s peak={peak:.2f}GB reserved={reserved:.2f}GB smi={smi_mb()}MB steps={steps} err={getattr(res,'error',None)}", flush=True)
    rows.append(dict(run=i, gen_s=gen_s, peak_gb=peak, reserved_gb=reserved, smi_mb=smi_mb(), ok=ok, audio=str(paths)))
json.dump(dict(tag=a.tag, config=a.config_path, quant=a.quantization, compile=a.compile, compile_target=a.compile_target, cpu_quant=a.cpu_quant,
               duration=a.duration, steps=steps, seed=a.seed, load_s=load_s, resident_gb=resident, load_peak_gb=load_peak,
               runs=rows), open(os.path.join(a.out, "probe.json"), "w"), indent=1)
