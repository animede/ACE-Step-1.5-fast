#!/usr/bin/env python3
"""Load two XL DiTs through the real initialize_service path (as the API server does) and generate with each."""
import argparse, json, os, sys, time, subprocess
PROJECT_ROOT = "/home/animede/ACE-Step-1.5-fast"; sys.path.insert(0, PROJECT_ROOT); os.chdir(PROJECT_ROOT)
p = argparse.ArgumentParser()
p.add_argument("--quantization", default="int8_weight_only"); p.add_argument("--compile", action="store_true")
p.add_argument("--offload", action="store_true", help="offload_to_cpu for VAE/TE (server lowvram setting)")
p.add_argument("--duration", type=int, default=60); p.add_argument("--seed", type=int, default=12345)
p.add_argument("--out", required=True)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
import torch
from loguru import logger; logger.remove(); logger.add(sys.stderr, level="INFO")
from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
def smi():
    out = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"]).decode()
    return next((int(m) for l in out.splitlines() for pid,m in [map(str.strip,l.split(","))] if int(pid)==os.getpid()), -1)
hs = {}
for name in ["acestep-v15-xl-turbo", "acestep-v15-xl-sft"]:
    h = AceStepHandler(); t0 = time.time()
    msg, ok = h.initialize_service(project_root=PROJECT_ROOT, config_path=name, device="cuda", use_flash_attention=False,
                                   compile_model=a.compile, offload_to_cpu=a.offload, offload_dit_to_cpu=False, quantization=a.quantization)
    assert ok, msg; torch.cuda.synchronize(); hs[name] = h
    print(f"[dual] loaded {name} in {time.time()-t0:.1f}s resident={torch.cuda.memory_allocated()/2**30:.2f}GB load_peak={torch.cuda.max_memory_allocated()/2**30:.2f}GB smi={smi()}MB", flush=True)
ex = json.load(open("examples/text2music/example_05.json")); rows=[]
for name, h in hs.items():
    turbo = "turbo" in name; steps = 8 if turbo else 32
    params = GenerationParams(caption=ex["caption"], lyrics=ex["lyrics"], bpm=ex["bpm"], keyscale=ex["keyscale"], vocal_language=ex["language"],
                              timesignature=ex["timesignature"], duration=a.duration, inference_steps=steps, guidance_scale=1.0 if turbo else 7.0, seed=a.seed, thinking=False)
    cfg = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[a.seed], audio_format="wav")
    for i in range(2):
        torch.cuda.reset_peak_memory_stats(); t1=time.time()
        res = generate_music(h, None, params, cfg, save_dir=os.path.join(a.out, name)); torch.cuda.synchronize()
        print(f"[dual] {name} run{i} ok={res.success} gen={time.time()-t1:.1f}s peak={torch.cuda.max_memory_allocated()/2**30:.2f}GB reserved={torch.cuda.memory_reserved()/2**30:.2f}GB smi={smi()}MB err={res.error}", flush=True)
        rows.append(dict(model=name, run=i, gen_s=time.time()-t1, peak_gb=torch.cuda.max_memory_allocated()/2**30, smi_mb=smi(), ok=res.success))
json.dump(dict(quant=a.quantization, compile=a.compile, offload=a.offload, duration=a.duration, runs=rows), open(os.path.join(a.out,"dual.json"),"w"), indent=1)
