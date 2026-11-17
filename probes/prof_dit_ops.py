"""Op-level CUDA-time profile of the XL-SFT DiT step with source attribution (eager, bf16, flash-attn)."""
import os, sys, json, time, linecache, collections, threading, torch
PROJECT_ROOT="/home/animede/ACE-Step-1.5-fast"; sys.path.insert(0, PROJECT_ROOT); os.chdir(PROJECT_ROOT)
from loguru import logger; logger.remove(); logger.add(sys.stderr, level="WARNING")
from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
import acestep.core.generation.handler.generate_music_execute as gme

class _InlineThread:  # run the generation in this thread so the profiler sees its ops
    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None): self._t, self._a, self._k = target, args, kwargs or {}
    def start(self): self._t(*self._a, **self._k)
    def join(self, timeout=None): pass
    def is_alive(self): return False
class _ThreadingShim:
    Thread = _InlineThread
    def __getattr__(self, name): return getattr(threading, name)
gme.threading = _ThreadingShim()

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
DUR = int(sys.argv[2]) if len(sys.argv) > 2 else 300
h = AceStepHandler()
msg, ok = h.initialize_service(project_root=PROJECT_ROOT, config_path="acestep-v15-xl-sft", device="cuda", use_flash_attention=True, compile_model=False, offload_to_cpu=True, offload_dit_to_cpu=False, quantization=None); assert ok, msg
h.vae = h.vae.to("cuda").to(h._get_vae_dtype("cuda")); h.resident_components = ("vae",)
ex = json.load(open("examples/text2music/example_05.json"))
params = GenerationParams(caption=ex["caption"], lyrics=ex["lyrics"], bpm=ex["bpm"], keyscale=ex["keyscale"], vocal_language=ex["language"], timesignature=ex["timesignature"], duration=DUR, inference_steps=STEPS, guidance_scale=7.0, seed=101, thinking=False)
cfg = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[101], audio_format="wav")
out = f"/tmp/claude-1000/-home-animede-story-studio/e9b66071-980f-4177-80ac-41c41f3bda23/scratchpad/probe/prof_dit_out"
r = generate_music(h, None, params, cfg, save_dir=out); assert r.success, r.error   # warmup
from torch.profiler import profile, ProfilerActivity
from torch._C._profiler import _ExperimentalConfig
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, record_shapes=True, experimental_config=_ExperimentalConfig(verbose=True)) as p:
    r = generate_music(h, None, params, cfg, save_dir=out)
assert r.success, r.error

_INTERESTING = ("modeling_acestep", "modeling_qwen3", "flash_attn", "cache_utils", "masking_utils", "integrations/flash")
def model_frame(stack):
    """Innermost frame in the model file, transformers' qwen3 module or flash-attn (else None)."""
    for fr in stack or []:
        if any(tag in fr for tag in _INTERESTING): return fr
    return None
def in_dit(stack):
    return any("modeling_acestep" in fr for fr in (stack or []))
def frame_key(fr):
    # "/path/modeling_acestep_v15_xl_base.py(1234): func"  ->  (file, line, func)
    if "(" not in fr or ")" not in fr: return fr, 0, fr
    path, rest = fr.split("(", 1); line, func = rest.split(")", 1)
    try: return path, int(line), func.strip(": ")
    except ValueError: return fr, 0, fr

# aggregate raw events (key_averages rows carry no stacks): (op, model source frame) -> self CUDA time
agg = collections.defaultdict(lambda: [0.0, 0])
n_ev = n_stack = 0
samples = collections.defaultdict(list)
for e in p.events():
    if not (e.name.startswith("aten::") or "flash" in e.name.lower()): continue
    t = getattr(e, "self_device_time_total", None)
    if t is None: t = getattr(e, "self_cuda_time_total", 0.0)
    if t <= 0: continue
    n_ev += 1
    st = getattr(e, "stack", None)
    if not in_dit(st): continue
    fr = model_frame(st)
    if fr is None: continue
    n_stack += 1
    a = agg[(e.name, fr)]; a[0] += t; a[1] += 1

dit_rows = [(k, v[0], v[1], fr) for (k, fr), v in agg.items()]
print(f"[dit] aten events with CUDA time={n_ev}, attributed to the model file={n_stack}")
total = sum(r[1] for r in dit_rows)
if total == 0:
    allrows = list(p.key_averages())
    with_stack = [e for e in allrows if e.stack]
    print(f"[dbg] rows={len(allrows)} with_stack={len(with_stack)} with_device_time={sum(1 for e in allrows if getattr(e,'self_device_time_total',0)>0)}")
    lin = [e for e in allrows if e.key in ("aten::linear","aten::mm","aten::addmm")]
    print(f"[dbg] linear rows={len(lin)}; sample stacks:")
    for e in lin[:3]: print("[dbg]   ", e.key, getattr(e,'self_device_time_total',0), getattr(e,'device_time_total',0), (e.stack or [])[:4])
    import collections as _c
    files=_c.Counter()
    for e in with_stack:
        for fr in e.stack:
            files[fr.split("(")[0].split("/")[-1]] += 1
    print("[dbg] stack files:", files.most_common(12))
    print("[dbg] thread shim used:", gme.threading.__class__.__name__)
    sys.exit(0)
print(f"[dit] steps={STEPS} dur={DUR}s  DiT CUDA self time total={total/1e6:.2f}s  per step={total/1e6/STEPS:.2f}s  kernels/step={sum(r[2] for r in dit_rows)/STEPS:.0f}")
gemm = sum(t for k,t,_,_ in dit_rows if k in ("aten::linear","aten::mm","aten::addmm","aten::matmul","aten::bmm"))
attn = sum(t for k,t,_,_ in dit_rows if "flash" in k.lower() or "scaled_dot" in k or "attention" in k.lower())
print(f"[dit] gemm {gemm/total*100:.1f}%  attention {attn/total*100:.1f}%  other(elementwise etc.) {(total-gemm-attn)/total*100:.1f}%")
by_op = collections.Counter(); by_line = collections.Counter(); cnt_line = collections.Counter()
for k, t, c, fr in dit_rows:
    by_op[k] += t
    path, line, func = frame_key(fr); by_line[(os.path.basename(path), line, func)] += t; cnt_line[(os.path.basename(path), line, func)] += c
print("[dit] --- by op (self CUDA time, share) ---")
for k, t in by_op.most_common(16): print(f"[dit] {t/1e6/STEPS:7.3f}s/step {t/total*100:5.1f}%  {k}")
print("[dit] --- by source line (all ops under that line) ---")
paths = {os.path.basename(r[3].split('(')[0]): r[3].split('(')[0] for r in dit_rows if '(' in r[3]}
for (f, line, func), t in by_line.most_common(20):
    src = linecache.getline(paths.get(f, f), line).strip() if line else ""
    print(f"[dit] {t/1e6/STEPS:7.3f}s/step {t/total*100:5.1f}% n={cnt_line[(f,line,func)]//STEPS:4d}/step  {f.replace('modeling_','')[:22]}:{func}:{line}  {src[:80]}")
