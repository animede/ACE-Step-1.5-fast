import json, sys, time, subprocess, urllib.request, threading
BASE="http://127.0.0.1:8011"
def post(path, body):
    r=urllib.request.Request(BASE+path, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}); return json.loads(urllib.request.urlopen(r, timeout=60).read())
ex=json.load(open("examples/text2music/example_05.json"))
peak={"mb":0}; stop=False
def sampler():
    while not stop:
        try:
            mb=int(subprocess.check_output(["nvidia-smi","-i","1","--query-gpu=memory.used","--format=csv,noheader,nounits"]).decode().strip()); peak["mb"]=max(peak["mb"],mb)
        except Exception: pass
        time.sleep(0.5)
threading.Thread(target=sampler,daemon=True).start()
for label, body in [("turbo+LM", dict(prompt=ex["caption"], lyrics=ex["lyrics"], audio_duration=60, inference_steps=8, thinking=True, seed=12345)),
                    ("sft", dict(prompt=ex["caption"], lyrics=ex["lyrics"], audio_duration=60, inference_steps=32, guidance_scale=7.0, thinking=False, seed=12345, model="acestep-v15-xl-sft"))]:
    peak["mb"]=0; t0=time.time(); r=post("/release_task", body); tid=r["data"]["task_id"]
    while True:
        q=post("/query_result", {"task_id_list":[tid]}); st=q["data"][0]["status"]
        if st in (1,2): break
        time.sleep(2)
    d=q["data"][0]; res=d.get("result"); 
    print(f"[api] {label}: status={st} wall={time.time()-t0:.1f}s peak_smi={peak['mb']}MB result={str(res)[:160]}", flush=True)
stop=True
