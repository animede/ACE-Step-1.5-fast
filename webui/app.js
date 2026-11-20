/* ACE-Step-1.5-fast 評価 UI: plain JS client for the REST API (release_task / query_result / ...). */
(() => {
  const $ = (id) => document.getElementById(id);
  const base = () => $("base").value.replace(/\/+$/, "");
  const status = (el, text, cls) => { el.textContent = text; el.className = "status" + (cls ? " " + cls : ""); };
  const history = JSON.parse(localStorage.getItem("acestep_fast_history") || "[]");

  $("base").value = localStorage.getItem("acestep_fast_base") || location.origin;
  $("base").addEventListener("change", () => localStorage.setItem("acestep_fast_base", base()));

  async function api(path, body, isForm) {
    const opts = body === undefined ? { method: "GET" }
      : { method: "POST", body: isForm ? body : JSON.stringify(body), headers: isForm ? {} : { "Content-Type": "application/json" } };
    const res = await fetch(base() + path, opts);
    const json = await res.json().catch(() => ({}));
    if (!res.ok || (json.code && json.code !== 200)) throw new Error(json.error || json.detail || `${res.status} ${path}`);
    return json.data !== undefined ? json.data : json;
  }

  // ---- server panel
  async function refreshHealth() {
    try {
      const h = await api("/health");
      status($("server-status"), `${h.status} / DiT: ${h.loaded_model} / LM: ${h.loaded_lm_model || "-"} / 初期化: ${h.models_initialized}`, "ok");
      await fillModels(h.loaded_model);
    } catch (e) { status($("server-status"), "接続できません: " + e.message, "err"); }
  }
  async function fillModels(defaultName) {
    const sel = $("model"); const current = sel.value; sel.innerHTML = "";
    let names = [];
    try {  // two response shapes exist: {models:[{name}]} and an OpenAI-style list [{id:"acestep/<name>"}]
      const m = await api("/v1/models");
      const list = Array.isArray(m) ? m : (m.models || []);
      names = list.map((x) => (x.id ? String(x.id).split("/").pop() : x.name)).filter(Boolean);
    } catch (e) { /* older servers */ }
    for (const n of ["acestep-v15-xl-sft", "acestep-v15-xl-turbo", "acestep-v15-turbo", "acestep-v15-base", defaultName]) if (n && !names.includes(n)) names.push(n);
    for (const n of names) { const o = document.createElement("option"); o.value = n; o.textContent = n + (n === defaultName ? "(既定)" : ""); sel.appendChild(o); }
    sel.value = names.includes(current) ? current : (defaultName || names[0]);
  }
  $("btn-health").onclick = refreshHealth;
  $("btn-stats").onclick = async () => {
    try { const s = await api("/v1/stats"); status($("server-status"), `jobs: 待ち ${s.jobs.queued} / 実行中 ${s.jobs.running} / 成功 ${s.jobs.succeeded} / 失敗 ${s.jobs.failed}, 平均 ${Number(s.avg_job_seconds).toFixed(1)}s`, "ok"); }
    catch (e) { status($("server-status"), e.message, "err"); }
  };

  // ---- model management (slots, residency, /v1/init)
  async function refreshResidency() {
    try {
      const r = await api("/v1/residency");
      status($("residency-mode"), r.mode === "exclusive" ? `排他常駐(${r.store})・GPU: ${r.active || "-"}` : "共有常駐(全モデルが GPU 上)", "ok");
      const tb = $("slots").querySelector("tbody"); tb.innerHTML = "";
      for (const s of r.slots) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${s.slot}</td><td>${s.model}</td><td class="${s.on_gpu ? "gpu" : ""}">${s.on_gpu ? "● GPU" : (s.loaded ? "退避中" : "未読込")}</td><td></td>`;
        if (r.mode === "exclusive" && s.loaded && !s.on_gpu) {
          const b = document.createElement("button"); b.className = "small"; b.textContent = "GPU に載せる";
          b.onclick = async () => { b.disabled = true; status($("models-status"), `${s.model} を GPU へ…`);
            try { await api("/v1/residency/activate", { slot: s.slot }); status($("models-status"), "切替完了", "ok"); }
            catch (e) { status($("models-status"), e.message, "err"); } await refreshResidency(); };
          tr.lastElementChild.appendChild(b);
        }
        tb.appendChild(tr);
      }
    } catch (e) { status($("residency-mode"), "取得できません: " + e.message, "err"); }
    try {
      const inv = await api("/v1/model_inventory"); const sel = $("inventory"); sel.innerHTML = "";
      for (const m of (inv.models || [])) { const o = document.createElement("option"); o.value = m.name; o.textContent = m.name + (m.is_loaded ? "(読込済)" : "") + (m.vram_ok === false ? "(このGPUには載らない: 約" + m.min_vram_gb + "GB必要)" : ""); o.disabled = m.vram_ok === false; sel.appendChild(o); }
    } catch (e) { /* inventory endpoint missing */ }
  }
  $("btn-residency").onclick = refreshResidency;
  $("btn-init").onclick = async () => {
    const model = $("inventory_custom").value.trim() || $("inventory").value; const slot = Number($("init_slot").value);
    if (!model) return;
    $("btn-init").disabled = true; status($("models-status"), `slot ${slot} に ${model} を読み込み中…(数十秒)`);
    try { const r = await api("/v1/init", { model, slot }); status($("models-status"), `完了: slot ${r.slot} = ${r.loaded_model}`, "ok"); }
    catch (e) { status($("models-status"), e.message, "err"); }
    $("btn-init").disabled = false; await refreshResidency(); await refreshHealth();
  };

  // ---- form <-> request
  const num = (id) => { const v = $(id).value.trim(); return v === "" ? null : Number(v); };
  const modelName = () => $("model_custom").value.trim() || $("model").value;
  function collect() {
    const b = {
      task_type: $("task_type").value, model: modelName(),
      prompt: $("prompt").value, lyrics: $("instrumental").checked ? "[Instrumental]" : $("lyrics").value,
      vocal_language: $("vocal_language").value, bpm: num("bpm"), key_scale: $("key_scale").value.trim(),
      time_signature: $("time_signature").value, audio_duration: num("audio_duration"),
      inference_steps: num("inference_steps"), guidance_scale: num("guidance_scale"), shift: num("shift"),
      infer_method: $("infer_method").value, use_random_seed: $("use_random_seed").checked, seed: $("seed").value.trim(),
      batch_size: num("batch_size"), audio_format: $("audio_format").value,
      thinking: $("thinking").checked, use_cot_caption: $("use_cot_caption").checked, use_cot_language: $("use_cot_language").checked,
      use_format: $("use_format").checked, lm_temperature: num("lm_temperature"), lm_cfg_scale: num("lm_cfg_scale"),
      lm_top_k: num("lm_top_k"), lm_top_p: num("lm_top_p"), lm_repetition_penalty: num("lm_repetition_penalty"),
      repainting_start: num("repainting_start"), repainting_end: num("repainting_end"), audio_cover_strength: num("audio_cover_strength"),
      instruction: $("instruction").value.trim(), timesteps: $("timesteps").value.trim() || null, use_adg: $("use_adg").checked,
      cfg_interval_start: num("cfg_interval_start"), cfg_interval_end: num("cfg_interval_end"),
      audio_code_string: $("audio_code_string").value.trim(),
    };
    for (const k of Object.keys(b)) if (b[k] === null || b[k] === "") delete b[k];
    return b;
  }
  function fillForm(d) {
    if (!d) return;
    if (d.caption !== undefined) $("prompt").value = d.caption;
    if (d.lyrics !== undefined) { $("lyrics").value = d.lyrics; $("instrumental").checked = false; }
    if (d.bpm) $("bpm").value = d.bpm;
    if (d.key_scale !== undefined) $("key_scale").value = d.key_scale || "";
    if (d.time_signature) $("time_signature").value = String(d.time_signature);
    if (d.duration) $("audio_duration").value = d.duration;
    if (d.vocal_language) $("vocal_language").value = d.vocal_language;
  }

  // ---- jobs
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  async function submit(body) {
    const ref = $("reference_audio").files[0], src = $("src_audio").files[0];
    if (ref || src) {
      const fd = new FormData();
      for (const [k, v] of Object.entries(body)) fd.append(k, typeof v === "boolean" ? String(v) : v);
      if (ref) fd.append("reference_audio", ref);
      if (src) fd.append("src_audio", src);
      return (await api("/release_task", fd, true)).task_id;
    }
    return (await api("/release_task", body)).task_id;
  }
  async function poll(taskId, onProgress) {
    for (;;) {
      const d = (await api("/query_result", { task_id_list: [taskId] }))[0];
      if (d.status === 1 || d.status === 2) return d;
      onProgress(d); await sleep(1000);
    }
  }
  async function runJob(body, label) {
    const btn = $("btn-generate"); btn.disabled = true; const t0 = performance.now();
    try {
      status($("job-status"), "投入中…");
      const taskId = await submit(body);
      const d = await poll(taskId, (p) => status($("job-status"), `${p.stage || "実行中"} ${Math.round((p.progress || 0) * 100)}%  ${((performance.now() - t0) / 1000).toFixed(0)}s`));
      const elapsed = (performance.now() - t0) / 1000;
      let items = d.result; if (typeof items === "string") { try { items = JSON.parse(items); } catch (e) { items = []; } }
      status($("job-status"), d.status === 1 ? `完了 ${elapsed.toFixed(1)}s` : `失敗 ${elapsed.toFixed(1)}s`, d.status === 1 ? "ok" : "err");
      addResult({ label, body, status: d.status, elapsed, items: items || [], error: d.error || null, at: new Date().toLocaleTimeString() });
    } catch (e) { status($("job-status"), "エラー: " + e.message, "err"); }
    finally { btn.disabled = false; }
  }
  $("btn-generate").onclick = () => runJob(collect(), `${modelName()} ${$("audio_duration").value}s ${$("inference_steps").value}step`);
  $("btn-sample-generate").onclick = () => {
    const b = collect(); b.sample_mode = true; b.sample_query = $("sample_query").value; delete b.prompt; delete b.lyrics;
    runJob(b, `簡易モード: ${$("sample_query").value.slice(0, 30)}`);
  };
  $("btn-random").onclick = async () => {
    try { status($("job-status"), "ランダム作成中…"); fillForm(await api("/create_random_sample", { sample_type: "custom_mode" })); status($("job-status"), "フォームに反映", "ok"); }
    catch (e) { status($("job-status"), e.message, "err"); }
  };
  $("btn-format").onclick = async () => {
    try {
      status($("job-status"), "LM で整形中…");
      const p = { duration: num("audio_duration"), bpm: num("bpm"), key: $("key_scale").value, time_signature: $("time_signature").value, language: $("vocal_language").value };
      fillForm(await api("/format_input", { prompt: $("prompt").value, lyrics: $("lyrics").value, temperature: num("lm_temperature") || 0.85, param_obj: JSON.stringify(p) }));
      status($("job-status"), "フォームに反映", "ok");
    } catch (e) { status($("job-status"), e.message, "err"); }
  };

  // ---- results / history
  function addResult(r) { history.unshift(r); history.splice(30); localStorage.setItem("acestep_fast_history", JSON.stringify(history)); render(); }
  function render() {
    const root = $("results"); root.innerHTML = "";
    history.forEach((r, i) => {
      const card = document.createElement("div"); card.className = "card" + (r.status === 1 ? "" : " failed");
      const head = document.createElement("div"); head.className = "head";
      head.innerHTML = `<span>${r.at}</span><b>${r.label}</b><span>${r.elapsed.toFixed(1)}s</span>`;
      card.appendChild(head);
      (r.items || []).forEach((it) => {
        const h = document.createElement("div"); h.className = "head";
        h.innerHTML = `<span>DiT: <b>${it.dit_model || "-"}</b></span><span>LM: ${it.lm_model || "-"}</span><span>seed: ${it.seed_value || "-"}</span>`;
        card.appendChild(h);
        if (it.file) { const a = document.createElement("audio"); a.controls = true; a.preload = "none"; a.src = base() + it.file; card.appendChild(a); }
        if (it.metas) { const pre = document.createElement("pre"); pre.textContent = JSON.stringify(it.metas); card.appendChild(pre); }
      });
      if (r.error) { const pre = document.createElement("pre"); pre.textContent = String(r.error).slice(0, 600); card.appendChild(pre); }
      const tools = document.createElement("div"); tools.className = "tools";
      const again = document.createElement("button"); again.textContent = "同じ設定をフォームへ"; again.className = "small";
      again.onclick = () => { const b = r.body; fillForm({ caption: b.prompt, lyrics: b.lyrics, bpm: b.bpm, key_scale: b.key_scale, time_signature: b.time_signature, duration: b.audio_duration, vocal_language: b.vocal_language });
        if (r.items && r.items[0] && r.items[0].seed_value) { $("seed").value = String(r.items[0].seed_value).split(",")[0]; $("use_random_seed").checked = false; } };
      tools.appendChild(again); card.appendChild(tools); root.appendChild(card);
    });
  }
  $("btn-clear").onclick = () => { history.length = 0; localStorage.removeItem("acestep_fast_history"); render(); };

  $("task_type").addEventListener("change", () => { $("audio-tasks").open = $("task_type").value !== "text2music"; });
  render(); refreshHealth(); refreshResidency();
})();
