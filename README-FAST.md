# ACE-Step-1.5-fast

ACE-Step 1.5 のフォーク。**16GB 級 GPU(RTX A4000)で XL(4B)モデルを動かすための高速化版 + API サーバ改良版**。
上流(ace-step/ACE-Step-1.5、git remote `upstream`)の 2026-05-04 時点 `6c1b2ef` を土台にしている。

## 上流との違い(要点)

- API サーバが `ACESTEP_QUANTIZATION` を受け取る(上流は量子化を一切渡さず常に bf16)。
- ローダ: CPU で量子化してから GPU へ(読み込みピーク = 常駐)、compile は `decoder.forward` に当てる。
- `ACESTEP_DIT_RESIDENCY=exclusive`: GPU に DiT を 1 本だけ置き、他はホスト側に退避して約 2 秒で入れ替える。退避先は `ACESTEP_DIT_RESIDENCY_STORE=mmap`(bf16 の safetensors をページキャッシュ経由で mmap。サーバの常駐メモリ約 4GB で 32GB のホストに収まる)または `pinned`(固定 RAM に 3 本で約 23GB)。
- `ACESTEP_FUSED_RMSNORM=exact`: DiT の RMSNorm を融合カーネルに(1 step −14%)。
- `ACESTEP_RESIDENT_COMPONENTS` / `ACESTEP_VAE_DECODE_CHUNK_SIZE` / `ACESTEP_MP3_COMPRESSION_LEVEL`: 拡散以外の削減。
- `ACESTEP_DIT_FUSIONS`(attention/MLP/adaLN の eager 融合)、`ACESTEP_QUANTIZATION=int8_plain`: 実装済みだが効果が小さく既定 OFF。

実測(A4000、XL-SFT、300 秒 1 曲、32 step、mp3・CoT あり): 84 秒 → 62〜65 秒。記録と注意点は
`docs/ja/低VRAM_XL_プローブ_20261116.md`。

## 評価用 Web UI(Gradio 不要)

サーバ起動後に `http://<host>:<port>/ui/` を開く。`webui/`(HTML + JS)を API サーバが配信し、`/release_task` 等の REST API だけで動く。
モデル選択(xl-sft / xl-turbo / 2B turbo)、通常生成・cover・repaint・簡易モード、LM の設定、結果の再生と履歴(seed の再利用)ができる。
「サーバのモデル管理」で、どの DiT が GPU 上かの確認、事前の切替(`POST /v1/residency/activate`)、slot への別モデルの読み込み(`POST /v1/init`、排他常駐では CPU に読んで退避)ができる。

## 起動

```bash
./run_api_server_fast.sh         # 16GB 向けの標準: XL-SFT 常駐 + XL-turbo / 2B turbo を要求時に入れ替え
./run_api_server_xl_swap.sh      # XL-SFT 常駐(bf16)+ XL-turbo を要求時に入れ替え(2 本版)
./run_api_server_multimodel.sh   # 複数 DiT を同時常駐(VRAM に余裕がある GPU 向け)
```

VRAM に余裕がある GPU では multimodel に `ACESTEP_FUSED_RMSNORM=exact ACESTEP_OFFLOAD_TO_CPU=0 ACESTEP_LM_OFFLOAD_TO_CPU=0 ACESTEP_VAE_DECODE_CHUNK_SIZE=512` を付ける。
