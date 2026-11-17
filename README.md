# ACE-Step-1.5-fast

[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)(オープンソースの音楽生成モデル)のフォーク。
**16GB 級 GPU(RTX A4000)で XL(4B)モデルを動かすための高速化版 + API サーバ改良版**で、Linux / CUDA に絞っています。
オリジナル(ace-step/ACE-Step-1.5、git remote `upstream`)の 2026-05-04 時点 `6c1b2ef` を土台にし、オリジナルの取り込みは予定していません。

- 実測(A4000、XL-SFT、300 秒 1 曲、32 step、mp3・CoT あり、音は bf16 のまま): **84 秒 → 62〜65 秒**
- サーバの常駐ホストメモリ 約 4GB(32GB の専用機で動く)
- 評価用の Web UI(Gradio 不要)を API サーバが `/ui/` で配信

## オリジナルとの違い

| 項目 | 内容 |
|---|---|
| 排他常駐 | `ACESTEP_DIT_RESIDENCY=exclusive`: GPU に DiT を 1 本だけ置き、他はホスト側に退避して約 2 秒で入れ替える。退避先は `ACESTEP_DIT_RESIDENCY_STORE=mmap`(bf16 の safetensors をページキャッシュ経由で mmap、常駐 RSS 約 4GB)または `pinned`(固定 RAM に 3 本で約 23GB) |
| RMSNorm 融合 | `ACESTEP_FUSED_RMSNORM=exact`: DiT の RMSNorm を融合カーネルに(1 step −14%、bf16 の丸め差のみ) |
| 拡散以外の削減 | `ACESTEP_RESIDENT_COMPONENTS`(VAE / テキストエンコーダを退避しない)、`ACESTEP_VAE_DECODE_CHUNK_SIZE`、`ACESTEP_MP3_COMPRESSION_LEVEL`(任意) |
| API サーバの修正 | `ACESTEP_QUANTIZATION` を受け取る(オリジナルは量子化を渡さず常に bf16)。ローダは CPU で量子化してから GPU へ、compile は `decoder.forward` に当てる |
| 同時実行 | LM(CoT)の呼び出しを直列化(オリジナルは同時実行で CUDA が壊れる)。別モデルへの切替は実行中のジョブを待ってから |
| モデル管理 API | `GET /v1/residency`、`POST /v1/residency/activate`(事前切替)、`POST /v1/init` の排他常駐対応(slot の入れ替え) |
| Web UI | `webui/`(HTML + JS)。生成・cover・repaint・簡易モード・LM 設定・結果の再生と履歴・モデル管理 |
| 実装済みだが既定 OFF | `ACESTEP_DIT_FUSIONS`(attention / MLP / adaLN の eager 融合、効果 ≤3%)、`ACESTEP_QUANTIZATION=int8_plain` |

実測と注意点はすべて [docs/ja/低VRAM_XL_プローブ_20261116.md](docs/ja/低VRAM_XL_プローブ_20261116.md) に記録しています。

## セットアップ

### Linux の場合

前提は NVIDIA ドライバ(CUDA 12.8 以上対応)と `git`、`curl` だけです。上から順に実行します。

```bash
git clone https://github.com/animede/ACE-Step-1.5-fast.git
cd ACE-Step-1.5-fast
./setup.sh
./run_api_server_fast.sh
```

- 起動後にブラウザで `http://localhost:8001/ui/` を開く。停止は `./close_api_server.sh --port 8001`。
- `./setup.sh` が uv の導入 → `.venv` の作成 → GPU 確認 → モデル取得(約 45GB、取得済みは飛ばす)を行う。`--large` で LM 4B も、`--no-models` で環境だけ。詳細は [docs/ja/INSTALL.md](docs/ja/INSTALL.md)。

### Windows の場合

前提は NVIDIA ドライバ(CUDA 12.8 以上対応)だけです。Python・CUDA Toolkit・Visual Studio は不要(uv が Python 3.11 を用意し、torch と flash-attn はビルド済み wheel)。コマンドプロンプトで上から順に実行します。

```bat
winget install --id Git.Git -e
git clone https://github.com/animede/ACE-Step-1.5-fast.git
cd ACE-Step-1.5-fast
setup.bat
run_api_server_fast.bat
```

- 起動後にブラウザで `http://localhost:8001/ui/` を開く。停止は `close_api_server.bat`。
- `setup.bat` が uv の導入(winget、無ければ公式インストーラ)→ `.venv` の作成 → GPU 確認 → モデル取得(約 45GB)を行う。途中で止まっても再実行すれば続きから。直後に `uv` が見つからないと言われたら、新しいコマンドプロンプトで `setup.bat` をもう一度。
- Git を入れたくない場合は、1〜2 行目の代わりに GitHub の「Code → Download ZIP」を展開してそのフォルダで続ける。
- 設定の上書きは起動前に `set ACESTEP_GPU_PROFILE=large` のように指定。Windows 版は実機での動作確認がまだなので、問題があれば Issue で報告してください。

## 起動

```bash
./run_api_server_fast.sh                             # 16GB 級(既定): XL-SFT 常駐 + XL-turbo / 2B turbo を要求時に入れ替え(mmap 退避)
ACESTEP_GPU_PROFILE=large ./run_api_server_fast.sh   # 40GB 以上: 3 本とも GPU に常駐、LM 4B、退避なし
./close_api_server.sh --port 8001                    # 停止
```

起動スクリプトはこの 1 本で、すべて環境変数で上書きできます(例: `ACESTEP_API_PORT=8002`、`ACESTEP_QUEUE_WORKERS=2`、`ACESTEP_CONFIG_PATH2=acestep-v15-base`)。
同時実行数とキューの考え方は [運用メモ_同時実行数とキュー.md](運用メモ_同時実行数とキュー.md)、起動コマンドの一覧は [起動コマンド.md](起動コマンド.md)。

## 使い方

- **Web UI**: サーバ起動後に `http://<host>:<port>/ui/` を開く。モデル選択(xl-sft / xl-turbo / 2B turbo)、通常生成・cover・repaint、簡易モード(説明から LM が曲を考える)、LM の設定、結果の再生と履歴(seed の再利用)、サーバのモデル管理(GPU 上のモデルの確認・事前切替・slot の入れ替え)。
- **REST API**: `POST /release_task` → `POST /query_result` → `GET /v1/audio`。リクエストの `model` でロード済みの DiT を選ぶ(未ロードのモデル名は primary に置き換わり、結果の `dit_model` で分かる)。仕様は [docs/ja/API.md](docs/ja/API.md) / [docs/en/API.md](docs/en/API.md)。
- **学習(LoRA / LoKr)**: API の `/train` 系ルートと `acestep/training_v2`。手順は [docs/ja/LoRA_Training_Tutorial.md](docs/ja/LoRA_Training_Tutorial.md)(オリジナルのまま。コマンドライン版 `train.py` は削除済み)。

## 外部アプリからの利用(API 互換性)

このサーバは「音楽生成 API」としてオリジナルの ACE-Step 1.5 と同じ契約で使えます。`/release_task` → `/query_result` → `/v1/audio` の流れ、`/format_input`、`/create_random_sample`、`/v1/models`、`/health`、`/v1/stats`、`/v1/init` のリクエスト・レスポンスの形は変えていません。
追加した `GET /v1/residency` と `POST /v1/residency/activate`(GPU 上のモデルの確認と事前切替)、外した `/v1/chat/completions`(OpenRouter 向け)、`model` と `seed` の注意点は [docs/ja/API.md の 12 節](docs/ja/API.md#12-このフォークace-step-15-fastでの追加と互換性) にまとめています。
この契約は単体テスト(`acestep/api/api_contract_test.py`)で守っています。

### 同梱アプリ: ACE-Step-client-pro(`client-pro/`)

この API を使う本格的な Web アプリの例で、このリポジトリに同梱しています(別プロセス、ポート 8888。簡単モードとカスタムモード、LLM による歌詞生成)。

![ACE-Step-client-pro の起動画面](client-pro/startupUI.png)
`./setup.sh` が専用の環境(`client-pro/.venv`)も作るので、起動は次の 2 つです。

ターミナル 1(API サーバ、起動したまま):

```bash
./run_api_server_fast.sh
```

ターミナル 2(Web アプリ。サーバが起動してから):

```bash
./run_client_pro.sh
```

ブラウザで `http://localhost:8888` を開きます。8888 が使用中なら `./run_client_pro.sh --port 8890`。

- 歌詞生成の LLM は外部の OpenAI 互換 API(既定 Ollama、`client-pro/.env` の `OPENAI_BASE_URL`)。接続できないときは CPU 上の GGUF(Qwen3-1.7B)にフォールバックします(GPU は使いません)。
- アプリは `model` を送らないので生成は常に primary(xl-sft)です。turbo は `/ui/` から使えます。
- 使い方の詳細は [client-pro/README.md](client-pro/README.md)。**ライセンスに注意**: `client-pro/` は PolyForm Noncommercial 1.0.0([client-pro/LICENSE](client-pro/LICENSE))で、それ以外(サーバ本体)は MIT です。

## モデル

| DiT | 大きさ | CFG | step | 用途 |
|---|---|---|---|---|
| `acestep-v15-turbo` | 2B | なし | 8 | 速い下書き |
| `acestep-v15-base` / `-sft` | 2B | あり | 50 | 2B の品質重視(base は extract / lego / complete 対応) |
| `acestep-v15-xl-turbo` | 4B | なし | 8 | XL の速い生成 |
| `acestep-v15-xl-sft` | 4B | あり | 50(実運用 32) | **最高品質(このフォークの主役)** |
| `acestep-v15-xl-base` | 4B | あり | 50 | XL の extract / lego / complete |

LM(CoT / thinking): `acestep-5Hz-lm-0.6B` / `1.7B` / `4B`(Qwen3 ベース)。16GB では 0.6B、余裕があれば 4B。
配布元: [Hugging Face ACE-Step](https://huggingface.co/ACE-Step)。

## 文書

- 日本語: [docs/ja/](docs/ja/)(API、推論、インストール、GPU 互換、LM バックエンド、XL 運用ガイド、低 VRAM プローブの記録)
- English: [docs/en/](docs/en/)(API, INFERENCE, INSTALL, DCW, LoRA / SFT training, musicians guide)

## 謝辞

このフォークは、ACE Studio と StepFun が共同で開発・公開した [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) の成果の上に成り立っています。
モデル(2B / XL の DiT、5Hz LM、VAE)、学習・推論のコード、API サーバの設計はすべてオリジナルのものです。
Junmin Gong、Yulin Song、Wenxiao Zhao、Sen Wang、Shengyuan Xu、Jing Guo をはじめとする ACE-Step チームと、
オープンソースで公開してくださったことに深く感謝します。ここで行ったのは 16GB 級 GPU 向けの最適化と運用面の整理であり、
音楽生成の品質はオリジナルの研究によるものです。

## ライセンスと引用

コードは [MIT](./LICENSE)(オリジナルと同じ)。ただし同梱の `client-pro/` は PolyForm Noncommercial 1.0.0(非商用)です。モデルの利用条件は各モデルカードに従ってください。オリジナルの研究への引用:

```BibTeX
@misc{gong2026acestep,
	title={ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation},
	author={Junmin Gong, Yulin Song, Wenxiao Zhao, Sen Wang, Shengyuan Xu, Jing Guo},
	howpublished={\url{https://github.com/ace-step/ACE-Step-1.5}},
	year={2026},
	note={GitHub repository}
}
```
