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
| 同梱アプリ | `easy-music/`: ワンクリック生成の Web アプリ Easy Music(別プロセス、`./run_easy_music.sh`) |
| 実装済みだが既定 OFF | `ACESTEP_DIT_FUSIONS`(attention / MLP / adaLN の eager 融合、効果 ≤3%)、`ACESTEP_QUANTIZATION=int8_plain` |

実測と注意点はすべて [docs/ja/低VRAM_XL_プローブ_20261116.md](docs/ja/低VRAM_XL_プローブ_20261116.md) に記録しています。

## セットアップ

### Linux の場合

前提は NVIDIA ドライバ(CUDA 12.8 以上対応)と `git`、`curl` だけです。上から順に実行します。

```bash
git clone https://github.com/animede/ACE-Step-1.5-fast.git
cd ACE-Step-1.5-fast
./setup.sh
```

ターミナル 1(API サーバ、起動したまま):

```bash
cd ACE-Step-1.5-fast
./run_api_server_fast.sh
```

ターミナル 2(同梱の Web アプリを使う場合。サーバが起動してから):

```bash
cd ACE-Step-1.5-fast
./run_easy_music.sh 8888
```

- ブラウザで `http://localhost:8001/ui/`(評価用 UI)または `http://localhost:8888`(Web アプリ)を開く。Web アプリのポートは引数で変えられる(例: `./run_easy_music.sh 8890`)。停止は `./close_api_server.sh --port 8001`。
- `./setup.sh` が uv の導入 → `.venv` の作成 → GPU 確認 → モデル取得(約 45GB、取得済みは飛ばす)を行う。`--large` で LM 4B も、`--12gb` で 2B だけ(約 15GB)、`--no-models` で環境だけ。詳細は [docs/ja/INSTALL.md](docs/ja/INSTALL.md)。

### Windows の場合

前提は NVIDIA ドライバ(CUDA 12.8 以上対応)だけです。Python・CUDA Toolkit・Visual Studio は不要(uv が Python 3.11 を用意し、torch と flash-attn はビルド済み wheel)。コマンドプロンプトで上から順に実行します。

```bat
winget install --id Git.Git -e --source winget
git clone https://github.com/animede/ACE-Step-1.5-fast.git
cd ACE-Step-1.5-fast
setup.bat
```

コマンドプロンプト 1(API サーバ、起動したまま):

```bat
cd ACE-Step-1.5-fast
run_api_server_fast.bat
```

コマンドプロンプト 2(同梱の Web アプリを使う場合。サーバが起動してから):

```bat
cd ACE-Step-1.5-fast
run_easy_music.bat 8888
```

- ブラウザで `http://localhost:8001/ui/`(評価用 UI)または `http://localhost:8888`(Web アプリ)を開く。Web アプリのポートは引数で変えられる(例: `run_easy_music.bat 8890`)。停止は `close_api_server.bat`。
- 歌詞生成の LLM(OpenAI 互換 API。既定は `http://localhost:11434/v1` = Ollama、無ければ CPU 上の GGUF = Gemma 4 E4B Q4_0 にフォールバック。同梱 LLM は llama.cpp のビルド済み `llama-server` を初回に自動取得するのでコンパイラ不要)を指定するには、起動引数か `easy-music\.env` を使う:

```bat
run_easy_music.bat 8888 --llm-url http://LLM_HOST:11434/v1 --llm-model gemma3:latest
```

`easy-music\.env` に書く場合は `OPENAI_BASE_URL=http://LLM_HOST:11434/v1`、`OPENAI_API_KEY=YOUR_KEY`、`OPENAI_CHAT_MODEL=gemma3:latest`(起動引数が優先)。ACE サーバは `--ace-url`、背景画像サーバは `--img-url` で同様に指定できる。
- `setup.bat` が uv の導入(winget、無ければ公式インストーラ)→ `.venv` の作成 → GPU 確認 → モデル取得(約 45GB)を行う。途中で止まっても再実行すれば続きから。直後に `uv` が見つからないと言われたら、新しいコマンドプロンプトで `setup.bat` をもう一度。
- Git を入れたくない場合は、1〜2 行目の代わりに GitHub の「Code → Download ZIP」を展開してそのフォルダで続ける。
- 設定の上書きは起動前に `set ACESTEP_GPU_PROFILE=large` のように指定。Windows 版は実機での動作確認がまだなので、問題があれば Issue で報告してください(同じ作りの [minimax-h3](https://github.com/animede/Diffusers_minimax-h3) の Windows 実機で見つかった問題、HF キャッシュの symlink 競合・winget のソース指定・バッチの改行コード・`python -c` の引用符は、こちらにも反映済み)。

## 起動

```bash
./run_api_server_fast.sh                             # 16GB 級(既定): XL-SFT 常駐 + XL-turbo / 2B turbo を要求時に入れ替え(mmap 退避)
ACESTEP_GPU_PROFILE=large ./run_api_server_fast.sh   # 40GB 以上: 3 本とも GPU に常駐、LM 4B、退避なし
ACESTEP_GPU_PROFILE=12gb ./run_api_server_fast.sh    # 12GB 級(RTX 3060 12GB など): 2B turbo 常駐 + 2B base を要求時に入れ替え、XL は使わない
CUDA_VISIBLE_DEVICES=1 ./run_api_server_fast.sh      # 複数 GPU の機械で 2 枚目を使う(番号は nvidia-smi -L)
./close_api_server.sh --port 8001                    # 停止
```

起動後に開くもの:

- **評価用 Web UI(Gradio の代わり)**: `http://localhost:8001/ui/`。API サーバが配信するので別の起動コマンドはなく、サーバが立ち上がれば使える。Windows も同じ。
- **Easy Music(本格的な Web アプリ)**: 別ターミナルで `./run_easy_music.sh`(Windows は `run_easy_music.bat`)を実行して `http://localhost:8888`。詳細は下の[同梱アプリ](#同梱アプリ-easy-musiceasy-music)。

GPU の指定は `CUDA_VISIBLE_DEVICES` で行う(Windows は `set CUDA_VISIBLE_DEVICES=1` のあと `run_api_server_fast.bat`)。指定した GPU がそのプロセスの 0 番になるので、VRAM 量によるプロファイルの自動判定もその GPU で行われる。`ACESTEP_DEVICE=cuda:1` のような指定は、VRAM 判定が 0 番を見るため使わない。

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

### 同梱アプリ: Easy Music(`easy-music/`)

この API を使う本格的な Web アプリの例で、このリポジトリに同梱しています(別プロセス、ポート 8888)。ジャンルを選んでワンクリックで曲を作る「簡単モード」、AI 作詞、ムード・ボーカルの指定、STEP 選択、インラインプレーヤー、履歴、背景画像の自動生成(別の画像生成サーバがある場合)、日英切替。開発元は [animede/easy_music](https://github.com/animede/easy_music)(MIT)。

![Easy Music の起動画面](easy-music/startupUI.png)

`./setup.sh` が専用の環境(`easy-music/.venv`)も作ります。

ターミナル 1(API サーバ、起動したまま):

```bash
cd ACE-Step-1.5-fast
./run_api_server_fast.sh
```

ターミナル 2(Web アプリ。サーバが起動してから。引数はポート):

```bash
cd ACE-Step-1.5-fast
./run_easy_music.sh 8888
```

ブラウザで `http://localhost:8888` を開きます。8888 が使用中なら `./run_easy_music.sh 8890`。

- 歌詞生成の LLM は外部の OpenAI 互換 API(Ollama、llama.cpp server、LM Studio、vLLM、OpenAI など)。接続できないときは CPU 上の GGUF(Gemma 4 E4B Q4_0、約 4.4GB、初回に自動ダウンロード、thinking は OFF)にフォールバックします(GPU は使いません)。同梱 LLM の実体は llama.cpp の GitHub Releases にあるビルド済み `llama-server`(CPU 版、約 25MB、固定タグ b12475)で、初回のフォールバック時に `easy-music/tools/llama/` へ自動取得して子プロセスとして起動します(コンパイラ不要。`easy-music/.venv/bin/python easy-music/scripts/fetch_llama_server.py` で事前取得も可)。既存のバイナリや GGUF を使うなら環境変数 `LLAMA_SERVER` / `LOCAL_LLM_GGUF`、ポートは `LOCAL_LLM_PORT`(既定 11440)で指定できます。

**LLM の指定方法**(起動引数 > `easy-music/.env` > 既定値 `http://localhost:11434/v1` の順に優先):

```bash
./run_easy_music.sh 8888 --llm-url http://LLM_HOST:11434/v1 --llm-model gemma3:latest
```

ホストとポートを分けるなら `--llm-host LLM_HOST --llm-port 11434`。恒久的に設定するなら `easy-music/.env` に次を書きます(認証不要のサーバなら `OPENAI_API_KEY` はダミーのまま)。

```
OPENAI_BASE_URL=http://LLM_HOST:11434/v1
OPENAI_API_KEY=YOUR_KEY
OPENAI_CHAT_MODEL=gemma3:latest
```

同じ要領で、ACE サーバは `--ace-url` / `ACE_STEP_API_URL`、背景画像サーバは `--img-url` / `IMAGE_GENERATION_URL` で指定できます。Windows は `run_easy_music.bat 8888 --llm-url ...`。
- 背景画像は任意で、`IMAGE_GENERATION_URL` に画像生成サーバを指定したときだけ使われます(無ければ曲の生成はそのまま動きます)。
- 使い方の詳細は [easy-music/README.md](easy-music/README.md)。同梱版は開発元の main を `git subtree` で取り込んだもので、`git subtree pull --prefix=easy-music https://github.com/animede/easy_music.git main` で追従できます。

## モデル

| DiT | 大きさ | CFG | step | VRAM(bf16 の重み) | ディスク | 用途 |
|---|---|---|---|---|---|---|
| `acestep-v15-turbo` | 2B | なし | 8 | 4.8GB | 4.5GB | 速い下書き |
| `acestep-v15-base` / `-sft` | 2B | あり | 50 | 4.8GB | 4.5GB | 2B の品質重視(base は extract / lego / complete 対応) |
| `acestep-v15-xl-turbo` | 4B | なし | 8 | 10.0GB | 19GB | XL の速い生成 |
| `acestep-v15-xl-sft` | 4B | あり | 50(実運用 32) | 10.0GB | 19GB | **最高品質(このフォークの主役)** |
| `acestep-v15-xl-base` | 4B | あり | 50 | 10.0GB | 19GB | XL の extract / lego / complete |

DiT 以外: LM(CoT / thinking)`acestep-5Hz-lm-0.6B` 1.3GB / `1.7B` 3.5GB / `4B` 8.4GB(Qwen3 ベース)、テキストエンコーダ `Qwen3-Embedding-0.6B` 1.2GB、VAE 0.3GB。16GB では LM 0.6B、余裕があれば 4B。
配布元: [Hugging Face ACE-Step](https://huggingface.co/ACE-Step)。`setup.sh` の既定(xl-sft、xl-turbo、turbo、LM 0.6B、TE、VAE)で約 45GB、`--large` で LM 4B が加わり約 53GB、`--12gb` は XL を取らず 2B turbo + 2B base(+ LM 1.7B を含む基本セット)で約 15GB。

### 必要 VRAM の目安(実測は RTX A4000 16GB、RTX PRO 6000 96GB)

| プロファイル | GPU に載るもの | 起動直後 | 生成中のピーク | ホスト側 |
|---|---|---|---|---|
| `16gb`(既定) | DiT 1 本(既定 xl-sft)+ VAE + LM 0.6B。TE は要求時だけ GPU へ(同時実行 2 以上では常駐、+1.2GB) | 約 10GB | XL-sft 300 秒 1 本で約 13.6GB、2B turbo 60 秒で約 12GB | サーバ常駐 約 4GB。退避中の DiT は `.cache/acestep/dit_residency/` の bf16 ファイル(XL 10GB、2B 4.8GB、初回起動時に作成、合計約 25GB のディスク)をページキャッシュ経由で読むので、RAM はその分あると切替が約 2 秒、無ければ NVMe から約 7 秒 |
| `12gb` | 2B turbo(常駐)+ 2B base(要求時に入れ替え)+ VAE + LM 0.6B。XL は読まない | 約 8GB | 2B turbo / base とも 60〜300 秒で約 8.5GB(A4000 で実測、12GB に約 3.5GB の余裕) | 16gb と同じ(退避ファイルは 2B 2 本で約 10GB) |
| `large` | 3 本の DiT 全部 + VAE + TE + LM 4B、退避なし | 約 35GB(重みの合計からの見込み) | 生成分(300 秒 1 本で 3〜4GB、同時実行数ぶん)が上乗せ。40GB 以上、同時実行 2〜4 なら 48GB 級 | サーバ常駐 約 4GB |

- 16GB では XL-sft の 300 秒は 1 並列まで(`ACESTEP_QUEUE_WORKERS=1`、既定)。2B turbo なら 2 並列も入る。
- 16GB で 600 秒(10 分)の曲を作るときは、VRAM の見積もりが先に拒否するか OOM になることがある。長い曲は `large` か、`ACESTEP_VAE_DECODE_CHUNK_SIZE=128` で decode のピークを下げる。
- 12GB の GPU は `12gb` プロファイル(2B のみ)。XL は bf16 の重みだけで 10GB あるので載らない。
- **8GB で動かす場合(未確認)**: `12gb` の 8.6GB の内訳は 2B turbo 4.8GB + LM 0.6B 1.3GB + VAE 0.3GB + テキストエンコーダ 1.2GB(要求時のみ)+ 作業領域 約 1GB。計算上は次の指定で 6.5〜7.3GB まで下がる。実機では未確認で、8GB カードは表示用に 0.5〜1GB 取られるので余裕は 1GB 前後。短い曲(60〜120 秒)向け。

  ```bash
  ACESTEP_GPU_PROFILE=12gb ACESTEP_LM_OFFLOAD_TO_CPU=1 ACESTEP_RESIDENT_COMPONENTS= ACESTEP_VAE_DECODE_CHUNK_SIZE=128 ./run_api_server_fast.sh
  ```

  - `ACESTEP_LM_OFFLOAD_TO_CPU=1`: LM(CoT)を拡散の前だけ GPU に置く。ピークは LM の段と拡散の段の大きい方になり、拡散の段は約 7.3GB。CoT を使わないなら `ACESTEP_INIT_LLM=0` で LM 自体を読まない。
  - `ACESTEP_RESIDENT_COMPONENTS=`(空): VAE も要求時だけ GPU へ(−0.3GB、1 曲あたり +1 秒程度)。
  - `ACESTEP_VAE_DECODE_CHUNK_SIZE=128`: decode のピークを −0.5GB(decode が約 2 秒遅くなる)。
  - それでも足りなければ `ACESTEP_QUANTIZATION=int8_weight_only`(2B の重み 4.8 → 2.4GB。速くはならず、音も bf16 と少し変わる)。

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

コードは [MIT](./LICENSE)(オリジナルと同じ)。同梱の Easy Music(`easy-music/`)も MIT です。モデルの利用条件は各モデルカードに従ってください。オリジナルの研究への引用:

```BibTeX
@misc{gong2026acestep,
	title={ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation},
	author={Junmin Gong, Yulin Song, Wenxiao Zhao, Sen Wang, Shengyuan Xu, Jing Guo},
	howpublished={\url{https://github.com/ace-step/ACE-Step-1.5}},
	year={2026},
	note={GitHub repository}
}
```
