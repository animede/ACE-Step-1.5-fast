# ACE-Step-1.5-fast

**最新版: 2026-11-21 18:10 JST** — README を全体に見直し(対象 GPU と OS の記述、オリジナルとの違いの表に最近の機能を追加、Easy Music の節を整理)。「実機の実測」の表(RTX 3060 12GB の入門機で 3 分 15 秒の曲を 1 分 15 秒、A4000 16GB で 40 秒前後)。変更の履歴は [GitHub の commits](https://github.com/animede/ACE-Step-1.5-fast/commits/main)。

[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)(オープンソースの音楽生成モデル)のフォーク。
**16GB 級 GPU(RTX A4000)で XL(4B)モデルを動かすための高速化版 + API サーバ改良版**です。12GB 級は 2B モデルで、8GB 級も指定付きで動きます。対象は NVIDIA GPU(CUDA)の Linux と Windows です。
オリジナル(ace-step/ACE-Step-1.5、git remote `upstream`)の 2026-05-04 時点 `6c1b2ef` を土台にし、オリジナルの取り込みは予定していません。

- 実測(A4000、XL-SFT、300 秒 1 曲、32 step、mp3・CoT あり、音は bf16 のまま): **84 秒 → 62〜65 秒**
- サーバの常駐ホストメモリ 約 4GB(32GB の専用機で動く)
- 評価用の Web UI(Gradio 不要)を API サーバが `/ui/` で配信
- ワンクリックで曲を作る Web アプリ [Easy Music](#同梱アプリ-easy-musiceasy-music) を同梱(`easy-music/`、別プロセス、ポート 8888。歌詞・キャプションの LLM は外部が無ければ同梱の Gemma 4 E4B を GPU で使う)

## オリジナルとの違い

| 項目 | 内容 |
|---|---|
| 排他常駐 | `ACESTEP_DIT_RESIDENCY=exclusive`: GPU に DiT を 1 本だけ置き、他はホスト側に退避して約 2 秒で入れ替える。退避先は `ACESTEP_DIT_RESIDENCY_STORE=mmap`(bf16 の safetensors をページキャッシュ経由で mmap、常駐 RSS 約 4GB)または `pinned`(固定 RAM に 3 本で約 23GB) |
| RMSNorm 融合 | `ACESTEP_FUSED_RMSNORM=exact`: DiT の RMSNorm を融合カーネルに(1 step −14%、bf16 の丸め差のみ) |
| 拡散以外の削減 | `ACESTEP_RESIDENT_COMPONENTS`(VAE / テキストエンコーダを退避しない)、`ACESTEP_VAE_DECODE_CHUNK_SIZE`、`ACESTEP_MP3_COMPRESSION_LEVEL`(任意) |
| API サーバの修正 | `ACESTEP_QUANTIZATION` を受け取る(オリジナルは量子化を渡さず常に bf16)。ローダは CPU で量子化してから GPU へ、compile は `decoder.forward` に当てる |
| 同時実行 | LM(CoT)の呼び出しを直列化(オリジナルは同時実行で CUDA が壊れる)。別モデルへの切替は実行中のジョブを待ってから |
| モデル管理 API | `GET /v1/residency`、`POST /v1/residency/activate`(事前切替)、`POST /v1/init` の排他常駐対応(slot の入れ替え)。VRAM に載らないモデルは `/v1/init` が読む前に断り、`/v1/models` に `vram_ok` を付ける |
| プロファイルの自動選択 | 起動スクリプトが VRAM を見て `12gb`(14GB 未満)/ `16gb` を選ぶ。8GB は指定付き([必要 VRAM の目安](#必要-vram-の目安実測は-rtx-a4000-16gbrtx-pro-6000-96gb)) |
| 生成の中止 | `POST /v1/cancel`。待ち行列中は即、生成中は今のステップの後で止まる。`/release_task` に任意の `task_id` |
| Windows 対応 | `setup.bat` / `run_api_server_fast.bat` / `run_easy_music.bat`。Windows 10 / 11 の実機で確認済み |
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

前提は NVIDIA ドライバ(CUDA 12.8 以上対応)だけです。Python・CUDA Toolkit・Visual Studio は不要(uv が Python 3.11 を用意し、torch と flash-attn はビルド済み wheel)。PowerShell でもコマンドプロンプトでも同じ(`.\` 付きで書いてあるのはそのため。PowerShell はカレントディレクトリの `.bat` を名前だけでは実行しない)。コマンドプロンプトで上から順に実行します。

```bat
winget install --id Git.Git -e --source winget
git clone https://github.com/animede/ACE-Step-1.5-fast.git
cd ACE-Step-1.5-fast
.\setup.bat
```

コマンドプロンプト 1(API サーバ、起動したまま):

```bat
cd ACE-Step-1.5-fast
.\run_api_server_fast.bat
```

コマンドプロンプト 2(同梱の Web アプリを使う場合。サーバが起動してから):

```bat
cd ACE-Step-1.5-fast
.\run_easy_music.bat 8888
```

- ブラウザで `http://localhost:8001/ui/`(評価用 UI)または `http://localhost:8888`(Web アプリ)を開く。Web アプリのポートは引数で変えられる(例: `.\run_easy_music.bat 8890`)。停止は `.\close_api_server.bat`。
- 歌詞を作る LLM は、外部の LLM サーバが無ければ同梱の Gemma 4 E4B を自動で取ってきて GPU で使う。指定の仕方は[同梱アプリ: Easy Music](#同梱アプリ-easy-musiceasy-music)。
- `setup.bat` が uv の導入(winget、無ければ公式インストーラ)→ `.venv` の作成 → GPU 確認 → モデル取得(約 45GB)を行う。途中で止まっても再実行すれば続きから。直後に `uv` が見つからないと言われたら、新しいウィンドウで `.\setup.bat` をもう一度。
- Git を入れたくない場合は、1〜2 行目の代わりに GitHub の「Code → Download ZIP」を展開してそのフォルダで続ける。
- 設定の上書きは起動前に `set ACESTEP_GPU_PROFILE=large`(PowerShell は `$env:ACESTEP_GPU_PROFILE = "large"`)のように指定。
- Windows 実機で動作を確認済み(RTX 3060 12GB の入門機と RTX 4060 Laptop 8GB のノート PC。数字は[実機の実測](#実機の実測))。12GB 以上は何も付けずに起動すればよい(VRAM を見て、12GB 級なら `12gb`、16GB 以上なら `16gb` を選ぶ)。**8GB は次の 5 つを最初から全部入れて起動**する。付けずに起動すると VRAM があふれて共有 GPU メモリに退避し、エラーにはならないまま極端に遅くなる。

```powershell
$env:ACESTEP_GPU_PROFILE = "12gb"
$env:ACESTEP_LM_OFFLOAD_TO_CPU = "1"
$env:ACESTEP_RESIDENT_COMPONENTS = ""
$env:ACESTEP_VAE_DECODE_CHUNK_SIZE = "128"
$env:ACESTEP_QUANTIZATION = "int8_weight_only"
.\run_api_server_fast.bat
```

- 問題があれば Issue で報告してください。

## 起動

```bash
./run_api_server_fast.sh                             # プロファイル無指定: VRAM を見て自動選択(14GB 未満 → 12gb、それ以上 → 16gb。起動時に 1 行表示)
ACESTEP_GPU_PROFILE=16gb ./run_api_server_fast.sh    # 16GB 級: XL-SFT 常駐 + XL-turbo / 2B turbo を要求時に入れ替え(mmap 退避)
ACESTEP_GPU_PROFILE=large ./run_api_server_fast.sh   # 40GB 以上: 3 本とも GPU に常駐、LM 4B、退避なし(自動では選ばれない)
ACESTEP_GPU_PROFILE=12gb ./run_api_server_fast.sh    # 12GB 級(RTX 3060 12GB など): 2B turbo 常駐 + 2B base を要求時に入れ替え、XL は使わない
# 8GB 級(RTX 4060 Laptop 8GB / RAM 16GB(DDR5-4800)の Windows で確認済み): 12gb をもとに LM・VAE を退避、decode chunk 128、2B を int8 に。詳細は「必要 VRAM の目安」
ACESTEP_GPU_PROFILE=12gb ACESTEP_LM_OFFLOAD_TO_CPU=1 ACESTEP_RESIDENT_COMPONENTS= ACESTEP_VAE_DECODE_CHUNK_SIZE=128 ACESTEP_QUANTIZATION=int8_weight_only ./run_api_server_fast.sh
CUDA_VISIBLE_DEVICES=1 ./run_api_server_fast.sh      # 複数 GPU の機械で 2 枚目を使う(番号は nvidia-smi -L)
./close_api_server.sh --port 8001                    # 停止
```

起動後に開くもの:

- **評価用 Web UI(Gradio の代わり)**: `http://localhost:8001/ui/`。API サーバが配信するので別の起動コマンドはなく、サーバが立ち上がれば使える。Windows も同じ。
- **Easy Music(本格的な Web アプリ)**: 別ターミナルで `./run_easy_music.sh`(Windows は `run_easy_music.bat`)を実行して `http://localhost:8888`。詳細は下の[同梱アプリ](#同梱アプリ-easy-musiceasy-music)。

GPU の指定は `CUDA_VISIBLE_DEVICES` で行う(Windows は `set CUDA_VISIBLE_DEVICES=1` のあと `run_api_server_fast.bat`)。指定した GPU がそのプロセスの 0 番になるので、起動スクリプトの VRAM によるプロファイル自動選択(nvidia-smi の容量が 14GB 未満なら `12gb`)もその GPU で行われる。`ACESTEP_DEVICE=cuda:1` のような指定は、VRAM 判定が 0 番を見るため使わない。

起動スクリプトはこの 1 本で、すべて環境変数で上書きできます(例: `ACESTEP_API_PORT=8002`、`ACESTEP_QUEUE_WORKERS=2`、`ACESTEP_CONFIG_PATH2=acestep-v15-base`)。
同時実行数とキューの考え方は [運用メモ_同時実行数とキュー.md](運用メモ_同時実行数とキュー.md)、起動コマンドの一覧は [起動コマンド.md](起動コマンド.md)。

## 使い方

- **Web UI**: サーバ起動後に `http://<host>:<port>/ui/` を開く。モデル選択(サーバに読み込んだ DiT から。`16gb` なら xl-sft / xl-turbo / 2B turbo、`12gb` なら 2B turbo / 2B base)、通常生成・cover・repaint、簡易モード(説明から LM が曲を考える)、LM の設定、結果の再生と履歴(seed の再利用)、サーバのモデル管理(GPU 上のモデルの確認・事前切替・slot の入れ替え)。
- **REST API**: `POST /release_task` → `POST /query_result` → `GET /v1/audio`。仕様は [docs/ja/API.md](docs/ja/API.md) / [docs/en/API.md](docs/en/API.md)。
  - リクエストの `model` でロード済みの DiT を選ぶ。読み込んでいないモデル名はエラー(`Model '...' is not loaded on this server`)で、黙って primary には置き換えない。先に `POST /v1/init {"model": ..., "slot": 2}` で読み込む(数十秒)。
  - `GET /v1/models` の各モデルに `is_loaded` / `vram_ok` / `min_vram_gb` が付く。VRAM に載らないモデル(12GB 級への XL など)は `/v1/init` が `VRAM too small` で断る。
  - 中止は `POST /v1/cancel {"task_id": ...}`。結果は status 2、error `cancelled by client`。`/release_task` に任意で `task_id`(UUID)を付けると、投入の応答を受け取れなかったときもその ID で中止できる(同じ ID は 409)。
- **学習(LoRA / LoKr)**: API の `/train` 系ルートと `acestep/training_v2`。手順は [docs/ja/LoRA_Training_Tutorial.md](docs/ja/LoRA_Training_Tutorial.md)(オリジナルのまま。コマンドライン版 `train.py` は削除済み)。

## 外部アプリからの利用(API 互換性)

このサーバは「音楽生成 API」としてオリジナルの ACE-Step 1.5 と同じ契約で使えます。`/release_task` → `/query_result` → `/v1/audio` の流れ、`/format_input`、`/create_random_sample`、`/v1/models`、`/health`、`/v1/stats`、`/v1/init` のリクエスト・レスポンスの形は変えていません。
追加した `GET /v1/residency` と `POST /v1/residency/activate`(GPU 上のモデルの確認と事前切替)、`POST /v1/cancel`(中止)、外した `/v1/chat/completions`(OpenRouter 向け)、`model` と `seed` の注意点は [docs/ja/API.md の 12 節](docs/ja/API.md#12-このフォークace-step-15-fastでの追加と互換性) にまとめています。
動きを 1 つ変えています。読み込んでいないモデル名を `model` に指定したとき、オリジナルは黙って primary で生成しますが、このフォークはエラーを返します(思ったのと違うモデルで生成されないように)。
この契約は単体テスト(`acestep/api/api_contract_test.py`)で守っています。

### 同梱アプリ: Easy Music(`easy-music/`)

この API を使う本格的な Web アプリの例で、このリポジトリに同梱しています(別プロセス、ポート 8888)。ジャンルを選んでワンクリックで曲を作る「簡単モード」、AI 作詞、ムード・ボーカルの指定、STEP 選択、インラインプレーヤー、履歴、背景画像の自動生成(別の画像生成サーバがある場合)、日英切替。開発元は [animede/easy_music](https://github.com/animede/easy_music)(MIT)。

![Easy Music の起動画面](easy-music/startupUI.png)

起動はセットアップの手順どおり、API サーバのあとに `./run_easy_music.sh 8888`(Windows は `.\run_easy_music.bat 8888`)で、ブラウザで `http://localhost:8888` を開きます。8888 が使用中なら別のポートを引数に。専用の環境(`easy-music/.venv`)は `setup.sh` / `setup.bat` が作ります。使い方の詳細は [easy-music/README.md](easy-music/README.md)。

#### 歌詞・キャプションの LLM

- 外部の OpenAI 互換 API(Ollama、llama.cpp server、LM Studio、vLLM、OpenAI など)を使い、つながらなければ同梱の Gemma 4 E4B(GGUF Q4_0、約 4.4GB、初回に自動ダウンロード、thinking は OFF)を使います。外部の既定は `http://localhost:11434/v1`(Ollama)。
- 同梱 LLM の実体は llama.cpp のビルド済み `llama-server`(固定タグ b12475)で、初回に `easy-music/tools/llama/` へ自動で取ってきて子プロセスとして動かします(コンパイラ不要。`easy-music/scripts/fetch_llama_server.py` で事前取得も可)。既存のバイナリや GGUF を使うなら `LLAMA_SERVER` / `LOCAL_LLM_GGUF`、ポートは `LOCAL_LLM_PORT`(既定 11440)。
- **GPU で動かす**(`LOCAL_LLM_GPU=auto`、既定): NVIDIA GPU があれば CUDA 版の `llama-server`(本体 + cudart、0.6〜0.8GB)を取り、空き VRAM に入る層だけ GPU に載せます(llama.cpp の `--fit`。全部載ると約 3GB + KV)。ACE に生成を投げる直前に止めて VRAM を返し、ACE の生成が終わるまでは起こしません(曲名や背景画像のプロンプトはその後に作る)。`LOCAL_LLM_GPU=off` で CPU 版、`all` / 層数で固定。`LOCAL_LLM_STOP_BEFORE_ACE=0` で止めない(VRAM 16GB 以上で同居できるとき)、`=1` で CPU 版でも止める(RAM 16GB で 4.4GB を返したいとき)。
- **呼び出しは 1 回**: 簡単モードでテーマも歌詞も空のまま生成を押すと、テーマ・歌詞・キャプションを 1 回の呼び出しでまとめて作ります(`/api/song_plan`)。失敗したときはテーマ → 歌詞 → キャプションの 3 回に戻ります。
- **疎通確認**: 起動時と 60 秒ごとに外部 LLM(`GET /v1/models`。生成はしない。指定モデルが一覧に無ければ使えない扱い)と画像サーバを確かめ、起動ログに `[起動確認] 外部 LLM: 接続不可(...)→ 同梱 LLM(Gemma 4 E4B)を使用` のように 1 行ずつ出します。使えない間は呼び出しのたびに試さず、つながれば次の確認で外部に戻ります。状態は `GET /api/services_status`。

指定は 起動引数 > `easy-music/.env` > 既定値 の順に優先します。

```bash
./run_easy_music.sh 8888 --llm-url http://LLM_HOST:11434/v1 --llm-model gemma3:latest
```

ホストとポートを分けるなら `--llm-host LLM_HOST --llm-port 11434`。`easy-music/.env` に書くなら次の 3 行です(認証不要のサーバなら `OPENAI_API_KEY` はダミーのまま)。

```
OPENAI_BASE_URL=http://LLM_HOST:11434/v1
OPENAI_API_KEY=YOUR_KEY
OPENAI_CHAT_MODEL=gemma3:latest
```

ACE サーバは `--ace-url` / `ACE_STEP_API_URL`、背景画像サーバは `--img-url` / `IMAGE_GENERATION_URL` で同じように指定できます。Windows は `.\run_easy_music.bat 8888 --llm-url ...`。

#### 背景画像

再生画面の背景画像は任意です。`IMAGE_GENERATION_URL` の画像生成サーバに届くときだけ作り、届かなければ画像のプロンプト作りごと飛ばします(曲の生成はそのまま動きます)。

#### 生成の流れと待ち時間

- **ACE サーバの準備待ち**: ACE サーバはモデルを読み込み終わるまで(初回は数分)接続を受け付けません。端末に `[API Server] READY / 準備完了` が出たら使えます。Easy Music はそれまで生成ボタンを薄くして「ACE サーバの準備を待っています」を出し、押しても生成しません。準備ができると自動で使えるようになり、モデル一覧も取り直します(以後も 30 秒ごとに確認)。
- **投入の待ち時間**: タスクを投入して ACE サーバの受付を最長 120 秒待ちます(`easy-music/.env` の `ACE_SUBMIT_TIMEOUT`)。超えたらサーバ側の生成も中止して「もう一度生成を」と出します。
- **中止ボタン**: 生成中だけ「⏹ 中止」が出ます。フロントの待ちをやめるだけでなく、ACE サーバの生成も `/v1/cancel` で止めます(今のステップの後、数秒以内)。
- **モデルの選択**: プルダウンにはサーバの `checkpoints/` にある DiT が全部並びます。読み込まれていないものは「(未読込・選ぶと切替)」で、選んで生成を押すとその場で読み込みます(XL で数十秒〜1 分。primary 以外のスロットを使うので既定モデルはそのまま)。この GPU に載らないもの(12GB 級で XL など)は「(このGPUには載らない)」で選べません。前回選んだモデルはブラウザに覚えておき、起動し直しても同じものを選びます。

同梱版は開発元の main を `git subtree` で取り込んだもので、`git subtree pull --prefix=easy-music https://github.com/animede/easy_music.git main` で追従できます。

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
| `16gb`(14GB 以上で自動) | DiT 1 本(既定 xl-sft)+ VAE + LM 0.6B。TE は要求時だけ GPU へ(同時実行 2 以上では常駐、+1.2GB) | 約 10GB | XL-sft 300 秒 1 本で約 13.6GB、2B turbo 60 秒で約 12GB | サーバ常駐 約 4GB。退避中の DiT は `.cache/acestep/dit_residency/` の bf16 ファイル(XL 10GB、2B 4.8GB、初回起動時に作成、合計約 25GB のディスク)をページキャッシュ経由で読むので、RAM はその分あると切替が約 2 秒、無ければ NVMe から約 7 秒 |
| `12gb`(14GB 未満で自動) | 2B turbo(常駐)+ 2B base(要求時に入れ替え)+ VAE + LM 0.6B。XL は読まない | 約 8GB | 2B turbo / base とも 60〜300 秒で約 8.5GB(A4000 で実測、12GB に約 3.5GB の余裕) | 16gb と同じ(退避ファイルは 2B 2 本で約 10GB) |
| `large`(指定したときだけ) | 3 本の DiT 全部 + VAE + TE + LM 4B、退避なし | 約 35GB(重みの合計からの見込み) | 生成分(300 秒 1 本で 3〜4GB、同時実行数ぶん)が上乗せ。40GB 以上、同時実行 2〜4 なら 48GB 級 | サーバ常駐 約 4GB |

- 16GB では XL-sft の 300 秒は 1 並列まで(`ACESTEP_QUEUE_WORKERS=1`、既定)。2B turbo なら 2 並列も入る。
- 16GB で 600 秒(10 分)の曲を作るときは、VRAM の見積もりが先に拒否するか OOM になることがある。長い曲は `large` か、`ACESTEP_VAE_DECODE_CHUNK_SIZE=128` で decode のピークを下げる。
- 12GB の GPU は `12gb` プロファイル(2B のみ、自動で選ばれる)。XL は bf16 の重みだけで 10GB あるので載らない。
- **8GB で動かす場合(RTX 4060 Laptop 8GB / i9-13900H / RAM 16GB(DDR5-4800)の Windows 11 ノート PC で確認済み)**: `12gb` の 8.6GB の内訳は 2B turbo 4.8GB + LM 0.6B 1.3GB + VAE 0.3GB + テキストエンコーダ 1.2GB(要求時のみ)+ 作業領域 約 1GB。8GB カードは表示用に 0.5〜1GB 取られるので、次の 5 つを**最初から全部**指定する(int8 抜きだと 6.5〜7.3GB の見積もりでも実機では足りず、Windows では VRAM があふれても OOM にならずに共有 GPU メモリへ退避して、極端に遅くなってフロントがタイムアウトする)。生成時間は下の[実機の実測](#実機の実測)。

  ```bash
  ACESTEP_GPU_PROFILE=12gb ACESTEP_LM_OFFLOAD_TO_CPU=1 ACESTEP_RESIDENT_COMPONENTS= ACESTEP_VAE_DECODE_CHUNK_SIZE=128 ACESTEP_QUANTIZATION=int8_weight_only ./run_api_server_fast.sh
  ```

  Windows(PowerShell)は上の[Windows の場合](#windows-の場合)にある `$env:` 形式の 5 行。

  - `ACESTEP_LM_OFFLOAD_TO_CPU=1`: LM(CoT)を拡散の前だけ GPU に置く。ピークは LM の段と拡散の段の大きい方になる。CoT を使わないなら `ACESTEP_INIT_LLM=0` で LM 自体を読まない。
  - `ACESTEP_RESIDENT_COMPONENTS=`(空): VAE も要求時だけ GPU へ(−0.3GB、1 曲あたり +1 秒程度)。
  - `ACESTEP_VAE_DECODE_CHUNK_SIZE=128`: decode のピークを −0.5GB(decode が約 2 秒遅くなる)。
  - `ACESTEP_QUANTIZATION=int8_weight_only`: 2B の重みを 4.8 → 2.4GB に。8GB ではこれが無いと足りない。速くはならず、音は bf16 と少し変わる。
  - RAM 16GB では Easy Music のローカル LLM(Gemma 4 E4B、約 4.4GB)と ACE サーバ(常駐約 4GB)が同居してぎりぎりなので、他の大きなアプリは閉じておく。
  - 8GB でも同梱 LLM の GPU 実行(`LOCAL_LLM_GPU=auto`)は使える: `--fit` が空き VRAM に入る層だけ載せ、ACE に投げる直前に止める。CPU の弱い機械ほど効く。

### 実機の実測

Easy Music の簡単モード(ジャンルを選んで生成、テーマ・歌詞・キャプションは LLM が作る)で、生成ボタンを押してから曲ができるまで。LLM の時間(テーマ・歌詞・キャプション)と ACE の時間(CoT の LM 0.6B、DiT、デコード)を含みます。曲の長さは自動(AI 推定)。

| 環境 | 起動・モデル | 曲の長さ → 生成時間 |
|---|---|---|
| RTX 3060 12GB / i3-10105(4 コア)/ RAM 32GB、Windows 10 | 何も付けない(自動で `12gb`)、2B turbo、同梱 LLM は GPU | 起動直後の 1 曲目: 3 分 25 秒 → 3 分 35 秒<br>続けて同じジャンル: 3 分 15 秒 → **1 分 15 秒**<br>ジャンルを変えて: 3 分 50 秒 → **2 分 1 秒** |
| RTX 4060 Laptop 8GB / i9-13900H / RAM 16GB(DDR5-4800)、Windows 11 | [8GB の 5 つの指定](#必要-vram-の目安実測は-rtx-a4000-16gbrtx-pro-6000-96gb)、2B turbo、同梱 LLM は CPU | 3 分 19 秒 → 2 分 50 秒(CPU 負荷 65% 前後) |
| RTX A4000 16GB / Core Ultra 7 265KF / RAM 128GB、Ubuntu 24.04 | `12gb` を指定、2B turbo、同梱 LLM は GPU | 起動直後の 1 曲目: 3 分 13 秒 → **39 秒**<br>続けて同じジャンル: 3 分 22 秒 → **40 秒**<br>ジャンルを変えて: 3 分 0 秒 → **40 秒** |
| 同上 | 何も付けない(自動で `16gb`)、2B turbo を選択 | 起動直後の 1 曲目: 3 分 35 秒 → 1 分 19 秒<br>続けて同じジャンル: 3 分 8 秒 → 39 秒<br>ジャンルを変えて: 3 分 35 秒 → 42 秒 |
| 同上 | 何も付けない(自動で `16gb`)、XL-SFT(既定モデル、50 ステップ) | 2 分 55 秒 → 1 分 26 秒 |

- 2 曲目からは、どの環境でも曲の長さより短い時間で作れるので、1 曲流している間に次を作る連続生成が成り立ちます。
- A4000 の内訳は、LLM(テーマ・歌詞・キャプションを 1 回で)約 7 秒、ACE の `format_input` 約 5 秒、ACE の生成が 2B turbo で 25〜30 秒、XL-SFT で 74 秒です。Linux の行は、簡単モードと同じリクエストを同じ順に送るスクリプトで測りました(ブラウザ操作の時間は含みません)。
- **起動直後の 1 曲目が遅い理由**: 同梱 LLM は起動時に空き VRAM を見て、入らない層を CPU に回します(llama.cpp の `--fit`)。A4000 の `16gb` は起動時に XL-SFT(約 10GB)を GPU に置くので、1 曲目の LLM は一部が CPU で動き、12.7 tok/s でした(全部 GPU なら 82 tok/s)。2B turbo で 1 曲作ると GPU 上のモデルが 2B turbo(4.8GB)に替わって空きが増え、2 曲目からは LLM が全部 GPU に載ります。`12gb` で起動すると最初から空きがあり、1 曲目から遅くなりません。3060 12GB は待機中の ACE で約 8GB を使うので空きが少なく、同じ理由で 1 曲目の LLM が i3 の CPU で動いて遅くなっていると見ています(推定)。
- 3060 の入門機では CPU が律速です。同梱 LLM を CPU で動かしていたときは、歌詞だけで 1 分 20 秒かかっていました。GPU 実行(既定)とテーマ・歌詞・キャプションの 1 回化で、LLM の時間が十数秒になりました。

## 文書

- 日本語: [docs/ja/](docs/ja/)(API、推論、インストール、GPU 互換、LM バックエンド、XL 運用ガイド、低 VRAM プローブの記録)
- English: [docs/en/](docs/en/)(API, INFERENCE, INSTALL, DCW, LoRA / SFT training, musicians guide)

## 謝辞

このフォークは、ACE Studio と StepFun が共同で開発・公開した [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) の成果の上に成り立っています。
モデル(2B / XL の DiT、5Hz LM、VAE)、学習・推論のコード、API サーバの設計はすべてオリジナルのものです。
Junmin Gong、Yulin Song、Wenxiao Zhao、Sen Wang、Shengyuan Xu、Jing Guo をはじめとする ACE-Step チームと、
オープンソースで公開してくださったことに深く感謝します。ここで行ったのは 16GB 級をはじめ 12GB / 8GB の GPU 向けの最適化と運用面の整理であり、
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
