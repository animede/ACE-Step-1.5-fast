# インストール（ACE-Step-1.5-fast）

対象: Linux x86_64 + NVIDIA GPU(CUDA 12.8 対応ドライバ)。Python 3.11 / 3.12(無ければ uv が自動で用意する)。

## 必要なもの

| 項目 | 目安 |
|---|---|
| GPU | 16GB(A4000 級)で XL を実行可。40GB 以上なら `ACESTEP_GPU_PROFILE=large` で全モデル常駐。12GB(RTX 3060 12GB など)は `ACESTEP_GPU_PROFILE=12gb` で 2B のみ(ピーク約 8.5GB) |
| ホストメモリ | 32GB 以上(16gb プロファイルの常駐は約 4GB + ページキャッシュ) |
| ディスク | 環境 約 8GB、チェックポイント 約 45GB(xl-sft 19 / xl-turbo 19 / 2B turbo 4.5 / LM 0.6B 1.3 / VAE・テキストエンコーダ 1.5)、退避用キャッシュ 24GB |
| ネットワーク | 初回に Hugging Face(失敗時 ModelScope)から取得 |

## 手順

```bash
git clone <このリポジトリ> ACE-Step-1.5-fast
cd ACE-Step-1.5-fast
./setup.sh              # uv の導入 → .venv の作成(uv.lock どおり)→ GPU 確認 → チェックポイント取得
./run_api_server_fast.sh
```

ブラウザで `http://localhost:8001/ui/` を開けば評価用 UI、API は `http://localhost:8001`(仕様は [API.md](API.md))。

- `./setup.sh --large`: LM 4B も取得(40GB 以上の GPU で `ACESTEP_GPU_PROFILE=large` を使う場合)。
- `./setup.sh --12gb`: XL を取らず 2B turbo + 2B base + LM 0.6B(基本セットの LM 1.7B 込みで約 15GB。12GB の GPU で `ACESTEP_GPU_PROFILE=12gb` を使う場合)。
- `./setup.sh --no-models`: 環境だけ作る(チェックポイントは初回起動時に自動取得される)。
- 何度実行しても安全(取得済みのモデルは飛ばす)。

## Windows

前提: Windows 10 / 11(64 bit)、NVIDIA ドライバ(CUDA 12.8 以上対応)、メモリ 32GB 以上、ディスク 80GB。Python・CUDA Toolkit・Visual Studio は不要。
コマンドプロンプトで上から順に実行する。

```bat
winget install --id Git.Git -e
git clone https://github.com/animede/ACE-Step-1.5-fast.git
cd ACE-Step-1.5-fast
setup.bat
run_api_server_fast.bat
```

```bat
close_api_server.bat
```

- 起動後は `http://localhost:8001/ui/`。別ポートの停止は `close_api_server.bat 8002`。
- `setup.bat`: uv の導入(winget があれば `winget install --id astral-sh.uv -e`、無ければ公式の PowerShell インストーラ)→ Python 3.11 と `.venv` の作成(`uv sync --frozen`)→ GPU 確認 → モデル取得。`setup.bat --large` で LM 4B も、`setup.bat --12gb` で 2B だけ、`setup.bat --no-models` で環境だけ。再実行は安全。直後に `uv` が見つからないと言われたら、新しいコマンドプロンプトでもう一度。
- Git を入れない場合は GitHub の「Code → Download ZIP」を展開し、そのフォルダで `setup.bat` 以降を実行。
- 設定は起動前に `set ACESTEP_GPU_PROFILE=large` のように指定(変数は 起動コマンド.md の表と同じ)。`PYTORCH_ALLOC_CONF` は Windows では付けていない。
- 実機での動作確認はまだ行っていないので、問題があれば Issue で報告を。

## 同梱アプリ Easy Music の起動と LLM の指定

`./setup.sh` / `setup.bat` は Easy Music(`easy-music/`)の環境も作ります。API サーバを起動したまま、別のターミナルで起動します(引数はポート)。

```bash
cd ACE-Step-1.5-fast
./run_easy_music.sh 8888
```

Windows は `run_easy_music.bat 8888`。ブラウザで `http://localhost:8888`。

歌詞生成に使う LLM は OpenAI 互換 API(Ollama、llama.cpp server、LM Studio、vLLM、OpenAI など)で、既定は `http://localhost:11434/v1`(Ollama)。接続できないときは CPU 上の GGUF(Gemma 4 E4B Q4_0、約 4.4GB、初回に自動ダウンロード、thinking OFF)に自動でフォールバックします。この同梱 LLM には `llama-cpp-python>=0.3.34` のソースビルドが必要です(Linux: gcc/g++ + cmake、Windows: Visual Studio Build Tools の C++ + CMake)。指定方法は 2 つで、起動引数 > `easy-music/.env` > 既定値 の順に優先されます。

起動引数:

```bash
./run_easy_music.sh 8888 --llm-url http://LLM_HOST:11434/v1 --llm-model gemma3:latest
```

(ホストとポートを分けるなら `--llm-host LLM_HOST --llm-port 11434`。Windows は `run_easy_music.bat 8888 --llm-url ...`)

`easy-music/.env`(恒久的な設定。認証不要のサーバなら `OPENAI_API_KEY` はダミーのまま):

```
OPENAI_BASE_URL=http://LLM_HOST:11434/v1
OPENAI_API_KEY=YOUR_KEY
OPENAI_CHAT_MODEL=gemma3:latest
```

同じ要領で、ACE サーバは `--ace-url` / `ACE_STEP_API_URL`(既定 `http://127.0.0.1:8001`)、背景画像サーバ(任意)は `--img-url` / `IMAGE_GENERATION_URL` で指定できます。

## 手で行う場合

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # uv
uv sync --frozen                                    # torch 2.10 (cu128) / flash-attn(ビルド済み wheel)/ その他を .venv へ
uv run acestep-download --model acestep-v15-xl-sft  # 主要モデル一式(VAE・テキストエンコーダ・2B turbo・LM 1.7B)も一緒に取得
uv run acestep-download --model acestep-v15-xl-turbo
uv run acestep-download --model acestep-5Hz-lm-0.6B
```

チェックポイントの置き場を変えるときは `.env` の `ACESTEP_CHECKPOINTS_DIR`(雛形は `.env.example`)。

## 困ったとき

| 症状 | 対処 |
|---|---|
| `cuda available: False` | ドライバが CUDA 12.8 に対応しているか(`nvidia-smi` の CUDA Version が 12.8 以上)を確認 |
| `flash_attn not importable` | サーバは SDPA に自動で切り替わる(1〜2 割遅い)。`uv sync --frozen` をやり直す |
| 起動時に OOM | 16GB では `ACESTEP_GPU_PROFILE=16gb`(既定)のままにし、他のプロセスが GPU を使っていないか確認 |
| 生成が 600 秒で打ち切られる | `.env` の `ACESTEP_GENERATION_TIMEOUT` を延ばす |

起動の設定(プロファイル・環境変数)は [起動コマンド.md](../../起動コマンド.md)、計測の記録は [低VRAM_XL_プローブ_20261116.md](低VRAM_XL_プローブ_20261116.md)。
