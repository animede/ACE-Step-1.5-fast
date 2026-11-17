# ACE-Step 1.5 Text-to-Audio 独立Webアプリ仕様書

## 概要

本ドキュメントは、ComfyUIに依存せず、ACE-Step 1.5のREST APIを直接使用して音楽生成Webアプリを構築するための技術仕様を記載します。

---

## 目次

1. [ACE-Step 1.5 アーキテクチャ](#1-ace-step-15-アーキテクチャ)
2. [GUI仕様](#2-gui仕様)
3. [入力パラメータ仕様](#3-入力パラメータ仕様)
   - [3.7 利用可能なAPIオプション一覧](#37-利用可能なapiオプション一覧)
4. [ACE-Step 1.5 推論パイプライン](#4-ace-step-15-推論パイプライン)
5. [LLM連携仕様（詳細）](#5-llm連携仕様詳細)
6. [モデル仕様](#6-モデル仕様)
7. [パフォーマンス](#7-パフォーマンス)
8. [REST API 詳細仕様](#8-rest-api-詳細仕様)
9. [独立Webアプリ実装ガイド](#9-独立webアプリ実装ガイド)
10. [参考リンク](#11-参考リンク)

---

## 1. ACE-Step 1.5 アーキテクチャ

### 1.1 システム構成図

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Web UI)                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ テーマ入力    │  │ 歌詞エディタ  │  │ 音楽メタデータ │          │
│  │ (シナリオ)    │  │ (Lyrics)     │  │ BPM/Key/拍子  │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│  ┌──────▼─────────────────▼─────────────────▼───────┐          │
│  │               API Client Layer                    │          │
│  └──────────────────────┬───────────────────────────┘          │
└─────────────────────────┼───────────────────────────────────────┘
                          │ HTTP/WebSocket
┌─────────────────────────▼───────────────────────────────────────┐
│                     Backend API Server                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ LLM Service │  │ Job Manager │  │ File Storage│             │
│  │ (作詞/タグ) │  │ (非同期処理) │  │ (音声保存)  │             │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┘             │
└─────────┼────────────────┼──────────────────────────────────────┘
          │                │
┌─────────▼────────────────▼──────────────────────────────────────┐
│                  ACE-Step 1.5 Inference Engine                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ace_step_1.5_turbo_aio.safetensors (All-in-One Model)   │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ • DiT (Diffusion Transformer)                           │   │
│  │ • DCAE (Deep Compression AutoEncoder)                   │   │
│  │ • Text Encoder (多言語対応)                              │   │
│  │ • VAE Decoder (Audio)                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 処理フロー

```
┌─────────────────────────────────────────────────────────────────┐
│                    完全自動生成フロー                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ① テーマ入力        ② AI作詞           ③ タグ生成              │
│  ┌──────────┐       ┌──────────┐       ┌──────────┐            │
│  │ "魔法少女 │ ───▶ │ LLM API  │ ───▶ │ LLM API  │            │
│  │ の物語"  │       │ 歌詞生成  │       │ ジャンル/ │            │
│  └──────────┘       └────┬─────┘       │ タグ推奨  │            │
│                          │              └────┬─────┘            │
│                          ▼                   ▼                   │
│                     ┌────────────────────────────┐              │
│                     │        歌詞 + タグ          │              │
│                     │ + BPM/Key/拍子 (自動抽出)   │              │
│                     └────────────┬───────────────┘              │
│                                  ▼                               │
│                     ④ ACE-Step 1.5 推論                         │
│                     ┌──────────────────────┐                    │
│                     │ TextEncode → Sample  │                    │
│                     │ → VAE Decode → MP3   │                    │
│                     └──────────┬───────────┘                    │
│                                ▼                                 │
│                     ⑤ 音声出力 (WAV/MP3)                        │
│                     ┌──────────────────────┐                    │
│                     │ 🎵 generated_music.mp3│                    │
│                     └──────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. GUI仕様

### 2.1 画面レイアウト

```
┌─────────────────────────────────────────────────────────────────┐
│ 🎵 ACE-Step 1.5 Music Generator                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │ 💡 テーマ / シナリオ      │  │ 🎵 タグ / ジャンル            │  │
│  │ ┌─────────────────────┐ │  │ ┌─────────────────────────┐ │  │
│  │ │ 夜空を見上げて願いを  │ │  │ │ Neo-Soul, warm, organic, │ │  │
│  │ │ 込める魔法少女の物語  │ │  │ │ female voice, emotional  │ │  │
│  │ └─────────────────────┘ │  │ └─────────────────────────┘ │  │
│  │                         │  │                             │  │
│  │ 📝 歌詞 / Lyrics        │  │ ┌─────────────────────────┐ │  │
│  │ ┌─────────────────────┐ │  │ │ 🎹 BPM     │ 120        │ │  │
│  │ │ [intro]             │ │  │ ├───────────┼────────────┤ │  │
│  │ │ [inst]              │ │  │ │ 🎼 拍子    │ 4/4        │ │  │
│  │ │                     │ │  │ ├───────────┼────────────┤ │  │
│  │ │ [verse]             │ │  │ │ 🎵 調      │ A minor    │ │  │
│  │ │ 星の海に浮かぶ願いが │ │  │ └───────────┴────────────┘ │  │
│  │ │ 夜空を駆け巡る      │ │  │                             │  │
│  │ │ ...                 │ │  │ ┌─────────────────────────┐ │  │
│  │ └─────────────────────┘ │  │ │ 🎧 生成結果               │ │  │
│  │                         │  │ │  ▶ ────────── 00:00    │ │  │
│  │                         │  │ └─────────────────────────┘ │  │
│  └─────────────────────────┘  └─────────────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 🌐 言語 │ ⏱️ 長さ │ 🔄 Steps │ 🎯 CFG │ 🎲 Seed        │   │
│  │  [ja]  │ [120秒] │   [8]   │ [1.0] │ [random]       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  │
│  │🎼 作詞 │  │🎵 タグ │  │✨ 生成 │  │🚀 AUTO │  │🔄 Reset│  │
│  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 入力フィールド仕様

| フィールド | タイプ | 必須 | 説明 | デフォルト | 範囲/オプション |
|-----------|--------|------|------|-----------|----------------|
| テーマ/シナリオ | text | ○ (AI作詞時) | 曲のテーマ・ストーリー | - | 自由入力 |
| 歌詞 (Lyrics) | textarea | △ | 歌詞テキスト（構造タグ付き） | - | [verse],[chorus]等 |
| タグ/ジャンル | textarea | ○ | ジャンル、スタイル、楽器等 | - | 自由入力 |
| BPM | number | ○ | テンポ（1分間の拍数） | 120 | 10-300 |
| 拍子 | select | ○ | 拍子記号 | 4 | 2, 3, 4, 6 |
| 調 (Key) | select | ○ | 音楽のキー | C major | C-B × major/minor |
| 言語 | select | ○ | 歌詞の言語 | ja | en,ja,zh,ko,es,fr,de,pt,ru,it... |
| 長さ (秒) | number | ○ | 生成する音楽の長さ | 30 | 10-600 |
| Steps | number | ○ | 推論ステップ数 | 8 | 4-50 |
| CFG | number | ○ | Classifier-Free Guidance | 1.0 | 0.5-5.0 |
| Seed | number | - | 乱数シード | random | 0-2^64 |
| LM Temperature | number | - | LM温度（多様性制御） | 0.85 | 0.0-2.0 |
| LM CFG Scale | number | - | LM CFGスケール | 2.5 | 0.0-20.0 |
| LM Top-P | number | - | 核サンプリング | 0.9 | 0.0-1.0 |
| CoT Caption | checkbox | - | Chain-of-Thoughtキャプション強化 | ON | ON/OFF |
| CoT Language | checkbox | - | Chain-of-Thought言語判定 | ON | ON/OFF |

### 2.4 カバーモード入力フィールド

| フィールド | タイプ | 必須 | 説明 | デフォルト |
|-----------|--------|------|------|----------|
| ソース音声 | file | ○ | 参照する音声ファイル (20〜45秒推奨) | - |
| カバー強度 | range | - | 声質参照の強さ | 0.80 (0.0-1.0) |

### 2.5 リペイントモード入力フィールド

| フィールド | タイプ | 必須 | 説明 | デフォルト |
|-----------|--------|------|------|----------|
| リペイント対象音声 | file | ○ | 再生成対象の音声ファイル | - |
| 開始位置 (秒) | number | ○ | リペイント開始秒 | 0.0 |
| 終了位置 (秒) | number | ○ | リペイント終了秒 | 30.0 |

### 2.3 歌詞構造タグ

ACE-Step 1.5が認識する構造タグ：

| タグ | 説明 | 推奨時間 |
|------|------|---------|
| `[intro]` | イントロ | 5-10秒 |
| `[verse]` | Aメロ/Bメロ | 15-25秒 |
| `[chorus]` | サビ | 20-30秒 |
| `[bridge]` | ブリッジ | 10-15秒 |
| `[outro]` | アウトロ | 5-10秒 |
| `[inst]` | インストゥルメンタル（歌なし） | 任意 |
| `[pre-chorus]` | プリコーラス | 10-15秒 |

---

## 3. REST API仕様

### 3.1 API エンドポイント一覧

| エンドポイント | メソッド | 説明 |
|---------------|---------|------|
| `/api/generate` | POST | 音楽生成ジョブを投入 |
| `/api/utility` | POST | ユーティリティ処理（作詞/タグ生成） |
| `/api/job/{job_id}` | GET | ジョブステータス取得 |
| `/ws/progress/{job_id}` | WebSocket | リアルタイム進捗通知 |
| `/output/{filename}` | GET | 生成された音声ファイル取得 |

### 3.2 音楽生成 API

#### リクエスト

```http
POST /api/generate
Content-Type: application/json

{
  "workflow": "ace_step_1_5_t2a",
  "tags": "Neo-Soul, warm, organic, female voice, emotional, live instrumentation",
  "lyrics": "[intro]\n[inst]\n\n[verse]\n星の海に浮かぶ願いが\n夜空を駆け巡る\n...",
  "duration": 120,
  "bpm": 120,
  "timesignature": "4",
  "language": "ja",
  "keyscale": "A minor",
  "steps": 8,
  "cfg": 1.0,
  "seed": 12345
}
```

#### パラメータ詳細

| パラメータ | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| workflow | string | ○ | `"ace_step_1_5_t2a"` 固定 |
| tags | string | ○ | ジャンル・スタイル・楽器等 |
| lyrics | string | △ | 歌詞（空の場合はインストゥルメンタル） |
| duration | int | ○ | 生成時間（秒） |
| bpm | int | ○ | テンポ (10-300) |
| timesignature | string | ○ | 拍子 ("2", "3", "4", "6") |
| language | string | ○ | 言語コード (en, ja, zh, ko, etc.) |
| keyscale | string | ○ | 調 ("C major", "A minor", etc.) |
| steps | int | - | 推論ステップ (デフォルト: 8) |
| cfg | float | - | CFG スケール (デフォルト: 1.0) |
| seed | int | - | 乱数シード (省略時はランダム) |

#### レスポンス

```json
{
  "job_id": "abc12345-6789-def0-1234-567890abcdef",
  "status": "queued",
  "message": "Job queued"
}
```

### 3.3 作詞 API (LLM)

#### リクエスト

```http
POST /api/utility
Content-Type: application/json

{
  "workflow": "lyrics_generate",
  "scenario": "夜空を見上げて願いを込める魔法少女の物語",
  "genre": "pop, female voice, emotional",
  "language": "Japanese"
}
```

#### レスポンス

```json
{
  "job_id": "lyrics-abc123",
  "status": "queued"
}
```

#### ジョブ完了時の結果

```json
{
  "status": "completed",
  "result": {
    "lyrics": "[intro]\n[inst]\n\n[verse]\n星の海に浮かぶ願いが...",
    "recommended_duration": 120,
    "parts_timing": {
      "intro": 5,
      "verse1": 25,
      "chorus1": 30,
      "verse2": 25,
      "chorus2": 30,
      "outro": 5
    }
  }
}
```

### 3.4 タグ生成 API (LLM)

#### リクエスト

```http
POST /api/utility
Content-Type: application/json

{
  "workflow": "lyrics_to_tags",
  "lyrics": "[verse]\n星の海に浮かぶ願いが\n夜空を駆け巡る..."
}
```

#### レスポンス（ジョブ完了時）

```json
{
  "status": "completed",
  "result": {
    "genre": "pop, j-pop, female voice, emotional ballad",
    "tags": "piano, strings, dreamy, hopeful, key: A minor, bpm: 90, slow tempo"
  }
}
```

### 3.5 ジョブステータス API

#### リクエスト

```http
GET /api/job/{job_id}
```

#### レスポンス

```json
{
  "job_id": "abc12345",
  "status": "completed",
  "progress": 100,
  "message": "Generation complete",
  "result": {
    "audio_files": [
      "/output/audio/ComfyUI_00001.mp3"
    ]
  }
}
```

### 3.6 WebSocket 進捗通知

```javascript
const ws = new WebSocket('ws://localhost:9999/ws/progress/abc12345');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // {
  //   "type": "progress",
  //   "progress": 45,
  //   "message": "Sampling step 4/8..."
  // }
  updateProgressBar(data.progress, data.message);
};
```

---

## 3.7 利用可能なAPIオプション一覧

### 3.7.1 TextEncodeAceStepAudio1.5 オプション

ACE-Step 1.5のテキストエンコーディングで使用可能なすべてのオプション：

| パラメータ | 型 | 範囲 | デフォルト | 説明 | フロント推奨 |
|-----------|-----|------|----------|------|------------|
| tags | string | - | "" | ジャンル、スタイル、楽器、ムード等 | ✅ テキストエリア |
| lyrics | string | - | "" | 歌詞（構造タグ付き） | ✅ テキストエリア |
| seed | int | 0 - 2^64-1 | 0 | 再現性のためのシード | ✅ 数値入力 + ランダムボタン |
| bpm | int | 10 - 300 | 120 | テンポ (Beats Per Minute) | ✅ スライダー + 数値入力 |
| duration | float | 0.0 - 2000.0 | 120.0 | 生成時間（秒） | ✅ スライダー (推奨: 30-180秒) |
| timesignature | enum | "2","3","4","6" | "4" | 拍子 | ✅ セレクト |
| language | enum | 23言語 | "en" | 歌詞の言語 | ✅ セレクト |
| keyscale | enum | 34種類 | "C major" | 調（キー） | ✅ セレクト |

### 3.7.2 KSampler オプション（推論品質調整）

| パラメータ | 型 | 範囲 | Turbo推奨 | 説明 | フロント推奨 |
|-----------|-----|------|----------|------|------------|
| steps | int | 1 - 100+ | **8** | 推論ステップ数 | △ 上級者向け |
| cfg | float | 0.0 - 30.0 | **1.0** | CFGスケール（ガイダンス強度） | △ 上級者向け |
| sampler_name | enum | 40+種類 | **"euler"** | サンプラーアルゴリズム | △ 上級者向け |
| scheduler | enum | 9種類 | **"simple"** | スケジューラー | △ 上級者向け |
| denoise | float | 0.0 - 1.0 | **1.0** | デノイズ強度 | ✗ 固定推奨 |

### 3.7.3 ModelSamplingAuraFlow オプション

| パラメータ | 型 | 範囲 | Turbo推奨 | 説明 | フロント推奨 |
|-----------|-----|------|----------|------|------------|
| shift | float | 0.0 - 100.0 | **3.0** | サンプリングシフト | ✗ 固定推奨 |

### 3.7.4 サンプラー一覧

```python
SAMPLER_NAMES = [
    # 基本サンプラー
    "euler",                    # ★推奨（Turboデフォルト）
    "euler_ancestral",          # ノイズ追加版
    "heun",                     # 2次精度
    "dpm_2",                    # DPM-Solver 2次
    
    # 高速サンプラー
    "lcm",                      # Latent Consistency Model用
    "ddim",                     # Denoising Diffusion IM
    
    # 高品質サンプラー
    "dpmpp_2m",                 # DPM++ 2M
    "dpmpp_2m_sde",             # DPM++ 2M SDE
    "dpmpp_3m_sde",             # DPM++ 3M SDE
    "uni_pc",                   # UniPC
    
    # その他（上級者向け）
    "euler_cfg_pp", "euler_ancestral_cfg_pp", "heunpp2",
    "exp_heun_2_x0", "exp_heun_2_x0_sde", "dpm_2_ancestral",
    "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m_cfg_pp", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu", "dpmpp_3m_sde_gpu", "ddpm",
    "ipndm", "ipndm_v", "deis", "res_multistep",
    "res_multistep_cfg_pp", "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp", "gradient_estimation",
    "gradient_estimation_cfg_pp", "er_sde", "seeds_2", "seeds_3",
    "sa_solver", "sa_solver_pece", "uni_pc_bh2"
]
```

### 3.7.5 スケジューラー一覧

```python
SCHEDULER_NAMES = [
    "simple",           # ★推奨（Turboデフォルト）
    "normal",           # 標準ノイズスケジュール
    "karras",           # Karrasスケジュール（滑らか）
    "exponential",      # 指数スケジュール
    "sgm_uniform",      # SGM均一
    "ddim_uniform",     # DDIM均一
    "beta",             # ベータスケジュール
    "linear_quadratic", # 線形二次
    "kl_optimal",       # KL最適化
]
```

### 3.7.6 言語コード一覧

```python
SUPPORTED_LANGUAGES = {
    "en": "English",      # ★デフォルト
    "ja": "Japanese",     # 日本語
    "zh": "Chinese",      # 中国語
    "ko": "Korean",       # 韓国語
    "es": "Spanish",      # スペイン語
    "de": "German",       # ドイツ語
    "fr": "French",       # フランス語
    "pt": "Portuguese",   # ポルトガル語
    "ru": "Russian",      # ロシア語
    "it": "Italian",      # イタリア語
    "nl": "Dutch",        # オランダ語
    "pl": "Polish",       # ポーランド語
    "tr": "Turkish",      # トルコ語
    "vi": "Vietnamese",   # ベトナム語
    "cs": "Czech",        # チェコ語
    "fa": "Persian",      # ペルシャ語
    "id": "Indonesian",   # インドネシア語
    "uk": "Ukrainian",    # ウクライナ語
    "hu": "Hungarian",    # ハンガリー語
    "ar": "Arabic",       # アラビア語
    "sv": "Swedish",      # スウェーデン語
    "ro": "Romanian",     # ルーマニア語
    "el": "Greek",        # ギリシャ語
}
```

### 3.7.7 キースケール（調）一覧

```python
KEYSCALES = [
    # メジャー（長調）
    "C major", "C# major", "Db major", "D major", "D# major",
    "Eb major", "E major", "F major", "F# major", "Gb major",
    "G major", "G# major", "Ab major", "A major", "A# major",
    "Bb major", "B major",
    
    # マイナー（短調）
    "C minor", "C# minor", "Db minor", "D minor", "D# minor",
    "Eb minor", "E minor", "F minor", "F# minor", "Gb minor",
    "G minor", "G# minor", "Ab minor", "A minor", "A# minor",
    "Bb minor", "B minor",
]
```

### 3.7.8 フロントエンド推奨UI構成

```
┌─────────────────────────────────────────────────────────────────┐
│ 🎵 ACE-Step 1.5 Music Generator                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  【基本設定】 ★ すべてのユーザーに表示                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ テーマ/シナリオ: [テキストエリア]              [🎼 AI作詞]  ││
│  │ 歌詞:           [大きなテキストエリア]                       ││
│  │ タグ/ジャンル:  [テキストエリア]               [🎵 タグ生成] ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │ 時間: [==●=====] 120秒    BPM: [==●===] 120                 ││
│  │ 拍子: [4/4 ▼]   言語: [日本語 ▼]   調: [A minor ▼]          ││
│  │ シード: [12345] [🎲 ランダム]                                ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  【上級設定】 ▶ 折りたたみ（デフォルト非表示）                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ ステップ数: [8]  CFG: [1.0]                                  ││
│  │ サンプラー: [euler ▼]  スケジューラー: [simple ▼]            ││
│  │ ※ Turbo最適値が設定済み。変更は品質に影響する可能性があります ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  [✨ Generate Music]                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.7.9 パラメータ検証ルール

```javascript
const VALIDATION_RULES = {
    duration: {
        min: 10,
        max: 300,      // 実用的な上限（長すぎるとVRAM不足）
        recommended: { min: 30, max: 180 }
    },
    bpm: {
        min: 10,
        max: 300,
        recommended: { min: 60, max: 200 }
    },
    steps: {
        min: 1,
        max: 50,       // Turboは8で十分
        turbo_default: 8
    },
    cfg: {
        min: 0.0,
        max: 10.0,     // 高すぎると品質低下
        turbo_default: 1.0
    },
    seed: {
        min: 0,
        max: Number.MAX_SAFE_INTEGER,  // JavaScriptの安全な最大整数
        random_value: -1  // -1でサーバー側でランダム生成
    }
};

function validateParams(params) {
    const errors = [];
    
    if (params.duration < 10 || params.duration > 2000) {
        errors.push("Duration must be between 10 and 2000 seconds");
    }
    if (params.bpm < 10 || params.bpm > 300) {
        errors.push("BPM must be between 10 and 300");
    }
    if (!KEYSCALES.includes(params.keyscale)) {
        errors.push("Invalid keyscale");
    }
    if (!["2", "3", "4", "6"].includes(params.timesignature)) {
        errors.push("Timesignature must be 2, 3, 4, or 6");
    }
    
    return errors;
}
```

### 3.7.10 ReferenceTimbreAudio（音色参照）※実験的

ACE-Step 1.5では、既存の音声から音色を参照して生成する機能があります（実験的機能）：

```python
# 参照音声からの音色抽出
class ReferenceTimbreAudio:
    """
    既存の音声ラテントを参照として、同様の音色で新しい音楽を生成
    
    使用例:
    1. 参照音声をVAEエンコード → latent取得
    2. ReferenceTimbreAudioでconditioningに付加
    3. 通常のサンプリングで生成
    
    注意: experimental（実験的）機能のため、結果は保証されません
    """
    inputs = {
        "conditioning": Conditioning,  # 必須: テキストエンコード結果
        "latent": Latent | None,       # オプション: 参照音声のラテント
    }
```

**フロントエンド実装推奨:**
- 上級設定内に配置
- 「参照音声をアップロード」オプションとして提供
- ファイルアップロード → VAEエンコード → latent取得の処理が必要
- 現状は実験的機能のため、非表示またはベータ表示を推奨

---

## 4. ACE-Step 1.5 推論パイプライン

### 4.1 ComfyUI非依存の推論フロー

ComfyUIワークフローを使わず、直接ACE-Stepを呼び出す場合のPythonコード例：

```python
import torch
from acestep import ACEStepPipeline

# パイプライン初期化
pipeline = ACEStepPipeline.from_pretrained(
    "ace_step_1.5_turbo_aio.safetensors",
    device="cuda",
    dtype=torch.bfloat16
)

# 生成パラメータ
params = {
    "tags": "Neo-Soul, warm, organic, female voice",
    "lyrics": "[verse]\n星の海に浮かぶ願いが...",
    "bpm": 120,
    "duration": 60.0,
    "timesignature": 4,
    "language": "ja",
    "keyscale": "A minor",
    "seed": 12345,
    "steps": 8,
    "cfg": 1.0,
}

# 推論実行
audio = pipeline.generate(**params)

# 保存
pipeline.save_audio(audio, "output.mp3", format="mp3")
```

### 4.2 推論パイプラインの内部処理

```
┌─────────────────────────────────────────────────────────────────┐
│                   ACE-Step 1.5 推論パイプライン                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: Text Encoding                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ TextEncodeAceStepAudio1.5                               │   │
│  │ • tags → CLIP Tokenize                                  │   │
│  │ • lyrics → Lyrics Tokenize                              │   │
│  │ • metadata (bpm, key, lang, timesig) → Embedding        │   │
│  │ → Conditioning (positive/negative)                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ▼                                   │
│  Step 2: Latent Initialization                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ EmptyAceStep1.5LatentAudio                              │   │
│  │ • seconds → latent_length = round(seconds * 48000 / 1920)│   │
│  │ • shape: [batch, 64, latent_length]                     │   │
│  │ • dtype: float32, device: cuda                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ▼                                   │
│  Step 3: Model Sampling                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ModelSamplingAuraFlow (shift=3)                         │   │
│  │ KSampler:                                               │   │
│  │ • sampler_name: "euler"                                 │   │
│  │ • scheduler: "simple"                                   │   │
│  │ • steps: 8 (turbo mode)                                 │   │
│  │ • cfg: 1.0                                              │   │
│  │ • denoise: 1.0                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ▼                                   │
│  Step 4: VAE Decode                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ VAEDecodeAudio                                          │   │
│  │ • latent → waveform (48kHz)                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ▼                                   │
│  Step 5: Save Audio                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ SaveAudioMP3 (quality: V0)                              │   │
│  │ • waveform → MP3 file                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. LLM連携仕様（詳細）

### 5.1 LLMクライアント設定

OpenAI互換APIを使用します（OpenAI、Ollama、vLLM等に対応）。

#### 環境変数設定

```bash
# OpenAI互換APIエンドポイント
export OPENAI_BASE_URL="http://localhost:11434/v1"  # Ollama例
export OPENAI_API_KEY="your-api-key"

# 使用するモデル（任意）
export OPENAI_CHAT_MODEL="gemma3:latest"  # デフォルト
```

#### Pythonクライアント初期化

```python
from openai import AsyncOpenAI

def get_openai_client():
    return AsyncOpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY"),
    )
```

#### chat_req関数

```python
async def chat_req(
    client: AsyncOpenAI,
    user_msg: str,
    role: str,  # system role
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """
    OpenAI互換APIにチャットリクエストを送信
    
    Args:
        client: AsyncOpenAI クライアント
        user_msg: ユーザーメッセージ
        role: システムロール（プロンプト）
        model: モデル名（デフォルト: OPENAI_CHAT_MODEL環境変数 or "gemma3:latest"）
        max_tokens: 最大トークン数
        temperature: 温度パラメータ
    
    Returns:
        str: LLMの応答テキスト
    """
    messages = [
        {"role": "system", "content": role},
        {"role": "user", "content": user_msg}
    ]
    
    effective_model = model or os.environ.get("OPENAI_CHAT_MODEL", "gemma3:latest")
    kwargs = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature

    completion = await client.chat.completions.create(
        model=effective_model,
        messages=messages,
        **kwargs,
    )
    return completion.choices[0].message.content
```

---

### 5.2 作詞API (lyrics_generate) 完全仕様

#### 5.2.1 システムプロンプト（完全版）

```python
LYRICS_GENERATE_SYSTEM_PROMPT = """You are a professional lyricist who creates song lyrics for AI music generation (ACE-Step).

CRITICAL OUTPUT RULES:
1. Start with a JSON metadata line, then output ONLY the lyrics with structure tags
2. Use structure tags: [intro], [verse], [chorus], [bridge], [outro], [inst] (for instrumental)
3. Do NOT include timestamps like (0:00-0:05) - ACE-Step does not use them
4. Do NOT include romanization in parentheses - write in the requested language ONLY
5. Write lyrics line by line, each line on its own

REQUIRED FIRST LINE FORMAT (JSON):
{"recommended_duration": 90, "parts": {"intro": 5, "verse1": 20, "chorus1": 25, "verse2": 20, "chorus2": 25, "outro": 5}}

- recommended_duration: Total song length in seconds (typically 60-180)
- parts: Estimated duration for each part in seconds

STRUCTURE GUIDELINES:
- [intro]: Keep short (1-2 lines) or use [inst] for instrumental (5-10 sec)
- [verse]: 4-6 lines per verse (15-25 sec)
- [chorus]: 4-6 lines, catchy and memorable (20-30 sec)
- [bridge]: Optional, 2-4 lines (10-15 sec)
- [outro]: Short ending, 1-2 lines or [inst] (5-10 sec)

STYLE GUIDELINES:
- Match the mood and genre specified
- Use natural, singable phrases
- If Japanese: Write only in Japanese (kanji/hiragana/katakana mix)
- If English: Write only in English
- Keep vocabulary natural for singing

EXAMPLE OUTPUT:
{"recommended_duration": 120, "parts": {"intro": 5, "verse1": 25, "chorus1": 30, "verse2": 25, "chorus2": 30, "outro": 5}}
[intro]
[inst]

[verse]
夜空に浮かぶ星を見つめて
あの日の約束を思い出す

[chorus]
君がいない世界で
僕は歩き続ける
いつかまた会える日まで
この歌を歌い続ける

[outro]
[inst]"""
```

#### 5.2.2 ユーザープロンプト

```python
def build_lyrics_user_prompt(scenario: str, genre: str, language: str) -> str:
    return f"""Write song lyrics for:
THEME: {scenario}
GENRE: {genre if genre else "pop"}
LANGUAGE: {language}

IMPORTANT: Start with the JSON metadata line containing recommended_duration and parts timing, then output the lyrics with structure tags."""
```

#### 5.2.3 レスポンスパース処理

```python
import re
import json

def parse_lyrics_response(response: str) -> dict:
    """
    LLMレスポンスから歌詞とメタデータを抽出
    
    Returns:
        {
            "lyrics": str,           # 歌詞テキスト
            "recommended_duration": int,  # 推奨時間（秒）
            "parts_timing": dict     # パート別時間
        }
    """
    recommended_duration = None
    parts_timing = None
    lyrics_text = response
    
    # パターン1: ```json ... ``` コードブロック内
    code_block_pattern = r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```'
    code_match = re.search(code_block_pattern, response, re.DOTALL)
    
    if code_match:
        json_str = code_match.group(1).strip()
        try:
            metadata = json.loads(json_str)
            recommended_duration = metadata.get('recommended_duration')
            parts_timing = metadata.get('parts')
            lyrics_text = re.sub(code_block_pattern, '', response, count=1).strip()
        except json.JSONDecodeError:
            pass
    else:
        # パターン2: 最初の行がJSON
        lines = response.strip().split('\n')
        first_line = lines[0].strip()
        if first_line.startswith('```'):
            first_line = first_line.replace('```json', '').replace('```', '').strip()
        
        if first_line.startswith('{'):
            try:
                metadata = json.loads(first_line)
                recommended_duration = metadata.get('recommended_duration')
                parts_timing = metadata.get('parts')
                lyrics_text = '\n'.join(lines[1:]).strip()
            except json.JSONDecodeError:
                pass
    
    # クリーンアップ
    lyrics_text = re.sub(r'^```\w*\n?', '', lyrics_text)
    lyrics_text = re.sub(r'\n?```$', '', lyrics_text)
    lyrics_text = lyrics_text.strip()
    
    return {
        "lyrics": lyrics_text,
        "recommended_duration": recommended_duration,
        "parts_timing": parts_timing
    }
```

#### 5.2.4 完全なAPI処理フロー

```python
async def process_lyrics_generate(scenario: str, genre: str, language: str) -> dict:
    """
    歌詞生成の完全な処理フロー
    
    Args:
        scenario: テーマ/シナリオ
        genre: ジャンル（オプション）
        language: 言語 (Japanese, English, etc.)
    
    Returns:
        {
            "lyrics": str,
            "scenario": str,
            "genre": str,
            "language": str,
            "recommended_duration": int,
            "parts_timing": dict
        }
    """
    client = get_openai_client()
    
    system_role = LYRICS_GENERATE_SYSTEM_PROMPT
    user_content = build_lyrics_user_prompt(scenario, genre, language)
    
    response = await chat_req(client, user_content, system_role)
    
    parsed = parse_lyrics_response(response)
    
    return {
        "lyrics": parsed["lyrics"],
        "scenario": scenario,
        "genre": genre,
        "language": language,
        "recommended_duration": parsed["recommended_duration"],
        "parts_timing": parsed["parts_timing"]
    }
```

---

### 5.3 タグ生成API (lyrics_to_tags) 完全仕様

#### 5.3.1 システムプロンプト（完全版）

```python
TAGS_GENERATE_SYSTEM_PROMPT = """You are a music genre and style expert. Analyze song lyrics and suggest appropriate genre and tags for AI music generation (ACE-Step).

OUTPUT FORMAT (EXACTLY as shown):
GENRE: [comma-separated genre/style list]
TAGS: [comma-separated tags list]

GENRE EXAMPLES:
- pop, rock, jazz, electronic, classical, folk, r&b, hip-hop, metal, country
- female voice, male voice, duet, choir
- acoustic, electric, orchestral
- ballad, upbeat, dance

TAGS EXAMPLES:
- Mood: emotional, upbeat, melancholic, dreamy, energetic, peaceful, dark, hopeful
- Instruments: piano, guitar, drums, violin, synthesizer, bass, strings
- Tempo: slow tempo, fast tempo, 120 bpm
- Key: C major, A minor, etc.
- Style: a cappella, lo-fi, cinematic, ambient

Output ONLY the two lines. No explanations."""
```

#### 5.3.2 ユーザープロンプト

```python
def build_tags_user_prompt(lyrics: str) -> str:
    # 歌詞が長い場合は先頭2000文字に制限
    truncated_lyrics = lyrics[:2000]
    return f"""Analyze these lyrics and suggest genre and tags:

{truncated_lyrics}

Output ONLY:
GENRE: ...
TAGS: ..."""
```

#### 5.3.3 レスポンスパース処理

```python
def parse_tags_response(response: str) -> dict:
    """
    LLMレスポンスからジャンルとタグを抽出
    
    Returns:
        {
            "genre": str,  # "pop, rock, female voice"
            "tags": str    # "emotional, piano, key: A minor, bpm: 90"
        }
    """
    genre = ""
    tags = ""
    
    for line in response.strip().split('\n'):
        line = line.strip()
        if line.upper().startswith('GENRE:'):
            genre = line[6:].strip()
        elif line.upper().startswith('TAGS:'):
            tags = line[5:].strip()
    
    return {"genre": genre, "tags": tags}
```

#### 5.3.4 タグからメタデータ自動抽出（フロントエンド）

```javascript
/**
 * タグ文字列からBPMとKeyを自動抽出してフォームにセット
 * @param {string} tags - 生成されたタグ文字列
 * @param {HTMLElement} card - カード要素
 */
function extractAndSetMetadataFromTags(tags, card) {
    // Key抽出: "key: A minor" または "key: C major" パターン
    const keyMatch = tags.match(/key:\s*([A-Ga-g][#b]?\s*(?:major|minor))/i);
    if (keyMatch) {
        const extractedKey = keyMatch[1].trim();
        const keyscaleSelect = card.querySelector('.keyscale-select');
        if (keyscaleSelect) {
            // セレクトボックスのオプションと照合
            const normalizedKey = extractedKey.charAt(0).toUpperCase() + 
                                  extractedKey.slice(1).toLowerCase();
            const options = Array.from(keyscaleSelect.options);
            const matchingOption = options.find(opt => 
                opt.value.toLowerCase() === extractedKey.toLowerCase() ||
                opt.value.toLowerCase() === normalizedKey.toLowerCase()
            );
            if (matchingOption) {
                keyscaleSelect.value = matchingOption.value;
                console.log(`Auto-set keyscale to: ${matchingOption.value}`);
            }
        }
    }
    
    // BPM抽出: "bpm: 120" または "tempo: 128" パターン
    const bpmMatch = tags.match(/(?:bpm|tempo):\s*(\d+)/i);
    if (bpmMatch) {
        const extractedBpm = parseInt(bpmMatch[1]);
        const bpmInput = card.querySelector('.bpm-input');
        if (bpmInput && extractedBpm >= 10 && extractedBpm <= 300) {
            bpmInput.value = extractedBpm;
            console.log(`Auto-set BPM to: ${extractedBpm}`);
        }
    }
}
```

---

### 5.4 完全なLLM連携シーケンス図

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LLM連携シーケンス図                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌─────────────┐  │
│  │ Frontend   │     │ API Server │     │ LLM Server │     │ ACE-Step    │  │
│  │ (WebUI)    │     │ (FastAPI)  │     │ (OpenAI互換)│     │ (推論)       │  │
│  └─────┬──────┘     └─────┬──────┘     └─────┬──────┘     └──────┬──────┘  │
│        │                  │                  │                   │         │
│        │ ① 🎼 作詞ボタン  │                  │                   │         │
│        ├─────────────────►│                  │                   │         │
│        │ POST /api/utility│                  │                   │         │
│        │ {workflow:       │                  │                   │         │
│        │  "lyrics_generate",                 │                   │         │
│        │  scenario: "...",│                  │                   │         │
│        │  genre: "...",   │                  │                   │         │
│        │  language: "..."}│                  │                   │         │
│        │                  │                  │                   │         │
│        │                  │ ② chat_req()    │                   │         │
│        │                  ├─────────────────►│                   │         │
│        │                  │ system: LYRICS_  │                   │         │
│        │                  │ GENERATE_SYSTEM_ │                   │         │
│        │                  │ PROMPT           │                   │         │
│        │                  │ user: scenario+  │                   │         │
│        │                  │ genre+language   │                   │         │
│        │                  │                  │                   │         │
│        │                  │ ③ JSON + 歌詞   │                   │         │
│        │                  │◄─────────────────┤                   │         │
│        │                  │                  │                   │         │
│        │ ④ lyrics +      │                  │                   │         │
│        │   recommended_   │                  │                   │         │
│        │   duration       │                  │                   │         │
│        │◄─────────────────┤                  │                   │         │
│        │                  │                  │                   │         │
│        │ ⑤ 🎵 タグ生成   │                  │                   │         │
│        ├─────────────────►│                  │                   │         │
│        │ POST /api/utility│                  │                   │         │
│        │ {workflow:       │                  │                   │         │
│        │  "lyrics_to_tags",                  │                   │         │
│        │  lyrics: "..."}  │                  │                   │         │
│        │                  │                  │                   │         │
│        │                  │ ⑥ chat_req()    │                   │         │
│        │                  ├─────────────────►│                   │         │
│        │                  │ system: TAGS_    │                   │         │
│        │                  │ GENERATE_SYSTEM_ │                   │         │
│        │                  │ PROMPT           │                   │         │
│        │                  │ user: lyrics     │                   │         │
│        │                  │                  │                   │         │
│        │                  │ ⑦ GENRE: ...    │                   │         │
│        │                  │    TAGS: ...     │                   │         │
│        │                  │◄─────────────────┤                   │         │
│        │                  │                  │                   │         │
│        │ ⑧ genre + tags  │                  │                   │         │
│        │   (key: A minor, │                  │                   │         │
│        │    bpm: 120)     │                  │                   │         │
│        │◄─────────────────┤                  │                   │         │
│        │                  │                  │                   │         │
│        │ ⑨ 自動抽出:     │                  │                   │         │
│        │   Key → keyscale │                  │                   │         │
│        │   BPM → bpm      │                  │                   │         │
│        │   Duration →     │                  │                   │         │
│        │   duration       │                  │                   │         │
│        │                  │                  │                   │         │
│        │ ⑩ ✨ Generateボタン                │                   │         │
│        ├─────────────────►│                  │                   │         │
│        │ POST /api/generate                  │                   │         │
│        │ {workflow:       │                  │                   │         │
│        │  "ace_step_1_5_t2a",                │                   │         │
│        │  tags, lyrics,   │                  │                   │         │
│        │  bpm, keyscale, ...}                │                   │         │
│        │                  │                  │                   │         │
│        │                  │                  │ ⑪ 音楽生成       │         │
│        │                  │                  ├─────────────────►│         │
│        │                  │                  │                   │         │
│        │                  │                  │ ⑫ audio.mp3      │         │
│        │                  │                  │◄──────────────────┤         │
│        │                  │                  │                   │         │
│        │ ⑬ 完了通知 +    │                  │                   │         │
│        │   audio URL      │                  │                   │         │
│        │◄─────────────────┤                  │                   │         │
│        │                  │                  │                   │         │
└────────┴──────────────────┴──────────────────┴───────────────────┴─────────┘
```

---

### 5.5 推奨LLMモデル

| 用途 | 推奨モデル | 備考 |
|------|----------|------|
| 作詞 (日本語) | gemma-3-27b-it, qwen2.5-32b | 日本語の自然な歌詞生成 |
| 作詞 (英語) | gpt-4o-mini, claude-3.5-sonnet | 高品質な英語歌詞 |
| タグ生成 | gemma-3-12b-it, llama3.1-8b | 軽量でも十分な精度 |
| Ollama推奨 | gemma3:latest | ローカル実行で高速 |

---

### 5.6 エラーハンドリング

```python
async def process_with_retry(func, *args, max_retries=3, **kwargs):
    """
    LLM API呼び出しのリトライ処理
    """
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

---

## 6. モデル仕様

### 6.1 チェックポイント

| ファイル名 | サイズ | 説明 |
|-----------|-------|------|
| ace_step_1.5_turbo_aio.safetensors | ~4GB | All-in-One (DiT + VAE + TextEncoder) |

### 6.2 対応言語

```python
SUPPORTED_LANGUAGES = [
    "en",  # English
    "ja",  # Japanese
    "zh",  # Chinese
    "es",  # Spanish
    "de",  # German
    "fr",  # French
    "pt",  # Portuguese
    "ru",  # Russian
    "it",  # Italian
    "nl",  # Dutch
    "pl",  # Polish
    "tr",  # Turkish
    "vi",  # Vietnamese
    "cs",  # Czech
    "fa",  # Persian
    "id",  # Indonesian
    "ko",  # Korean
    "uk",  # Ukrainian
    "hu",  # Hungarian
    "ar",  # Arabic
    "sv",  # Swedish
    "ro",  # Romanian
    "el",  # Greek
]
```

### 6.3 対応キー（調）

```python
SUPPORTED_KEYS = [
    f"{root} {quality}"
    for quality in ["major", "minor"]
    for root in ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]
]
# 結果: ["C major", "C minor", "C# major", "C# minor", ...]
```

---

## 7. パフォーマンス

### 7.1 推論時間（参考値）

| GPU | Steps | Duration | 推論時間 | RTF |
|-----|-------|----------|---------|-----|
| RTX 4090 | 8 | 60秒 | ~1.7秒 | 34x |
| RTX 3090 | 8 | 60秒 | ~4.7秒 | 12x |
| A100 | 8 | 60秒 | ~2.2秒 | 27x |

※ RTF (Real-Time Factor): 値が高いほど高速

### 7.2 VRAM使用量

| 設定 | VRAM |
|------|------|
| 通常 | ~10GB |
| CPU Offload | ~8GB |
| bf16 | ~8GB |

---

## 8. REST API 詳細仕様

### 8.1 エンドポイント一覧

| メソッド | エンドポイント | 説明 |
|---------|---------------|------|
| POST | `/api/utility` | LLM処理（作詞・タグ生成） |
| POST | `/api/generate` | 音楽生成 |
| GET | `/ws/progress/{job_id}` | WebSocket進捗監視 |
| GET | `/output/{filename}` | 生成済みファイル取得 |

### 8.2 作詞API

**エンドポイント:** `POST /api/utility`

#### リクエスト

```json
{
  "workflow": "lyrics_generate",
  "scenario": "夜空を見上げて願いを込める魔法少女の物語",
  "genre": "pop, female voice, emotional",
  "language": "Japanese"
}
```

| フィールド | 型 | 必須 | 説明 |
|-----------|-----|-----|------|
| workflow | string | ✅ | `"lyrics_generate"` 固定 |
| scenario | string | ✅ | テーマ/シナリオ |
| genre | string | - | ジャンル（省略時: "pop"） |
| language | string | ✅ | 言語名 ("Japanese", "English", etc.) |

#### レスポンス

```json
{
  "status": "ok",
  "lyrics": "[intro]\n[inst]\n\n[verse]\n夜空に浮かぶ星を見つめて...",
  "scenario": "夜空を見上げて願いを込める魔法少女の物語",
  "genre": "pop, female voice, emotional",
  "language": "Japanese",
  "recommended_duration": 120,
  "parts_timing": {
    "intro": 5,
    "verse1": 25,
    "chorus1": 30,
    "verse2": 25,
    "chorus2": 30,
    "outro": 5
  }
}
```

### 8.3 タグ生成API

**エンドポイント:** `POST /api/utility`

#### リクエスト

```json
{
  "workflow": "lyrics_to_tags",
  "lyrics": "[verse]\n夜空に浮かぶ星を見つめて\nあの日の約束を思い出す\n..."
}
```

| フィールド | 型 | 必須 | 説明 |
|-----------|-----|-----|------|
| workflow | string | ✅ | `"lyrics_to_tags"` 固定 |
| lyrics | string | ✅ | 歌詞テキスト |

#### レスポンス

```json
{
  "status": "ok",
  "genre": "j-pop, anime, female voice, emotional ballad",
  "tags": "piano, strings, dreamy, hopeful, key: A minor, bpm: 90, slow tempo, cinematic"
}
```

### 8.4 音楽生成API

**エンドポイント:** `POST /api/generate`

#### リクエスト

```json
{
  "workflow": "ace_step_1_5_t2a",
  "tags": "j-pop, female voice, emotional, piano, strings",
  "lyrics": "[verse]\n夜空に浮かぶ星を見つめて\n...",
  "duration": 120,
  "bpm": 90,
  "timesignature": "4",
  "language": "ja",
  "keyscale": "A minor",
  "seed": -1
}
```

| フィールド | 型 | 範囲/値 | デフォルト | 説明 |
|-----------|-----|--------|----------|------|
| workflow | string | - | - | `"ace_step_1_5_t2a"` 固定 |
| tags | string | - | - | ジャンル/スタイルタグ |
| lyrics | string | - | "" | 歌詞（構造タグ付き） |
| duration | int | 10-2000 | 60 | 生成時間（秒） |
| bpm | int | 10-300 | 120 | テンポ |
| timesignature | string | "2","3","4","6" | "4" | 拍子 |
| language | string | 言語コード | "en" | 言語（歌詞解釈用） |
| keyscale | string | "C major"等 | "C major" | 調 |
| seed | int | -1 or >= 0 | -1 | シード（-1=ランダム） |

#### レスポンス（即時）

```json
{
  "status": "queued",
  "job_id": "abc123-def456-...",
  "message": "Generation queued"
}
```

### 8.5 WebSocket進捗監視

**エンドポイント:** `ws://{host}/ws/progress/{job_id}`

#### メッセージ形式

```json
// 進捗更新
{
  "type": "progress",
  "step": 4,
  "total_steps": 8,
  "percent": 50
}

// 完了
{
  "type": "completed",
  "status": "ok",
  "output_url": "/output/ace_step_00001_.mp3",
  "duration": 120
}

// エラー
{
  "type": "error",
  "message": "Generation failed: ..."
}
```

### 8.6 TypeScript型定義

```typescript
// リクエスト型
interface LyricsGenerateRequest {
  workflow: "lyrics_generate";
  scenario: string;
  genre?: string;
  language: string;
}

interface TagsGenerateRequest {
  workflow: "lyrics_to_tags";
  lyrics: string;
}

interface MusicGenerateRequest {
  workflow: "ace_step_1_5_t2a";
  tags: string;
  lyrics?: string;
  duration?: number;      // 10-2000, default 60
  bpm?: number;           // 10-300, default 120
  timesignature?: "2" | "3" | "4" | "6";
  language?: string;      // 言語コード
  keyscale?: string;      // "C major", "A minor", etc.
  seed?: number;          // -1 = random
}

// レスポンス型
interface LyricsGenerateResponse {
  status: "ok" | "error";
  lyrics: string;
  scenario: string;
  genre: string;
  language: string;
  recommended_duration?: number;
  parts_timing?: Record<string, number>;
}

interface TagsGenerateResponse {
  status: "ok" | "error";
  genre: string;
  tags: string;
}

interface MusicGenerateResponse {
  status: "queued" | "error";
  job_id: string;
  message?: string;
}

interface ProgressMessage {
  type: "progress" | "completed" | "error";
  step?: number;
  total_steps?: number;
  percent?: number;
  output_url?: string;
  duration?: number;
  message?: string;
}
```

### 8.7 cURL例

```bash
# 作詞
curl -X POST http://localhost:8188/api/utility \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": "lyrics_generate",
    "scenario": "夜空を見上げる魔法少女",
    "genre": "pop, female voice",
    "language": "Japanese"
  }'

# タグ生成
curl -X POST http://localhost:8188/api/utility \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": "lyrics_to_tags",
    "lyrics": "[verse]\n夜空に浮かぶ星を見つめて..."
  }'

# 音楽生成
curl -X POST http://localhost:8188/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": "ace_step_1_5_t2a",
    "tags": "j-pop, female voice, emotional",
    "lyrics": "[verse]\n夜空に浮かぶ星を見つめて...",
    "duration": 120,
    "bpm": 90,
    "keyscale": "A minor",
    "language": "ja"
  }'
```

---

## 9. 独立Webアプリ実装ガイド

### 9.1 最小構成

```
ace-step-webapp/
├── backend/
│   ├── app.py              # FastAPI サーバー
│   ├── inference.py        # ACE-Step推論ラッパー
│   ├── llm_service.py      # LLM API (作詞/タグ)
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
└── models/
    └── ace_step_1.5_turbo_aio.safetensors
```

### 9.2 バックエンド実装例 (FastAPI)

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uuid

app = FastAPI()

class GenerateRequest(BaseModel):
    tags: str
    lyrics: str = ""
    duration: int = 60
    bpm: int = 120
    timesignature: str = "4"
    language: str = "ja"
    keyscale: str = "C major"
    steps: int = 8
    cfg: float = 1.0
    seed: int = None

@app.post("/api/generate")
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    background_tasks.add_task(run_inference, job_id, request)
    return {"job_id": job_id, "status": "queued"}

async def run_inference(job_id: str, request: GenerateRequest):
    # ACE-Step推論実行
    # ... 
    pass
```

### 9.3 フロントエンド実装例

```javascript
class ACEStepClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }
    
    async generate(params) {
        const response = await fetch(`${this.baseUrl}/api/generate`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(params)
        });
        return response.json();
    }
    
    async composeLyrics(scenario, genre, language) {
        const response = await fetch(`${this.baseUrl}/api/utility`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                workflow: 'lyrics_generate',
                scenario, genre, language
            })
        });
        return response.json();
    }
    
    monitorProgress(jobId, onProgress, onComplete) {
        const ws = new WebSocket(`ws://${this.baseUrl.replace('http://', '')}/ws/progress/${jobId}`);
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.status === 'completed') {
                onComplete(data);
            } else {
                onProgress(data);
            }
        };
    }
}
```

---

## 10. 参考リンク

- [ACE-Step GitHub](https://github.com/ace-step/ACE-Step)
- [ACE-Step Technical Report](https://arxiv.org/abs/2506.00045)
- [Hugging Face Model](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B)

---

## ✅ Standalone Webアプリ開発完了

この仕様書に基づいて **ace_step_client_pro** Webアプリを開発しました。

### 実装内容

| 機能 | 状態 | 備考 |
|------|------|------|
| テーマ入力 → AI作詞 | ✅ 完了 | LLM API連携 |
| タグ自動生成 | ✅ 完了 | LLM API連携 |
| 音楽生成 | ✅ 完了 | ACE-Step API連携 |
| パラメータ設定 | ✅ 完了 | STEP/CFG/SEED/BPM/調/拍子 |
| ビジュアライザー | ✅ 完了 | Web Audio API |
| 音声プロキシ | ✅ 完了 | CORS対応 |
| 詳細設定アコーディオン | ✅ 完了 | CFG/SEED |
| サーバー情報表示 | ✅ 完了 | モデル/キュー状態 |

### 開発で判明した修正点

1. **パラメータ名の訂正**:
   - ✗ `infer_step` → ✅ `inference_steps`
   - ✗ `task_id` → ✅ `task_ids` または `task_id_list`

2. **キースケール**:
   - 34種類対応（C〜B × major/minor、#/b含む）

3. **拍子の選択肢**:
   - `2/2`, `2/4`, `3/4`, `4/4`, `5/4`, `6/8`, `7/8`, `12/8`

4. **CORS対応**:
   - ACE-Step APIから直接音声取得時はCORSエラー
   - バックエンドでプロキシAPIが必要

5. **Turboモデル推奨値**:
   - inference_steps: **8**
   - guidance_scale: **3.0**（効果限定的）

### パス

アプリ: `ace_step_client_pro/`  
README: `../README.md`

---

