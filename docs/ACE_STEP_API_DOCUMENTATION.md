# ACE-Step 1.5 REST API 完全仕様書

**Version:** 1.0  
**Date:** 2026年2月7日  
**Base URL:** `http://localhost:8001`

> **✅ Standalone Webアプリ開発完了**: このドキュメントに基づいて `ace_step_client_pro` Webアプリを開発しました。  
> 詳細は [../README.md](../README.md) を参照。

---

## 目次

1. [概要](#1-概要)
2. [サーバー起動方法](#2-サーバー起動方法)
3. [認証](#3-認証)
4. [レスポンス形式](#4-レスポンス形式)
5. [エンドポイント一覧](#5-エンドポイント一覧)
6. [エンドポイント詳細](#6-エンドポイント詳細)
7. [パラメータ詳細リファレンス](#7-パラメータ詳細リファレンス)
8. [エラーハンドリング](#8-エラーハンドリング)
9. [使用例](#9-使用例)
10. [Webアプリ開発ガイド](#10-webアプリ開発ガイド)
11. [付録](#付録)
    - [A. パラメータエイリアス一覧](#a-パラメータエイリアス一覧)
    - [B. サポート言語一覧](#b-サポート言語一覧)
    - [C. モデル一覧](#c-モデル一覧)
    - [D. 実環境テスト結果](#d-実環境テスト結果2026年2月5日実施)

---

## 1. 概要

ACE-Step 1.5は高品質な音楽生成AIです。REST APIを通じて以下の機能を提供します：

- **テキストから音楽生成** - プロンプトと歌詞から音楽を生成
- **カバー生成** - 既存音声のスタイル変換
- **リペイント** - 音声の特定部分を再生成
- **メタデータ抽出** - 音声からBPM、調などを抽出

### アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│                    クライアント                          │
│              (Webアプリ / モバイル / CLI)                │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP/REST
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  ACE-Step API Server                     │
│                   (FastAPI / Uvicorn)                    │
│                   Port: 8001                             │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │   5Hz LM    │  │     DiT     │  │       VAE       │  │
│  │  (0.6B-4B)  │  │   (Turbo)   │  │   (Oobleck)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 2. サーバー起動方法

### 基本起動

```bash
# uvを使用
uv run acestep-api

# Pythonを直接使用
python acestep/api_server.py
```

### 起動オプション

```bash
# ポート指定
python acestep/api_server.py --port 8001

# ホスト指定（外部アクセス許可）
python acestep/api_server.py --host 0.0.0.0

# API認証キー設定
python acestep/api_server.py --api-key sk-your-secret-key

# モデル指定
python acestep/api_server.py --config-path acestep-v15-turbo --lm-model-path acestep-5Hz-lm-1.7B
```

### 環境変数

| 変数名 | デフォルト | 説明 |
|--------|---------|------|
| `ACESTEP_API_HOST` | `127.0.0.1` | サーバーホスト |
| `ACESTEP_API_PORT` | `8001` | サーバーポート |
| `ACESTEP_API_KEY` | (なし) | API認証キー（空=認証なし） |
| `ACESTEP_CONFIG_PATH` | `acestep-v15-turbo` | DiTモデルパス |
| `ACESTEP_LM_MODEL_PATH` | `acestep-5Hz-lm-0.6B` | LMモデルパス |
| `ACESTEP_QUEUE_MAXSIZE` | `200` | 最大キューサイズ |
| `ACESTEP_OFFLOAD_TO_CPU` | `false` | CPUオフロード |

---

## 3. 認証

### 認証方法

API認証が有効な場合、以下の2つの方法で認証できます：

#### 方法A: リクエストボディに `ai_token`

```json
{
  "ai_token": "your-api-key",
  "prompt": "upbeat pop song",
  ...
}
```

#### 方法B: Authorization ヘッダー

```http
Authorization: Bearer your-api-key
```

### 認証なしの場合

`ACESTEP_API_KEY` が設定されていない場合、認証は不要です。

---

## 4. レスポンス形式

### 統一レスポンス構造

すべてのAPIレスポンスは以下の形式でラップされます：

```json
{
  "data": { ... },
  "code": 200,
  "error": null,
  "timestamp": 1700000000000,
  "extra": null
}
```

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `data` | any | 実際のレスポンスデータ |
| `code` | int | ステータスコード（200=成功） |
| `error` | string/null | エラーメッセージ（成功時はnull） |
| `timestamp` | int | レスポンス時刻（ミリ秒） |
| `extra` | any | 追加情報（通常null） |

### タスクステータスコード

| コード | ステータス | 説明 |
|--------|----------|------|
| `0` | queued/running | タスクがキュー中または実行中 |
| `1` | succeeded | 生成成功、結果取得可能 |
| `2` | failed | 生成失敗 |

---

## 5. エンドポイント一覧

| エンドポイント | メソッド | 説明 |
|---------------|---------|------|
| `/release_task` | POST | 音楽生成タスクを作成 |
| `/query_result` | POST | タスク結果をバッチクエリ |
| `/format_input` | POST | LLMでcaption/lyricsを強化 |
| `/create_random_sample` | POST | ランダムサンプルパラメータ取得 |
| `/v1/models` | GET | 利用可能なモデル一覧 |
| `/v1/audio` | GET | 音声ファイルダウンロード |
| `/v1/stats` | GET | サーバー統計情報 |
| `/health` | GET | ヘルスチェック |

---

## 6. エンドポイント詳細

### 6.1 POST /release_task

音楽生成タスクを作成します。

#### リクエスト

**Content-Type:** `application/json` または `multipart/form-data`

```json
{
  "prompt": "upbeat pop song with catchy melody",
  "lyrics": "[Verse 1]\nWalking down the street...\n\n[Chorus]\nThis is the moment...",
  "thinking": true,
  "vocal_language": "en",
  "audio_duration": 60,
  "batch_size": 2,
  "audio_format": "mp3"
}
```

#### レスポンス

```json
{
  "data": {
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "queued",
    "queue_position": 1
  },
  "code": 200,
  "error": null,
  "timestamp": 1700000000000,
  "extra": null
}
```

---

### 6.2 POST /query_result

タスク結果をバッチでクエリします。

> ℹ️ **パラメータ名**: この環境では `task_id_list`（配列）が確実に動作します。`task_ids` は実装差分により `data: []` が返る場合があります。

#### リクエスト

```json
{
  "task_ids": ["550e8400-e29b-41d4-a716-446655440000"]
}
```

または

```json
{
  "task_id_list": ["550e8400-e29b-41d4-a716-446655440000"]
}
```

#### 複数タスクの同時クエリ

```json
{
  "task_id_list": [
    "31583ff6-21dd-4ca6-a867-2e0336830dde",
    "e4ec5bde-760e-4bdd-a0e0-8f9b08ecd3aa"
  ]
}
```

#### レスポンス

```json
{
  "data": [
    {
      "task_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": 1,
      "result": "[{\"file\": \"/v1/audio?path=...\", \"status\": 1, \"metas\": {...}}]"
    }
  ],
  "code": 200,
  "error": null,
  "timestamp": 1700000000000,
  "extra": null
}
```

**result フィールド（JSONパース後）:**

> ⚠️ **注意**: `result` フィールドはJSON文字列として返されます。使用前に `JSON.parse()` が必要です。

```json
[
  {
    "file": "/v1/audio?path=%2Fhome%2Fuser%2F.cache%2Facestep%2Ftmp%2Fapi_audio%2F149b2c12-28a5-3e87-f310-8e9a0f0ebc19.mp3",
    "wave": "",
    "status": 1,
    "create_time": 1770282320,
    "env": "development",
    "prompt": "calm piano music",
    "lyrics": "[Instrumental]",
    "metas": {
      "bpm": 54,
      "duration": 20.0,
      "genres": "N/A",
      "keyscale": "C major",
      "timesignature": "2",
      "prompt": "calm piano music",
      "lyrics": "[Instrumental]"
    },
    "generation_info": "**🎯 Average Time per Track: 0.72s** (2 track(s))\n\n**🤖 LM-Generated Metadata:**\n- **BPM:** 54\n- **Refined Caption:** A gentle and introspective solo piano piece...\n...",
    "seed_value": "2359985563,2203890097",
    "lm_model": "acestep-5Hz-lm-0.6B",
    "dit_model": "acestep-v15-turbo"
  }
]
```

**ステータスコード:**

| status | 意味 |
|--------|------|
| 0 | 処理中（queued/running） |
| 1 | 成功（succeeded） |
| 2 | 失敗（failed） |
```

---

### 6.3 POST /format_input

LLMを使用してcaptionとlyricsを強化します。

#### リクエスト

```json
{
  "prompt": "pop rock",
  "lyrics": "Walking down the street",
  "temperature": 0.85
}
```

#### レスポンス

```json
{
  "data": {
    "caption": "Upbeat pop rock anthem with electric guitars and driving drums",
    "lyrics": "[Verse 1]\nWalking down the street today\nFeeling the rhythm in my way...",
    "bpm": 120,
    "key_scale": "G Major",
    "time_signature": "4",
    "duration": 180,
    "vocal_language": "en"
  },
  "code": 200,
  "error": null,
  "timestamp": 1700000000000,
  "extra": null
}
```

---

### 6.4 POST /create_random_sample

ランダムなサンプルパラメータを取得します。

#### リクエスト

```json
{
  "sample_type": "simple_mode"
}
```

#### レスポンス

```json
{
  "data": {
    "caption": "Melodic indie folk with acoustic guitar",
    "lyrics": "[Verse 1]\nIn the morning light...",
    "bpm": 95,
    "key_scale": "D Major",
    "time_signature": "4",
    "duration": 180,
    "vocal_language": "en"
  },
  "code": 200,
  "error": null,
  "timestamp": 1700000000000,
  "extra": null
}
```

---

### 6.5 GET /v1/models

利用可能なDiTモデル一覧を取得します。

#### レスポンス

```json
{
  "data": {
    "models": [
      {
        "name": "acestep-v15-turbo",
        "is_default": true
      }
    ],
    "default_model": "acestep-v15-turbo"
  },
  "code": 200,
  "error": null,
  "timestamp": 1700000000000,
  "extra": null
}
```

---

### 6.6 GET /v1/audio

生成された音声ファイルをダウンロードします。

#### リクエストパラメータ

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `path` | string | URLエンコードされたファイルパス |

#### 使用例

```
GET /v1/audio?path=%2Ftmp%2Fapi_audio%2Fabc123.mp3
```

#### レスポンス

音声ファイル（バイナリ）

---

### 6.7 GET /v1/stats

サーバー統計情報を取得します。

#### レスポンス

```json
{
  "data": {
    "jobs": {
      "total": 100,
      "queued": 5,
      "running": 1,
      "succeeded": 90,
      "failed": 4
    },
    "queue_size": 5,
    "queue_maxsize": 200,
    "avg_job_seconds": 8.5
  },
  "code": 200,
  "error": null,
  "timestamp": 1700000000000,
  "extra": null
}
```

---

### 6.8 GET /health

サーバーヘルスチェックを実行します。

#### レスポンス

```json
{
  "data": {
    "status": "ok",
    "service": "ACE-Step API",
    "version": "1.0"
  },
  "code": 200,
  "error": null,
  "timestamp": 1700000000000,
  "extra": null
}
```

---

## 7. パラメータ詳細リファレンス

### 7.1 基本パラメータ

| パラメータ | 型 | デフォルト | 必須 | 説明 |
|-----------|-----|---------|------|------|
| `prompt` | string | `""` | △ | 音楽の説明プロンプト。ジャンル、楽器、雰囲気などを記述。エイリアス: `caption` |
| `lyrics` | string | `""` | △ | 歌詞内容。構造タグ `[Verse]`, `[Chorus]` などを使用可能。インストの場合は `[Instrumental]` |
| `thinking` | bool | `false` | ✗ | `true`: 5Hz LMで音声コード生成（高品質）。`false`: DiTのみ（高速） |
| `vocal_language` | string | `"en"` | ✗ | 歌詞言語。`en`, `zh`, `ja`, `ko`, `es`, `fr`, `de`, `it`, `pt`, `ru` など |
| `audio_format` | string | `"mp3"` | ✗ | 出力形式: `mp3`, `wav`, `flac` |

### 7.2 サンプル/説明モードパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|---------|------|
| `sample_mode` | bool | `false` | LMで自動的にcaption/lyrics/metasを生成 |
| `sample_query` | string | `""` | 自然言語での音楽説明。エイリアス: `description`, `desc`, `sampleQuery` |
| `use_format` | bool | `false` | LMでcaption/lyricsを強化。エイリアス: `format`, `useFormat` |
| `model` | string | null | 使用するDiTモデル名（例: `acestep-v15-turbo`） |

### 7.3 音楽属性パラメータ

| パラメータ | 型 | デフォルト | 範囲 | 説明 |
|-----------|-----|---------|------|------|
| `bpm` | int | null (自動) | 30-300 | テンポ（BPM） |
| `key_scale` | string | `""` (自動) | - | 調。例: `"C Major"`, `"Am"`, `"F# minor"` |
| `time_signature` | string | `""` (自動) | 2,3,4,6 | 拍子。`2`=2/4, `3`=3/4, `4`=4/4, `6`=6/8 |
| `audio_duration` | float | null (自動) | 10-600 | 生成時間（秒）。-1で自動 |

**エイリアス:**
- `key_scale`: `keyscale`, `keyScale`, `key`
- `time_signature`: `timesignature`, `timeSignature`
- `audio_duration`: `duration`, `audioDuration`, `target_duration`

### 7.4 生成制御パラメータ

| パラメータ | 型 | デフォルト | 範囲 | 説明 |
|-----------|-----|---------|------|------|
| `inference_steps` | int | `8` | Turbo: 1-20, Base: 1-200 | 推論ステップ数。Turboは8推奨 |
| `guidance_scale` | float | `7.0` | 1.0-20.0 | ガイダンス強度（Baseモデルのみ有効） |
| `seed` | int | `-1` | - | シード値。-1でランダム |
| `use_random_seed` | bool | `true` | - | ランダムシード使用 |
| `batch_size` | int | `2` | 1-8 | 同時生成数 |

### 7.5 高度なDiTパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|---------|------|
| `shift` | float | `3.0` | タイムステップシフト係数（1.0-5.0）。Baseモデルのみ有効 |
| `infer_method` | string | `"ode"` | 推論方法: `"ode"`（Euler、高速）, `"sde"`（確率的） |
| `timesteps` | string | null | カスタムタイムステップ（カンマ区切り）。例: `"0.97,0.76,0.615,0.5,0.395,0.28,0.18,0.085,0"` |
| `use_adg` | bool | `false` | Adaptive Dual Guidance（Baseモデルのみ） |
| `cfg_interval_start` | float | `0.0` | CFG適用開始比率（0.0-1.0） |
| `cfg_interval_end` | float | `1.0` | CFG適用終了比率（0.0-1.0） |

### 7.6 5Hz LMパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|---------|------|
| `lm_model_path` | string | null | LMモデル名。例: `acestep-5Hz-lm-0.6B` |
| `lm_backend` | string | `"vllm"` | LMバックエンド: `"vllm"`（高速）, `"pt"`（互換性） |
| `lm_temperature` | float | `0.85` | サンプリング温度（0.0-2.0）。高い=より創造的 |
| `lm_cfg_scale` | float | `2.5` | CFGスケール（1.0-3.0） |
| `lm_top_k` | int | null | Top-Kサンプリング。0/nullで無効 |
| `lm_top_p` | float | `0.9` | Top-Pサンプリング（0.0-1.0）。1.0以上で無効 |
| `lm_repetition_penalty` | float | `1.0` | 繰り返しペナルティ |
| `lm_negative_prompt` | string | `"NO USER INPUT"` | CFG用ネガティブプロンプト |

### 7.7 CoT（Chain-of-Thought）パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|---------|------|
| `use_cot_caption` | bool | `true` | LMでcaptionを書き換え/強化 |
| `use_cot_language` | bool | `true` | LMで歌詞言語を検出 |
| `constrained_decoding` | bool | `true` | FSMベースの制約デコーディング有効化 |
| `constrained_decoding_debug` | bool | `false` | デバッグログ有効化 |
| `allow_lm_batch` | bool | `true` | LMバッチ処理許可 |

### 7.8 編集/参照音声パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|---------|------|
| `task_type` | string | `"text2music"` | タスクタイプ（下記参照） |
| `reference_audio_path` | string | null | 参照音声パス（スタイル転送用） |
| `src_audio_path` | string | null | ソース音声パス（cover/repaint用） |
| `instruction` | string | (自動) | タスク指示。未指定時はtask_typeから自動生成 |
| `repainting_start` | float | `0.0` | リペイント開始時間（秒） |
| `repainting_end` | float | null | リペイント終了時間（秒）。-1で音声終端まで |
| `audio_cover_strength` | float | `1.0` | カバー強度（0.0-1.0）。低い値（0.2）でスタイル転送 |

### 7.9 タスクタイプ

| タスクタイプ | 説明 | 必須入力 |
|-------------|------|---------|
| `text2music` | テキストから音楽生成（デフォルト） | prompt または lyrics |
| `cover` | 既存音声のスタイル変換 | src_audio_path, prompt |
| `repaint` | 特定区間の再生成 | src_audio_path, repainting_start/end |
| `lego` | 楽器トラック追加（Baseのみ） | src_audio_path, track_name |
| `extract` | 楽器トラック抽出（Baseのみ） | src_audio_path, track_name |
| `complete` | 不完全トラックの補完（Baseのみ） | src_audio_path, track_names |

---

## 8. エラーハンドリング

### HTTPステータスコード

| コード | 説明 |
|--------|------|
| `200` | 成功 |
| `400` | 不正なリクエスト（JSONエラー、必須パラメータ不足） |
| `401` | 認証エラー（APIキー不正/未設定） |
| `404` | リソースが見つからない |
| `415` | サポートされていないContent-Type |
| `429` | サーバービジー（キュー満杯） |
| `500` | 内部サーバーエラー |

### エラーレスポンス形式

```json
{
  "detail": "Error message describing the issue"
}
```

または

```json
{
  "data": null,
  "code": 500,
  "error": "LLM not initialized",
  "timestamp": 1700000000000,
  "extra": null
}
```

---

## 9. 使用例

### 9.1 JavaScript (Fetch API)

```javascript
// ACE-Step API Client
class AceStepClient {
  constructor(baseUrl = 'http://localhost:8001', apiKey = null) {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
  }

  async request(endpoint, options = {}) {
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers
    };
    
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  // 音楽生成タスク作成
  async createTask(params) {
    return this.request('/release_task', {
      method: 'POST',
      body: JSON.stringify(params)
    });
  }

  // タスク結果クエリ
  async queryResults(taskIds) {
    return this.request('/query_result', {
      method: 'POST',
      body: JSON.stringify({ task_id_list: taskIds })
    });
  }

  // ヘルスチェック
  async health() {
    return this.request('/health');
  }

  // モデル一覧
  async getModels() {
    return this.request('/v1/models');
  }

  // 統計情報
  async getStats() {
    return this.request('/v1/stats');
  }

  // ランダムサンプル取得
  async getRandomSample(sampleType = 'simple_mode') {
    return this.request('/create_random_sample', {
      method: 'POST',
      body: JSON.stringify({ sample_type: sampleType })
    });
  }

  // 入力フォーマット
  async formatInput(prompt, lyrics, temperature = 0.85) {
    return this.request('/format_input', {
      method: 'POST',
      body: JSON.stringify({ prompt, lyrics, temperature })
    });
  }

  // 音声ダウンロードURL生成
  getAudioUrl(path) {
    return `${this.baseUrl}/v1/audio?path=${encodeURIComponent(path)}`;
  }

  // タスク完了まで待機
  async waitForCompletion(taskId, intervalMs = 2000, timeoutMs = 300000) {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeoutMs) {
      const result = await this.queryResults([taskId]);
      const task = result.data[0];
      
      if (task.status === 1) {
        // 成功
        return JSON.parse(task.result);
      } else if (task.status === 2) {
        // 失敗
        throw new Error('Task failed');
      }
      
      // 待機
      await new Promise(resolve => setTimeout(resolve, intervalMs));
    }
    
    throw new Error('Timeout waiting for task completion');
  }
}

// 使用例
async function generateMusic() {
  const client = new AceStepClient('http://localhost:8001');
  
  // タスク作成
  const taskResponse = await client.createTask({
    prompt: 'upbeat pop song with catchy melody',
    lyrics: '[Verse 1]\nWalking down the street today\n\n[Chorus]\nThis is the moment we live for',
    thinking: true,
    vocal_language: 'en',
    audio_duration: 60,
    batch_size: 2,
    audio_format: 'mp3'
  });
  
  console.log('Task created:', taskResponse.data.task_id);
  
  // 完了待機
  const results = await client.waitForCompletion(taskResponse.data.task_id);
  
  // 結果表示
  for (const audio of results) {
    console.log('Audio URL:', client.getAudioUrl(audio.file));
    console.log('Metadata:', audio.metas);
  }
}
```

### 9.2 Python (requests)

```python
import requests
import time
import json
from typing import Optional, List, Dict, Any
from urllib.parse import urlencode


class AceStepClient:
    """ACE-Step REST API Client"""
    
    def __init__(self, base_url: str = "http://localhost:8001", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
    
    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        response = requests.request(method, url, headers=self._get_headers(), **kwargs)
        response.raise_for_status()
        return response.json()
    
    def health(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        return self._request("GET", "/health")
    
    def get_models(self) -> Dict[str, Any]:
        """利用可能なモデル一覧"""
        return self._request("GET", "/v1/models")
    
    def get_stats(self) -> Dict[str, Any]:
        """サーバー統計情報"""
        return self._request("GET", "/v1/stats")
    
    def create_task(self, **params) -> Dict[str, Any]:
        """音楽生成タスク作成"""
        return self._request("POST", "/release_task", json=params)
    
    def query_results(self, task_ids: List[str]) -> Dict[str, Any]:
        """タスク結果クエリ"""
        return self._request("POST", "/query_result", json={"task_id_list": task_ids})
    
    def format_input(self, prompt: str, lyrics: str = "", temperature: float = 0.85) -> Dict[str, Any]:
        """入力フォーマット"""
        return self._request("POST", "/format_input", json={
            "prompt": prompt,
            "lyrics": lyrics,
            "temperature": temperature
        })
    
    def get_random_sample(self, sample_type: str = "simple_mode") -> Dict[str, Any]:
        """ランダムサンプル取得"""
        return self._request("POST", "/create_random_sample", json={"sample_type": sample_type})
    
    def get_audio_url(self, path: str) -> str:
        """音声ダウンロードURL生成"""
        return f"{self.base_url}/v1/audio?{urlencode({'path': path})}"
    
    def download_audio(self, path: str, save_path: str) -> str:
        """音声ファイルダウンロード"""
        url = self.get_audio_url(path)
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        with open(save_path, "wb") as f:
            f.write(response.content)
        
        return save_path
    
    def wait_for_completion(
        self, 
        task_id: str, 
        interval: float = 2.0, 
        timeout: float = 300.0
    ) -> List[Dict[str, Any]]:
        """タスク完了まで待機"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = self.query_results([task_id])
            task = result["data"][0]
            
            if task["status"] == 1:
                # 成功
                return json.loads(task["result"])
            elif task["status"] == 2:
                # 失敗
                raise Exception("Task failed")
            
            time.sleep(interval)
        
        raise TimeoutError("Timeout waiting for task completion")
    
    def generate_music(
        self,
        prompt: str,
        lyrics: str = "",
        thinking: bool = True,
        vocal_language: str = "en",
        audio_duration: float = 60,
        batch_size: int = 2,
        audio_format: str = "mp3",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """音楽生成（タスク作成から完了まで）"""
        # タスク作成
        task_response = self.create_task(
            prompt=prompt,
            lyrics=lyrics,
            thinking=thinking,
            vocal_language=vocal_language,
            audio_duration=audio_duration,
            batch_size=batch_size,
            audio_format=audio_format,
            **kwargs
        )
        
        task_id = task_response["data"]["task_id"]
        print(f"Task created: {task_id}")
        
        # 完了待機
        results = self.wait_for_completion(task_id)
        
        return results


# 使用例
if __name__ == "__main__":
    client = AceStepClient("http://localhost:8001")
    
    # ヘルスチェック
    health = client.health()
    print(f"Server status: {health['data']['status']}")
    
    # 音楽生成
    results = client.generate_music(
        prompt="upbeat pop song with catchy melody and driving drums",
        lyrics="""[Verse 1]
Walking down the street today
Feeling the rhythm in my way

[Chorus]
This is the moment we live for
Dancing through the night once more""",
        thinking=True,
        vocal_language="en",
        audio_duration=60,
        batch_size=2
    )
    
    # 結果表示
    for i, audio in enumerate(results):
        print(f"\n--- Audio {i+1} ---")
        print(f"URL: {client.get_audio_url(audio['file'])}")
        print(f"BPM: {audio['metas'].get('bpm')}")
        print(f"Key: {audio['metas'].get('keyscale')}")
        
        # ダウンロード
        client.download_audio(audio["file"], f"output_{i+1}.mp3")
        print(f"Saved to: output_{i+1}.mp3")
```

### 9.3 cURL

```bash
# ヘルスチェック
curl http://localhost:8001/health
# レスポンス例: {"data":{"status":"ok","service":"ACE-Step API","version":"1.0"},"code":200,...}

# モデル一覧
curl http://localhost:8001/v1/models
# レスポンス例: {"data":{"models":[{"name":"acestep-v15-turbo","is_default":true}],...}

# サーバー統計
curl http://localhost:8001/v1/stats
# レスポンス例: {"data":{"jobs":{"total":3,"succeeded":3,"failed":0},"avg_job_seconds":2.17,...}

# 基本的な音楽生成
curl -X POST http://localhost:8001/release_task \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "upbeat pop song with catchy melody",
    "lyrics": "[Verse 1]\nWalking down the street today",
    "thinking": true,
    "vocal_language": "en",
    "audio_duration": 60,
    "batch_size": 2
  }'
# レスポンス例: {"data":{"task_id":"e4ec5bde-760e-4bdd-a0e0-8f9b08ecd3aa","status":"queued","queue_position":1},...}

# 自然言語説明から生成
curl -X POST http://localhost:8001/release_task \
  -H 'Content-Type: application/json' \
  -d '{
    "sample_query": "a soft Japanese love song for a quiet evening",
    "thinking": true
  }'

# タスク結果クエリ（⚠️ パラメータ名は task_id_list）
curl -X POST http://localhost:8001/query_result \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id_list": ["e4ec5bde-760e-4bdd-a0e0-8f9b08ecd3aa"]
  }'
# 注意: task_id や task_ids ではなく task_id_list を使用
# 結果が空配列の場合は、タスクがまだ処理中か、キャッシュに保存されていません

# 複数タスクの同時クエリ
curl -X POST http://localhost:8001/query_result \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id_list": ["task-id-1", "task-id-2", "task-id-3"]
  }'

# 音声ダウンロード（URLエンコードされたパスを使用）
curl "http://localhost:8001/v1/audio?path=%2Fhome%2Fuser%2F.cache%2Facestep%2Ftmp%2Fapi_audio%2F149b2c12-28a5-3e87-f310-8e9a0f0ebc19.mp3" -o output.mp3
# ファイル確認: file output.mp3  →  Audio file with ID3 version 2.4.0, MPEG ADTS, layer III, 48 kHz, Stereo

# 認証付きリクエスト
curl -X POST http://localhost:8001/release_task \
  -H 'Authorization: Bearer sk-your-api-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "jazz piano trio",
    "thinking": true
  }'
```

---

## 10. Webアプリ開発ガイド

### 10.1 推奨アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│                    フロントエンド                        │
│            (React / Vue / Vanilla JS)                   │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  音楽生成フォーム │  │      結果表示エリア          │  │
│  │  - プロンプト    │  │  - オーディオプレーヤー      │  │
│  │  - 歌詞         │  │  - メタデータ表示            │  │
│  │  - パラメータ    │  │  - ダウンロードボタン        │  │
│  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │ Fetch API / Axios
                      ▼
┌─────────────────────────────────────────────────────────┐
│               バックエンド (オプション)                   │
│           (Node.js / Python / 直接接続)                 │
├─────────────────────────────────────────────────────────┤
│  - API認証の秘匿                                         │
│  - タスク管理                                            │
│  - ユーザー管理                                          │
│  - 生成履歴保存                                          │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP
                      ▼
┌─────────────────────────────────────────────────────────┐
│                 ACE-Step API Server                      │
│                  http://localhost:8001                   │
└─────────────────────────────────────────────────────────┘
```

### 10.2 フロントエンド実装パターン

#### React コンポーネント例

```jsx
import React, { useState, useEffect, useCallback } from 'react';

const API_BASE = 'http://localhost:8001';

function MusicGenerator() {
  const [prompt, setPrompt] = useState('');
  const [lyrics, setLyrics] = useState('');
  const [loading, setLoading] = useState(false);
  const [taskId, setTaskId] = useState(null);
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);

  // タスク作成
  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE}/release_task`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          lyrics,
          thinking: true,
          vocal_language: 'en',
          audio_duration: 60,
          batch_size: 2,
          audio_format: 'mp3'
        })
      });
      
      const data = await response.json();
      setTaskId(data.data.task_id);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  // ポーリングでタスク状態確認
  useEffect(() => {
    if (!taskId) return;
    
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE}/query_result`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ task_id_list: [taskId] })
        });
        
        const data = await response.json();
        const task = data.data[0];
        
        if (task.status === 1) {
          // 成功
          setResults(JSON.parse(task.result));
          setLoading(false);
          setTaskId(null);
          clearInterval(interval);
        } else if (task.status === 2) {
          // 失敗
          setError('Generation failed');
          setLoading(false);
          setTaskId(null);
          clearInterval(interval);
        }
      } catch (err) {
        setError(err.message);
        setLoading(false);
        clearInterval(interval);
      }
    }, 2000);
    
    return () => clearInterval(interval);
  }, [taskId]);

  return (
    <div className="music-generator">
      <h1>ACE-Step Music Generator</h1>
      
      <div className="input-section">
        <textarea
          placeholder="Music description (e.g., upbeat pop song with guitar)"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        
        <textarea
          placeholder="Lyrics (optional)"
          value={lyrics}
          onChange={(e) => setLyrics(e.target.value)}
        />
        
        <button onClick={handleGenerate} disabled={loading}>
          {loading ? 'Generating...' : 'Generate Music'}
        </button>
      </div>
      
      {error && <div className="error">{error}</div>}
      
      <div className="results-section">
        {results.map((audio, index) => (
          <div key={index} className="audio-result">
            <audio controls src={`${API_BASE}/v1/audio?path=${encodeURIComponent(audio.file)}`} />
            <div className="metadata">
              <p>BPM: {audio.metas?.bpm}</p>
              <p>Key: {audio.metas?.keyscale}</p>
              <p>Duration: {audio.metas?.duration}s</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default MusicGenerator;
```

### 10.3 CORS設定

ACE-Step APIサーバーはデフォルトでCORSが有効です。異なるオリジンからアクセスする場合：

```python
# api_server.py でCORSを確認/設定
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では制限推奨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 10.4 ベストプラクティス

#### パフォーマンス最適化

1. **thinking=true を使用** - 高品質な結果のため
2. **適切なbatch_size** - 2-4が推奨（多すぎると遅くなる）
3. **ポーリング間隔** - 2-3秒が適切
4. **タイムアウト設定** - 長い曲は5分以上かかる場合あり

#### ユーザー体験

1. **進捗表示** - ローディングインジケーター
2. **推定時間表示** - `/v1/stats` から `avg_job_seconds` を取得
3. **エラーハンドリング** - 分かりやすいエラーメッセージ
4. **プリセット機能** - `/create_random_sample` でサンプル提供

#### セキュリティ

1. **APIキーの秘匿** - フロントエンドに直接埋め込まない
2. **バックエンド経由** - プロキシサーバーを使用
3. **レート制限** - 過度なリクエストを防止

### 10.5 トラブルシューティング

| 問題 | 原因 | 解決策 |
|------|------|--------|
| CORS エラー | オリジン不一致 | CORS設定を確認 |
| 401 Unauthorized | APIキー不正 | 認証設定を確認 |
| 429 Too Many Requests | キュー満杯 | 待機してリトライ |
| タスクがいつまでも完了しない | サーバー負荷 | タイムアウト設定、stats確認 |
| 音声が再生されない | パス不正/形式問題 | URLエンコード確認、形式確認 |

---

## 付録

### A. パラメータエイリアス一覧

| 標準名 | エイリアス |
|--------|-----------|
| `prompt` | `caption` |
| `sample_query` | `sampleQuery`, `description`, `desc` |
| `use_format` | `useFormat`, `format` |
| `key_scale` | `keyscale`, `keyScale`, `key` |
| `time_signature` | `timesignature`, `timeSignature` |
| `audio_duration` | `duration`, `audioDuration`, `target_duration`, `targetDuration` |
| `vocal_language` | `vocalLanguage`, `language` |
| `inference_steps` | `inferenceSteps` |
| `guidance_scale` | `guidanceScale` |
| `use_random_seed` | `useRandomSeed` |
| `audio_code_string` | `audioCodeString` |
| `audio_cover_strength` | `audioCoverStrength` |
| `task_type` | `taskType` |
| `infer_method` | `inferMethod` |
| `use_tiled_decode` | `useTiledDecode` |
| `constrained_decoding` | `constrainedDecoding`, `constrained` |
| `use_cot_caption` | `cot_caption`, `cot-caption` |
| `use_cot_language` | `cot_language`, `cot-language` |
| `allow_lm_batch` | `allowLmBatch`, `parallel_thinking` |

### B. サポート言語一覧

| コード | 言語 |
|--------|------|
| `en` | English |
| `zh` | 中文 (Chinese) |
| `ja` | 日本語 (Japanese) |
| `ko` | 한국어 (Korean) |
| `es` | Español (Spanish) |
| `fr` | Français (French) |
| `de` | Deutsch (German) |
| `it` | Italiano (Italian) |
| `pt` | Português (Portuguese) |
| `ru` | Русский (Russian) |
| `ar` | العربية (Arabic) |
| `hi` | हिंदी (Hindi) |
| `bn` | বাংলা (Bengali) |
| `th` | ไทย (Thai) |
| `vi` | Tiếng Việt (Vietnamese) |
| `id` | Indonesian |
| `tr` | Türkçe (Turkish) |
| `nl` | Nederlands (Dutch) |
| `pl` | Polski (Polish) |
| `unknown` | 自動検出 |

### C. モデル一覧

#### DiTモデル

| モデル | 推論ステップ | 品質 | 速度 |
|--------|-------------|------|------|
| `acestep-v15-turbo` | 8 | Very High | Very Fast |
| `acestep-v15-turbo-shift3` | 8 | Very High | Very Fast |
| `acestep-v15-base` | 50 | Medium | Slow |
| `acestep-v15-sft` | 50 | High | Slow |

#### LMモデル

| モデル | パラメータ数 | 品質 | VRAM |
|--------|------------|------|------|
| `acestep-5Hz-lm-0.6B` | 0.6B | Medium | 6-12GB |
| `acestep-5Hz-lm-1.7B` | 1.7B | High | 12-16GB |
| `acestep-5Hz-lm-4B` | 4B | Very High | 16GB+ |

---

### D. 実環境テスト結果（2026年2月5日実施）

#### テスト環境

| 項目 | 値 |
|------|-----|
| サーバーアドレス | `http://YOUR_ACE_HOST:8001` |
| DiTモデル | `acestep-v15-turbo` |
| LMモデル | `acestep-5Hz-lm-0.6B` |

#### エンドポイント動作確認

| エンドポイント | メソッド | 状態 | 備考 |
|--------------|---------|------|------|
| `/health` | GET | ✅ 正常 | `{"status": "ok"}` |
| `/v1/models` | GET | ✅ 正常 | モデル一覧取得 |
| `/v1/stats` | GET | ✅ 正常 | ジョブ統計取得 |
| `/create_random_sample` | POST | ✅ 正常 | ランダムサンプル生成 |
| `/release_task` | POST | ✅ 正常 | タスク送信 |
| `/query_result` | POST | ✅ 正常 | 結果取得（⚠️ `task_id_list` を使用） |
| `/v1/audio` | GET | ✅ 正常 | 音声ダウンロード |

#### 生成パフォーマンス

| テスト | 内容 | 生成時間 | ファイルサイズ |
|--------|------|---------|--------------|
| ピアノ曲 | `calm piano music` (20秒, インスト) | **1.43秒** | 313KB (MP3, 48kHz Stereo) |
| ポップス | `upbeat pop song, female vocal` (30秒, ボーカル) | **2.32秒** | 469KB (MP3, 48kHz Stereo) |

#### 実際のレスポンス例

**`/release_task` レスポンス:**
```json
{
  "data": {
    "task_id": "e4ec5bde-760e-4bdd-a0e0-8f9b08ecd3aa",
    "status": "queued",
    "queue_position": 1
  },
  "code": 200,
  "error": null,
  "timestamp": 1770282213025,
  "extra": null
}
```

**`/v1/stats` レスポンス:**
```json
{
  "data": {
    "jobs": {
      "total": 3,
      "queued": 0,
      "running": 0,
      "succeeded": 3,
      "failed": 0
    },
    "queue_size": 0,
    "queue_maxsize": 200,
    "avg_job_seconds": 2.175
  }
}
```

#### 重要な知見

1. **`/query_result` のパラメータ名**
   - 使用可能: `task_ids` または `task_id_list`（配列形式）
   - 結果が空配列の場合は、タスクがまだ完了していないか、キャッシュに保存されていない可能性

2. **音声ファイルのパス**
   - `file` フィールドは既にURLエンコード済みの相対パス
   - 例: `/v1/audio?path=%2Fhome%2Fuser%2F.cache%2Facestep%2Ftmp%2Fapi_audio%2F149b2c12.mp3`
   - `BASE_URL + file` でダウンロード可能
   - **CORS注意**: Webアプリから直接アクセスするとCORSエラーが発生。バックエンドでプロキシが必要

3. **生成速度**
   - Turboモデル + 0.6B LMで、20-30秒の楽曲が約2秒で生成
   - `thinking=true` でも十分高速

4. **出力形式**
   - デフォルトMP3: 48kHz Stereo, 64kbps
   - ID3v2.4.0タグ付き

5. **inference_steps パラメータ**
   - Turboモデル: デフォルト **8**
   - Baseモデル: デフォルト **60**
   - ⨯ `infer_step` ではなく `inference_steps` が正しいパラメータ名

---

## 🛠️ Standalone Webアプリ開発記録

このドキュメントに基づいて **ace_step_client_pro** Webアプリを開発しました。

### 実装内容

- FastAPIバックエンド + Vanilla JSフロントエンド
- LLM連携によるAI作詞・タグ生成
- 音声プロキシAPI（CORS対応）
- ビジュアライザー付きプレイヤー
- 34種のキースケール対応
- STEP/CFG/SEEDパラメータ

### 詳細

パス: `ace_step_client_pro/`  
README: [../README.md](../README.md)

---

**ドキュメント終了**
