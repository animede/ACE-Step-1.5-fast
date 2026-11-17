# ACESTEP_LM_BACKEND（vllm / pt）とは

## 結論（要点）
`ACESTEP_LM_BACKEND` は **5Hz LM（言語モデル）の推論バックエンド**を選ぶ環境変数です。

- `vllm`：`nano-vllm` を使う高速バックエンド（GPU/CUDA前提）
- `pt`：PyTorch（Transformers）で実行する互換性重視バックエンド

「vllm を指定して“切る意味”」は、LM 推論を **高速化したいなら vllm**、**互換性/安定性を優先するなら pt** を選べる点にあります。

## どこで使われているか（コード上の根拠）
- サーバ初期化時の LM ロード：`acestep/api_server.py`
  - `ACESTEP_LM_BACKEND` を読み取り、`vllm`/`pt` 以外は `vllm` に丸める
- リクエスト単位の LM バックエンド指定：`acestep/api_server.py`
  - リクエストの `lm_backend` があればそれが優先され、無ければ `ACESTEP_LM_BACKEND` が使われる
- 実際の LM 初期化・フォールバック：`acestep/llm_inference.py`
  - `backend == "vllm"` のとき `nano-vllm` での初期化を試し、失敗した場合は **PyTorch にフォールバック**

補足：UI/ドキュメント上でも `vllm (faster) / pt (more compatible)` と説明されている箇所があります（例：`acestep/gradio_ui/i18n/*.json`）。

## 優先順位（リクエスト vs 環境変数）
API が `lm_backend` を受け取れる場合、一般に次の優先順位で決まります。

1. リクエストボディの `lm_backend`
2. 環境変数 `ACESTEP_LM_BACKEND`
3. 既定値 `vllm`

つまり「環境変数で vllm を指定している」状態でも、クライアント側が `lm_backend: "pt"` を送ればリクエスト単位で切り替え可能です（ただしサーバ実装・ルーティング設定に依存）。

## vllm と pt の違い（実用上）
### vllm（nano-vllm）
- 目的：LM 推論の高速化・スループット改善
- 前提：CUDA が利用可能であること
- 依存：`nanovllm` が import できること（`nano-vllm` が未インストールだと失敗）
- 失敗時：実装上、初期化失敗を検出すると PyTorch へフォールバックする設計

### pt（PyTorch）
- 目的：互換性・安定性重視
- 前提：CPU でも動くが、速度は環境次第（GPU 推奨）
- 依存：通常の PyTorch/Transformers の依存関係

## 「vllm 指定を切る（ptにする）」意味が大きいケース
- `nano-vllm` を入れていない/入れられない（環境制約）
- CUDA 周りや特定GPU環境で `vllm` 初期化が不安定
- まずは確実に動く経路で検証したい（デバッグ・切り分け）
- vllm のメモリ利用挙動（`gpu_memory_utilization` 等）が環境に合わない

※ なお「環境変数を unset する」だけだと既定が `vllm` なので、pt にしたい場合は **明示的に `ACESTEP_LM_BACKEND=pt`** を推奨します。

## LM を“使う/使わない”のスイッチではない点に注意
`ACESTEP_LM_BACKEND` は **LM の実装方式**であって、LM を無効化するスイッチではありません。

LM をそもそも使わない/軽くしたい場合は、次の設定と組み合わせて考えます。

- リクエストの `thinking: false`（LM による高品質生成を抑制）
- `ACESTEP_INIT_LLM`（サーバ起動時に LM を初期化するか）
- `ACESTEP_LM_OFFLOAD_TO_CPU`（LM のオフロード）
- `ACESTEP_LM_MODEL_PATH`（LM モデルサイズの選択）

## 設定例
### 起動時に vllm を使う
```bash
export ACESTEP_LM_BACKEND=vllm
```

### 起動時に pt を使う（互換性優先）
```bash
export ACESTEP_LM_BACKEND=pt
```

### リクエスト単位で pt を指定（対応している場合）
```json
{
  "thinking": true,
  "lm_backend": "pt"
}
```

## よくあるトラブルと対処
### 1) vllm を選んだが `nano-vllm` が無い
- 症状：ログに `nano-vllm is not installed` 相当が出る
- 対処：
  - `nano-vllm` をインストールする、または
  - `ACESTEP_LM_BACKEND=pt` にする（確実）

### 2) CUDA が無い/認識されない
- 症状：`CUDA is not available` 相当
- 対処：
  - GPU 環境を修正、または
  - `ACESTEP_LM_BACKEND=pt`（CPUでも動く可能性はあるが遅い）

## 参考ファイル
- `acestep/api_server.py`
- `acestep/llm_inference.py`
- `docs/ja/API.md`（`lm_backend` / `ACESTEP_LM_BACKEND` の項目）
