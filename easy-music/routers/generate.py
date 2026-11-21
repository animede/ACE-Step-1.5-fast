"""
音楽生成エンドポイント
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
import logging
import re
import httpx
from urllib.parse import urlencode

from services.ace_step_client import ace_step_client, TaskResult, SUPPORTED_LANGUAGES, SUPPORTED_KEY_SCALES
from config import settings

router = APIRouter(prefix="/api", tags=["generate"])
logger = logging.getLogger("uvicorn.error")


async def _release_local_llm() -> None:
    """ACE に生成を投げる直前に、同梱 LLM(CUDA 版 llama-server)を止めて VRAM を返し、「ACE 生成中」を立てる。

    12GB の GPU では ACE の拡散ピーク(約 8.5GB)と E4B(約 3GB+)が同時に入らないため。生成中フラグが立っている間は
    LLM の呼び出し(背景画像のプロンプトなど)は GPU での起動を待つ。フラグは /api/status が完了を返したとき、
    generate_and_wait の終了時、タスク作成に失敗したときに下ろす(保険として 20 分で自動解除)。
    LOCAL_LLM_STOP_BEFORE_ACE=0 で停止も待ちも無効。生成中フラグは task_id が返ったあと _ace_started() で立てる。
    """
    try:
        from services.local_llm_service import local_llm_service
        loop = asyncio.get_event_loop()
        stopped = await loop.run_in_executor(None, local_llm_service.release_before_ace)
        if stopped:
            logger.info("[generate] local LLM stopped before ACE generation (VRAM released)")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[generate] could not stop local LLM: {e}")


def _new_task_id() -> str:
    """ACE に渡す task_id(UUID)。投入の応答を受け取れなくても、この ID で中止できる。"""
    import uuid
    return str(uuid.uuid4())


async def _submit_error(e: Exception, task_id: str = "", what: str = "音楽生成タスク") -> HTTPException:
    """ACE へのタスク投入で起きた例外を、理由と次にやることが読める HTTPException にする。

    応答待ちの時間切れ(settings.ace_submit_timeout)では、サーバ側にタスクが作られている可能性があるので
    その task_id で /v1/cancel を送ってから返す(生成を続けさせると次の生成がそのぶん待たされる)。
    """
    log = logging.getLogger("uvicorn.error")
    if isinstance(e, httpx.TimeoutException):
        wait = int(settings.ace_submit_timeout)
        log.error("ACE-Step API の応答待ちがタイムアウト(%s、%d 秒、%s): %r", what, wait, settings.ace_step_api_url, e)
        cancelled = ""
        if task_id:
            try:
                r = await ace_step_client.cancel_task(task_id)
                state = ((r or {}).get("data") or {}).get("status") if isinstance(r, dict) else None
                log.info("[submit timeout] cancel %s -> %s", task_id, state or r)
                cancelled = "サーバ側の生成も中止しました。" if state in ("cancelled", "cancelling") else ""
            except Exception as ce:  # noqa: BLE001
                log.warning("[submit timeout] cancel %s failed: %s", task_id, ce)
            finally:
                _ace_finished()
        return HTTPException(status_code=504, detail=(
            f"ACE サーバが {wait} 秒以内に{what}を受け付けませんでした。{cancelled}もう一度「生成」を押してください"
            "(続けて起きるときは ACE サーバの端末を確認)"))
    log.exception("%s の作成に失敗: %s: %s", what, type(e).__name__, e)
    return HTTPException(status_code=500, detail=f"{what}の作成に失敗: {type(e).__name__}: {e}"[:400])

def _ace_started() -> None:
    """ACE タスクが作れたときに呼ぶ(task_id が返った後)。この間、同梱 LLM は GPU で起こさない。"""
    try:
        from services.local_llm_service import local_llm_service
        local_llm_service.ace_started()
    except Exception:  # noqa: BLE001
        pass


def _ace_finished() -> None:
    """ACE タスクが終わった(成功・失敗・作成失敗)ときに呼ぶ。LLM の GPU 起動を再び許す。"""
    try:
        from services.local_llm_service import local_llm_service
        local_llm_service.ace_finished()
    except Exception:  # noqa: BLE001
        pass


# SFT モデルのバッチサイズ上限（VRAM 安全対策: CFGありで実効バッチ×2のため）
SFT_MAX_BATCH_SIZE = 2


def clamp_batch_for_sft(model: Optional[str], batch_size: int) -> int:
    """SFT モデル選択時はバッチサイズを制限する（OOM 防止）"""
    if model and "sft" in model.lower() and batch_size > SFT_MAX_BATCH_SIZE:
        logging.info(f"[SFT guard] batch_size {batch_size} → {SFT_MAX_BATCH_SIZE} (model={model})")
        return SFT_MAX_BATCH_SIZE
    return batch_size


# =============================================================================
# Caption Sanitizer — ACE-Step 1.5 制約に合わせる
# =============================================================================

def sanitize_caption(text: str) -> str:
    """
    ACE-Step 1.5 API に安全に送る前にキャプションをサニタイズ。
    
    制約 (acestep/handler.py, constrained_logits_processor.py):
    - text_encoder max_length = 256 tokens
    - constrained decoder: caption < 512 tokens
    - バッククォート禁止
    - 改行は避ける（YAML値としてパースされる）
    - ピリオドで終わるのが理想的
    """
    if not text:
        return text
    # バッククォート除去
    text = text.replace('`', '')
    # 改行→スペース
    text = re.sub(r'[\n\r]+', ' ', text)
    # 制御文字除去
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Markdown記法除去
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # 連続スペース整理
    text = re.sub(r'\s{2,}', ' ', text).strip()
    # 480文字制限（安全マージン）
    if len(text) > 480:
        cut = text[:480].rfind('. ')
        if cut > 200:
            text = text[:cut + 1]
        else:
            text = text[:480].rstrip()
            last_space = text.rfind(' ')
            if last_space > 200:
                text = text[:last_space]
    # ピリオド終端
    if text and not text.endswith('.'):
        text = text.rstrip(',;:') + '.'
    return text


# =============================================================================
# Request/Response Models
# =============================================================================

class GenerateRequest(BaseModel):
    """音楽生成リクエスト"""
    prompt: str = Field(default="", description="音楽の説明（ジャンル、楽器、雰囲気など）")
    lyrics: str = Field(default="", description="歌詞（構造タグ付き）")
    thinking: bool = Field(default=True, description="LMで高品質生成")
    model: Optional[str] = Field(default=None, description="モデル名（省略時はサーバーデフォルト）")
    vocal_language: str = Field(default="ja", description="歌詞言語")
    audio_duration: int = Field(default=60, ge=-1, le=300, description="生成時間（秒） -1でLM自動推定")
    bpm: Optional[int] = Field(default=None, ge=30, le=300, description="テンポ")
    key_scale: Optional[str] = Field(default=None, description="調")
    time_signature: str = Field(default="4", description="拍子")
    batch_size: int = Field(default=1, ge=1, le=4, description="バッチサイズ")
    audio_format: str = Field(default="mp3", description="出力形式")
    seed: Optional[int] = Field(default=None, description="シード値")
    inference_steps: int = Field(default=150, ge=1, le=200, description="推論ステップ数")
    instrumental: bool = Field(default=False, description="インストゥルメンタルモード")
    sample_query: Optional[str] = Field(default=None, description="おまかせ生成: 自然言語で説明するとLMが全パラメータを自動生成")
    guidance_scale: float = Field(default=3.0, ge=0.0, le=20.0, description="CFGスケール")
    shift: Optional[float] = Field(default=None, ge=1.0, le=5.0, description="タイムステップシフト（セマンティクス vs ディテール）")
    infer_method: Optional[str] = Field(default=None, description="推論方式: ODE(決定論的) / SDE(確率的)")
    # LM パラメータ
    lm_temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="LM温度")
    lm_cfg_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0, description="LM CFGスケール")
    lm_top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="LM Top-P")
    use_cot_caption: Optional[bool] = Field(default=None, description="CoTキャプション使用")
    use_cot_language: Optional[bool] = Field(default=None, description="CoT言語判定使用")
    use_cot_metas: Optional[bool] = Field(default=None, description="CoTメタデータ推定使用")
    lm_negative_prompt: Optional[str] = Field(default=None, description="LMネガティブプロンプト")


class FormatInputRequest(BaseModel):
    """プロンプト強化リクエスト"""
    prompt: str = Field(default="", description="プロンプト")
    lyrics: str = Field(default="", description="歌詞")
    temperature: float = Field(default=0.85, ge=0.0, le=2.0, description="温度")


class GenerateResponse(BaseModel):
    """音楽生成レスポンス"""
    task_id: str
    status: str
    message: str = ""
    queue_position: Optional[int] = None
    seed: Optional[int] = None


class TaskStatusResponse(BaseModel):
    """タスクステータスレスポンス"""
    task_id: str
    status: int  # 0=processing, 1=succeeded, 2=failed
    status_text: str
    progress: Optional[int] = None  # サーバ報告の進捗（0-100）、なければNull
    queue_position: Optional[int] = None
    results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class GenerateAndWaitRequest(GenerateRequest):
    """生成して完了を待つリクエスト"""
    timeout: float = Field(default=300.0, description="タイムアウト（秒）")


class GenerateAndWaitResponse(BaseModel):
    """生成完了レスポンス"""
    success: bool
    results: List[Dict[str, Any]] = []
    error: Optional[str] = None


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/generate", response_model=GenerateResponse)
async def generate_music(request: GenerateRequest):
    """
    音楽生成タスクを作成
    
    タスクIDを返し、完了を待たない非同期処理
    """
    submit_task_id = ""
    try:
        extra_params: Dict[str, Any] = {}
        if request.model:
            extra_params["model"] = request.model
        if request.lm_temperature is not None:
            extra_params["lm_temperature"] = request.lm_temperature
        if request.lm_cfg_scale is not None:
            extra_params["lm_cfg_scale"] = request.lm_cfg_scale
        if request.lm_top_p is not None:
            extra_params["lm_top_p"] = request.lm_top_p
        if request.use_cot_caption is not None:
            extra_params["use_cot_caption"] = request.use_cot_caption
        if request.use_cot_language is not None:
            extra_params["use_cot_language"] = request.use_cot_language
        if request.use_cot_metas is not None:
            extra_params["use_cot_metas"] = request.use_cot_metas
        if request.lm_negative_prompt:
            extra_params["lm_negative_prompt"] = request.lm_negative_prompt
        if request.shift is not None:
            extra_params["shift"] = request.shift
        if request.infer_method:
            extra_params["infer_method"] = request.infer_method
        if request.instrumental:
            extra_params["instrumental"] = True
        if request.sample_query:
            extra_params["sample_query"] = request.sample_query

        # seedが未指定の場合はランダム生成して追跡できるようにする
        import random
        effective_seed = request.seed if request.seed is not None else random.randint(0, 2**31 - 1)
        # SFT モデル選択時のバッチサイズ制限
        effective_batch = clamp_batch_for_sft(request.model, request.batch_size)

        await _release_local_llm()
        submit_task_id = _new_task_id()
        result = await ace_step_client.release_task(
            task_id=submit_task_id,
            prompt=sanitize_caption(request.prompt),
            lyrics=request.lyrics,
            thinking=request.thinking,
            vocal_language=request.vocal_language,
            audio_duration=request.audio_duration,
            bpm=request.bpm,
            key_scale=request.key_scale,
            time_signature=request.time_signature,
            batch_size=effective_batch,
            audio_format=request.audio_format,
            seed=effective_seed,
            inference_steps=request.inference_steps,
            guidance_scale=request.guidance_scale,
            **extra_params
        )
        
        data = result.get("data", {})
        task_id = data.get("task_id", "")
        
        if not task_id:
            raise HTTPException(status_code=500, detail="Failed to create task")
        
        _ace_started()
        return GenerateResponse(
            task_id=task_id,
            status="queued",
            message="Task created successfully",
            queue_position=data.get("queue_position"),
            seed=effective_seed
        )
    
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        logging.getLogger("uvicorn.error").error(
            "ACE-Step API 接続失敗 (%s): %s", settings.ace_step_api_url, e
        )
        raise HTTPException(
            status_code=503,
            detail=f"ACE-Step APIに接続できません ({settings.ace_step_api_url})。"
                   f" --ace-url オプションでACE-Step APIのURLを指定してください。"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise await _submit_error(e, submit_task_id)


@router.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    """生成の中止。ACE サーバの /v1/cancel を呼び、「ACE 生成中」のフラグも下ろす(同梱 LLM を再び GPU で起こせるように)。"""
    try:
        result = await ace_step_client.cancel_task(task_id)
    except Exception as e:
        return {"success": False, "error": f"ACE サーバに接続できません: {e}"}
    finally:
        _ace_finished()
    code = result.get("code", 200) if isinstance(result, dict) else 500
    if code != 200:
        return {"success": False, "error": result.get("error") or f"HTTP {code}"}
    data = result.get("data", {}) or {}
    logger.info(f"[cancel] task {task_id} -> {data.get('status')}")
    return {"success": True, "task_id": task_id, "status": data.get("status")}


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    タスクステータスを取得
    
    一時的な接続エラーの場合は processing を返す（ポーリング継続可能）
    """
    try:
        result = await ace_step_client.query_result([task_id])
        
        if result.get("code") != 200:
            # ACE-Step API がエラーを返した場合も processing として扱う
            # （タスク投入直後はまだ認識されていない可能性がある）
            import logging
            logging.getLogger("uvicorn.error").warning(
                "query_result returned code=%s for task %s: %s",
                result.get("code"), task_id, result.get("error", "")
            )
            return TaskStatusResponse(
                task_id=task_id,
                status=0,
                status_text="processing"
            )
        
        data_list = result.get("data", [])
        if not data_list:
            return TaskStatusResponse(
                task_id=task_id,
                status=0,
                status_text="processing"
            )
        
        task_data = data_list[0]
        status = task_data.get("status", 0)
        
        status_map = {0: "processing", 1: "succeeded", 2: "failed"}
        status_text = status_map.get(status, "unknown")
        if status in (1, 2):
            _ace_finished()
        
        # サーバが進捗を送る場合はそのまま転送（将来対応）
        server_progress = task_data.get("progress")
        queue_position = task_data.get("queue_position")
        
        results = None
        error = None
        
        if status == 1:
            # 成功
            import json
            result_json = task_data.get("result", "[]")
            if isinstance(result_json, str):
                results = json.loads(result_json)
            else:
                results = result_json
            
            # URLを追加
            for r in results:
                if r.get("file"):
                    r["url"] = ace_step_client.get_audio_url(r["file"])
        
        elif status == 2:
            # 失敗 — result が JSON文字列の場合はパースして読みやすいメッセージに
            raw_error = task_data.get("result", "Unknown error")
            if isinstance(raw_error, str):
                try:
                    import json
                    parsed = json.loads(raw_error)
                    if isinstance(parsed, list):
                        error = "生成に失敗しました（サーバーエラー）"
                    elif isinstance(parsed, dict):
                        error = parsed.get("error") or parsed.get("message") or "生成に失敗しました"
                    else:
                        error = str(parsed)
                except (json.JSONDecodeError, TypeError):
                    error = raw_error
            elif isinstance(raw_error, list):
                error = "生成に失敗しました（サーバーエラー）"
            elif isinstance(raw_error, dict):
                error = raw_error.get("error") or raw_error.get("message") or "生成に失敗しました"
            else:
                error = str(raw_error)
        
        return TaskStatusResponse(
            task_id=task_id,
            status=status,
            status_text=status_text,
            progress=server_progress,
            queue_position=queue_position,
            results=results,
            error=error
        )
    
    except HTTPException:
        raise
    except Exception as e:
        # 一時的な接続エラー（httpx.ConnectError等）は processing として返す
        # フロントエンドは次のポーリングで再試行する
        import logging
        logging.getLogger("uvicorn.error").warning(
            "Transient error polling task %s: %s", task_id, e
        )
        return TaskStatusResponse(
            task_id=task_id,
            status=0,
            status_text="processing"
        )


@router.post("/generate_and_wait", response_model=GenerateAndWaitResponse)
async def generate_and_wait(request: GenerateAndWaitRequest):
    """
    音楽を生成し、完了まで待機
    
    同期的に結果を返す
    """
    submit_task_id = ""
    try:
        extra_params: Dict[str, Any] = {}
        if request.model:
            extra_params["model"] = request.model

        # タスク作成（SFT バッチ制限適用）
        effective_batch = clamp_batch_for_sft(request.model, request.batch_size)
        await _release_local_llm()
        submit_task_id = _new_task_id()
        result = await ace_step_client.release_task(
            task_id=submit_task_id,
            prompt=request.prompt,
            lyrics=request.lyrics,
            thinking=request.thinking,
            vocal_language=request.vocal_language,
            audio_duration=request.audio_duration,
            bpm=request.bpm,
            key_scale=request.key_scale,
            time_signature=request.time_signature,
            batch_size=effective_batch,
            audio_format=request.audio_format,
            seed=request.seed,
            **extra_params
        )
        
        data = result.get("data", {})
        task_id = data.get("task_id", "")
        
        if not task_id:
            return GenerateAndWaitResponse(
                success=False,
                error="Failed to create task"
            )
        
        # 完了を待機(この間 LLM は GPU で起こさない)
        _ace_started()
        try:
            task_results = await ace_step_client.wait_for_completion(
                task_id=task_id,
                timeout=request.timeout
            )
        finally:
            _ace_finished()
        
        # 結果を整形
        results = []
        for tr in task_results:
            results.append({
                "url": tr.url,
                "file": tr.file,
                "prompt": tr.prompt,
                "lyrics": tr.lyrics,
                "metas": {
                    "bpm": tr.metas.bpm if tr.metas else None,
                    "duration": tr.metas.duration if tr.metas else None,
                    "keyscale": tr.metas.keyscale if tr.metas else None,
                },
                "seed": tr.seed_value
            })
        
        return GenerateAndWaitResponse(
            success=True,
            results=results
        )
    
    except TimeoutError as e:
        return GenerateAndWaitResponse(
            success=False,
            error=str(e)
        )
    except Exception as e:
        if isinstance(e, httpx.TimeoutException):  # 投入の応答待ち切れ: サーバ側も中止して理由を返す
            return GenerateAndWaitResponse(success=False, error=(await _submit_error(e, submit_task_id)).detail)
        logging.getLogger("uvicorn.error").exception("generate_and_wait failed: %s: %s", type(e).__name__, e)
        return GenerateAndWaitResponse(
            success=False,
            error=f"{type(e).__name__}: {e}"
        )


@router.get("/languages")
async def get_languages():
    """サポート言語一覧を取得"""
    return {"languages": SUPPORTED_LANGUAGES}


@router.get("/key_scales")
async def get_key_scales():
    """サポートキースケール一覧を取得"""
    return {"key_scales": SUPPORTED_KEY_SCALES}


@router.get("/ace_ready")
async def ace_ready():
    """ACE サーバが生成を受け付けられるか。フロントは準備ができるまで「準備中」を出して生成を待たせる。

    ACE はモデルを読み込み終わるまでポートを開かないので、起動中と未起動は区別できない(どちらも接続不可)。
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{ace_step_client.base_url}/health", headers=ace_step_client._build_headers())
        data = (r.json() or {}).get("data", {}) if r.status_code == 200 else {}
        if r.status_code == 200 and data.get("models_initialized", True):
            return {"ready": True, "url": ace_step_client.base_url}
        return {"ready": False, "reason": "initializing", "url": ace_step_client.base_url}
    except Exception as e:  # noqa: BLE001
        return {"ready": False, "reason": "unreachable", "detail": f"{type(e).__name__}", "url": ace_step_client.base_url}


@router.get("/health")
async def health_check():
    """ACE-Step APIのヘルスチェック"""
    try:
        result = await ace_step_client.health_check()
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/random_sample")
async def get_random_sample():
    """ランダムサンプルパラメータを取得（おまかせ生成用）"""
    try:
        result = await ace_step_client.get_random_sample("simple_mode")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def get_models():
    """ACE-Step APIのモデル情報を取得"""
    try:
        result = await ace_step_client.get_models()
        data = result.get("data", {})
        return {
            "success": True,
            "models": data.get("models", []),
            "default_model": data.get("default_model", "unknown")
        }
    except Exception as e:
        return {"success": False, "error": str(e), "models": [], "default_model": "unknown"}


class LoadModelRequest(BaseModel):
    model: str = Field(..., description="読み込む DiT モデル名(例: acestep-v15-turbo)")


def pick_slot_for_model(slots: List[Dict[str, Any]], model: str) -> Optional[int]:
    """入れ替えに使うスロットを選ぶ。primary(1)は既定の生成に使うので触らない。

    優先: 未読込のスロット → GPU に載っていないスロット → 番号の大きいスロット。
    スロットが 1 本しか無い(ACESTEP_CONFIG_PATH2 未設定)なら None。
    """
    candidates = [s for s in slots if int(s.get("slot", 0)) != 1]
    if not candidates:
        return None
    for s in candidates:
        if s.get("model") == model:
            return int(s["slot"])
    candidates.sort(key=lambda s: (bool(s.get("loaded")), bool(s.get("on_gpu")), -int(s["slot"])))
    return int(candidates[0]["slot"])


@router.post("/models/load")
async def load_model(req: LoadModelRequest):
    """プルダウンで選んだモデルがスロットに無いとき、primary 以外のスロットに読み込んで切り替える。

    返り値: {success, loaded, slot, model, message}。VRAM に載らない・スロットが無いときは success=False と理由。
    """
    model = (req.model or "").strip()
    if not model:
        return {"success": False, "error": "model が空です"}
    try:
        inv = (await ace_step_client.get_models()).get("data", {}) or {}
    except Exception as e:
        return {"success": False, "error": f"ACE サーバに接続できません: {e}"}
    entry = next((m for m in inv.get("models", []) if m.get("name") == model), None)
    if entry is None:
        return {"success": False, "error": f"{model} は checkpoints/ にありません。setup で取得してください"}
    if entry.get("is_loaded"):
        return {"success": True, "loaded": True, "model": model, "slot": None, "message": "読み込み済み"}
    if entry.get("vram_ok") is False:
        return {"success": False, "error": f"{model} はこの GPU には載りません(約 {entry.get('min_vram_gb')}GB 必要)。2B 系のモデルを選んでください"}
    try:
        slots = ((await ace_step_client.get_residency()).get("data", {}) or {}).get("slots", []) or []
    except Exception as e:
        return {"success": False, "error": f"スロット情報を取得できません: {e}"}
    slot = pick_slot_for_model(slots, model)
    if slot is None:
        return {"success": False, "error": "このサーバはスロットが 1 本だけなので起動後の切り替えはできません。起動時に ACESTEP_CONFIG_PATH で指定してください"}
    logger.info(f"[models/load] {model} → slot {slot} (slots={slots})")
    try:
        result = await ace_step_client.init_model(model, slot)
    except Exception as e:
        return {"success": False, "error": str(e)}
    data = result.get("data", {}) if isinstance(result, dict) else {}
    return {"success": True, "loaded": True, "model": data.get("loaded_model") or model, "slot": data.get("slot", slot), "message": "切り替えました"}


@router.get("/stats")
async def get_stats():
    """ACE-Step APIの統計情報を取得"""
    try:
        result = await ace_step_client.get_stats()
        data = result.get("data", {})
        return {
            "success": True,
            "jobs": data.get("jobs", {}),
            "queue_size": data.get("queue_size", 0),
            "avg_job_seconds": data.get("avg_job_seconds", 0)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/format_input")
async def format_input(request: FormatInputRequest):
    """
    LLMでプロンプト/歌詞を強化・補完

    ACE-Step の /format_input を呼び出し、caption/lyrics/BPM等を返す
    """
    try:
        # ACE の format_input は LM(1.3GB)とテキストエンコーダ(1.2GB)を GPU に上げる。同梱 LLM が GPU に
        # 残っていると 8GB では合計が越えて共有 GPU メモリへあふれるので、ここでも先に止める(次の呼び出しで起き直す)
        await _release_local_llm()
        result = await ace_step_client.format_input(
            prompt=request.prompt,
            lyrics=request.lyrics,
            temperature=request.temperature,
        )
        data = result.get("data", {})
        return {
            "success": True,
            "caption": data.get("caption", ""),
            "lyrics": data.get("lyrics", ""),
            "bpm": data.get("bpm"),
            "key_scale": data.get("keyscale", data.get("key_scale", "")),
            "time_signature": data.get("timesignature", data.get("time_signature", "")),
            "duration": data.get("duration"),
            "vocal_language": data.get("language", data.get("vocal_language", "")),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/generate_cover", response_model=GenerateResponse)
async def generate_cover(
    src_audio: UploadFile = File(..., description="ソース音声ファイル"),
    prompt: str = Form(default="", description="音楽の説明"),
    lyrics: str = Form(default="", description="歌詞"),
    thinking: bool = Form(default=True),
    model: Optional[str] = Form(default=None),
    vocal_language: str = Form(default="ja"),
    audio_duration: int = Form(default=60),
    bpm: Optional[int] = Form(default=None),
    key_scale: Optional[str] = Form(default=None),
    time_signature: str = Form(default="4"),
    batch_size: int = Form(default=1),
    audio_format: str = Form(default="mp3"),
    seed: Optional[int] = Form(default=None),
    inference_steps: int = Form(default=8),
    guidance_scale: float = Form(default=3.0),
    shift: Optional[float] = Form(default=None),
    infer_method: Optional[str] = Form(default=None),
    audio_cover_strength: float = Form(default=0.8, description="カバー強度 0.0～1.0"),
    lm_temperature: Optional[float] = Form(default=None),
    lm_cfg_scale: Optional[float] = Form(default=None),
    lm_top_p: Optional[float] = Form(default=None),
    use_cot_caption: Optional[bool] = Form(default=None),
    use_cot_language: Optional[bool] = Form(default=None),
    lm_negative_prompt: Optional[str] = Form(default=None),
):
    """
    カバーモードで音楽生成タスクを作成

    ソース音声をアップロードし、声質・スタイルを参照して新しい音楽を生成
    """
    submit_task_id = ""
    try:
        src_audio_bytes = await src_audio.read()
        src_audio_filename = src_audio.filename or "upload.mp3"

        # カバー強度 < 1.0 のときは CFG=1.0 を強制
        effective_cfg = guidance_scale
        if audio_cover_strength < 1.0:
            effective_cfg = 1.0

        # SFT モデル選択時のバッチサイズ制限
        effective_batch = clamp_batch_for_sft(model, batch_size)

        # パラメータを構築
        form_fields = {
            "prompt": sanitize_caption(prompt),
            "lyrics": lyrics,
            "thinking": str(thinking).lower(),
            "vocal_language": vocal_language,
            "audio_duration": str(audio_duration),
            "time_signature": time_signature,
            "batch_size": str(effective_batch),
            "audio_format": audio_format,
            "inference_steps": str(inference_steps),
            "guidance_scale": str(effective_cfg),
            "task_type": "cover",
            "audio_cover_strength": str(audio_cover_strength),
        }
        if model:
            form_fields["model"] = model
        if bpm is not None:
            form_fields["bpm"] = str(bpm)
        if key_scale:
            form_fields["key_scale"] = key_scale
        if seed is not None:
            form_fields["seed"] = str(seed)
        if lm_temperature is not None:
            form_fields["lm_temperature"] = str(lm_temperature)
        if lm_cfg_scale is not None:
            form_fields["lm_cfg_scale"] = str(lm_cfg_scale)
        if lm_top_p is not None:
            form_fields["lm_top_p"] = str(lm_top_p)
        if use_cot_caption is not None:
            form_fields["use_cot_caption"] = str(use_cot_caption).lower()
        if use_cot_language is not None:
            form_fields["use_cot_language"] = str(use_cot_language).lower()
        if lm_negative_prompt:
            form_fields["lm_negative_prompt"] = lm_negative_prompt
        if shift is not None:
            form_fields["shift"] = str(shift)
        if infer_method:
            form_fields["infer_method"] = infer_method

        await _release_local_llm()
        submit_task_id = _new_task_id()
        form_fields["task_id"] = submit_task_id
        result = await ace_step_client.release_task_multipart(
            form_fields=form_fields,
            src_audio_bytes=src_audio_bytes,
            src_audio_filename=src_audio_filename,
        )

        data = result.get("data", {})
        task_id = data.get("task_id", "")

        if not task_id:
            raise HTTPException(status_code=500, detail="Failed to create cover task")

        _ace_started()
        return GenerateResponse(
            task_id=task_id,
            status="queued",
            message="Cover task created successfully",
            queue_position=data.get("queue_position")
        )
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        logging.getLogger("uvicorn.error").error(
            "ACE-Step API 接続失敗 (%s): %s", settings.ace_step_api_url, e
        )
        raise HTTPException(
            status_code=503,
            detail=f"ACE-Step APIに接続できません ({settings.ace_step_api_url})。"
                   f" --ace-url オプションでACE-Step APIのURLを指定してください。"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise await _submit_error(e, submit_task_id)


@router.post("/generate_repaint", response_model=GenerateResponse)
async def generate_repaint(
    src_audio: UploadFile = File(..., description="リペイント対象の音声ファイル"),
    repainting_start: float = Form(default=0.0, description="リペイント開始位置（秒）"),
    repainting_end: float = Form(default=0.0, description="リペイント終了位置（秒）"),
    prompt: str = Form(default="", description="音楽の説明"),
    lyrics: str = Form(default="", description="歌詞"),
    thinking: bool = Form(default=True),
    model: Optional[str] = Form(default=None),
    vocal_language: str = Form(default="ja"),
    audio_duration: int = Form(default=60),
    bpm: Optional[int] = Form(default=None),
    key_scale: Optional[str] = Form(default=None),
    time_signature: str = Form(default="4"),
    batch_size: int = Form(default=1),
    audio_format: str = Form(default="mp3"),
    seed: Optional[int] = Form(default=None),
    inference_steps: int = Form(default=8),
    guidance_scale: float = Form(default=3.0),
    lm_temperature: Optional[float] = Form(default=None),
    lm_cfg_scale: Optional[float] = Form(default=None),
    lm_top_p: Optional[float] = Form(default=None),
    use_cot_caption: Optional[bool] = Form(default=None),
    use_cot_language: Optional[bool] = Form(default=None),
    lm_negative_prompt: Optional[str] = Form(default=None),
    shift: Optional[float] = Form(default=None),
    infer_method: Optional[str] = Form(default=None),
):
    """
    リペイントモードで音楽の一部を再生成

    元の音声をアップロードし、指定区間のみを再生成する
    """
    submit_task_id = ""
    try:
        src_audio_bytes = await src_audio.read()
        src_audio_filename = src_audio.filename or "upload.mp3"

        # SFT モデル選択時のバッチサイズ制限
        effective_batch = clamp_batch_for_sft(model, batch_size)

        form_fields = {
            "prompt": sanitize_caption(prompt),
            "lyrics": lyrics,
            "thinking": str(thinking).lower(),
            "vocal_language": vocal_language,
            "audio_duration": str(audio_duration),
            "time_signature": time_signature,
            "batch_size": str(effective_batch),
            "audio_format": audio_format,
            "inference_steps": str(inference_steps),
            "guidance_scale": str(guidance_scale),
            "task_type": "repaint",
            "repainting_start": str(repainting_start),
            "repainting_end": str(repainting_end),
        }
        if model:
            form_fields["model"] = model
        if bpm is not None:
            form_fields["bpm"] = str(bpm)
        if key_scale:
            form_fields["key_scale"] = key_scale
        if seed is not None:
            form_fields["seed"] = str(seed)
        if lm_temperature is not None:
            form_fields["lm_temperature"] = str(lm_temperature)
        if lm_cfg_scale is not None:
            form_fields["lm_cfg_scale"] = str(lm_cfg_scale)
        if lm_top_p is not None:
            form_fields["lm_top_p"] = str(lm_top_p)
        if use_cot_caption is not None:
            form_fields["use_cot_caption"] = str(use_cot_caption).lower()
        if use_cot_language is not None:
            form_fields["use_cot_language"] = str(use_cot_language).lower()
        if lm_negative_prompt:
            form_fields["lm_negative_prompt"] = lm_negative_prompt
        if shift is not None:
            form_fields["shift"] = str(shift)
        if infer_method:
            form_fields["infer_method"] = infer_method

        await _release_local_llm()
        submit_task_id = _new_task_id()
        form_fields["task_id"] = submit_task_id
        result = await ace_step_client.release_task_multipart(
            form_fields=form_fields,
            src_audio_bytes=src_audio_bytes,
            src_audio_filename=src_audio_filename,
        )

        data = result.get("data", {})
        task_id = data.get("task_id", "")

        if not task_id:
            raise HTTPException(status_code=500, detail="Failed to create repaint task")

        _ace_started()
        return GenerateResponse(
            task_id=task_id,
            status="queued",
            message="Repaint task created successfully"
        )
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        logging.getLogger("uvicorn.error").error(
            "ACE-Step API 接続失敗 (%s): %s", settings.ace_step_api_url, e
        )
        raise HTTPException(
            status_code=503,
            detail=f"ACE-Step APIに接続できません ({settings.ace_step_api_url})。"
                   f" --ace-url オプションでACE-Step APIのURLを指定してください。"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise await _submit_error(e, submit_task_id)


# ---------------------------------------------------------------------------
# LRC（歌詞タイミング）生成
# ---------------------------------------------------------------------------
_lrc_cache: Dict[str, Any] = {}   # key = audio_url → LRC JSON


@router.post("/lrc")
async def generate_lrc_endpoint(payload: dict):
    """
    音声 URL ＋ 歌詞テキストから LRC タイムスタンプを生成する。

    Request body:
        { "audio_url": "/v1/audio?path=...", "lyrics": "..." }
    Returns:
        { "lrc": [ {"time": 12.3, "text": "...", "section": "verse"}, ... ] }
        歌詞なし / 解析不可: { "lrc": null }
    """
    audio_url = payload.get("audio_url", "")
    lyrics = payload.get("lyrics", "")

    if not audio_url or not lyrics or lyrics.strip() == "[inst]":
        return {"lrc": None}

    # キャッシュ確認
    if audio_url in _lrc_cache:
        return {"lrc": _lrc_cache[audio_url]}

    # 音声ダウンロード（ACE-Step サーバから）
    # audio_url のパターン:
    #   "/v1/audio?path=..."                 → ACE-Step 直接パス
    #   "/api/audio?path=..."                → 自サーバプロキシ URL → ACE-Step に変換
    #   "http://192.168.x.x:8001/v1/audio?path=..." → 完全 URL
    if audio_url.startswith("/api/audio"):
        # 自サーバプロキシ URL → ACE-Step URL に変換
        import urllib.parse
        parsed = urllib.parse.urlparse(audio_url)
        qs = urllib.parse.parse_qs(parsed.query)
        ace_path = qs.get("path", [""])[0]
        if ace_path:
            full_url = f"{settings.ace_step_api_url}/v1/audio?path={ace_path}"
        else:
            full_url = f"{settings.ace_step_api_url}{audio_url}"
    elif audio_url.startswith("/"):
        full_url = f"{settings.ace_step_api_url}{audio_url}"
    else:
        full_url = audio_url
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(full_url, timeout=60.0)
            resp.raise_for_status()
            audio_bytes = resp.content
    except Exception as e:
        logger.warning("LRC: 音声ダウンロード失敗: %s", e)
        return {"lrc": None}

    # エネルギー解析 + LRC 生成（CPU 処理なのでブロッキングだが <1秒）
    from services.lrc_service import generate_lrc, lrc_lines_to_json

    loop = asyncio.get_event_loop()
    try:
        lrc_lines = await loop.run_in_executor(
            None, generate_lrc, audio_bytes, lyrics
        )
    except Exception as e:
        logger.warning("LRC: 解析失敗: %s", e)
        return {"lrc": None}

    if lrc_lines is None:
        _lrc_cache[audio_url] = None
        return {"lrc": None}

    result = lrc_lines_to_json(lrc_lines)
    _lrc_cache[audio_url] = result
    return {"lrc": result}


@router.get("/audio")
async def proxy_audio(path: str):
    """
    ACE-Step APIからの音声ファイルをプロキシ
    CORSの問題を回避するため
    """
    audio_url = f"{settings.ace_step_api_url}/v1/audio?{urlencode({'path': path})}"
    last_err = None

    for attempt in range(3):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(audio_url, timeout=60.0)

                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail="Audio not found")

                # Content-Typeを取得
                content_type = response.headers.get("content-type", "audio/mpeg")

                return StreamingResponse(
                    iter([response.content]),
                    media_type=content_type,
                    headers={
                        "Accept-Ranges": "bytes",
                        "Content-Length": str(len(response.content))
                    }
                )
        except httpx.RequestError as e:
            last_err = e
            if attempt < 2:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            raise HTTPException(status_code=502, detail=f"Failed to fetch audio: {str(e)}")
