/**
 * ACE-Step Standalone - Frontend JavaScript
 */

console.log('ACE-Step app.js loaded');

// =============================================================================
// API Functions
// =============================================================================

/**
 * 音声URLをプロキシ経由にURLへ変換
 * CORSの問題を回避するため、ACE-Step APIの音声をバックエンド経由で取得
 */
function convertAudioUrl(url) {
    if (!url) return url;
    
    // 外部URL（ACE-Step API）の場合、プロキシ経由に変換
    try {
        const urlObj = new URL(url);
        // /v1/audio?path=... の形式の場合
        if (urlObj.pathname === '/v1/audio') {
            const path = urlObj.searchParams.get('path');
            if (path) {
                return `/api/audio?path=${encodeURIComponent(path)}`;
            }
        }
    } catch (e) {
        // 相対URLの場合はそのまま返す
    }
    
    return url;
}

/**
 * APIリクエストを送信
 */
async function apiRequest(endpoint, method = 'GET', data = null) {
    console.log(`API Request: ${method} ${endpoint}`, data);
    
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json',
        },
    };
    
    if (data) {
        options.body = JSON.stringify(data);
    }
    
    try {
        const response = await fetch(endpoint, options);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('API Error:', errorText);
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('API Response:', result);
        return result;
    } catch (e) {
        console.error('API Request failed:', e);
        throw e;
    }
}

// =============================================================================
// UI Helper Functions
// =============================================================================

function _normalizeModelName(modelName) {
    return (modelName || '').toString().trim().toLowerCase();
}

function getRecommendedStepsForModel(modelName) {
    const m = _normalizeModelName(modelName);
    if (!m) return null;

    // User requirement: turbo=8, base=50
    if (m.includes('turbo')) return 8;
    if (m.includes('base')) return 50;
    // Treat other non-turbo variants similarly to base unless explicitly handled
    if (m.includes('sft')) return 50;
    return null;
}

function applyRecommendedSteps(modelName) {
    const stepInput = document.getElementById('inference_steps');
    if (!stepInput) return;

    const recommended = getRecommendedStepsForModel(modelName);
    if (recommended == null) return;

    // Force overwrite mode: always set recommended STEP when model changes.
    stepInput.value = recommended;

    stepInput.title = `推論ステップ数（モデル推奨: ${recommended}）`;
}

function getEffectiveRequestedModel() {
    const modelSelect = document.getElementById('model');
    const requested = modelSelect?.value || '';
    return requested || (window.__aceStepDefaultModel || '');
}

function refreshStepsFromModelSelection() {
    applyRecommendedSteps(getEffectiveRequestedModel());
}

/**
 * ステータスメッセージを表示
 */
function showStatus(message, type = 'info') {
    const statusEl = document.getElementById('status-message');
    statusEl.textContent = message;
    statusEl.className = `status-message ${type}`;
    statusEl.style.display = 'block';
}

/**
 * ステータスメッセージを非表示
 */
function hideStatus() {
    document.getElementById('status-message').style.display = 'none';
}

/**
 * 進捗バーを表示/更新
 */
function showProgress(percent, text = '処理中...') {
    const container = document.getElementById('progress-container');
    const fill = document.getElementById('progress-fill');
    const textEl = document.getElementById('progress-text');
    
    container.style.display = 'block';
    fill.style.width = `${percent}%`;
    // パーセンテージを表示テキストに含める
    const pct = Math.round(percent);
    textEl.textContent = pct > 0 && pct < 100 ? `${text} [${pct}%]` : text;
}

/**
 * 進捗バーを非表示
 */
function hideProgress() {
    document.getElementById('progress-container').style.display = 'none';
}

/**
 * 生成時間を推定（秒）
 * 実測データ: steps=8, dur=10s → LM 3.18s + DiT 0.48s = 3.66s
 *
 * ACE-Step 処理フェーズ:
 *   1. キュー待ち: 0-数秒（queue_position による）
 *   2. LM (CoT) フェーズ: ~3-5秒
 *   3. DiT フェーズ: steps × duration に比例（実測: 0.006s/step/秒音声）
 *   4. VAE デコード + 転送: ~1-2秒
 */
function estimateGenerationTime(params) {
    const steps = params.inference_steps || 150;
    const duration = params.audio_duration || 60;
    const batch = params.batch_size || 1;
    const thinking = params.thinking !== false;
    
    // LM フェーズ: thinking=true で ~3-5秒、false でも ~1-2秒
    const lmTime = thinking ? 4 : 1.5;
    
    // DiT フェーズ: 実測 steps=8, dur=10 → 0.48s
    // → 0.48 / (8 * 10) = 0.006 s/step/秒音声
    const ditTime = steps * duration * 0.006;
    
    // バッチサイズで DiT 時間がほぼ線形に増加
    const batchFactor = 1 + (batch - 1) * 0.8;
    
    // オーバーヘッド（VAEデコード、MP3エンコード、転送）
    const overhead = 2;
    
    // 安全マージン 1.3x（キュー・GPU温度・負荷変動）
    const estimated = (lmTime + ditTime * batchFactor + overhead) * 1.3;
    
    // 最低8秒、最大600秒
    return Math.max(8, Math.min(600, estimated));
}

/**
 * 滑らかな進捗カーブ（3フェーズ対応）
 * フェーズ1 (0-15%): LM処理 — 最初の~4秒で素早く進む
 * フェーズ2 (15-85%): DiT生成 — メインの生成時間
 * フェーズ3 (85-95%): 仕上げ — ゆっくり収束
 * 予測時間超過後も95%で待機し、完了時に100%へジャンプ
 */
function smoothProgress(elapsedSec, expectedSec) {
    if (expectedSec <= 0) return 5;
    const ratio = elapsedSec / expectedSec;
    
    if (ratio <= 0) return 5;
    
    if (ratio <= 0.15) {
        // フェーズ1: 急速に15%まで（LM処理感）
        return 5 + (ratio / 0.15) * 15;
    } else if (ratio <= 0.85) {
        // フェーズ2: 線形に85%まで（DiT生成のメインフェーズ）
        const phaseRatio = (ratio - 0.15) / 0.70;
        return 20 + phaseRatio * 55;
    } else if (ratio <= 1.0) {
        // フェーズ3: イーズアウトで85%まで
        const phaseRatio = (ratio - 0.85) / 0.15;
        const eased = 1 - Math.pow(1 - phaseRatio, 2);
        return 75 + eased * 13;
    } else {
        // 予測超過: 88%→95%をゆっくり漸近
        const overRatio = Math.min(ratio - 1, 2);
        return Math.min(95, 88 + overRatio * 3.5);
    }
}

/**
 * 経過時間を読みやすい文字列に
 */
function formatElapsed(sec) {
    if (sec < 60) return `${sec}秒`;
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return s > 0 ? `${m}分${s}秒` : `${m}分`;
}

/**
 * 生成フェーズの表示テキスト
 */
function getProgressPhaseText(elapsedSec, expectedSec, prefix = '生成中') {
    const ratio = expectedSec > 0 ? elapsedSec / expectedSec : 0;
    const remaining = Math.max(0, Math.round(expectedSec - elapsedSec));
    
    if (ratio < 0.12) {
        return `${prefix}... (LM処理中)`;
    } else if (ratio < 0.85) {
        return remaining > 0
            ? `${prefix}... (残り約${formatElapsed(remaining)})`
            : `${prefix}... (${formatElapsed(elapsedSec)}経過)`;
    } else if (ratio < 1.0) {
        return `${prefix}... (仕上げ処理中)`;
    } else {
        return `${prefix}... (${formatElapsed(elapsedSec)}経過 — まもなく完了)`;
    }
}

// ビジュアライザー用のグローバル変数
let audioContext = null;
let analyser = null;
let dataArray = null;
let animationId = null;
let isVisualizerInitialized = false;

// 生成結果のグローバル変数
let currentResults = [];

/**
 * オーディオプレイヤーを表示
 */
function showAudioPlayer(url, metadata = {}) {
    const playerSection = document.getElementById('player-section');
    const audio = document.getElementById('audio');
    const downloadBtn = document.getElementById('download-btn');
    
    audio.src = url;
    downloadBtn.href = url;
    
    // 再生セクションを表示
    playerSection.classList.add('visible');
    
    // スムーズにスクロール
    playerSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    
    // ビジュアライザーを初期化
    initVisualizer();

    // 生成後は自動再生（ブラウザの制限で失敗する場合あり）
    setTimeout(() => {
        const playPromise = audio.play();
        if (playPromise && typeof playPromise.catch === 'function') {
            playPromise.catch(() => {
                showStatus('🎧 再生を開始できませんでした。再生ボタンを押してください。', 'info');
            });
        }
    }, 0);
}

/**
 * ビジュアライザーを初期化
 */
function initVisualizer() {
    const audio = document.getElementById('audio');
    
    if (isVisualizerInitialized) return;
    
    // 再生開始時にAudioContextを作成（ユーザー操作後のみ許可）
    audio.addEventListener('play', () => {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            
            const source = audioContext.createMediaElementSource(audio);
            source.connect(analyser);
            analyser.connect(audioContext.destination);
            
            const bufferLength = analyser.frequencyBinCount;
            dataArray = new Uint8Array(bufferLength);
        }
        
        if (audioContext.state === 'suspended') {
            audioContext.resume();
        }
        
        drawVisualizer();
    });
    
    audio.addEventListener('pause', () => {
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
    });
    
    audio.addEventListener('ended', () => {
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
        clearVisualizer();
    });
    
    isVisualizerInitialized = true;
}

/**
 * ビジュアライザーを描画
 */
function drawVisualizer() {
    const canvas = document.getElementById('visualizer');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    function draw() {
        animationId = requestAnimationFrame(draw);
        
        analyser.getByteFrequencyData(dataArray);
        
        // 背景をクリア（グラデーション）
        const gradient = ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, '#1a1a2e');
        gradient.addColorStop(1, '#16213e');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
        
        const barCount = 64;
        const barWidth = (width / barCount) * 0.8;
        const barGap = (width / barCount) * 0.2;

        // 高音域の上位1/4は表示しない（低〜中域に寄せる）
        const maxBin = Math.max(1, Math.floor(dataArray.length * 0.75));
        
        for (let i = 0; i < barCount; i++) {
            const dataIndex = Math.floor(i * maxBin / barCount);
            const barHeight = (dataArray[dataIndex] / 255) * height * 0.9;
            
            const x = i * (barWidth + barGap);
            const y = height - barHeight;
            
            // カラフルなグラデーション
            const hue = (i / barCount) * 360;
            const saturation = 80 + (dataArray[dataIndex] / 255) * 20;
            const lightness = 50 + (dataArray[dataIndex] / 255) * 20;
            
            // グロー効果
            ctx.shadowBlur = 15;
            ctx.shadowColor = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
            
            // バーを描画
            const barGradient = ctx.createLinearGradient(x, y, x, height);
            barGradient.addColorStop(0, `hsl(${hue}, ${saturation}%, ${lightness}%)`);  
            barGradient.addColorStop(1, `hsl(${hue}, ${saturation}%, ${lightness - 20}%)`);
            
            ctx.fillStyle = barGradient;
            ctx.beginPath();
            ctx.roundRect(x, y, barWidth, barHeight, 3);
            ctx.fill();
        }
        
        ctx.shadowBlur = 0;
    }
    
    draw();
}

/**
 * ビジュアライザーをクリア
 */
function clearVisualizer() {
    const canvas = document.getElementById('visualizer');
    const ctx = canvas.getContext('2d');
    
    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
    gradient.addColorStop(0, '#1a1a2e');
    gradient.addColorStop(1, '#16213e');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

/**
 * ボタンを無効化
 */
function disableButton(id, loading = true) {
    const btn = document.getElementById(id);
    btn.disabled = true;
    if (loading) {
        btn.dataset.originalText = btn.innerHTML;
        btn.innerHTML = '<span class="spinner"></span> 処理中...';
    }
}

/**
 * ボタンを有効化
 */
function enableButton(id) {
    const btn = document.getElementById(id);
    btn.disabled = false;
    if (btn.dataset.originalText) {
        btn.innerHTML = btn.dataset.originalText;
    }
}

/**
 * 構造タグを挿入
 */
function insertTag(tag) {
    const textarea = document.getElementById('lyrics');
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const text = textarea.value;
    
    textarea.value = text.substring(0, start) + tag + '\n' + text.substring(end);
    textarea.focus();
    textarea.selectionStart = textarea.selectionEnd = start + tag.length + 1;
}

/**
 * ジャンルボタンクリックでキャプション生成
 * テーマ・歌詞があればそれも加味してLLMでナラティブキャプションを生成
 */
async function genreCaption(genre) {
    const theme = document.getElementById('theme').value.trim();
    const lyrics = document.getElementById('lyrics').value.trim();
    const language = document.getElementById('language')?.value || 'ja';
    const langMap = { 'ja': 'Japanese', 'en': 'English', 'zh': 'Chinese', 'ko': 'Korean', 'es': 'Spanish', 'fr': 'French', 'de': 'German' };
    const langName = langMap[language] || 'Japanese';

    // 全ジャンルボタンを無効化
    const allBtns = document.querySelectorAll('.genre-suggestion');
    allBtns.forEach(b => b.classList.add('disabled'));
    const clickedBtn = event?.target;
    if (clickedBtn) clickedBtn.textContent = genre + ' …';

    try {
        showStatus(`🎵 ${genre} キャプションを生成中...`, 'info');

        const result = await apiRequest('/api/caption', 'POST', {
            theme: theme,
            lyrics: lyrics,
            tags: genre,
            language: langName
        });

        if (result.success && result.caption) {
            document.getElementById('prompt').value = result.caption;
            showStatus(`✅ ${genre} キャプション生成完了！`, 'success');
        } else {
            // フォールバック: 単純にジャンル名を追加
            addGenreSimple(genre);
            showStatus(`⚠️ LLMエラー。ジャンル名のみ追加しました`, 'error');
        }
    } catch (e) {
        // LLM接続失敗時は従来動作にフォールバック
        addGenreSimple(genre);
        showStatus(`⚠️ LLM未接続。ジャンル名のみ追加しました`, 'error');
    } finally {
        allBtns.forEach(b => b.classList.remove('disabled'));
        if (clickedBtn) clickedBtn.textContent = genre.charAt(0).toUpperCase() + genre.slice(1);
    }
}

/**
 * ジャンル名を単純に追加（フォールバック用）
 */
function addGenreSimple(genre) {
    const promptEl = document.getElementById('prompt');
    const current = promptEl.value.trim();
    if (current) {
        promptEl.value = current + ', ' + genre;
    } else {
        promptEl.value = genre;
    }
}

/**
 * アコーディオンをトグル
 */
function toggleAccordion(id) {
    const accordion = document.getElementById(id);
    accordion.classList.toggle('open');

    const isOpen = accordion.classList.contains('open');
    const toggles = document.querySelectorAll(`[data-accordion-target="${id}"]`);
    toggles.forEach((toggle) => {
        toggle.classList.toggle('open', isOpen);
        toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    });
}

/**
 * フォームをリセット
 */
function resetForm() {
    document.getElementById('theme').value = '';
    document.getElementById('lyrics').value = '';
    document.getElementById('prompt').value = '';
    document.getElementById('audio_duration').value = 150;
    document.getElementById('bpm').value = 120;
    document.getElementById('inference_steps').value = 150;
    document.getElementById('guidance_scale').value = 3.0;
    document.getElementById('key_scale').value = '';
    document.getElementById('seed').value = '';
    
    hideStatus();
    hideProgress();
    
    // 再生セクションを非表示
    document.getElementById('player-section').classList.remove('visible');
    
    // 結果をクリア
    currentResults = [];
    document.getElementById('results-grid').innerHTML = '';
    document.getElementById('track-indicator').textContent = '';
    
    // 音声を停止
    const audio = document.getElementById('audio');
    audio.pause();
    audio.src = '';
}

// =============================================================================
// Main Functions
// =============================================================================

/**
 * AI作詞
 */
async function generateLyrics() {
    console.log('generateLyrics called');
    const theme = document.getElementById('theme').value.trim();
    console.log('Theme:', theme);
    
    if (!theme) {
        showStatus('テーマを入力してください', 'error');
        return;
    }
    
    const btn = document.getElementById('lyrics-btn');
    const originalText = btn.innerHTML;
    
    try {
        // ボタンをローディング状態に
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> 生成中...';
        showStatus('歌詞を生成中...（LLMに問い合わせ中）', 'info');
        
        const language = document.getElementById('language').value;
        const languageMap = {
            'ja': 'Japanese',
            'en': 'English',
            'zh': 'Chinese',
            'ko': 'Korean',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German'
        };
        
        const result = await apiRequest('/api/lyrics', 'POST', {
            theme: theme,
            genre: document.getElementById('prompt').value,
            language: languageMap[language] || 'Japanese'
        });
        
        if (result.success) {
            document.getElementById('lyrics').value = result.lyrics;
            
            // 推奨秒数を設定
            if (result.recommended_duration) {
                const duration = Math.min(180, Math.max(10, result.recommended_duration));
                document.getElementById('audio_duration').value = duration;
                const durationValueEl = document.getElementById('duration_value');
                if (durationValueEl) {
                    durationValueEl.textContent = duration;
                }
            }
            
            showStatus('✅ 歌詞を生成しました！', 'success');
        } else {
            showStatus('❌ 歌詞生成に失敗: ' + (result.error || '不明なエラー'), 'error');
        }
    } catch (e) {
        showStatus('❌ エラー: ' + e.message, 'error');
    } finally {
        // ボタンを元に戻す
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

/**
 * タグ生成
 */
async function generateTags() {
    const theme = document.getElementById('theme').value.trim();
    const lyrics = document.getElementById('lyrics').value.trim();
    
    if (!theme && !lyrics) {
        showStatus('テーマまたは歌詞を入力してください', 'error');
        return;
    }
    
    const btn = document.getElementById('tags-btn');
    const originalText = btn.innerHTML;
    
    try {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> 生成中...';
        showStatus('タグを生成中...（LLMに問い合わせ中）', 'info');
        
        const result = await apiRequest('/api/tags', 'POST', {
            theme: theme,
            lyrics: lyrics,
            language: 'Japanese'
        });
        
        if (result.success) {
            // タグを設定
            const tags = [];
            if (result.genre) tags.push(result.genre);
            if (result.tags) tags.push(result.tags);
            document.getElementById('prompt').value = tags.join(', ');
            
            // BPMを設定
            if (result.bpm) {
                document.getElementById('bpm').value = result.bpm;
            }
            
            // 調を設定
            if (result.key_scale) {
                document.getElementById('key_scale').value = result.key_scale;
            }
            
            showStatus('✅ タグを生成しました！', 'success');
        } else {
            showStatus('❌ タグ生成に失敗: ' + (result.error || '不明なエラー'), 'error');
        }
    } catch (e) {
        showStatus('❌ エラー: ' + e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

/**
 * 一括生成（AUTO）
 */
async function autoGenerate() {
    const theme = document.getElementById('theme').value.trim();
    
    if (!theme) {
        showStatus('テーマを入力してください', 'error');
        return;
    }
    
    const btn = document.getElementById('auto-btn');
    const originalText = btn.innerHTML;
    
    try {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> 一括生成中...';
        showStatus('歌詞とタグを生成中...（LLMに問い合わせ中、30秒〜1分かかります）', 'info');
        
        const language = document.getElementById('language').value;
        const languageMap = {
            'ja': 'Japanese',
            'en': 'English',
            'zh': 'Chinese',
            'ko': 'Korean',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German'
        };
        
        const result = await apiRequest('/api/full_generate', 'POST', {
            theme: theme,
            genre: document.getElementById('prompt').value,
            language: languageMap[language] || 'Japanese'
        });
        
        if (result.success) {
            // 歌詞を設定
            document.getElementById('lyrics').value = result.lyrics;
            
            // タグを設定
            const tags = [];
            if (result.genre) tags.push(result.genre);
            if (result.tags) tags.push(result.tags);
            document.getElementById('prompt').value = tags.join(', ');
            
            // 推奨秒数を設定
            if (result.recommended_duration) {
                const duration = Math.min(300, Math.max(10, result.recommended_duration));
                document.getElementById('audio_duration').value = duration;
            }
            
            // BPMを設定
            if (result.bpm) {
                document.getElementById('bpm').value = result.bpm;
            }
            
            // 調を設定
            if (result.key_scale) {
                document.getElementById('key_scale').value = result.key_scale;
            }
            
            showStatus('✅ 歌詞とタグを生成しました！「音楽を生成」をクリックして音楽を作成できます', 'success');
        } else {
            showStatus('❌ 生成に失敗: ' + (result.error || '不明なエラー'), 'error');
        }
    } catch (e) {
        showStatus('❌ エラー: ' + e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

// =============================================================================
// Caption Generation (Narrative Caption via External LLM)
// =============================================================================

/**
 * 外部LLMでナラティブキャプションを生成
 * ACE-Step の format_input と同等の詳細な楽曲描写を生成する
 */
async function generateCaption() {
    const theme = document.getElementById('theme').value.trim();
    const lyrics = document.getElementById('lyrics').value.trim();
    const tags = document.getElementById('prompt').value.trim();
    const language = document.getElementById('language')?.value || 'ja';

    if (!theme && !lyrics && !tags) {
        showStatus('テーマ、歌詞、またはタグのいずれかを入力してください', 'error');
        return;
    }

    const btn = document.getElementById('caption-btn');
    const originalText = btn.innerHTML;

    try {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> 生成中...';
        showStatus('📖 キャプション生成中...（外部LLMに問い合わせ中）', 'info');

        // 言語コードを言語名に変換
        const langMap = { 'ja': 'Japanese', 'en': 'English', 'zh': 'Chinese', 'ko': 'Korean', 'es': 'Spanish', 'fr': 'French', 'de': 'German' };
        const langName = langMap[language] || 'Japanese';

        const result = await apiRequest('/api/caption', 'POST', {
            theme: theme,
            lyrics: lyrics,
            tags: tags,
            language: langName
        });

        if (result.success && result.caption) {
            document.getElementById('prompt').value = result.caption;
            showStatus('✅ キャプション生成完了！タグ欄を詳細な楽曲描写に置き換えました', 'success');
        } else {
            showStatus('❌ キャプション生成に失敗: ' + (result.error || '不明なエラー'), 'error');
        }
    } catch (e) {
        showStatus('❌ エラー: ' + e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

// =============================================================================
// Format Input (Prompt Enhancement)
// =============================================================================

/**
 * LLMでプロンプト・歌詞を強化（format_input API）
 */
async function formatInput() {
    const prompt = document.getElementById('prompt').value.trim();
    const lyrics = document.getElementById('lyrics').value.trim();

    if (!prompt && !lyrics) {
        showStatus('タグまたは歌詞を入力してください（強化元のテキストが必要です）', 'error');
        return;
    }

    const btn = document.getElementById('format-btn');
    const originalText = btn.innerHTML;

    try {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> AI強化中...';
        showStatus('LLMでプロンプトを強化中...（数秒〜十数秒かかります）', 'info');

        const result = await apiRequest('/api/format_input', 'POST', {
            prompt: prompt,
            lyrics: lyrics,
            temperature: 0.85,
        });

        if (result.success) {
            let updated = [];
            if (result.caption) {
                document.getElementById('prompt').value = result.caption;
                updated.push('タグ');
            }
            if (result.lyrics) {
                document.getElementById('lyrics').value = result.lyrics;
                updated.push('歌詞');
            }
            if (result.bpm) {
                document.getElementById('bpm').value = result.bpm;
                updated.push('BPM');
            }
            if (result.key_scale) {
                const ksEl = document.getElementById('key_scale');
                if (ksEl.querySelector(`option[value="${result.key_scale}"]`)) {
                    ksEl.value = result.key_scale;
                    updated.push('調');
                }
            }
            if (result.time_signature) {
                const tsEl = document.getElementById('time_signature');
                if (tsEl.querySelector(`option[value="${result.time_signature}"]`)) {
                    tsEl.value = result.time_signature;
                    updated.push('拍子');
                }
            }
            if (result.duration) {
                document.getElementById('audio_duration').value = Math.min(300, Math.max(10, result.duration));
                updated.push('長さ');
            }

            showStatus(`✨ AI強化完了！ 更新: ${updated.join(', ') || '（変化なし）'}`, 'success');
        } else {
            showStatus('❌ AI強化に失敗: ' + (result.error || '不明なエラー'), 'error');
        }
    } catch (e) {
        showStatus('❌ エラー: ' + e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

/**
 * 音楽生成
 */
async function generateMusic() {
    const prompt = document.getElementById('prompt').value.trim();
    const lyrics = document.getElementById('lyrics').value.trim();
    
    if (!prompt && !lyrics) {
        showStatus('タグまたは歌詞を入力してください', 'error');
        return;
    }
    
    try {
        disableButton('generate-btn');
        hideStatus();
        showProgress(0, 'タスクを作成中...');
        
        // パラメータを収集
        const params = {
            prompt: prompt,
            lyrics: lyrics,
            thinking: document.getElementById('thinking').checked,
            vocal_language: document.getElementById('language').value,
            audio_duration: parseInt(document.getElementById('audio_duration').value),
            bpm: parseInt(document.getElementById('bpm').value),
            inference_steps: parseInt(document.getElementById('inference_steps').value),
            guidance_scale: parseFloat(document.getElementById('guidance_scale').value),
            time_signature: document.getElementById('time_signature').value,
            batch_size: parseInt(document.getElementById('batch_size').value),
            audio_format: document.getElementById('audio_format').value,
        };

        // モデル（空=サーバーデフォルト）
        const model = document.getElementById('model')?.value;
        if (model) {
            params.model = model;
        }
        
        // キースケール
        const keyScale = document.getElementById('key_scale').value;
        if (keyScale) {
            params.key_scale = keyScale;
        }
        
        // シード
        const seed = document.getElementById('seed').value;
        if (seed) {
            params.seed = parseInt(seed);
        }

        // LM パラメータ
        const lmTemp = document.getElementById('lm_temperature').value;
        if (lmTemp !== '') params.lm_temperature = parseFloat(lmTemp);
        const lmCfg = document.getElementById('lm_cfg_scale').value;
        if (lmCfg !== '') params.lm_cfg_scale = parseFloat(lmCfg);
        const lmTopP = document.getElementById('lm_top_p').value;
        if (lmTopP !== '') params.lm_top_p = parseFloat(lmTopP);
        // CoT チェックボックス（デフォルトtrue なので、unchecked の時だけ false を送る）
        if (!document.getElementById('use_cot_caption').checked) {
            params.use_cot_caption = false;
        }
        if (!document.getElementById('use_cot_language').checked) {
            params.use_cot_language = false;
        }
        
        // タスク作成
        const createResult = await apiRequest('/api/generate', 'POST', params);
        const taskId = createResult.task_id;
        
        if (!taskId) {
            throw new Error('タスクの作成に失敗しました');
        }
        
        // キュー位置表示
        const queuePos = createResult.queue_position;
        const queueText = queuePos && queuePos > 1 ? ` (キュー: ${queuePos}番目)` : '';
        showProgress(5, `タスク作成完了。生成中...${queueText}`);
        
        // 予測生成時間を計算
        const expectedTime = estimateGenerationTime(params);
        console.log(`Estimated generation time: ${expectedTime.toFixed(1)}s`);
        
        // ポーリングで完了を待つ
        const maxPolls = 300;  // 5分（安全上限）
        let polls = 0;
        
        while (polls < maxPolls) {
            await sleep(1000);
            polls++;
            
            let statusResult;
            try {
                statusResult = await apiRequest(`/api/status/${taskId}`);
            } catch (pollError) {
                // 一時的なエラーはスキップして次のポーリングへ
                console.warn(`Poll ${polls} failed (will retry):`, pollError.message);
                continue;
            }
            
            if (statusResult.status === 1) {
                // 成功
                showProgress(100, '生成完了！');
                
                if (statusResult.results && statusResult.results.length > 0) {
                    showResults(statusResult.results);

                    const result = statusResult.results[0];
                    const count = statusResult.results.length;
                    const requestedModel = params.model || '';
                    const usedDitModel = result.dit_model || '';
                    const usedLmModel = result.lm_model || '';
                    const modelParts = [];
                    if (requestedModel) modelParts.push(`要求: ${requestedModel}`);
                    if (usedDitModel) modelParts.push(`DiT: ${usedDitModel}`);
                    if (usedLmModel) modelParts.push(`LM: ${usedLmModel}`);

                    const trackInfo = count > 1 ? `${count}トラック生成完了` : '生成完了';
                    showStatus(
                        modelParts.length > 0
                            ? `✅ ${trackInfo}！ (${modelParts.join(' / ')}) [${formatElapsed(polls)}]`
                            : `✅ ${trackInfo}！ [${formatElapsed(polls)}]`,
                        'success'
                    );
                }
                
                hideProgress();
                enableButton('generate-btn');
                return;
            } else if (statusResult.status === 2) {
                // 失敗
                let errMsg = statusResult.error || '生成に失敗しました';
                try { const p = JSON.parse(errMsg); if (Array.isArray(p) || typeof p === 'object') errMsg = '生成に失敗しました（サーバーエラー）'; } catch(_){}
                throw new Error(errMsg);
            }
            
            // 時間ベースの滑らかな進捗推定
            const progress = smoothProgress(polls, expectedTime);
            const progressText = getProgressPhaseText(polls, expectedTime);
            showProgress(progress, progressText);
        }
        
        throw new Error('タイムアウト: 生成に時間がかかりすぎています');
        
    } catch (e) {
        showStatus('エラー: ' + e.message, 'error');
        hideProgress();
        enableButton('generate-btn');
    }
}

/**
 * スリープ関数
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// =============================================================================
// Batch Results & Metadata
// =============================================================================

/**
 * 生成結果を表示（バッチ対応 + メタデータ）
 */
function showResults(results) {
    currentResults = results;
    if (!results || results.length === 0) return;

    // メインプレイヤーに最初のトラックをロード
    playTrack(0);

    // 結果グリッドを構築
    const grid = document.getElementById('results-grid');
    grid.innerHTML = '';

    results.forEach((result, index) => {
        const card = createResultCard(result, index);
        grid.appendChild(card);
    });

    // 再生セクションを表示
    const playerSection = document.getElementById('player-section');
    playerSection.classList.add('visible');
    playerSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * 結果カードのHTML要素を作成
 */
function createResultCard(result, index) {
    const card = document.createElement('div');
    card.className = 'result-card' + (index === 0 ? ' active' : '');
    card.id = `result-card-${index}`;

    const metas = result.metas || {};
    // seed_value はカンマ区切りで複数seed（バッチ分）が入っている場合がある
    const seeds = (result.seed_value || '').split(',');
    const seed = (seeds[index] || seeds[0] || '-').trim();

    const metaItems = [];
    if (metas.bpm) metaItems.push(`🎹 ${metas.bpm} BPM`);
    if (metas.keyscale && metas.keyscale !== 'N/A') metaItems.push(`🎵 ${metas.keyscale}`);
    if (metas.duration) metaItems.push(`⏱️ ${metas.duration}秒`);
    if (metas.timesignature) {
        const tsMap = { '2': '2/4', '3': '3/4', '4': '4/4', '6': '6/8' };
        metaItems.push(`🎼 ${tsMap[metas.timesignature] || metas.timesignature}`);
    }

    // generation_info から生成時間を抽出
    if (result.generation_info) {
        const match = result.generation_info.match(/Average Time per Track:\s*([\d.]+)s/);
        if (match) metaItems.push(`⚡ ${match[1]}秒/曲`);
    }

    const audioUrl = convertAudioUrl(result.url);

    card.innerHTML = `
        <div class="result-card-header">
            <button class="btn btn-primary btn-sm" onclick="playTrack(${index})">
                ▶ Track ${index + 1}
            </button>
            <span class="seed-badge" title="Seed値（同じ値で再現可能）">🎲 ${seed}</span>
            <a class="btn btn-secondary btn-sm" href="${audioUrl}" download title="ダウンロード">⬇️</a>
        </div>
        <div class="result-meta">
            ${metaItems.map(item => `<span class="result-meta-item">${item}</span>`).join('')}
        </div>
        ${result.dit_model || result.lm_model ? `
        <div class="result-models">
            ${result.dit_model ? `<span class="model-badge">DiT: ${result.dit_model}</span>` : ''}
            ${result.lm_model ? `<span class="model-badge">LM: ${result.lm_model}</span>` : ''}
        </div>` : ''}
    `;

    return card;
}

/**
 * 特定のトラックをメインプレイヤーで再生
 */
function playTrack(index) {
    if (!currentResults || index >= currentResults.length) return;

    const result = currentResults[index];
    const audioUrl = convertAudioUrl(result.url);

    const audio = document.getElementById('audio');
    const downloadBtn = document.getElementById('download-btn');
    const indicator = document.getElementById('track-indicator');

    audio.src = audioUrl;
    downloadBtn.href = audioUrl;

    // トラック番号表示
    if (currentResults.length > 1) {
        indicator.textContent = `Track ${index + 1} / ${currentResults.length}`;
    } else {
        indicator.textContent = '';
    }

    // 再生セクションを表示
    const playerSection = document.getElementById('player-section');
    playerSection.classList.add('visible');

    // ビジュアライザーを初期化
    initVisualizer();

    // アクティブカードのハイライト
    document.querySelectorAll('.result-card').forEach((card, i) => {
        card.classList.toggle('active', i === index);
    });

    // 自動再生
    setTimeout(() => {
        const playPromise = audio.play();
        if (playPromise && typeof playPromise.catch === 'function') {
            playPromise.catch(() => {});
        }
    }, 100);
}

// =============================================================================
// Cover Mode
// =============================================================================

/**
 * カバーモードで音楽生成
 */
async function generateCover() {
    const fileInput = document.getElementById('cover_audio');
    if (!fileInput.files || fileInput.files.length === 0) {
        showStatus('ソース音声ファイルを選択してください', 'error');
        return;
    }

    const file = fileInput.files[0];
    // ファイルサイズチェック (50MB以下)
    if (file.size > 50 * 1024 * 1024) {
        showStatus('ファイルサイズが大きすぎます（50MB以下）', 'error');
        return;
    }

    const btn = document.getElementById('cover-btn');
    const originalText = btn.innerHTML;

    try {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> アップロード中...';
        hideStatus();
        showProgress(0, 'カバータスクを作成中...');

        const formData = new FormData();
        formData.append('src_audio', file);
        formData.append('prompt', document.getElementById('prompt').value.trim());
        formData.append('lyrics', document.getElementById('lyrics').value.trim());
        formData.append('thinking', document.getElementById('thinking').checked);
        formData.append('vocal_language', document.getElementById('language').value);
        formData.append('audio_duration', document.getElementById('audio_duration').value);
        formData.append('bpm', document.getElementById('bpm').value);
        formData.append('inference_steps', document.getElementById('inference_steps').value);
        formData.append('time_signature', document.getElementById('time_signature').value);
        formData.append('batch_size', document.getElementById('batch_size').value);
        formData.append('audio_format', document.getElementById('audio_format').value);

        // カバー強度と CFG の連動制約
        const coverStrength = parseFloat(document.getElementById('cover_strength').value);
        formData.append('audio_cover_strength', coverStrength);
        if (coverStrength < 1.0) {
            // カバー強度 < 1.0 のときは CFG=1.0 を強制
            formData.append('guidance_scale', '1.0');
        } else {
            formData.append('guidance_scale', document.getElementById('guidance_scale').value);
        }

        const model = document.getElementById('model')?.value;
        if (model) formData.append('model', model);
        const keyScale = document.getElementById('key_scale').value;
        if (keyScale) formData.append('key_scale', keyScale);
        const seed = document.getElementById('seed').value;
        if (seed) formData.append('seed', seed);

        // LM パラメータ
        const lmTemp = document.getElementById('lm_temperature').value;
        if (lmTemp !== '') formData.append('lm_temperature', lmTemp);
        const lmCfg = document.getElementById('lm_cfg_scale').value;
        if (lmCfg !== '') formData.append('lm_cfg_scale', lmCfg);
        const lmTopP = document.getElementById('lm_top_p').value;
        if (lmTopP !== '') formData.append('lm_top_p', lmTopP);
        if (!document.getElementById('use_cot_caption').checked) formData.append('use_cot_caption', 'false');
        if (!document.getElementById('use_cot_language').checked) formData.append('use_cot_language', 'false');

        // multipart/form-data で送信（Content-Typeは自動設定）
        const response = await fetch('/api/generate_cover', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errText}`);
        }

        const createResult = await response.json();
        const taskId = createResult.task_id;

        if (!taskId) throw new Error('タスクの作成に失敗しました');

        btn.innerHTML = '<span class="spinner"></span> 生成中...';
        showProgress(5, 'カバー生成中...');

        // カバー生成の予測時間を計算
        const coverParams = {
            inference_steps: parseInt(document.getElementById('inference_steps').value) || 150,
            audio_duration: parseInt(document.getElementById('audio_duration').value) || 60,
            batch_size: parseInt(document.getElementById('batch_size').value) || 1,
        };
        const expectedTime = estimateGenerationTime(coverParams) * 1.2; // カバーは少し長め

        // ポーリングで完了を待つ
        const maxPolls = 300;
        let polls = 0;

        while (polls < maxPolls) {
            await sleep(1000);
            polls++;

            let statusResult;
            try {
                statusResult = await apiRequest(`/api/status/${taskId}`);
            } catch (pollError) {
                console.warn(`Cover poll ${polls} failed (will retry):`, pollError.message);
                continue;
            }

            if (statusResult.status === 1) {
                showProgress(100, 'カバー生成完了！');
                if (statusResult.results && statusResult.results.length > 0) {
                    showResults(statusResult.results);
                    const count = statusResult.results.length;
                    showStatus(
                        `✅ カバー${count > 1 ? count + 'トラック' : ''}生成完了！ [${formatElapsed(polls)}]`,
                        'success'
                    );
                }
                hideProgress();
                return;
            } else if (statusResult.status === 2) {
                let errMsg = statusResult.error || 'カバー生成に失敗しました';
                try { const p = JSON.parse(errMsg); if (Array.isArray(p) || typeof p === 'object') errMsg = 'カバー生成に失敗しました（サーバーエラー）'; } catch(_){}
                throw new Error(errMsg);
            }

            // 時間ベースの滑らかな進捗推定
            const progress = smoothProgress(polls, expectedTime);
            const progressText = getProgressPhaseText(polls, expectedTime, 'カバー生成中');
            showProgress(progress, progressText);
        }

        throw new Error('タイムアウト: 生成に時間がかかりすぎています');

    } catch (e) {
        showStatus('エラー: ' + e.message, 'error');
        hideProgress();
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

// =============================================================================
// Repaint Mode
// =============================================================================

/**
 * リペイントモードで音楽の一部を再生成
 */
async function generateRepaint() {
    const fileInput = document.getElementById('repaint_audio');
    if (!fileInput.files || fileInput.files.length === 0) {
        showStatus('リペイント対象の音声ファイルを選択してください', 'error');
        return;
    }

    const file = fileInput.files[0];
    if (file.size > 50 * 1024 * 1024) {
        showStatus('ファイルサイズが大きすぎます（50MB以下）', 'error');
        return;
    }

    const repaintStart = parseFloat(document.getElementById('repaint_start').value) || 0;
    const repaintEnd = parseFloat(document.getElementById('repaint_end').value) || 0;
    if (repaintEnd <= repaintStart) {
        showStatus('終了位置は開始位置より大きくしてください', 'error');
        return;
    }

    const btn = document.getElementById('repaint-btn');
    const originalText = btn.innerHTML;

    try {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> アップロード中...';
        hideStatus();
        showProgress(0, 'リペイントタスクを作成中...');

        const formData = new FormData();
        formData.append('src_audio', file);
        formData.append('repainting_start', repaintStart);
        formData.append('repainting_end', repaintEnd);
        formData.append('prompt', document.getElementById('prompt').value.trim());
        formData.append('lyrics', document.getElementById('lyrics').value.trim());
        formData.append('thinking', document.getElementById('thinking').checked);
        formData.append('vocal_language', document.getElementById('language').value);
        formData.append('audio_duration', document.getElementById('audio_duration').value);
        formData.append('bpm', document.getElementById('bpm').value);
        formData.append('inference_steps', document.getElementById('inference_steps').value);
        formData.append('guidance_scale', document.getElementById('guidance_scale').value);
        formData.append('time_signature', document.getElementById('time_signature').value);
        formData.append('batch_size', document.getElementById('batch_size').value);
        formData.append('audio_format', document.getElementById('audio_format').value);

        const model = document.getElementById('model')?.value;
        if (model) formData.append('model', model);
        const keyScale = document.getElementById('key_scale').value;
        if (keyScale) formData.append('key_scale', keyScale);
        const seed = document.getElementById('seed').value;
        if (seed) formData.append('seed', seed);

        // LM パラメータ
        const lmTemp = document.getElementById('lm_temperature').value;
        if (lmTemp !== '') formData.append('lm_temperature', lmTemp);
        const lmCfg = document.getElementById('lm_cfg_scale').value;
        if (lmCfg !== '') formData.append('lm_cfg_scale', lmCfg);
        const lmTopP = document.getElementById('lm_top_p').value;
        if (lmTopP !== '') formData.append('lm_top_p', lmTopP);
        if (!document.getElementById('use_cot_caption').checked) formData.append('use_cot_caption', 'false');
        if (!document.getElementById('use_cot_language').checked) formData.append('use_cot_language', 'false');

        const response = await fetch('/api/generate_repaint', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errText}`);
        }

        const createResult = await response.json();
        const taskId = createResult.task_id;
        if (!taskId) throw new Error('タスクの作成に失敗しました');

        btn.innerHTML = '<span class="spinner"></span> 生成中...';
        showProgress(5, 'リペイント生成中...');

        // リペイント区間の時間推定
        const repaintDuration = repaintEnd - repaintStart;
        const repaintParams = {
            inference_steps: parseInt(document.getElementById('inference_steps').value) || 150,
            audio_duration: Math.max(10, repaintDuration),
            batch_size: parseInt(document.getElementById('batch_size').value) || 1,
        };
        const expectedTime = estimateGenerationTime(repaintParams);

        const maxPolls = 300;
        let polls = 0;

        while (polls < maxPolls) {
            await sleep(1000);
            polls++;

            let statusResult;
            try {
                statusResult = await apiRequest(`/api/status/${taskId}`);
            } catch (pollError) {
                console.warn(`Repaint poll ${polls} failed (will retry):`, pollError.message);
                continue;
            }

            if (statusResult.status === 1) {
                showProgress(100, 'リペイント生成完了！');
                if (statusResult.results && statusResult.results.length > 0) {
                    showResults(statusResult.results);
                    const count = statusResult.results.length;
                    showStatus(
                        `✅ リペイント${count > 1 ? count + 'トラック' : ''}生成完了！ (区間: ${repaintStart}s〜${repaintEnd}s) [${formatElapsed(polls)}]`,
                        'success'
                    );
                }
                hideProgress();
                return;
            } else if (statusResult.status === 2) {
                let errMsg = statusResult.error || 'リペイント生成に失敗しました';
                try { const p = JSON.parse(errMsg); if (Array.isArray(p) || typeof p === 'object') errMsg = 'リペイント生成に失敗しました（サーバーエラー）'; } catch(_){}
                throw new Error(errMsg);
            }

            const progress = smoothProgress(polls, expectedTime);
            const progressText = getProgressPhaseText(polls, expectedTime, 'リペイント生成中');
            showProgress(progress, progressText);
        }

        throw new Error('タイムアウト: 生成に時間がかかりすぎています');

    } catch (e) {
        showStatus('エラー: ' + e.message, 'error');
        hideProgress();
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

// =============================================================================
// Random Sample
// =============================================================================

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('ACE-Step Standalone initialized');
    
    // カスタムリサイズハンドルの初期化
    initResizeHandles();
    
    // サーバー情報を取得して表示
    loadServerInfo();

    // モデル選択が変わったら推奨STEPを反映
    const modelSelect = document.getElementById('model');
    if (modelSelect) {
        modelSelect.addEventListener('change', () => {
            refreshStepsFromModelSelection();
        });
    }
});

/**
 * サーバー情報（モデル・統計）を取得して表示
 */
async function loadServerInfo() {
    try {
        // モデル情報を取得
        const modelResult = await apiRequest('/api/models');
        if (modelResult.success) {
            const modelName = modelResult.default_model || 'unknown';
            window.__aceStepDefaultModel = modelName;
            const modelDisplay = modelName.replace('acestep-', '').replace('-', ' ');
            const isTurbo = modelName.includes('turbo');
            
            const modelEl = document.getElementById('model-info');
            modelEl.innerHTML = `🤖 モデル: <strong>${modelDisplay}</strong>${isTurbo ? ' ⚡' : ''}`;
            modelEl.title = `フルネーム: ${modelName}${isTurbo ? '\nTurboモデル: STEP 8推奨' : '\nBaseモデル: STEP 60推奨'}`;

            // デフォルト/選択モデルに応じて推奨STEPを反映
            refreshStepsFromModelSelection();

            // モデルセレクタを更新（存在する場合のみ）
            const modelSelect = document.getElementById('model');
            if (modelSelect) {
                const previousValue = modelSelect.value;
                const models = Array.isArray(modelResult.models) ? modelResult.models : [];

                // 一旦クリアして「自動（デフォルト）」を入れる
                modelSelect.innerHTML = '';
                modelSelect.add(new Option('自動（デフォルト）', ''));

                models.forEach((m) => {
                    const name = (typeof m === 'string') ? m : (m?.name || m?.model || m?.id || '');
                    if (!name) return;
                    modelSelect.add(new Option(name, name));
                });

                // ユーザーが既に選んでいた値があれば維持（なければ自動）
                if (previousValue) {
                    modelSelect.value = previousValue;
                } else {
                    modelSelect.value = '';
                }

                // モデル一覧反映後にもう一度推奨STEPを更新
                refreshStepsFromModelSelection();
            }
        } else {
            document.getElementById('model-info').innerHTML = '🤖 モデル: <span style="color: var(--error-color);">接続エラー</span>';
        }
        
        // 統計情報を取得
        const statsResult = await apiRequest('/api/stats');
        if (statsResult.success) {
            document.getElementById('queue-info').innerHTML = `📊 キュー: <strong>${statsResult.queue_size}</strong>`;
            const avgTime = statsResult.avg_job_seconds ? statsResult.avg_job_seconds.toFixed(1) : '-';
            document.getElementById('avg-time-info').innerHTML = `⏱️ 平均: <strong>${avgTime}秒</strong>`;
        }
    } catch (e) {
        console.error('Failed to load server info:', e);
        document.getElementById('model-info').innerHTML = '🤖 モデル: <span style="color: var(--error-color);">接続エラー</span>';
    }
}

/**
 * カスタムリサイズハンドルを初期化
 */
function initResizeHandles() {
    const resizeHandles = document.querySelectorAll('.resize-handle');
    
    resizeHandles.forEach(handle => {
        const container = handle.closest('.textarea-container');
        const textarea = container.querySelector('textarea');
        
        if (!textarea) return;
        
        let isResizing = false;
        let startY = 0;
        let startHeight = 0;
        
        handle.addEventListener('mousedown', (e) => {
            isResizing = true;
            startY = e.clientY;
            startHeight = textarea.offsetHeight;
            document.body.style.cursor = 'ns-resize';
            document.body.style.userSelect = 'none';
            e.preventDefault();
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;
            
            const deltaY = e.clientY - startY;
            const newHeight = Math.max(150, startHeight + deltaY);
            textarea.style.height = newHeight + 'px';
        });
        
        document.addEventListener('mouseup', () => {
            if (isResizing) {
                isResizing = false;
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
            }
        });
        
        // タッチ対応
        handle.addEventListener('touchstart', (e) => {
            isResizing = true;
            startY = e.touches[0].clientY;
            startHeight = textarea.offsetHeight;
            e.preventDefault();
        });
        
        document.addEventListener('touchmove', (e) => {
            if (!isResizing) return;
            
            const deltaY = e.touches[0].clientY - startY;
            const newHeight = Math.max(150, startHeight + deltaY);
            textarea.style.height = newHeight + 'px';
        });
        
        document.addEventListener('touchend', () => {
            isResizing = false;
        });
    });
}
