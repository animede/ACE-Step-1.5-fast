"""Generation timing summary for API results (moved from the removed Gradio UI package)."""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_generation_info(
    lm_metadata: Optional[Dict[str, Any]],
    time_costs: Dict[str, float],
    seed_value: str,
    inference_steps: int,
    num_audios: int,
    audio_format: str = "flac",
) -> str:
    """Build a compact generation timing summary (markdown-ish text).

    Args:
        lm_metadata: LM metadata (unused, kept for call compatibility).
        time_costs: Unified time-cost dictionary from the handler.
        seed_value: Seed string (unused, kept for call compatibility).
        inference_steps: Diffusion steps (unused, kept for call compatibility).
        num_audios: Number of generated audios.
        audio_format: Output format name used in the conversion line.

    Returns:
        The summary, or an empty string when there is nothing to report.
    """
    if not time_costs or num_audios <= 0:
        return ""
    songs_label = f"({num_audios} song{'s' if num_audios > 1 else ''})"
    parts = []
    lm_total = time_costs.get("lm_total_time", 0.0)
    dit_total = time_costs.get("dit_total_time_cost", 0.0)
    gen_total = lm_total + dit_total
    if gen_total > 0:
        lines = [f"**🎵 Total generation time {songs_label}: {gen_total:.2f}s**", f"- {gen_total / num_audios:.2f}s per song"]
        if lm_total > 0:
            lines.append(f"- LM phase {songs_label}: {lm_total:.2f}s")
        if dit_total > 0:
            lines.append(f"- DiT phase {songs_label}: {dit_total:.2f}s")
        parts.append("\n".join(lines))
    conversion = time_costs.get("audio_conversion_time", 0.0)
    scoring = time_costs.get("auto_score_time", 0.0)
    lrc = time_costs.get("auto_lrc_time", 0.0)
    proc_total = conversion + scoring + lrc
    if proc_total > 0:
        fmt_label = audio_format.upper() if audio_format != "wav32" else "WAV 32-bit"
        lines = [f"**🔧 Total processing time {songs_label}: {proc_total:.2f}s**"]
        if conversion > 0:
            lines.append(f"- to {fmt_label} {songs_label}: {conversion:.2f}s")
        if scoring > 0:
            lines.append(f"- scoring {songs_label}: {scoring:.2f}s")
        if lrc > 0:
            lines.append(f"- LRC detection {songs_label}: {lrc:.2f}s")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


_build_generation_info = build_generation_info  # name used by api_server's call site
