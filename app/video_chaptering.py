# app/video_chaptering.py
"""
Module for generating video chapters from pre-processed ASR and OCR segments.
Designed to be integrated into the Celery task pipeline.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Azure AI Inference imports (the working ones)
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# OpenAI import
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Use the logger from the tasks module
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
# These can now be set from the main app's environment/config
SERVICE_TYPE = os.getenv("CHAPTER_SERVICE_TYPE", "openai")  # "openai" or "azure"
OPENAI_MODEL = os.getenv("CHAPTER_OPENAI_MODEL", "gpt-4o-mini")
AZURE_MODEL = os.getenv("CHAPTER_AZURE_MODEL", "Meta-Llama-3.1-8B-Instruct")

# ─────────────────────────
# Utilities
# ─────────────────────────
def sec_to_hms(sec: int) -> str:
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def clean_chapter_titles(chapters: Dict[str, str]) -> Dict[str, str]:
    """
    Clean up chapter titles by removing filler words and improving clarity.
    """
    cleaned = {}
    filler_words = ['那', '所以', '這個', '那個', '就是', '呢', '啊', '喔', '然後', '接著']
    for ts, title in chapters.items():
        for word in filler_words:
            title = title.replace(word, '')
        title = re.sub(r'[。，、！？]+$', '', title.strip())
        title = re.sub(r'\s+', ' ', title)
        if 0 < len(title) < 4:
            title = title.capitalize()
        cleaned[ts] = title
    return cleaned

# ─────────────────────────
# Token counting for Llama 3.1
# ─────────────────────────
def count_tokens_llama(text: str) -> int:
    """
    Approximate token counting for Llama 3.1 (4 chars ≈ 1 token for Chinese/English mix)
    """
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    non_chinese_len = len(text) - chinese_chars
    return chinese_chars + max(1, non_chinese_len // 4)

def truncate_text_by_tokens(text: str, max_tokens: int = 120000) -> str:
    """
    Truncate text to approximately max_tokens, preserving complete sentences
    """
    if count_tokens_llama(text) <= max_tokens:
        return text
    logger.warning(f"Truncating transcript from {count_tokens_llama(text):,} tokens to {max_tokens:,} tokens")
    sentences = re.split(r'(?<=[。！？.!?])', text)
    truncated = ""
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = count_tokens_llama(sentence)
        if current_tokens + sentence_tokens > max_tokens:
            break
        truncated += sentence
        current_tokens += sentence_tokens
    return truncated

# ─────────────────────────
# Chapter policy
# ─────────────────────────
def chapter_policy(duration_sec: int) -> Tuple[int, Tuple[int, int], int]:
    if duration_sec < 30 * 60:
        return 90,  (6, 12), 30
    if duration_sec < 60 * 60:
        return 180, (8, 16), 40
    if duration_sec < 120 * 60:
        return 300, (10, 20), 50
    if duration_sec < 180 * 60:
        return 540, (12, 24), 60
    return 600, (14, 28), 80

# ─────────────────────────
# Parsing
# ─────────────────────────
CHAPTER_LINE_RE = re.compile(
    r"""
    ^\s*
    (?:[\-\*\u2022]\s*)?
    \[?(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]?\s*
    (?:[-–—:]\s*)?
    (?P<title>.+?)
    \s*$
    """,
    re.VERBOSE,
)

def _normalize_ts(ts: str) -> str:
    parts = ts.split(":")
    if len(parts) == 2:
        return f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)}"
    if len(parts) == 3:
        return f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"
    return ts

def parse_chapters_from_output(output_text: str) -> Dict[str, str]:
    chapters: Dict[str, str] = {}
    for line in output_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = CHAPTER_LINE_RE.match(line)
        if not m:
            continue
        ts = _normalize_ts(m.group("ts").strip())
        title = m.group("title").strip()
        if title:
            chapters.setdefault(ts, title)
    if chapters:
        return chapters

    # loose fallback
    tmp: Dict[str, str] = {}
    for line in output_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.search(r"(\d{1,2}:\d{2}(?::\d{2})?)", line)
        if not m:
            continue
        ts = _normalize_ts(m.group(1))
        title = line[m.end():].lstrip(" -–—:\t").strip() or line[:m.start()].strip()
        if title:
            tmp.setdefault(ts, title)
    return tmp

# ─────────────────────────
# Balancing
# ─────────────────────────
def globally_balance_chapters(
    chapters: Dict[str, str],
    duration_sec: int,
    min_gap_sec: int,
    target_range: Tuple[int, int],
    max_caps: int,
) -> Dict[str, str]:
    def ts_to_s(ts: str) -> int:
        p = ts.split(":")
        if len(p) == 2:
            return int(p[0]) * 60 + int(p[1])
        if len(p) == 3:
            return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])
        return 0

    def s_to_ts(sec: int) -> str:
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    cands = [(ts_to_s(ts), ts, t.strip()) for ts, t in chapters.items() if 0 <= ts_to_s(ts) <= duration_sec]
    cands.sort(key=lambda x: x[0])
    if not cands:
        return {}

    dedup = []
    for s, ts, title in cands:
        if dedup and (s - dedup[-1][0]) < min_gap_sec:
            if len(title) > len(dedup[-1][2]):
                dedup[-1] = (s, ts, title)
        else:
            dedup.append((s, ts, title))

    t_low, t_high = target_range
    if t_low <= len(dedup) <= t_high:
        return {ts: title for _, ts, title in dedup}

    if len(dedup) > t_high:
        selected = []
        segment_length = duration_sec // t_high
        for i in range(t_high):
            segment_start = i * segment_length
            segment_end = (i + 1) * segment_length
            segment_chapters = [c for c in dedup if segment_start <= c[0] < segment_end]
            if segment_chapters:
                chosen = min(segment_chapters, key=lambda c: c[0] - segment_start)
                selected.append(chosen)
        return {ts: title for _, ts, title in selected}

    return {ts: title for _, ts, title in dedup[:max_caps]}

# ─────────────────────────
# Prompt builder - MODIFIED TO USE SEGMENTS
# ─────────────────────────
def build_transcript_from_segments(audio_segments: List[Dict]) -> str:
    """Convert audio segments into a continuous transcript text."""
    transcript_lines = []
    for seg in audio_segments:
        start = seg.get('start', 0)
        text = seg.get('text', '').strip()
        if text:
            # Format: [HH:MM:SS] Text
            timestamp = sec_to_hms(int(start))
            transcript_lines.append(f"[{timestamp}] {text}")
    return "\n".join(transcript_lines)

def build_ocr_context_from_segments(ocr_segments: List[Dict]) -> str:
    """Convert OCR segments into a descriptive context string."""
    if not ocr_segments:
        return ""
    
    context_lines = ["# 從投影片與螢幕捕捉到的相關文字："]
    for seg in ocr_segments:
        start = int(seg.get('start', 0))
        text = seg.get('text', '').strip()
        if text:
            timestamp = sec_to_hms(start)
            context_lines.append(f"*   於 {timestamp} 左右捕捉到:")
            # Split the combined text back into bullet points for readability
            for line in text.split():
                if line.strip():
                    context_lines.append(f"    - 「{line.strip()}」")
    
    return "\n".join(context_lines)

def build_prompt_body(transcript: str, duration_sec: int, ocr_context: str = "") -> str:
    # ... [keep the original function content unchanged] ...
    # (This function is already perfect for our use)
    duration_hms = sec_to_hms(int(duration_sec))
    min_gap_sec, (t_low, t_high), max_caps = chapter_policy(duration_sec)
    prompt = (
        "你是一位資深的教育內容編輯專家。你的任務是為以下影片逐字稿生成清晰、專業且簡潔的 YouTube 章節標題（**繁體中文**）。\n\n"
        f"影片總長度：{duration_hms}。請生成 **{t_low}–{t_high} 個章節**（最多 {max_caps} 個）。\n"
        f"每個章節間隔至少 **{min_gap_sec//60} 分鐘**（若主題延續請勿切分）。\n\n"
        "## 分析步驟：\n"
        "1.  **辨識主題：** 分析逐字稿" + ("與螢幕文字" if ocr_context else "") +
        "，確定課程主題（例如：『AutoCAD』、『平面設計』、『專案管理』、『Python 程式設計』、『行銷』）。\n"
        "2.  **提取關鍵主題：** 找出影片中所教授的主要課程、模組或技能。\n"
        "3.  **創建章節標題：** 用清晰的標題總結每個段落，反映其核心教學價值。\n\n"
    )
    if ocr_context.strip():
        prompt += (
            "## 螢幕文字輔助資訊：\n"
            "以下文字是從影片中的投影片、程式碼編輯器、或軟體介面擷取而來。請將這些關鍵字詞與術語融入你的章節標題中，使其更加精準。\n"
            f"{ocr_context}\n\n"
        )
    prompt += (
        "## 通用章節標題指南：\n"
        "1.  **總結，而非抄寫：** 不要直接複製逐字稿的句子。提煉核心主題。\n"
        "2.  **使用相關術語：** 為所識別的主題使用適當的技術或概念術語（例如：『圖層』、『約束』、『排版』、『SWOT 分析』、『函式』）。\n"
        "3.  **清晰簡潔：** 標題長度應為 5-12 個字，讓學生能立即理解該段落的內容。\n"
        "4.  **結構（可選）：** 如果影片有清晰的結構，可以自然地使用以下前綴：\n"
        "    - 介紹、概述、結論\n"
        "    - 理論：...、概念：...、說明：...\n"
        "    - 實作：...、教學：...、演示：...、練習：...\n"
        "    - 專案：...、案例研究：...、範例：...\n"
        "    - **請勿強制使用這些前綴。** 僅在符合內容時自然使用。\n"
        "5.  **均勻分佈：** 章節必須覆蓋整部影片，而不是只集中在開頭。\n\n"
        "## 輸出格式要求（必須嚴格遵守）：\n"
        "`HH:MM:SS - 標題`（不要編號、不要額外說明、不要裝飾符號）\n\n"
        "## 實際影片逐字稿內容：\n"
        f"{transcript}"
    )
    return prompt

# ─────────────────────────
# Client Initialization
# ─────────────────────────
def initialize_client(service_type: str, **kwargs):
    if service_type == "azure":
        return ChatCompletionsClient(
            endpoint=kwargs["endpoint"],
            credential=AzureKeyCredential(kwargs["key"]),
            api_version=kwargs.get("api_version", "2024-05-01-preview")
        )
    elif service_type == "openai":
        openai.api_key = kwargs["api_key"]
        if kwargs.get("base_url"):
            openai.base_url = kwargs["base_url"]
        return {"api_key": kwargs["api_key"], "base_url": kwargs.get("base_url")}
    else:
        raise ValueError(f"Unknown service type: {service_type}")

# ─────────────────────────
# LLM API Calls with retry logic
# ─────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def call_llm(
    service_type: str,
    client,
    system_message: str,
    user_message: str,
    model: str,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    top_p: float = 0.9
):
    if service_type == "azure":
        return client.complete(
            messages=[
                SystemMessage(content=system_message),
                UserMessage(content=user_message),
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            model=model
        )
    elif service_type == "openai":
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response
    else:
        raise ValueError(f"Unknown service type: {service_type}")

# ─────────────────────────
# MAIN FUNCTION FOR PIPELINE
# ─────────────────────────
def generate_chapters(
    audio_segments: List[Dict],
    ocr_segments: List[Dict],
    duration: float,
    video_id: str,
    run_dir: Optional[Path] = None
) -> Dict[str, str]:
    """
    Main function to generate chapters from pre-processed segments.
    
    Args:
        audio_segments: List of dicts with 'start', 'end', 'text' keys
        ocr_segments: List of dicts with 'start', 'end', 'text' keys  
        duration: Video duration in seconds
        video_id: Unique identifier for the video
        run_dir: Optional directory to save intermediate files for debugging
    
    Returns:
        Dict of chapter timestamps to titles: {'HH:MM:SS': 'Chapter Title'}
    """
    chapters: Dict[str, str] = {}
    output_text = ""
    full_prompt = ""

    # Use provided run_dir or create a temporary one
    if run_dir is None:
        run_dir = Path(f"/tmp/chapter_generation/{video_id}_{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Starting chapter generation for video {video_id} (duration: {duration}s)")
        
        # 1) Build inputs from segments
        transcript = build_transcript_from_segments(audio_segments)
        ocr_context = build_ocr_context_from_segments(ocr_segments)
        
        logger.info(f"Built transcript: {len(transcript)} chars, OCR context: {len(ocr_context)} chars")
        
        # Save intermediate files for debugging
        if run_dir:
            with open(run_dir / "audio_segments.json", "w", encoding="utf-8") as f:
                json.dump(audio_segments, f, ensure_ascii=False, indent=2)
            with open(run_dir / "ocr_segments.json", "w", encoding="utf-8") as f:
                json.dump(ocr_segments, f, ensure_ascii=False, indent=2)
            with open(run_dir / "transcript.txt", "w", encoding="utf-8") as f:
                f.write(transcript)
            with open(run_dir / "ocr_context.txt", "w", encoding="utf-8") as f:
                f.write(ocr_context)

        # 2) Initialize client
        service_type = SERVICE_TYPE
        model = OPENAI_MODEL if service_type == "openai" else AZURE_MODEL
        
        if service_type == "azure":
            azure_endpoint = os.getenv("AZURE_AI_ENDPOINT")
            azure_key = os.getenv("AZURE_AI_KEY")
            if not azure_endpoint or not azure_key:
                raise RuntimeError("Azure AI credentials not found in environment variables")
            client = initialize_client(
                service_type="azure",
                endpoint=azure_endpoint,
                key=azure_key,
                api_version=os.getenv("AZURE_API_VERSION", "2024-05-01-preview")
            )
        else:  # openai
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise RuntimeError("OpenAI API key not found in environment variables")
            client = initialize_client(
                service_type="openai",
                api_key=openai_api_key,
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
            )

        # 3) Build prompt with token truncation
        prompt_template = build_prompt_body("", duration, ocr_context)
        template_tokens = count_tokens_llama(prompt_template)
        max_transcript_tokens = 120000 - template_tokens
        truncated_transcript = truncate_text_by_tokens(transcript, max_transcript_tokens)
        full_prompt = build_prompt_body(truncated_transcript, duration, ocr_context)
        
        logger.info(f"Prompt tokens: {count_tokens_llama(full_prompt):,}")

        # 4) Call LLM
        enhanced_system_message = (
            "你是一個協助創建影片章節的助手。請分析提供的教育類逐字稿和從螢幕擷取的文字，"
            "並『只輸出』章節清單。每一行必須嚴格遵循格式：`HH:MM:SS - 標題`（繁體中文）。"
            "回應中請勿包含任何其他文字、評論或解釋。"
        )
        
        logger.info(f"Calling {service_type} API for chapter generation...")
        t0 = time.time()
        
        resp = call_llm(
            service_type=service_type,
            client=client,
            system_message=enhanced_system_message,
            user_message=full_prompt,
            model=model,
            max_tokens=2048,
            temperature=0.2,
            top_p=0.9
        )
        
        dt = time.time() - t0
        logger.info(f"LLM API call completed in {dt:.2f}s")

        # 5) Parse response
        if service_type == "azure":
            output_text = resp.choices[0].message.content
        else:  # openai
            output_text = resp.choices[0].message.content
        
        raw_chapters = parse_chapters_from_output(output_text)
        if not raw_chapters:
            raise RuntimeError("LLM returned no parseable chapters")
        
        raw_chapters = clean_chapter_titles(raw_chapters)
        logger.info(f"Parsed {len(raw_chapters)} raw chapters")

        # 6) Balance chapters
        min_gap_sec, target_range, max_caps = chapter_policy(duration)
        chapters = globally_balance_chapters(
            raw_chapters, duration, min_gap_sec, target_range, max_caps
        )
        
        if not chapters:
            raise RuntimeError("No chapters left after balancing")
        
        logger.info(f"Successfully generated {len(chapters)} balanced chapters")

        # Save final outputs
        if run_dir:
            with open(run_dir / "llm_output.txt", "w", encoding="utf-8") as f:
                f.write(output_text)
            with open(run_dir / "chapters.json", "w", encoding="utf-8") as f:
                json.dump(chapters, f, ensure_ascii=False, indent=2)
            with open(run_dir / "full_prompt.txt", "w", encoding="utf-8") as f:
                f.write(full_prompt)

        return chapters

    except Exception as e:
        logger.error(f"Chapter generation failed: {e}", exc_info=True)
        # Fallback: create time-based chapters
        fallback_chapters = {}
        for i in range(0, int(duration), 300):  # Every 5 minutes
            fallback_chapters[sec_to_hms(i)] = f"章節 {(i // 300) + 1}"
        logger.info(f"Created {len(fallback_chapters)} fallback chapters")
        return fallback_chapters

# Keep the fallback function for internal use
def fallback_chapters(duration_sec: int, chapters_dict: Dict[str, str]):
    """Create fallback chapters every 5 minutes"""
    try:
        for i in range(0, duration_sec, 300):
            chapters_dict[sec_to_hms(i)] = f"章節 {(i // 300) + 1}"
        logger.info(f"Created {len(chapters_dict)} fallback chapters")
    except Exception as e:
        logger.error(f"Failed to create fallback chapters: {e}")
