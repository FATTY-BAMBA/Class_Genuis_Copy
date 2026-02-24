# app/chapter_generation.py
"""
Module for generating video chapters from *raw* ASR and OCR inputs.
- No ASR/OCR preprocessing/merging; we use what you pass in.
- Enforces a ~128k token prompt budget (approx) for large-context models.
- Parses, lightly cleans, balances chapters, and converts titles to Traditional Chinese.
- Exposes RAW LLM output so you can inspect what the model produced before any parsing/balancing.

CLI examples:
    python video_chaptering.py --asr-file raw_asr.txt --duration 3600 --video-id test_01
    python video_chaptering.py --asr-file raw_asr.txt --ocr-file ocr_raw.txt --duration 1800 --video-id test_02
    # Show both RAW and FINAL in console:
    python video_chaptering.py --asr-file raw_asr.txt --duration 1800 --video-id debug_run --debug
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PASS3_JSON_SCHEMA = """
{
  "SuggestedUnits": [
    {
      "UnitNo": 1,
      "ParentUnitNo": null,
      "Title": "章節標題（繁體中文，反映講師實際講了什麼）",
      "Time": "HH:MM:SS（必須是逐字稿中存在的時間戳）",
      "trigger_quote": "從逐字稿中該時間點±2分鐘內複製的原文（10-30字）",
      "asr_keywords": ["關鍵詞1", "關鍵詞2", "關鍵詞3"]
    }
  ],
  "CourseSummary": {
    "topic": "...",
    "core_content": "...",
    "learning_objectives": "...",
    "target_audience": "...",
    "difficulty": "..."
  }
}
""".strip()


# Optional Azure AI Inference imports (only if used)
try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential
except Exception:  # optional at runtime
    ChatCompletionsClient = None  # type: ignore
    SystemMessage = None  # type: ignore
    UserMessage = None  # type: ignore
    AzureKeyCredential = None  # type: ignore

# Optional OpenAI import (only if used)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Optional Simplified→Traditional conversion (OpenCC preferred)
try:
    from opencc import OpenCC
    _opencc = OpenCC('s2t')
except Exception:
    _opencc = None

logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
@dataclass
class ChapterConfig:
    """Configuration for chapter generation service"""
    service_type: str = os.getenv("CHAPTER_SERVICE_TYPE", "openai")  # "openai" or "azure"
    openai_model: str = os.getenv("CHAPTER_OPENAI_MODEL", "gpt-4o-mini")
    azure_model: str = os.getenv("CHAPTER_AZURE_MODEL", "Meta-Llama-3.1-8B-Instruct")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    azure_endpoint: Optional[str] = os.getenv("AZURE_AI_ENDPOINT")
    azure_key: Optional[str] = os.getenv("AZURE_AI_KEY")
    azure_api_version: str = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")

def validate_config(config: ChapterConfig) -> bool:
    """Validate that required configuration is present"""
    if config.service_type == "azure":
        if not config.azure_endpoint or not config.azure_key:
            logger.error("Azure AI credentials not configured. Set AZURE_AI_ENDPOINT and AZURE_AI_KEY.")
            return False
    elif config.service_type == "openai":
        if not config.openai_api_key:
            logger.error("OpenAI API key not configured. Set OPENAI_API_KEY.")
            return False
    else:
        logger.error(f"Unknown service type: {config.service_type}")
        return False
    return True

def get_content_hash(transcript: str, ocr_context: str, duration: float) -> str:
    """Generate hash for content to enable caching"""
    content = f"{transcript}{ocr_context}{duration}"
    return hashlib.md5(content.encode()).hexdigest()

# ─────────────────────────
# Utilities
# ─────────────────────────
CHAPTER_LINE_RE = re.compile(
    r"""
    ^\s*
    (?:[\-\*•]\s*)?
    \[?(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]?\s*
    (?:[\-–—:]\s*)?
    (?P<title>.+?)
    \s*$
    """,
    re.VERBOSE,
)

ASR_TS_RE = re.compile(
    r"""^\s*
        \[?(\d{1,2}:\d{2}:\d{2})\]?
        \s*(?:[:\-–—]\s*|\s+)
    """,
    re.VERBOSE,
)
    
def ts_to_seconds_hms(ts: str) -> int:
    try:
        h, m, s = ts.strip().split(":")
        h, m, s = int(h), int(m), int(s)
        if h < 0 or m < 0 or s < 0 or m >= 60 or s >= 60:
            return -1
        return h * 3600 + m * 60 + s
    except Exception:
        return -1

def extract_asr_timestamps_sorted(raw_asr_text: str) -> List[str]:
    """Return sorted unique HH:MM:SS timestamps found in raw ASR."""
    seen = set()
    for line in (raw_asr_text or "").splitlines():
        m = ASR_TS_RE.match(line)
        if m:
            seen.add(_normalize_ts(m.group(1)))
    out = sorted(seen, key=ts_to_seconds_hms)
    return out

def get_first_last_asr_ts(raw_asr_text: str, duration_sec: int) -> Tuple[str, str]:
    """
    Returns (first_ts, last_ts) from raw ASR timestamps if present,
    otherwise falls back to 00:00:00 and full duration.
    """
    ts_sorted = extract_asr_timestamps_sorted(raw_asr_text)
    if ts_sorted:
        return ts_sorted[0], ts_sorted[-1]
    return "00:00:00", sec_to_hms(int(duration_sec))

def pick_anchor_timestamps(asr_ts_sorted: List[str], k: int = 12) -> List[str]:
    """
    Pick k timestamps spread across the whole ASR (including the tail).
    Always includes first and last if available.
    """
    if not asr_ts_sorted:
        return []
    if len(asr_ts_sorted) <= k:
        return asr_ts_sorted

    idxs = [0]
    # evenly spaced indices
    for i in range(1, k - 1):
        idxs.append(round(i * (len(asr_ts_sorted) - 1) / (k - 1)))
    idxs.append(len(asr_ts_sorted) - 1)

    # unique + ordered
    idxs = sorted(set(idxs))
    return [asr_ts_sorted[i] for i in idxs if 0 <= i < len(asr_ts_sorted)]

def chapters_coverage_ratio(
    suggested_units_structured: List[Dict[str, Any]],
    last_asr_sec: int
) -> float:
    """Compute last chapter time / last ASR time."""
    if not suggested_units_structured or last_asr_sec <= 0:
        return 1.0
    last_ch_ts = suggested_units_structured[-1].get("Time")
    last_ch_sec = ts_to_seconds_hms(str(last_ch_ts or ""))
    if last_ch_sec < 0:
        return 0.0
    return last_ch_sec / last_asr_sec


def sec_to_hms(sec: int) -> str:
    """Convert seconds to HH:MM:SS format"""
    if sec < 0:
        sec = 0
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _is_cjk(ch: str) -> bool:
    return '一' <= ch <= '鿿'

def clean_chapter_titles(chapters: Dict[str, str]) -> Dict[str, str]:
    """
    Clean up chapter titles by removing filler words and improving clarity.
    For Chinese titles, avoid English-centric capitalization; if over-trimmed (<4 chars),
    fall back to the original title.
    """
    cleaned: Dict[str, str] = {}
    filler_words = ['那', '所以', '這個', '那個', '就是', '呢', '啊', '喔', '然後', '接著']
    for ts, original_title in chapters.items():
        title = original_title
        for word in filler_words:
            title = title.replace(word, '')
        title = re.sub(r'[。，""、！？\.!?,]+$', '', title.strip())
        title = re.sub(r'\s+', ' ', title)

        # If cleaning made it too short, revert to the original
        if 0 < len(title) < 4:
            title = original_title.strip()

        cleaned[ts] = title 
    return cleaned

def clean_suggested_unit_title(title: str, max_length: int = 50) -> str:
    """
    Clean a SuggestedUnit title for client delivery.
    Removes raw ASR fragments, verbose descriptions, and redundant repetitions.
    """
    if not title or not isinstance(title, str):
        return title or ""
    
    title = title.strip()
    
    if " - " in title:
        parts = title.split(" - ", 1)
        category = parts[0].strip()
        subtitle = parts[1].strip()
        
        # Raw ASR speech markers → drop subtitle
        raw_markers = [
            '會跟你', '跟你們', '我們來', '我們要', '你們要',
            '今天是', '第一天', '第一次', '接下來',
            '那個', '這個', '就是', '所以', '然後',
            '好的', '好了', '對不對', '是不是',
            '老師', '大家', '開賣',
        ]
        has_raw = any(m in subtitle for m in raw_markers)
        
        # Verbose instructional phrases → drop subtitle
        verbose_markers = ['學生', '應能', '在課堂上', '將所學', '進行',
                           '深入講解', '掌握基本', '利用']
        has_verbose = any(m in subtitle for m in verbose_markers)
        
        # Category repeats in subtitle → redundant
        has_redundancy = category in subtitle
        
        # Short clean subtitle (≤12 chars, no markers) → keep as-is
        is_clean = (len(subtitle) <= 12
                    and not has_raw
                    and not has_verbose
                    and not has_redundancy)
        
        if is_clean:
            title = f"{category} - {subtitle}"
        elif has_raw:
            title = category
        elif has_redundancy:
            remainder = subtitle.replace(category, '').strip()
            remainder = re.sub(r'^[的與和及：:]+', '', remainder).strip()
            remainder = re.sub(r'[的與和及]+$', '', remainder).strip()
            if remainder and 2 <= len(remainder) <= 10:
                title = f"{category} - {remainder}"
            else:
                title = category
        elif has_verbose or len(subtitle) > 15:
            title = category
        else:
            title = f"{category} - {subtitle}"
    
    # Pure raw ASR sentence (no " - " prefix)
    elif len(title) > 15:
        markers = ['，', '。', '同學', '老師', '今天', '第一天',
                   '你們', '我們', '上課']
        if any(m in title for m in markers):
            for delim in ['，', '、', '。']:
                if delim in title:
                    first = title.split(delim)[0].strip()
                    if 4 <= len(first) <= max_length:
                        title = first
                        break
    
    if len(title) > max_length:
        title = title[:max_length - 1] + "…"
    
    return title or "章節"


def clean_all_suggested_units(suggested_units: list) -> list:
    """Clean all SuggestedUnit titles. Call before building client payload."""
    if not suggested_units:
        return suggested_units
    cleaned = [
        {**unit, "Title": clean_suggested_unit_title(unit.get("Title", ""))}
        for unit in suggested_units
    ]
    logger.info(f"🧹 Cleaned {len(cleaned)} SuggestedUnit titles")
    return cleaned

def count_tokens_llama(text: str) -> int:
    """Approximate token counting for mixed Chinese/English (≈1 token per CJK char; 1/4 per other chars)"""
    if not text:
        return 0
    chinese_chars = sum(1 for char in text if _is_cjk(char))
    non_chinese_len = len(text) - chinese_chars
    return chinese_chars + max(1, non_chinese_len // 4)

def truncate_text_by_tokens(text: str, max_tokens: int = 120_000) -> str:
    """Truncate text to approximately max_tokens, preserving sentence boundaries where possible"""
    if max_tokens <= 0:
        return ""
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

def stratified_sample_asr(
    raw_asr_text: str,
    duration_sec: int,
    token_budget: int = 100_000,
    bucket_sec: int = 300,
    lines_per_bucket: int = 3,
    head_tail_sec: int = 120,
) -> Tuple[str, Dict[str, Any]]:
    """
    Time-bucket sampling of ASR transcript for full timeline coverage.
    Instead of truncating from the beginning, selects representative lines
    from each time bucket across the entire video.
    
    Returns: (sampled_asr_text, sampler_stats)
    """
    if not raw_asr_text:
        return "", {"method": "empty", "lines_selected": 0}
    
    # If already within budget, return as-is
    if count_tokens_llama(raw_asr_text) <= token_budget:
        return raw_asr_text, {
            "method": "full_transcript",
            "lines_selected": len(raw_asr_text.splitlines()),
            "within_budget": True,
        }
    
    # Parse ASR lines into (seconds, full_line_text)
    parsed_lines: List[Tuple[int, str]] = []
    for line in raw_asr_text.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        m = ASR_TS_RE.match(line_stripped)
        if m:
            ts = _normalize_ts(m.group(1))
            sec = ts_to_seconds_hms(ts)
            if sec >= 0:
                parsed_lines.append((sec, line_stripped))
        else:
            # Non-timestamped line — attach to previous timestamp
            if parsed_lines:
                prev_sec, prev_text = parsed_lines[-1]
                parsed_lines[-1] = (prev_sec, prev_text + "\n" + line_stripped)
    
    if not parsed_lines:
        return truncate_text_by_tokens(raw_asr_text, token_budget), {
            "method": "fallback_truncate",
            "reason": "no_timestamps_found",
        }
    
    # Cue phrases that signal topic transitions (high-value lines)
    CUE_PATTERNS = re.compile(
        r'接下來|總結|重點|小結|我們來看|現在開始|休息|'
        r'第[一二三四五六七八九十\d]+[個部章節]|'
        r'Q\s*&?\s*A|問[與和]答|'
        r'首先|其次|最後|'
        r'這個部分|下一個|進入'
    )
    
    # Create time buckets
    max_sec = max(sec for sec, _ in parsed_lines)
    effective_duration = max(max_sec, duration_sec)
    num_buckets = max(1, effective_duration // bucket_sec + 1)
    
    # Assign lines to buckets
    buckets: Dict[int, List[Tuple[int, str, bool]]] = {}
    for sec, text in parsed_lines:
        bucket_idx = min(sec // bucket_sec, num_buckets - 1)
        is_cue = bool(CUE_PATTERNS.search(text))
        if bucket_idx not in buckets:
            buckets[bucket_idx] = []
        buckets[bucket_idx].append((sec, text, is_cue))
    
    # Select lines from each bucket
    selected: List[Tuple[int, str]] = []
    head_cutoff = head_tail_sec
    tail_cutoff = max(0, effective_duration - head_tail_sec)
    
    for bucket_idx in sorted(buckets.keys()):
        bucket_lines = buckets[bucket_idx]
        bucket_start_sec = bucket_idx * bucket_sec
        
        # Always include first line in bucket (anchor)
        picks: List[Tuple[int, str]] = [(bucket_lines[0][0], bucket_lines[0][1])]
        
        # Add cue lines (topic transitions)
        cue_lines = [(sec, text) for sec, text, is_cue in bucket_lines if is_cue]
        for cl in cue_lines[:2]:  # max 2 cue lines per bucket
            if cl not in picks:
                picks.append(cl)
        
        # If in head or tail window, include more lines
        if bucket_start_sec < head_cutoff or bucket_start_sec >= tail_cutoff:
            for sec, text, _ in bucket_lines[:lines_per_bucket + 2]:
                entry = (sec, text)
                if entry not in picks:
                    picks.append(entry)
        
        # Cap per bucket
        for p in picks[:lines_per_bucket + 2]:
            selected.append(p)
    
    # Sort by time, deduplicate
    selected.sort(key=lambda x: x[0])
    seen_texts = set()
    deduped: List[Tuple[int, str]] = []
    for sec, text in selected:
        # Normalize for dedup (first 50 chars)
        norm = text[:50].strip()
        if norm not in seen_texts:
            seen_texts.add(norm)
            deduped.append((sec, text))
    
    # Trim to token budget (drop from middle if needed, keep head + tail)
    result_text = "\n".join(text for _, text in deduped)
    if count_tokens_llama(result_text) <= token_budget:
        sampler_stats = {
            "method": "stratified_time_bucket",
            "duration_sec": effective_duration,
            "bucket_sec": bucket_sec,
            "num_buckets": num_buckets,
            "buckets_with_data": len(buckets),
            "lines_selected": len(deduped),
            "head_tail_sec": head_tail_sec,
            "final_tokens": count_tokens_llama(result_text),
        }
        return result_text, sampler_stats
    
    # Still over budget — trim from middle, preserve head and tail
    head_count = len(deduped) // 4
    tail_count = len(deduped) // 4
    head_lines = deduped[:head_count]
    tail_lines = deduped[-tail_count:] if tail_count > 0 else []
    middle_lines = deduped[head_count:len(deduped) - tail_count if tail_count > 0 else len(deduped)]
    
    # Keep adding middle lines until we hit budget
    final_lines = list(head_lines)
    remaining_budget = token_budget - count_tokens_llama("\n".join(t for _, t in head_lines)) - count_tokens_llama("\n".join(t for _, t in tail_lines))
    
    for sec, text in middle_lines:
        line_tokens = count_tokens_llama(text)
        if remaining_budget - line_tokens < 0:
            break
        final_lines.append((sec, text))
        remaining_budget -= line_tokens
    
    final_lines.extend(tail_lines)
    final_lines.sort(key=lambda x: x[0])
    
    result_text = "\n".join(text for _, text in final_lines)
    
    sampler_stats = {
        "method": "stratified_time_bucket",
        "duration_sec": effective_duration,
        "bucket_sec": bucket_sec,
        "num_buckets": num_buckets,
        "buckets_with_data": len(buckets),
        "lines_selected": len(final_lines),
        "head_tail_sec": head_tail_sec,
        "final_tokens": count_tokens_llama(result_text),
        "trimmed_from_middle": True,
    }
    return result_text, sampler_stats


# ─────────────────────────
# Chapter policy & parsing
# ─────────────────────────
def chapter_policy(duration_sec: int) -> Tuple[int, Tuple[int, int], int]:
    """Determine chapter generation parameters based on video duration"""
    if duration_sec < 30 * 60:     # < 30 min
        return 120, (5, 10), 30    # min_gap: 2 min, target: 5-10 chapters
    elif duration_sec < 60 * 60:   # < 1 hour  
        return 180, (6, 12), 40    # min_gap: 3 min, target: 6-12 chapters
    elif duration_sec < 120 * 60:  # < 2 hours
        return 240, (8, 16), 50    # min_gap: 4 min, target: 8-16 chapters
    elif duration_sec < 180 * 60:  # < 3 hours
        return 300, (10, 20), 60   # min_gap: 5 min, target: 10-20 chapters
    else:                           # 3+ hours
        return 360, (12, 24), 80   # min_gap: 6 min, target: 12-24 chapters

def _normalize_ts(ts: str) -> str:
    """Normalize timestamp format to HH:MM:SS"""
    parts = ts.split(":")
    if len(parts) == 2:
        return f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)}"
    if len(parts) == 3:
        return f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"
    return ts

def parse_chapters_from_output(output_text: str) -> Dict[str, str]:
    """Parse chapter timestamps and titles from LLM output"""
    chapters: Dict[str, str] = {}
    
    # Direct parsing for "HH:MM:SS - Title" format
    for line in output_text.splitlines():
        line = line.strip()
        if not line:
            continue
        
        # Look for the pattern "HH:MM:SS - Title"
        if ' - ' in line:
            parts = line.split(' - ', 1)
            if len(parts) == 2:
                ts = parts[0].strip()
                title = parts[1].strip()
                # Validate timestamp format
                if re.fullmatch(r'\d{2}:\d{2}:\d{2}', ts):
                    chapters[ts] = title
    
    # If no chapters found, try the original regex approach
    if not chapters:
        for line in output_text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = CHAPTER_LINE_RE.match(line)
            if m:
                ts = _normalize_ts(m.group("ts").strip())
                title = m.group("title").strip()
                if title:
                    chapters.setdefault(ts, title)
    
    return chapters

def parse_summary_from_output(output_text: str) -> Dict[str, str]:
    """Extract the structured summary from the LLM output"""
    summary = {}
    lines = output_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('課程主題：'):
            summary['topic'] = line.replace('課程主題：', '').strip()
        elif line.startswith('核心內容：'):
            summary['core_content'] = line.replace('核心內容：', '').strip()
        elif line.startswith('學習目標：'):
            summary['learning_objectives'] = line.replace('學習目標：', '').strip()
        elif line.startswith('適合對象：'):
            summary['target_audience'] = line.replace('適合對象：', '').strip()
        elif line.startswith('難度級別：'):
            summary['difficulty'] = line.replace('難度級別：', '').strip()
    
    # Apply Traditional Chinese conversion to summary fields
    if _opencc:
        for key in summary:
            summary[key] = to_traditional(summary[key])
    
    return summary

def _extract_json_blob(text: str) -> Optional[str]:
    """
    Try to extract JSON from:
    - raw JSON
    - ```json ... ```
    - ``` ... ```
    - or: first {...} / [...] span inside surrounding text
    Returns JSON string or None.
    """
    if not text:
        return None

    # ```json ... ```
    m = re.search(r"```json\s*([\[{].*?[\]}])\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # ``` ... ```
    m = re.search(r"```\s*([\[{].*?[\]}])\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    s = text.strip()

    # raw json only
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        return s

    # NEW: attempt to extract the first JSON object/array embedded in other text
    first_obj = s.find("{")
    last_obj = s.rfind("}")
    if 0 <= first_obj < last_obj:
        candidate = s[first_obj:last_obj + 1].strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate

    first_arr = s.find("[")
    last_arr = s.rfind("]")
    if 0 <= first_arr < last_arr:
        candidate = s[first_arr:last_arr + 1].strip()
        if candidate.startswith("[") and candidate.endswith("]"):
            return candidate

    return None


def safe_load_json(text: str) -> Optional[Any]:
    blob = _extract_json_blob(text)
    if not blob:
        return None
    try:
        return json.loads(blob)
    except Exception:
        return None

def _is_hms(ts: str) -> bool:
    return bool(re.fullmatch(r"\d{2}:\d{2}:\d{2}", (ts or "").strip()))

# --- Client unit parsing helpers (back-compat) ---
_CLIENT_UNIT_RE = re.compile(r"\[\s*單元\s*(\d+)\s*[:：]")

def _extract_client_unit_no_from_title(title: str) -> Optional[int]:
    m = _CLIENT_UNIT_RE.search(title or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def normalize_suggested_units(
    suggested_units: Any,
    units: Optional[List[Dict]] = None
) -> List[Dict[str, Any]]:
    """
    Normalize SuggestedUnits list:
    - enforce fields
    - ensure Time is HH:MM:SS
    - ParentUnitNo is optional and used ONLY for chapter hierarchy (do not validate it vs client Units)
    - If client units are provided:
        - validate ClientUnitNo against Units[].UnitNo
        - fill ClientUnitTitle from Units[].Title
    - sort by Time
    - renumber UnitNo sequentially
    """
    if not isinstance(suggested_units, list):
        return []

    valid_client_units = None
    unit_title_by_no: Dict[int, str] = {}
    if units:
        valid_client_units = set()
        for u in units:
            try:
                uno = int(u.get("UnitNo"))
                valid_client_units.add(uno)
                unit_title_by_no[uno] = str(u.get("Title") or "").strip()
            except Exception:
                pass

    out: List[Dict[str, Any]] = []
    for su in suggested_units:
        if not isinstance(su, dict):
            continue

        title = str(su.get("Title") or "").strip()
        ts = str(su.get("Time") or "").strip()
        if not title or not _is_hms(ts):
            continue

        # If title already starts with a unit prefix, strip it to avoid double-prefixing later
        title = re.sub(r'^\s*\[\s*單元\s*\d+\s*(?:[:：][^\]]+)?\]\s*', '', title).strip()

        # ParentUnitNo is chapter-hierarchy only now (optional; sanitize int/null)
        parent = su.get("ParentUnitNo", None)
        if parent is not None:
            try:
                parent = int(parent)
            except Exception:
                parent = None

        # NEW: ClientUnitNo mapping (validate vs client Units if provided)
        client_unit_no = su.get("ClientUnitNo", None)
        if client_unit_no is not None:
            try:
                client_unit_no = int(client_unit_no)
            except Exception:
                client_unit_no = None

        # Back-compat fallback: parse from title "[單元N：...]"
        if client_unit_no is None:
            client_unit_no = _extract_client_unit_no_from_title(title)

        if valid_client_units is None:
            # No client Units provided => allow null mapping
            client_unit_no = None
            client_unit_title = None
        else:
            if client_unit_no not in valid_client_units:
                client_unit_no = None
            client_unit_title = unit_title_by_no.get(client_unit_no) if client_unit_no else None

        # Preserve asr_keywords evidence field if present
        asr_kw = su.get("asr_keywords", [])
        if not isinstance(asr_kw, list):
            asr_kw = []
        
        out.append({
            "UnitNo": 0,  # will renumber
            "ParentUnitNo": parent,
            "Title": title,
            "Time": ts,
            "ClientUnitNo": client_unit_no,
            "ClientUnitTitle": client_unit_title,
            "trigger_quote": str(su.get("trigger_quote", "")).strip(),
            "asr_keywords": asr_kw,
        })

    out.sort(key=lambda x: x["Time"])
    for i, su in enumerate(out, 1):
        su["UnitNo"] = i
    return out


# ─── FIX 4: Multi-Segment Unit Timestamps ───
def back_calculate_unit_timestamps(
    suggested_units_structured: List[Dict[str, Any]],
    client_units: Optional[List[Dict[str, Any]]],
    gap_merge_sec: int = 900,  # 15 min: merge segments closer than this
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Back-calculate timestamps for client Units, supporting multi-segment/interleaving.
    
    A client unit can appear in multiple non-contiguous segments of the video.
    Each segment is a continuous run of chapters tagged to that unit.
    
    Returns:
        Tuple of:
        - enriched_units: Units with Time, Segments, and metadata
        - diagnostics: Validation results and statistics
    """
    if not client_units:
        return [], {
            "units_provided": False,
            "validation_passed": True,
            "message": "No client Units to process"
        }
    
    if not suggested_units_structured:
        return client_units, {
            "units_provided": True,
            "validation_passed": False,
            "error": "No SuggestedUnits available for mapping"
        }
    
    # Build mapping: ClientUnitNo -> list of chapter times (sorted)
    unit_chapters_map: Dict[int, List[Dict[str, Any]]] = {}
    unmapped_chapters = 0
    ext_chapters = 0
    admin_chapters = 0
    
    for su in suggested_units_structured:
        client_unit_no = su.get("ClientUnitNo")
        if client_unit_no is None:
            unmapped_chapters += 1
        elif client_unit_no == 0:
            ext_chapters += 1
        elif client_unit_no == -1:
            admin_chapters += 1
        elif client_unit_no > 0:
            if client_unit_no not in unit_chapters_map:
                unit_chapters_map[client_unit_no] = []
            unit_chapters_map[client_unit_no].append(su)
    
    # Sort chapters within each unit by time
    for unit_no in unit_chapters_map:
        unit_chapters_map[unit_no].sort(key=lambda x: ts_to_seconds_hms(x.get("Time", "")))
    
    # Enrich client units
    enriched_units = []
    units_with_multiple_segments = 0
    
    for unit in client_units:
        unit_no = unit.get("UnitNo")
        enriched_unit = dict(unit)
        
        if unit_no not in unit_chapters_map:
            logger.warning(f"⚠️ Unit {unit_no} ('{unit.get('Title')}') has NO mapped chapters in video!")
            enriched_unit.update({
                "Time": "",
                "EndTime": None,
                "Duration": None,
                "SuggestedUnitCount": 0,
                "Segments": [],
                "FirstChapter": None,
                "LastChapter": None,
            })
            enriched_units.append(enriched_unit)
            continue
        
        chapters = unit_chapters_map[unit_no]
        
        # Build segments by merging adjacent chapters
        segments: List[Dict[str, Any]] = []
        current_segment_start = ts_to_seconds_hms(chapters[0].get("Time", ""))
        current_segment_end = current_segment_start
        current_segment_chapters: List[str] = [chapters[0].get("Title", "")]
        
        for i in range(1, len(chapters)):
            ch_sec = ts_to_seconds_hms(chapters[i].get("Time", ""))
            if ch_sec < 0:
                continue
            
            gap = ch_sec - current_segment_end
            
            if gap <= gap_merge_sec:
                # Extend current segment
                current_segment_end = ch_sec
                current_segment_chapters.append(chapters[i].get("Title", ""))
            else:
                # Close current segment, start new one
                segments.append({
                    "Start": sec_to_hms(current_segment_start),
                    "End": sec_to_hms(current_segment_end),
                    "DurationSec": current_segment_end - current_segment_start,
                    "ChapterCount": len(current_segment_chapters),
                })
                current_segment_start = ch_sec
                current_segment_end = ch_sec
                current_segment_chapters = [chapters[i].get("Title", "")]
        
        # Close last segment
        segments.append({
            "Start": sec_to_hms(current_segment_start),
            "End": sec_to_hms(current_segment_end),
            "DurationSec": current_segment_end - current_segment_start,
            "ChapterCount": len(current_segment_chapters),
        })
        
        if len(segments) > 1:
            units_with_multiple_segments += 1
        
        total_active_sec = sum(seg["DurationSec"] for seg in segments)
        first_start = ts_to_seconds_hms(segments[0]["Start"])
        last_end = ts_to_seconds_hms(segments[-1]["End"])
        span_sec = last_end - first_start
        
        enriched_unit.update({
            "Time": segments[0]["Start"],  # Client-facing: earliest start
            "EndTime": segments[-1]["End"],
            "Duration": sec_to_hms(total_active_sec),
            "SpanDuration": sec_to_hms(span_sec) if span_sec > 0 else None,
            "SuggestedUnitCount": len(chapters),
            "Segments": segments,
            "IsContiguous": len(segments) == 1,
            "FirstChapter": chapters[0].get("Title", ""),
            "LastChapter": chapters[-1].get("Title", ""),
        })
        
        enriched_units.append(enriched_unit)
        
        seg_info = f"{len(segments)} segment{'s' if len(segments) > 1 else ''}"
        logger.info(
            f"✅ Unit {unit_no}: {unit.get('Title')}\n"
            f"   → {seg_info}, {len(chapters)} chapters\n"
            f"   → Time: {segments[0]['Start']} → {segments[-1]['End']}\n"
            f"   → Active: {sec_to_hms(total_active_sec)}"
        )
    
    # Diagnostics (no ORDER VIOLATION for interleaving!)
    diagnostics = {
        "units_provided": True,
        "total_units": len(client_units),
        "units_found": sum(1 for u in enriched_units if u.get("Time")),
        "units_missing": sum(1 for u in enriched_units if not u.get("Time")),
        "units_with_multiple_segments": units_with_multiple_segments,
        "total_suggested_units": len(suggested_units_structured),
        "mapped_to_client_units": sum(len(chs) for chs in unit_chapters_map.values()),
        "ext_chapters": ext_chapters,
        "admin_chapters": admin_chapters,
        "unmapped_chapters": unmapped_chapters,
        "validation_passed": True,  # No more ORDER VIOLATION errors
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("📍 UNIT TIMESTAMP BACK-CALCULATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Units found: {diagnostics['units_found']}/{diagnostics['total_units']}")
    logger.info(f"⚠️ Units missing: {diagnostics['units_missing']}")
    logger.info(f"🔀 Units with multiple segments: {units_with_multiple_segments}")
    logger.info(f"📊 Chapters: {diagnostics['mapped_to_client_units']} mapped, "
                f"{ext_chapters} EXT, {admin_chapters} ADMIN, {unmapped_chapters} unmapped")
    logger.info("=" * 60 + "\n")
    
    return enriched_units, diagnostics


# ─── FIX 6: Timestamp Snapping to Real ASR Lines ───
def snap_chapters_to_asr_timestamps(
    suggested_units: List[Dict[str, Any]],
    raw_asr_text: str,
    tolerance_sec: int = 30,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Snap each chapter's Time to the nearest real ASR timestamp.
    This kills round-number fakes (01:10:00, 01:20:00, 01:30:00).
    
    If two chapters snap to the same ASR line, the second one is bumped
    to the next closest unused ASR timestamp.
    
    Returns: (snapped_units, snap_diagnostics)
    """
    if not suggested_units or not raw_asr_text:
        return suggested_units, {"snapped": False, "reason": "empty input"}
    
    # Build sorted list of (seconds, timestamp_str) from ASR
    asr_ts_list: List[Tuple[int, str]] = []
    seen_secs = set()
    for line in raw_asr_text.splitlines():
        m = ASR_TS_RE.match(line.strip())
        if m:
            ts = _normalize_ts(m.group(1))
            sec = ts_to_seconds_hms(ts)
            if sec >= 0 and sec not in seen_secs:
                seen_secs.add(sec)
                asr_ts_list.append((sec, ts))
    
    asr_ts_list.sort(key=lambda x: x[0])
    
    if not asr_ts_list:
        return suggested_units, {"snapped": False, "reason": "no ASR timestamps found"}
    
    asr_secs = [s for s, _ in asr_ts_list]
    
    def _find_nearest(target_sec: int, used: set) -> Optional[Tuple[int, str]]:
        """Find nearest unused ASR timestamp within tolerance."""
        best = None
        best_dist = tolerance_sec + 1
        for s, ts in asr_ts_list:
            dist = abs(s - target_sec)
            if dist < best_dist and s not in used:
                best = (s, ts)
                best_dist = dist
        return best
    
    used_secs: set = set()
    snapped = []
    snap_count = 0
    no_snap_count = 0
    
    for su in suggested_units:
        ts_str = str(su.get("Time", "")).strip()
        ch_sec = ts_to_seconds_hms(ts_str)
        
        if ch_sec < 0:
            snapped.append(su)
            continue
        
        nearest = _find_nearest(ch_sec, used_secs)
        
        if nearest:
            new_sec, new_ts = nearest
            used_secs.add(new_sec)
            dist = abs(ch_sec - new_sec)
            
            if dist > 0:
                snap_count += 1
                logger.debug(
                    f"📌 Snapped chapter '{su.get('Title', '')[:30]}' "
                    f"from {ts_str} → {new_ts} (Δ{dist}s)"
                )
            
            su_copy = dict(su)
            su_copy["Time"] = new_ts
            su_copy["_original_time"] = ts_str
            su_copy["_snap_distance_sec"] = dist
            snapped.append(su_copy)
        else:
            # No ASR timestamp within tolerance — keep original but flag it
            no_snap_count += 1
            su_copy = dict(su)
            su_copy["_snap_failed"] = True
            snapped.append(su_copy)
            logger.warning(
                f"⚠️ No ASR timestamp within ±{tolerance_sec}s of {ts_str} "
                f"for chapter '{su.get('Title', '')[:30]}'"
            )
    
    diagnostics = {
        "snapped": True,
        "total_chapters": len(suggested_units),
        "chapters_snapped": snap_count,
        "chapters_no_snap_needed": len(suggested_units) - snap_count - no_snap_count,
        "chapters_snap_failed": no_snap_count,
        "tolerance_sec": tolerance_sec,
        "asr_timestamps_available": len(asr_ts_list),
    }
    
    if snap_count > 0:
        logger.info(
            f"📌 Timestamp snapping: {snap_count} snapped, "
            f"{no_snap_count} failed (±{tolerance_sec}s tolerance, "
            f"{len(asr_ts_list)} ASR timestamps available)"
        )
    
    return snapped, diagnostics


# ─── FIX 2: Bigram Quote Validator ───
def validate_chapters_against_asr(
    suggested_units: List[Dict[str, Any]],
    raw_asr_text: str,
    tolerance_sec: int = 180,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Post-generation validation: check that each chapter's trigger_quote
    actually exists in the ASR near the claimed timestamp.
    trigger_quote is INTERNAL only — never sent to client.
    """
    if not suggested_units or not raw_asr_text:
        return suggested_units, {"validated": False, "reason": "empty input"}

    asr_lines: Dict[int, str] = {}
    for line in raw_asr_text.splitlines():
        m = ASR_TS_RE.match(line)
        if m:
            ts = _normalize_ts(m.group(1))
            sec = ts_to_seconds_hms(ts)
            if sec >= 0:
                text = line[m.end():].strip().lstrip(':- ').strip()
                if text:
                    asr_lines[sec] = text

    if not asr_lines:
        return suggested_units, {"validated": False, "reason": "no ASR timestamps found"}

    validated = []
    hallucination_count = 0
    total_with_quotes = 0

    for su in suggested_units:
        quote = str(su.get("trigger_quote", "")).strip()
        ts = str(su.get("Time", "")).strip()
        ch_sec = ts_to_seconds_hms(ts)

        if not quote or len(quote) < 4:
            su["_quote_validated"] = None
            validated.append(su)
            continue

        total_with_quotes += 1

        window_text = ""
        for asr_sec, asr_text_line in asr_lines.items():
            if abs(asr_sec - ch_sec) <= tolerance_sec:
                window_text += asr_text_line + " "

        if window_text:
            # Normalize: strip punctuation and whitespace for comparison
            def _normalize_for_match(s: str) -> str:
                return re.sub(r'[\s\u3000，。！？、；：""''（）\[\]【】…—\-\.\,\!\?\;\:\"\'\(\)]+', '', s)
            
            norm_quote = _normalize_for_match(quote)
            norm_window = _normalize_for_match(window_text)
            
            # Check 1: Exact substring match (strongest signal)
            if norm_quote in norm_window:
                ratio = 1.0
                match_method = "exact_substr"
            else:
                # Check 2: Bigram Jaccard similarity (good for Chinese)
                def _bigrams(s: str) -> set:
                    return {s[i:i+2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}
                
                q_bigrams = _bigrams(norm_quote)
                w_bigrams = _bigrams(norm_window)
                
                if q_bigrams and w_bigrams:
                    intersection = q_bigrams & w_bigrams
                    union = q_bigrams | w_bigrams
                    ratio = len(intersection) / max(len(union), 1)
                    match_method = "bigram_jaccard"
                else:
                    ratio = 0.0
                    match_method = "empty"
            
            # Dynamic threshold based on quote length
            if len(norm_quote) < 12:
                threshold = 0.40  # Short quotes need higher similarity
            elif len(norm_quote) < 20:
                threshold = 0.30
            else:
                threshold = 0.20  # Longer quotes can tolerate lower Jaccard
        else:
            ratio = 0.0
            threshold = 0.40
            match_method = "no_window"

        su["_quote_validated"] = ratio >= threshold
        if ratio < threshold:
            hallucination_count += 1
            logger.warning(
                f"⚠️ QUOTE VALIDATION FAILED: {ts} "
                f"title='{su.get('Title', '')[:40]}' "
                f"quote='{quote[:30]}...' method={match_method} "
                f"score={ratio:.2f} threshold={threshold:.2f}"
            )

        validated.append(su)

    hallucination_rate = hallucination_count / max(total_with_quotes, 1)

    diagnostics = {
        "validated": True,
        "total_chapters": len(suggested_units),
        "chapters_with_quotes": total_with_quotes,
        "quotes_passed": total_with_quotes - hallucination_count,
        "quotes_failed": hallucination_count,
        "hallucination_rate": round(hallucination_rate, 3),
        "threshold_exceeded": hallucination_rate > 0.3,
    }

    if hallucination_rate > 0.3:
        logger.warning(
            f"⚠️ HIGH HALLUCINATION RATE: {hallucination_rate:.0%} "
            f"({hallucination_count}/{total_with_quotes} chapters failed)"
        )
    else:
        logger.info(
            f"✅ Quote validation: {total_with_quotes - hallucination_count}/"
            f"{total_with_quotes} chapters grounded in ASR"
        )

    return validated, diagnostics


# ─── FIX 3H: Updated strip_internal_fields ───
def strip_internal_fields(suggested_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Strip internal fields (trigger_quote, _quote_validated, _tag_reason) before client payload.
    These must NEVER appear in webhook responses or client-facing output.
    """
    if not suggested_units:
        return suggested_units

    INTERNAL_FIELDS = {"trigger_quote", "_quote_validated", "_tag_reason", "asr_keywords",
                       "_original_time", "_snap_distance_sec", "_snap_failed"}

    cleaned = []
    for su in suggested_units:
        clean_su = {
            k: v for k, v in su.items()
            if k not in INTERNAL_FIELDS and not k.startswith("_")
        }
        cleaned.append(clean_su)

    return cleaned


# ─── FIX 5: Chapter Window Index ───
def build_chapter_windows(
    suggested_units_structured: List[Dict[str, Any]],
    raw_asr_text: str,
    ocr_context: str = "",
    pad_before_sec: int = 90,
    pad_after_sec: int = 30,
) -> List[Dict[str, Any]]:
    """
    Build per-chapter ASR+OCR text windows for downstream QA/notes generation.
    
    Each window contains the ASR text from [chapter_start - pad_before] to 
    [next_chapter_start + pad_after], plus any OCR content near that timestamp.
    """
    if not suggested_units_structured or not raw_asr_text:
        return []
    
    # Parse ASR into (sec, text) pairs
    asr_lines: List[Tuple[int, str]] = []
    for line in raw_asr_text.splitlines():
        m = ASR_TS_RE.match(line.strip())
        if m:
            ts = _normalize_ts(m.group(1))
            sec = ts_to_seconds_hms(ts)
            text = line.strip()[m.end():].strip().lstrip(':- ').strip()
            if sec >= 0 and text:
                asr_lines.append((sec, text))
    
    # Parse OCR if provided (simple timestamp-based)
    ocr_lines: List[Tuple[int, str]] = []
    if ocr_context:
        for line in ocr_context.splitlines():
            # Match "* HH:MM:SS：text" format
            m_ocr = re.match(r'\*\s*(\d{2}:\d{2}:\d{2})[：:]\s*(.*)', line.strip())
            if m_ocr:
                sec = ts_to_seconds_hms(m_ocr.group(1))
                text = m_ocr.group(2).strip()
                if sec >= 0 and text:
                    ocr_lines.append((sec, text))
    
    # Build windows
    windows: List[Dict[str, Any]] = []
    
    for i, su in enumerate(suggested_units_structured):
        ch_sec = ts_to_seconds_hms(su.get("Time", ""))
        if ch_sec < 0:
            continue
        
        # Determine window boundaries
        window_start = max(0, ch_sec - pad_before_sec)
        if i + 1 < len(suggested_units_structured):
            next_sec = ts_to_seconds_hms(suggested_units_structured[i + 1].get("Time", ""))
            if next_sec > 0:
                window_end = next_sec + pad_after_sec
            else:
                window_end = ch_sec + 900  # 15 min default
        else:
            window_end = ch_sec + 1800  # Last chapter: 30 min window
        
        # Extract ASR text in window
        asr_snippet = " ".join(
            text for sec, text in asr_lines 
            if window_start <= sec <= window_end
        )
        
        # Extract OCR text in window
        ocr_snippet = " ".join(
            text for sec, text in ocr_lines
            if window_start <= sec <= window_end
        )
        
        windows.append({
            "UnitNo": su.get("UnitNo"),
            "Time": su.get("Time", ""),
            "Title": su.get("Title", ""),
            "ClientUnitNo": su.get("ClientUnitNo"),
            "StartSec": window_start,
            "EndSec": window_end,
            "ASRTokens": count_tokens_llama(asr_snippet),
            "ASRSnippet": asr_snippet[:5000],  # Cap per window for storage
            "OCRSnippet": ocr_snippet[:2000] if ocr_snippet else "",
        })
    
    logger.info(f"📖 Built {len(windows)} chapter windows (avg {sum(w['ASRTokens'] for w in windows) // max(len(windows), 1)} tokens/window)")
    return windows


# ─── FIX 3G: Updated suggested_units_to_chapters_dict (EXT/ADMIN aware) ───
def suggested_units_to_chapters_dict(
    suggested_units: List[Dict[str, Any]],
    *,
    duration_sec: Optional[int] = None,
    bump_limit_sec: int = 120,
) -> Dict[str, str]:
    """
    Convert SuggestedUnits -> chapters dict, preventing timestamp key collisions
    WITHOUT producing non-HH:MM:SS keys (so later validation won't drop them).
    """

    def bump_ts(ts: str, delta: int) -> Optional[str]:
        base = ts_to_seconds_hms(ts)
        if base < 0:
            return None
        bumped = base + delta
        if duration_sec is not None:
            if bumped < 0 or bumped > duration_sec - 1:
                return None
        return sec_to_hms(bumped)

    chapters: Dict[str, str] = {}
    used = set()

    for su in suggested_units:
        ts = str(su.get("Time") or "").strip()
        title = str(su.get("Title") or "").strip()
        cu = su.get("ClientUnitNo")
        cut = su.get("ClientUnitTitle")
        # Only prefix with unit info for actual client units (positive numbers)
        if cu is not None and isinstance(cu, int) and cu > 0 and cut:
            prefix = f"[單元{cu}：{cut}] "
        elif cu is not None and isinstance(cu, int) and cu > 0:
            prefix = f"[單元{cu}] "
        else:
            # EXT (0), ADMIN (-1), or unmapped → no unit prefix
            prefix = ""

        if not _is_hms(ts) or not title:
            continue

        candidate = ts

        if candidate in used:
            placed = False

            # 1) bump forward
            for d in range(1, bump_limit_sec + 1):
                cand2 = bump_ts(ts, d)
                if cand2 and cand2 not in used:
                    candidate = cand2
                    placed = True
                    break

            # 2) bump backward if forward didn't work
            if not placed:
                for d in range(1, bump_limit_sec + 1):
                    cand2 = bump_ts(ts, -d)
                    if cand2 and cand2 not in used:
                        candidate = cand2
                        placed = True
                        break

            if not placed:
                logger.warning(
                    "⚠️ Could not place unique HH:MM:SS timestamp for chapter (ts=%s, title=%s). Dropping.",
                    ts, title[:80]
                )
                continue

        used.add(candidate)
        chapters[candidate] = prefix + title

    return chapters


def validate_and_normalize_timestamps(
    chapters: Dict[str, str], 
    duration_sec: int,
    video_id: str = "unknown"
) -> Dict[str, str]:
    """
    Validate and normalize chapter timestamps to ensure:
    1. All timestamps are in HH:MM:SS format
    2. All timestamps are within video duration
    3. Timestamps are distributed across the video
    4. Suspicious clustering is detected and logged
    """
    if not chapters:
        return {}

    def ts_to_seconds(ts: str) -> int:
        try:
            parts = ts.split(':')
            if len(parts) == 3:
                h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
                if h < 0 or m < 0 or s < 0 or m >= 60 or s >= 60:
                    return -1
                return h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m, s = int(parts[0]), int(parts[1])
                if m < 0 or s < 0 or m >= 60 or s >= 60:
                    return -1
                return m * 60 + s
            return -1
        except Exception:
            return -1
        
    validated = {}
    timestamps_in_seconds = []
    
    for ts, title in chapters.items():
        ts_normalized = _normalize_ts(ts)
        ts_sec = ts_to_seconds(ts_normalized)
        
        if ts_sec > duration_sec:
            logger.warning(f"⚠️ [{video_id}] Skipping chapter at {ts_normalized} ({ts_sec}s) - exceeds duration ({duration_sec}s)")
            continue
        
        if ts_sec < 0:
            logger.warning(f"⚠️ [{video_id}] Skipping chapter at {ts_normalized} - invalid timestamp")
            continue
        
        validated[ts_normalized] = title
        timestamps_in_seconds.append(ts_sec)
    
    if not validated:
        logger.error(f"❌ [{video_id}] No valid chapters after timestamp validation!")
        return {}
    
    first_chapter_sec = min(timestamps_in_seconds)
    last_chapter_sec = max(timestamps_in_seconds)
    chapter_span_sec = last_chapter_sec - first_chapter_sec
    chapter_span_percent = (chapter_span_sec / duration_sec) * 100 if duration_sec > 0 else 0
    
    logger.info(f"📊 [{video_id}] Chapter span: {sec_to_hms(int(chapter_span_sec))} ({chapter_span_percent:.1f}% of video)")
    
    if chapter_span_percent < 10 and duration_sec > 600:
        logger.warning("=" * 70)
        logger.warning(f"⚠️  [{video_id}] SUSPICIOUS CHAPTER CLUSTERING DETECTED!")
        logger.warning(f"⚠️  All {len(validated)} chapters are in first {chapter_span_percent:.1f}% of video")
        logger.warning(f"⚠️  Video duration: {sec_to_hms(int(duration_sec))} ({duration_sec}s)")
        logger.warning(f"⚠️  Chapter range: {sec_to_hms(int(first_chapter_sec))} to {sec_to_hms(int(last_chapter_sec))}")
        logger.warning("=" * 70)
        
        logger.warning(f"⚠️  Chapters generated:")
        for ts, title in sorted(validated.items(), key=lambda x: ts_to_seconds(x[0]))[:5]:
            logger.warning(f"     {ts} - {title[:60]}")
        if len(validated) > 5:
            logger.warning(f"     ... and {len(validated) - 5} more")
    
    chapters_in_first_minute = sum(1 for ts_sec in timestamps_in_seconds if ts_sec < 60)
    if chapters_in_first_minute > len(timestamps_in_seconds) * 0.5:
        logger.warning(f"⚠️ [{video_id}] {chapters_in_first_minute}/{len(timestamps_in_seconds)} chapters in first 60 seconds - possible timestamp error")
    
    logger.info(f"✅ [{video_id}] Validated {len(validated)}/{len(chapters)} chapters")
    logger.info(f"   First: {sec_to_hms(int(first_chapter_sec))} | Last: {sec_to_hms(int(last_chapter_sec))} | Span: {chapter_span_percent:.1f}%")
    
    return validated

def globally_balance_chapters(
    chapters: Dict[str, str],
    duration_sec: int,
    min_gap_sec: int,
    target_range: Tuple[int, int],
    max_caps: int,
) -> Dict[str, str]:
    """Balance chapters with content-aware merging"""
    
    def extract_module_tag(title: str) -> str:
        match = re.match(r'\[([^\]]+)\]', title)
        return match.group(1) if match else ""

    def ts_to_s(ts: str) -> int:
        try:
            p = ts.strip().split(":")
            if len(p) == 2:
                m, s = int(p[0]), int(p[1])
                if m < 0 or s < 0 or m >= 60 or s >= 60:
                    return -1
                return m * 60 + s
            if len(p) == 3:
                h, m, s = int(p[0]), int(p[1]), int(p[2])
                if h < 0 or m < 0 or s < 0 or m >= 60 or s >= 60:
                    return -1
                return h * 3600 + m * 60 + s
            return -1
        except Exception:
            return -1

    cands = []
    for ts, t in chapters.items():
        s = ts_to_s(ts)
        if 0 <= s <= duration_sec:
            cands.append((s, ts, t.strip()))
        
    cands.sort(key=lambda x: x[0])
    if not cands:
        return {}

    dedup = []
    adaptive_min_gap = max(120, min_gap_sec // 2)
    
    logger.info(f"📊 Chapter balancing: {len(chapters)} input chapters, merge threshold = {adaptive_min_gap}s (policy min_gap: {min_gap_sec}s)")
    
    for s, ts, title in cands:
        if not dedup:
            dedup.append((s, ts, title))
            continue
            
        time_gap = s - dedup[-1][0]
        
        should_merge = False
        prev_module = extract_module_tag(dedup[-1][2])
        curr_module = extract_module_tag(title)
        
        if time_gap < adaptive_min_gap:
            should_merge = True
        elif prev_module and curr_module and prev_module == curr_module:
            should_merge = time_gap < min_gap_sec * 0.7
        elif time_gap < min_gap_sec // 2:
            should_merge = True
            
        if should_merge:
            prev_s, prev_ts, prev_title = dedup[-1]
            if curr_module and not prev_module:
                dedup[-1] = (prev_s, prev_ts, title)
            elif len(title) > len(prev_title) * 1.3:
                dedup[-1] = (prev_s, prev_ts, title)
        else:
            dedup.append((s, ts, title))

    t_low, t_high = target_range
    
    logger.info(f"Chapter balancing: {len(chapters)} → {len(dedup)} (target: {t_low}-{t_high})")
    
    if t_low <= len(dedup) <= t_high:
        return {ts: title for _, ts, title in dedup}

    if len(dedup) > t_high:
        selected = []
        segment_length = max(1, duration_sec // t_high)
        for i in range(t_high):
            segment_start = i * segment_length
            segment_end = (i + 1) * segment_length if i < t_high - 1 else duration_sec + 1
            segment_chapters = [c for c in dedup if segment_start <= c[0] < segment_end]
            if segment_chapters:
                segment_center = (segment_start + segment_end) // 2
                chosen = min(segment_chapters, key=lambda c: abs(c[0] - segment_center))
                selected.append(chosen)
        selected.sort(key=lambda x: x[0])
        return {ts: title for _, ts, title in selected}

    return {ts: title for _, ts, title in dedup[:max_caps]}

# ─────────────────────────
# OCR handling (optional legacy "segments" mode)
# ─────────────────────────
def load_ocr_segments(file_obj, filename: str) -> List[Dict]:
    """
    Accepts:
      - JSON array:        [ { "start": 0, "end": 3, "text": "..." }, ... ]
      - Wrapped JSON:      { "segments": [ {...}, ... ] }
      - JSON Lines (JSONL): one JSON object per line
      - Plain text (.txt): whole file becomes a single segment at t=0
    """
    try:
        data = json.load(file_obj)
        if isinstance(data, dict) and "segments" in data and isinstance(data["segments"], list):
            segments = data["segments"]
        elif isinstance(data, list):
            segments = data
        else:
            return []
        out = []
        for item in segments:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            start = int(item.get("start", 0))
            end = int(item.get("end", start))
            out.append({"start": start, "end": end, "text": text})
        return out
    except json.JSONDecodeError:
        pass

    try:
        file_obj.seek(0)
        segments = []
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                segments = None
                break
            if not isinstance(obj, dict):
                continue
            text = str(obj.get("text", "")).strip()
            if not text:
                continue
            start = int(obj.get("start", 0))
            end = int(obj.get("end", start))
            segments.append({"start": start, "end": end, "text": text})
        if segments is not None:
            return segments
    except Exception:
        pass

    try:
        file_obj.seek(0)
        txt = file_obj.read().strip()
    except Exception:
        txt = ""
    return [{"start": 0, "end": 0, "text": txt}] if txt else []

def build_ocr_context_from_segments(ocr_segments: List[Dict]) -> str:
    if not ocr_segments:
        return ""
    lines = ["# 螢幕/投影片擷取文字（原始）："]
    for seg in ocr_segments:
        start = int(seg.get("start", 0))
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        lines.append(f"* {sec_to_hms(start)}：{text}")
    return "\n".join(lines)

# ─────────────────────────
# Simplified→Traditional conversion
# ─────────────────────────
_S2T_FALLBACK_MAP = {
    "体": "體", "台": "臺", "后": "後", "广": "廣", "画": "畫", "录": "錄", "观": "觀",
    "面": "麵", "发": "發", "门": "門", "问": "問", "类": "類", "网": "網", "图": "圖",
    "书": "書", "记": "記", "读": "讀", "党": "黨", "术": "術", "层": "層", "约": "約",
}

def to_traditional(text: str) -> str:
    if not text:
        return text
    if _opencc is not None:
        try:
            return _opencc.convert(text)
        except Exception:
            pass
    return ''.join(_S2T_FALLBACK_MAP.get(ch, ch) for ch in text)

def ensure_traditional_chapters(chapters: Dict[str, str]) -> Dict[str, str]:
    return {ts: to_traditional(title) for ts, title in chapters.items()}

# ─────────────────────────
# Client Initialization & LLM call
# ─────────────────────────
def initialize_client(service_type: str, **kwargs) -> Any:
    if service_type == "azure":
        if ChatCompletionsClient is None or AzureKeyCredential is None:
            raise RuntimeError("Azure dependencies are not available in this environment.")
        return ChatCompletionsClient(
            endpoint=kwargs["endpoint"],
            credential=AzureKeyCredential(kwargs["key"]),
            api_version=kwargs.get("api_version", "2024-05-01-preview"),
        )
    elif service_type == "openai":
        if OpenAI is None:
            raise RuntimeError("OpenAI client is not available in this environment.")
        return OpenAI(
            api_key=kwargs["api_key"],
            base_url=kwargs.get("base_url", "https://api.openai.com/v1/"),
        )
    else:
        raise ValueError(f"Unknown service type: {service_type}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
)
def call_llm(
    service_type: str,
    client: Any,
    system_message: str,
    user_message: str,
    model: str,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> Any:
    if service_type == "azure":
        return client.complete(
            messages=[
                SystemMessage(content=system_message),
                UserMessage(content=user_message),
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            model=model,
        )
    elif service_type == "openai":
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response
    else:
        raise ValueError(f"Unknown service type: {service_type}")


# ─────────────────────────
# Prompt builder
# ─────────────────────────

def build_prompt_body(
    transcript: str,
    duration_sec: int,
    ocr_context: str = "",
    video_title: Optional[str] = None,
    first_ts_override: Optional[str] = None,
    last_ts_override: Optional[str] = None,
) -> str:
    duration_hms = sec_to_hms(int(duration_sec))
    min_gap_sec, (t_low, t_high), max_caps = chapter_policy(int(duration_sec))
    
    timestamps: List[str] = []
    for line in transcript.splitlines():
        m = ASR_TS_RE.match(line)
        if m:
            timestamps.append(_normalize_ts(m.group(1)))
    first_ts = first_ts_override or (timestamps[0] if timestamps else "00:00:00")
    last_ts  = last_ts_override  or (timestamps[-1] if timestamps else duration_hms)

    video_title_context = ""
    if video_title:
        clean_title = re.sub(r'\.(mp4|avi|mov|mkv|webm|flv|m4v)$', '', video_title, flags=re.IGNORECASE)
        video_title_context = f"""
        
# 📚 課程檔案資訊
檔名：{clean_title}
請參考檔名理解課程主題、章節編號、涵蓋內容等重要資訊，並據此設計章節結構。
"""
    
    prompt = f"""
# 教育章節設計專家 - 時間戳記精準對應版
你是資深線上課程設計專家，負責將教學影片轉化為專業教育章節結構。

{video_title_context}
# 🚨 最重要的規則 - 時間戳記必須精準對應
**逐字稿實際時間範圍：{first_ts} 到 {last_ts}**

## 絕對禁止的行為：
❌ 生成 00:00:00 章節（除非逐字稿真的從 00:00:00 開始）
❌ 規律時間間隔：每15分鐘、每30分鐘等固定模式
❌ 憑空想像時間點（必須對應逐字稿中的實際時間戳）
❌ 忽略逐字稿的時間範圍

## 必須遵守的規則：
✅ 第一個章節時間 >= {first_ts}（逐字稿開始時間）
✅ 最後一個章節時間 <= {last_ts}（逐字稿結束時間）  
✅ 每個章節時間必須接近逐字稿中實際討論該主題的時間戳（±60秒內）
✅ 基於內容自然轉折點，而非固定間隔

# 如何找到真實的章節轉折點：
## 語言信號詞（講師轉換話題）：
- 「接下來我們要講...」「現在進入...」「首先...第二...」
- 「我們來看一下...」「這個部分完成後，我們來看...」
- 「有了基礎概念，現在來實際操作...」
- 「問與答時間」「總結一下」「我們來練習...」

## 教學內容轉換：
- 新概念/技術的首次詳細解釋
- 理論講解 → 實際操作的轉換
- 不同工具/軟體的切換時間點
- 範例演示的開始與結束
- 練習題/互動環節的開始

## 視覺/操作轉換（參考OCR）：
- 畫面切換到新投影片/軟體界面
- 開始實際操作示範
- 檔案開啟/工具切換的時間點

# 錯誤示範 vs 正確做法：
## ❌ 錯誤（絕對避免）：
00:00:00 - 課程介紹
00:15:00 - 基礎概念  
00:30:00 - 進階應用
00:45:00 - 實作練習

## ✅ 正確（基於實際內容）：
{first_ts} - 課程開場與學習目標說明
[尋找逐字稿中第一個主題轉換的時間戳] - 第一個主要概念講解
[尋找逐字稿中理論轉實作的時間戳] - 實際操作演示開始
[尋找逐字稿中重要範例的時間戳] - 關鍵範例分析

# 影片資訊
- 總時長: {duration_hms}
- 逐字稿時間範圍: {first_ts} 到 {last_ts}
- 目標章節: {t_low}-{t_high} 個學習單元
- 最小間隔: {min_gap_sec//60} 分鐘

# 分析步驟：
1. **識別時間範圍**：確認逐字稿從 {first_ts} 開始，到 {last_ts} 結束
2. **通讀內容**：理解整體教學流程和知識架構
3. **標記轉折**：找出 {t_low}-{t_high} 個最重要的主題轉換點
4. **時間對應**：每個章節時間必須對應逐字稿中實際討論的時間
5. **標題精準**：用具體術語描述該時間點開始的教學內容

# 內容資料
## 主要逐字稿（包含真實時間戳）：
{transcript}

## 輔助視覺內容：
{ocr_context if ocr_context else "（無螢幕內容參考）"}

# 輸出格式
## 第一部分：章節列表
嚴格遵守：`HH:MM:SS - 具體章節標題`
- 時間戳必須是逐字稿中實際存在或非常接近（±60秒內）的時間
- 標題用繁體中文，具體描述該時間點開始的教學內容

## 第二部分：課程摘要（章節列表完成後，空一行輸出）
請提供結構化的課程摘要，格式如下：

課程主題：[主要教學領域，如：Python程式設計、Premiere Pro剪輯]
核心內容：[列出6-12個主要教學概念，以頓號分隔，涵蓋整個課程的關鍵知識點]
學習目標：[學生完成後應具備的能力]
適合對象：[目標學員背景]
難度級別：[初級/中級/高級]

# 最終檢查
生成每個章節前，問自己：
1. 這個時間點在逐字稿中是否有對應的內容轉換？
2. 章節時間是否在 {first_ts} 到 {last_ts} 範圍內？
3. 標題是否準確反映從這個時間點開始的教學內容？

完成章節後，檢查摘要：
1. 課程主題是否準確反映核心教學內容？
2. 核心內容是否包含最重要的2-3個技術點？
3. 學習目標是否具體可衡量？
"""
    return prompt

def build_educational_context(section_title: Optional[str], units: Optional[List[Dict]]) -> str:
    if not section_title and not units:
        return ""
    
    context_parts = []
    
    if section_title:
        context_parts.append(f"# 📚 課程單元資訊")
        context_parts.append(f"本影片屬於課程單元：**{section_title}**")
        context_parts.append("")
    
    if units:
        context_parts.append(f"## 預定教學單元結構 ({len(units)} 個單元)")
        context_parts.append("講師計劃在本課程中涵蓋以下教學單元：")
        context_parts.append("")
        for unit in units:
            context_parts.append(f"{unit['UnitNo']}. {unit['Title']}")
        
        context_parts.append("")
        context_parts.append("## 章節設計指引")
        context_parts.append("✅ 優先考慮這些預定單元作為主要章節分組")
        context_parts.append("✅ 在逐字稿中尋找講師實際講解這些單元的時間點")
        context_parts.append(f"✅ 目標：創建 {len(units) * 2} 到 {len(units) * 4} 個章節")
        context_parts.append("✅ 章節標題建議格式：[單元N：單元名稱] 具體內容")
        context_parts.append("")
        context_parts.append("**範例格式：**")
        context_parts.append("00:05:30 - [單元1：廚具規劃] 廚房工作三角原理與動線設計")
        context_parts.append("00:18:45 - [單元1：廚具規劃] 廚具尺寸標準與人體工學考量")
        context_parts.append("00:32:10 - [單元2：天花板大樣圖] 大樣圖繪製基本規範與圖例說明")
        context_parts.append("")
    
    return "\n".join(context_parts)


# ─── FIX 3.0: Unit Descriptor Generation ───
def generate_unit_descriptors(
    units: List[Dict],
    section_title: Optional[str],
    asr_sample: str,
    client: Any,
    config: "ChapterConfig",
) -> Optional[List[Dict[str, Any]]]:
    """
    Auto-expand vague client unit titles into rich descriptors for tagging.
    One lightweight LLM call (~800 tokens, ~$0.001).
    """
    if not units or len(units) == 0:
        return None
    
    unit_lines = []
    for u in units:
        unit_lines.append(f"  {u.get('UnitNo', '?')}. {u.get('Title', '未命名')}")
    units_text = "\n".join(unit_lines)
    
    section_context = f"課程名稱/段落：{section_title}" if section_title else "（未提供課程名稱）"
    asr_excerpt = asr_sample[:2000] if asr_sample else "（無逐字稿摘要）"
    
    descriptor_prompt = f"""你是教學內容分析專家。以下是一門課程的單元列表。請為每個單元生成描述詞，幫助後續自動分類。

{section_context}

【教學單元】
{units_text}

【逐字稿摘要（參考用，了解本課程實際內容）】
{asr_excerpt}

【任務】
為每個單元生成：
1. intent：這個單元預計涵蓋什麼內容（1-2句話）
2. keywords_positive：在逐字稿中聽到這些詞時，很可能正在教這個單元（8-20個詞）
3. keywords_negative：聽到這些詞時，通常不是這個單元的內容（5-10個詞）

【重要原則】
- 根據單元標題和課程名稱推斷內容，不要只複製標題文字
- keywords 要具體（如「三腳架」「穩定器」），不要太泛（如「影片」「教學」）
- 如果單元標題太模糊無法推斷內容（如「第一章」「重點」），keywords 可以少一些，這是正常的
- keywords_negative 用來區分相似但不同的單元

【輸出格式（只輸出 JSON）】
{{
  "unit_descriptors": [
    {{
      "unit_no": 1,
      "unit_title": "設備",
      "intent": "介紹影片拍攝所需的硬體設備與器材選擇",
      "keywords_positive": ["相機", "手機", "鏡頭", "穩定器", "三腳架", "收音", "麥克風", "燈光", "器材", "設備"],
      "keywords_negative": ["景別", "構圖", "剪輯", "後期", "轉場"]
    }}
  ]
}}

只輸出 JSON，禁止其他文字。
"""
    
    logger.info("🏷️ Generating unit descriptors for evidence-based tagging...")
    
    try:
        t0 = time.time()
        resp = call_llm(
            service_type=config.service_type,
            client=client,
            system_message=(
                "你是教學內容分析專家。根據單元標題和課程資訊，生成用於自動分類的描述詞。"
                "只輸出 JSON，禁止其他文字。"
            ),
            user_message=descriptor_prompt,
            model=config.openai_model if config.service_type == "openai" else config.azure_model,
            max_tokens=800,
            temperature=0.2,
        )
        
        elapsed = time.time() - t0
        resp_text = (resp.choices[0].message.content
                    if config.service_type == "openai"
                    else resp.choices[0].message.content)
        
        data = safe_load_json(resp_text)
        
        if isinstance(data, dict) and "unit_descriptors" in data:
            descriptors = data["unit_descriptors"]
            if isinstance(descriptors, list) and len(descriptors) > 0:
                logger.info(f"✅ Unit descriptors generated in {elapsed:.1f}s for {len(descriptors)} units:")
                for desc in descriptors:
                    kw_count = len(desc.get("keywords_positive", []))
                    logger.info(
                        f"   Unit {desc.get('unit_no')}: {desc.get('unit_title', '?')} "
                        f"→ {kw_count} positive keywords, intent: {desc.get('intent', '?')[:60]}"
                    )
                return descriptors
        
        logger.warning(f"⚠️ Unit descriptor generation returned unexpected format, skipping")
        return None
        
    except Exception as e:
        logger.warning(f"⚠️ Unit descriptor generation failed: {e}. Tagging will proceed without descriptors.")
        return None


def format_descriptors_for_tagging(descriptors: List[Dict[str, Any]]) -> str:
    """Format unit descriptors into a text block for the tagging prompt."""
    if not descriptors:
        return ""
    
    lines = ["【單元描述詞（用於判斷章節歸屬的客觀依據）】"]
    lines.append("標記章節時，請對照以下描述詞。只有當章節內容匹配多個 positive keywords 時，才歸入該單元。")
    lines.append("")
    
    for desc in descriptors:
        unit_no = desc.get("unit_no", "?")
        title = desc.get("unit_title", "?")
        intent = desc.get("intent", "")
        kw_pos = desc.get("keywords_positive", [])
        kw_neg = desc.get("keywords_negative", [])
        
        lines.append(f"📌 單元 {unit_no}：{title}")
        if intent:
            lines.append(f"   涵蓋範圍：{intent}")
        if kw_pos:
            lines.append(f"   ✅ 相關詞彙：{', '.join(kw_pos)}")
        if kw_neg:
            lines.append(f"   ❌ 不相關（屬於其他單元）：{', '.join(kw_neg)}")
        lines.append("")
    
    lines.append("⚠️ 如果章節內容不匹配任何單元的 2 個以上 positive keywords，且 intent 不吻合，")
    lines.append("   必須標記為 0（延伸教學內容）或 -1（非教學內容），不可強行歸入。")
    
    return "\n".join(lines)


# ─────────────────────────
# Hierarchical Multi-Pass Generation
# ─────────────────────────

def should_use_hierarchical(duration: float, transcript_length: int) -> bool:
    return (duration >= 1800 and transcript_length >= 5000 and duration <= 14400)


def hierarchical_multipass_generation(
    raw_asr_text: str,
    duration: float,
    ocr_context: str,
    video_title: Optional[str],
    section_title: Optional[str],
    units: Optional[List[Dict]],
    client: Any,
    config: ChapterConfig,
    progress_callback: Optional[Callable[[str, int], None]] = None
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    """
    Three-pass hierarchical generation for high-quality educational chapters.
    Returns: (raw_llm_text, chapters, metadata)
    """
    
    ASR_LIMIT = 100_000
    OCR_LIMIT = 15_000
    
    asr_tokens = count_tokens_llama(raw_asr_text)
    ocr_tokens = count_tokens_llama(ocr_context) if ocr_context else 0
    total_content_tokens = asr_tokens + ocr_tokens
    
    logger.info("=" * 60)
    logger.info("\U0001f393 HIERARCHICAL MULTI-PASS CHAPTER GENERATION")
    logger.info("   Strategy: ASR-primary (timing) + OCR-supporting (detail)")
    logger.info("=" * 60)
    logger.info(f"\U0001f4ca Original ASR tokens: {asr_tokens:,} (limit: {ASR_LIMIT:,})")
    logger.info(f"\U0001f4ca Original OCR tokens: {ocr_tokens:,} (limit: {OCR_LIMIT:,})")
    logger.info(f"\U0001f4ca Video duration: {sec_to_hms(int(duration))}")
    
    # FIX 1: Stratified sampling
    asr_text, sampler_stats = stratified_sample_asr(
        raw_asr_text, int(duration), token_budget=ASR_LIMIT
    )
    logger.info(f"\U0001f9ea ASR Sampler: {sampler_stats.get('method', 'unknown')} — "
                f"{sampler_stats.get('lines_selected', '?')} lines selected from "
                f"{sampler_stats.get('buckets_with_data', '?')}/{sampler_stats.get('num_buckets', '?')} buckets")
    ocr_text = truncate_text_by_tokens(ocr_context, OCR_LIMIT) if ocr_context else ""

    asr_ts_sorted = extract_asr_timestamps_sorted(raw_asr_text)
    asr_end_ts = asr_ts_sorted[-1] if asr_ts_sorted else sec_to_hms(int(duration))
    asr_end_sec = ts_to_seconds_hms(asr_end_ts)
    if asr_end_sec <= 0:
        asr_end_sec = int(duration)
        asr_end_ts = sec_to_hms(asr_end_sec)
    anchors = pick_anchor_timestamps(asr_ts_sorted, k=14)

    logger.info(f"\U0001f9fe ASR time coverage: first={asr_ts_sorted[0] if asr_ts_sorted else 'N/A'} "
                f"last={asr_end_ts} (asr_ts_count={len(asr_ts_sorted)})")
    logger.info(f"\U0001f9f7 Anchor timestamps (spread): {anchors}")

    asr_used = count_tokens_llama(asr_text)
    ocr_used = count_tokens_llama(ocr_text)
    content_used = asr_used + ocr_used
    asr_coverage = (asr_used / asr_tokens * 100) if asr_tokens > 0 else 100
    ocr_coverage = (ocr_used / ocr_tokens * 100) if ocr_tokens > 0 else 100
    
    logger.info(f"✅ Using per pass: ASR={asr_used:,} ({asr_coverage:.1f}%), OCR={ocr_used:,} ({ocr_coverage:.1f}%), Total={content_used:,}")
    
    # ==================== PASS 1 ====================
    logger.info("\n" + "-" * 60)
    logger.info("\U0001f50d PASS 1: Course Structure Analysis")
    logger.info("-" * 60)
    
    if progress_callback:
        progress_callback("analyzing_course_structure", 40)
    
    # NOTE: video_title is NOT passed to PASS 1/2/3 to prevent title-based hallucination.
    # It is only used for unit validation and logging.

    # NOTE: section_title and units are logged but NOT passed to PASS 1/2/3.
    # They are reserved for the TAGGING step (STEP 2) to prevent syllabus-hallucination.
    if section_title or units:
        logger.info("=" * 60)
        logger.info("\U0001f4da EDUCATIONAL METADATA PROVIDED (reserved for tagging step only)")
        if section_title:
            logger.info(f"   \U0001f4d6 Section: {section_title}")
        if units:
            logger.info(f"   \U0001f4d1 Units: {len(units)} predefined learning units")
            for unit in units:
                logger.info(f"      {unit['UnitNo']}. {unit['Title']}")
        logger.info("=" * 60)
        
    structure_prompt = f"""
作為資深教學設計專家，分析這個{sec_to_hms(int(duration))}教學影片逐字稿的整體架構。

⚠️ 重要：只根據逐字稿中講師「實際說的內容」來分析，不要根據影片標題或課程名稱推測。

【內容類型識別】
1. 哪些時段是正式教學內容？（理論講解、技能示範、案例分析）
2. 哪些時段是行政事務？（點名、設備測試、休息、課堂管理）
3. 哪些時段是題外話或閒聊？（與課程主題無關的討論）

【知識架構分析】
- 講師實際涵蓋了哪些主題？按時間順序列出
- 各主題的大約時間佔比

【教學方法識別】
- 理論講解 vs. 實例演示 vs. 操作練習 的比例分佈

完整逐字稿：
{asr_text}

視覺輔助內容：
{ocr_text if ocr_text else "無視覺輔助內容"}
"""
    
    t0 = time.time()
    try:
        structure_response = call_llm(
            service_type=config.service_type, client=client,
            system_message="你是課程架構分析專家。你只根據逐字稿中講師實際說的內容來分析，絕不根據影片標題或課程名稱推測。你會誠實區分正式教學、行政事務和題外話。",
            user_message=structure_prompt,
            model=config.openai_model if config.service_type == "openai" else config.azure_model,
            max_tokens=1200, temperature=0.3
        )
        structure_text = structure_response.choices[0].message.content
        logger.info(f"✅ PASS 1 completed in {time.time() - t0:.1f}s ({len(structure_text)} chars)")
    except Exception as e:
        logger.error(f"❌ PASS 1 failed: {e}", exc_info=True)
        raise
    
    # ==================== PASS 2 ====================
    logger.info("\n" + "-" * 60)
    logger.info("\U0001f4da PASS 2: Learning Modules Identification")
    logger.info("-" * 60)
    
    if progress_callback:
        progress_callback("identifying_learning_modules", 60)
    
    modules_prompt = f"""
基於課程結構分析：
{structure_text}

現在識別具體的學習模塊（7-12個），每個模塊應滿足：
1. 有明確的學習目標
2. 包含完整的教學閉環
3. 時長合理（10-30分鐘）

【模塊邊界識別策略】
第一優先：講師的重大主題轉換（ASR）
第二優先：教學邏輯的結構轉換
第三優先：視覺結構變化（OCR）

完整逐字稿：
{asr_text}

視覺輔助內容：
{ocr_text if ocr_text else "無視覺輔助內容"}

請輸出格式：
模塊名稱 ~ 起始時間戳(HH:MM:SS) ~ 結束時間戳(HH:MM:SS) ~ 核心學習點 ~ 教學方法
"""
    
    t0 = time.time()
    try:
        modules_response = call_llm(
            service_type=config.service_type, client=client,
            system_message="你是課程模塊設計師，擅長將教學內容分解為邏輯連貫的學習單元。",
            user_message=modules_prompt,
            model=config.openai_model if config.service_type == "openai" else config.azure_model,
            max_tokens=1500, temperature=0.2
        )
        modules_text = modules_response.choices[0].message.content
        logger.info(f"✅ PASS 2 completed in {time.time() - t0:.1f}s ({len(modules_text)} chars)")
    except Exception as e:
        logger.error(f"❌ PASS 2 failed: {e}", exc_info=True)
        raise

    # ==================== VALIDATE CLIENT UNITS ====================
    unit_validation_result = None
    validated_units = units

    if units and isinstance(units, list) and len(units) > 0:
        logger.info("\n" + "=" * 60)
        logger.info("\U0001f50d VALIDATING CLIENT UNITS")
        logger.info("=" * 60)
        try:
            from app.qa_generation import validate_units_relevance, UNIT_VALIDATION_THRESHOLD
            bare_units = [{"UnitNo": u.get("UnitNo"), "Title": u.get("Title")} for u in units]
            _module_topics = []
            for _line in (modules_text or "").splitlines():
                _line_s = _line.strip()
                if "~" in _line_s:
                    _parts = _line_s.split("~")
                    _topic = re.sub(r'^模塊\d+[：:]\s*', '', _parts[0].strip())
                    _topic = re.sub(r'^\d+\.\s*', '', _topic)
                    if _topic and len(_topic) >= 2:
                        _module_topics.append(_topic)
            content_from_passes = {"main_topics": _module_topics[:8], "key_concepts": [], "technical_terms": []}
            is_valid, score, reason = validate_units_relevance(
                units=bare_units, content_analysis=content_from_passes, chapters={},
                video_title=video_title or "", threshold=UNIT_VALIDATION_THRESHOLD,
                structure_analysis=structure_text, modules_analysis=modules_text,
                section_title=section_title,
            )
            unit_validation_result = {"is_valid": is_valid, "score": score, "reason": reason,
                                      "threshold": UNIT_VALIDATION_THRESHOLD, "units_provided": len(units)}
            if is_valid:
                logger.info(f"✅ CLIENT UNITS VALIDATED (score={score:.2f})")
                validated_units = units
            else:
                logger.warning(f"❌ CLIENT UNITS REJECTED (score={score:.2f}, reason={reason})")
                validated_units = None
        except Exception as e:
            logger.error(f"Unit validation error: {e}", exc_info=True)
            unit_validation_result = {"is_valid": True, "score": 1.0, "reason": f"Validation error: {str(e)[:50]}",
                                      "threshold": 0.0, "units_provided": len(units), "error": str(e)}
            validated_units = units
        logger.info("=" * 60 + "\n")

    # ==================== BUILD EDUCATIONAL CONTEXT ====================
    if validated_units:
        educational_context = build_educational_context(section_title, validated_units)
    else:
        educational_context = ""

    # ==================== PASS 3 ====================
    logger.info("\n" + "-" * 60)
    logger.info("\U0001f4d1 PASS 3: Detailed Chapter Generation")
    logger.info("-" * 60)
    
    if progress_callback:
        progress_callback("generating_detailed_chapters", 80)
    
    # PASS 3: No unit_context_hint — section title and units are reserved for STEP 2 tagging only.
    # This prevents "syllabus hallucination" where the model generates textbook chapters
    # instead of describing what the instructor actually said.

    _min_gap_sec, (_t_low, _t_high), _max_caps = chapter_policy(int(duration))
    _hard_min = max(_t_low, int(duration) // 900)
    _hard_min = max(_hard_min, 8)

    # Build anchor timestamps hint for PASS 3
    _anchor_hint = ""
    if anchors:
        _anchor_hint = f"""
## 逐字稿中的實際時間戳樣本（你的 Time 必須接近這些時間戳之一）
{', '.join(anchors)}
"""

    chapters_prompt = f"""
# \U0001f6a8 最重要原則：從逐字稿內容出發，禁止憑空想像

## 你的角色
你是影片章節分割專家。你的唯一任務是在逐字稿中找到自然的主題轉換點。
你必須描述講師「實際講了什麼」，而不是你認為講師「應該講什麼」。

## 絕對禁止
❌ 根據影片標題或課程名稱猜測章節內容
❌ 生成通用教科書式章節（如「基礎概念」「進階應用」「實作練習」）
❌ 章節數量少於 {_hard_min} 個
❌ 相鄰章節超過 20 分鐘
❌ 使用整數分鐘時間戳（如 01:00:00, 01:10:00, 01:20:00）除非逐字稿中確實存在
❌ 編造 trigger_quote 或 asr_keywords（必須是逐字稿中的原文）

## 必須遵守
✅ 每個章節的 Title 必須反映講師在該時間點「實際說了什麼」
✅ 每個 Time 必須接近逐字稿中一個真實的時間戳（±30秒內）
✅ 每個章節必須附帶 trigger_quote（從逐字稿該時間點±2分鐘內複製 10-30 字原文）
✅ 每個章節必須附帶 asr_keywords（從逐字稿中複製 2-3 個該段出現的關鍵詞）
✅ 章節均勻分佈，約每 10-15 分鐘一個
✅ 行政事務（點名、設備測試、休息）也要建立章節，標題如實描述

## 證據規則（EVIDENCE RULE）
如果你無法從逐字稿中找到支持某個章節的 trigger_quote 和 asr_keywords，
則 **不要建立該章節**。寧可少一個章節，也不要製造一個虛假的章節。

## 背景參考（僅供理解整體結構，不要用來決定章節標題）
{structure_text[:600] if structure_text else ""}

{modules_text[:600] if modules_text else ""}

{_anchor_hint}

## 逐字稿
{asr_text}

## 視覺輔助內容
{ocr_text if ocr_text else "（無）"}

## 影片總時長：{sec_to_hms(int(duration))}
## 最少章節數：{_hard_min}

## 輸出格式（只輸出 JSON）

{PASS3_JSON_SCHEMA}
"""
    
    t0 = time.time()
    try:
        final_response = call_llm(
            service_type=config.service_type, client=client,
            system_message=(
                "你是細心的章節設計師。你只根據逐字稿內容生成章節，絕不根據課程名稱猜測。"
                "每個章節的 Time 必須接近逐字稿中的真實時間戳。"
                "每個章節必須有來自逐字稿的 trigger_quote 和 asr_keywords 作為證據。"
                "請只輸出一個 JSON 物件，包含 SuggestedUnits 與 CourseSummary。"
                "禁止輸出任何其他文字、禁止 ```。"
            ),
            user_message=chapters_prompt,
            model=config.openai_model if config.service_type == "openai" else config.azure_model,
            max_tokens=3000, temperature=0.1
        )
        final_text = final_response.choices[0].message.content
        logger.info(f"✅ PASS 3 completed in {time.time() - t0:.1f}s ({len(final_text)} chars)")
    except Exception as e:
        logger.error(f"❌ PASS 3 failed: {e}", exc_info=True)
        raise
    
    # Parse results
    data = safe_load_json(final_text)
    suggested_units_structured: List[Dict[str, Any]] = []
    course_summary: Dict[str, Any] = {}

    if isinstance(data, dict):
        suggested_units_structured = normalize_suggested_units(data.get("SuggestedUnits"), units=validated_units)
        suggested_units_structured = clean_all_suggested_units(suggested_units_structured)
        cs = data.get("CourseSummary")
        if isinstance(cs, dict):
            course_summary = cs
    elif isinstance(data, list):
        suggested_units_structured = normalize_suggested_units(data, units=validated_units)
        suggested_units_structured = clean_all_suggested_units(suggested_units_structured)

    if _opencc and course_summary:
        for k, v in list(course_summary.items()):
            if isinstance(v, str):
                course_summary[k] = to_traditional(v)

    logger.info(f"\U0001f4ca Generated {len(suggested_units_structured)} chapters (before snapping & validation)")

    # FIX 6: Timestamp snapping — force chapters to real ASR timestamps
    snap_diagnostics = {}
    if suggested_units_structured:
        suggested_units_structured, snap_diagnostics = snap_chapters_to_asr_timestamps(
            suggested_units_structured, raw_asr_text, tolerance_sec=30
        )
        # If strict snapping missed too many, retry with wider tolerance
        if snap_diagnostics.get("chapters_snap_failed", 0) > len(suggested_units_structured) * 0.3:
            logger.warning("⚠️ >30% chapters failed strict snap, retrying with ±90s tolerance")
            suggested_units_structured, snap_diagnostics = snap_chapters_to_asr_timestamps(
                suggested_units_structured, raw_asr_text, tolerance_sec=90
            )

    # FIX 2: Quote validation
    quote_diagnostics = {}
    if suggested_units_structured:
        suggested_units_structured, quote_diagnostics = validate_chapters_against_asr(
            suggested_units_structured, raw_asr_text, tolerance_sec=180
        )
        if quote_diagnostics.get("threshold_exceeded"):
            logger.warning(f"⚠️ Quote validation threshold exceeded (rate={quote_diagnostics['hallucination_rate']:.0%})")
    
    # ────────── STEP 2: Semantic unit tagging with FIX 3.0 descriptors ──────────
    unit_descriptors = None
    coverage_policy = None
    
    if validated_units and len(validated_units) >= 1 and suggested_units_structured:
        logger.info(f"\n\U0001f3f7️  STEP 2: Semantic unit tagging for {len(suggested_units_structured)} chapters → {len(validated_units)} units")
        
        # FIX 3.0: Generate unit descriptors for evidence-based tagging
        # Use a small stratified sample (not prefix slice) so descriptors reflect full lecture
        descriptor_asr_sample, _ = stratified_sample_asr(
            raw_asr_text, int(duration), token_budget=3000
        )
        unit_descriptors = generate_unit_descriptors(
            units=validated_units,
            section_title=section_title,
            asr_sample=descriptor_asr_sample,
            client=client,
            config=config,
        )
        descriptors_text = format_descriptors_for_tagging(unit_descriptors) if unit_descriptors else ""
        
        # Build chapter list for tagging prompt
        chapter_lines = []
        for ch in suggested_units_structured:
            chapter_lines.append(f"  {ch['UnitNo']}. [{ch['Time']}] {ch['Title']}")
        chapters_list_text = "\n".join(chapter_lines)
        
        unit_lines = []
        for u in validated_units:
            unit_lines.append(f"  {u['UnitNo']}. {u['Title']}")
        units_list_text = "\n".join(unit_lines)
        
        # FIX 3A: Updated tagging prompt with EXT/ADMIN + descriptors + matched_keywords
        tagging_prompt = f"""你是教學內容分類專家。以下是一個教學影片的章節列表和客戶定義的教學單元。

【影片章節（已按時間排序）】
{chapters_list_text}

【客戶教學單元】
{units_list_text}

{descriptors_text}

【任務】
為每個章節標記最合適的類別。

【分類規則（嚴格遵守）】
1. 只有在章節內容與教學單元有「直接語義關聯」時，才標記為該單元的編號
2. 標記為 0 = 教學延伸內容（EXT）：影片中有教學價值但不屬於任何客戶單元
3. 標記為 -1 = 非教學內容（ADMIN）：課程介紹、行政事務、休息、設備設定
4. 寧可標記為 0 也不要強行歸入不相關的客戶單元
5. 同一單元可有多個章節，也可沒有任何章節
6. 每個章節只能歸入一個類別

【輸出格式（只輸出 JSON）】
{{
  "tags": [
    {{"chapter": 1, "unit": -1, "matched_keywords": [], "reason": "課前設備測試"}},
    {{"chapter": 2, "unit": 1, "matched_keywords": ["相機", "穩定器", "三腳架"], "reason": "講解拍攝器材"}},
    {{"chapter": 3, "unit": 0, "matched_keywords": [], "reason": "剪輯主題不在客戶單元中"}},
    {{"chapter": 4, "unit": 2, "matched_keywords": ["景別", "構圖", "近景"], "reason": "鏡頭語言屬於景別單元"}}
  ]
}}

其中：
- chapter = 章節編號
- unit = 客戶教學單元編號（正數）、0（延伸教學內容 EXT）、-1（非教學 ADMIN）
- matched_keywords = 該章節匹配到的 positive keywords
- reason = 簡短分類理由（10字內）

⚠️ 硬性規則：如果 matched_keywords 少於 2 個且 intent 不明確吻合 → 必須標記為 0 或 -1

必須為每個章節都提供標記。只輸出 JSON。
"""
        
        logger.info(f"\U0001f4e4 Tagging prompt: ~{count_tokens_llama(tagging_prompt):,} tokens")
        t0_tag = time.time()
        
        try:
            tag_response = call_llm(
                service_type=config.service_type, client=client,
                system_message="你是教學內容分類專家。根據章節的內容主題判斷它屬於哪個教學單元。非教學內容標記為 -1，延伸內容標記為 0。只輸出 JSON。",
                user_message=tagging_prompt,
                model=config.openai_model if config.service_type == "openai" else config.azure_model,
                max_tokens=1000, temperature=0.0
            )
            tag_elapsed = time.time() - t0_tag
            logger.info(f"✅ Unit tagging completed in {tag_elapsed:.1f}s")
            
            tag_text = tag_response.choices[0].message.content
            tag_data = safe_load_json(tag_text)
            
            # FIX 3B: Parse tags with EXT/ADMIN distinction + matched_keywords
            if isinstance(tag_data, dict) and "tags" in tag_data:
                tags_list = tag_data["tags"]
                valid_unit_nos = {int(u["UnitNo"]) for u in validated_units}
                valid_unit_titles = {int(u["UnitNo"]): str(u.get("Title", "")).strip() for u in validated_units}
                
                tag_map: Dict[int, Optional[int]] = {}
                tag_reasons: Dict[int, str] = {}
                tag_keywords: Dict[int, List[str]] = {}
                ext_count = 0
                admin_count = 0
                
                for tag_entry in tags_list:
                    if isinstance(tag_entry, dict):
                        ch_no = tag_entry.get("chapter")
                        u_no = tag_entry.get("unit")
                        reason = str(tag_entry.get("reason", ""))[:30]
                        matched_kw = tag_entry.get("matched_keywords", [])
                        if not isinstance(matched_kw, list):
                            matched_kw = []
                        if ch_no is not None and u_no is not None:
                            try:
                                ch_no = int(ch_no)
                                u_no = int(u_no)
                            except (ValueError, TypeError):
                                continue
                            tag_reasons[ch_no] = reason
                            tag_keywords[ch_no] = matched_kw
                            if u_no in valid_unit_nos:
                                tag_map[ch_no] = u_no
                            elif u_no == 0:
                                tag_map[ch_no] = 0
                                ext_count += 1
                            elif u_no == -1:
                                tag_map[ch_no] = -1
                                admin_count += 1
                            else:
                                tag_map[ch_no] = 0
                                ext_count += 1
                
                # FIX 3C: Apply tags with EXT/ADMIN distinction
                tagged_count = 0
                ext_tagged = 0
                admin_tagged = 0
                untagged_count = 0
                for su in suggested_units_structured:
                    ch_no = su.get("UnitNo")
                    if ch_no in tag_map:
                        mapped_unit = tag_map[ch_no]
                        su["_tag_reason"] = tag_reasons.get(ch_no, "")
                        if mapped_unit is not None and mapped_unit > 0:
                            su["ClientUnitNo"] = mapped_unit
                            su["ClientUnitTitle"] = valid_unit_titles.get(mapped_unit, "")
                            tagged_count += 1
                        elif mapped_unit == 0:
                            su["ClientUnitNo"] = 0
                            su["ClientUnitTitle"] = "延伸教學內容"
                            ext_tagged += 1
                        elif mapped_unit == -1:
                            su["ClientUnitNo"] = -1
                            su["ClientUnitTitle"] = None
                            admin_tagged += 1
                        else:
                            su["ClientUnitNo"] = None
                            su["ClientUnitTitle"] = None
                            untagged_count += 1
                    else:
                        su["ClientUnitNo"] = None
                        su["ClientUnitTitle"] = None
                        untagged_count += 1
                
                # FIX 3D: Updated logging
                logger.info(
                    f"\U0001f3f7️  Tagging results: {tagged_count} → client units, "
                    f"{ext_tagged} → EXT (educational), {admin_tagged} → ADMIN, "
                    f"{untagged_count} untagged"
                )
                
                for vu in validated_units:
                    uno = int(vu["UnitNo"])
                    unit_chapters = [su for su in suggested_units_structured if su.get("ClientUnitNo") == uno]
                    if unit_chapters:
                        first_ts = unit_chapters[0]["Time"]
                        last_ts = unit_chapters[-1]["Time"]
                        logger.info(f"   Unit {uno} ({vu['Title']}): {len(unit_chapters)} chapters, {first_ts} → {last_ts}")
                    else:
                        logger.warning(f"   Unit {uno} ({vu['Title']}): 0 chapters (not found in video)")
            else:
                logger.warning("⚠️ Unit tagging response did not contain valid 'tags' array")
                
        except Exception as e:
            logger.warning(f"⚠️ Unit tagging LLM call failed: {e}. Chapters will have no unit tags.")
    
    # FIX 3E: Coverage Policy
    if validated_units and suggested_units_structured:
        total_chapters = len(suggested_units_structured)
        mapped_to_units = sum(1 for su in suggested_units_structured 
                             if su.get("ClientUnitNo") is not None and su.get("ClientUnitNo", 0) > 0)
        ext_chapters_count = sum(1 for su in suggested_units_structured if su.get("ClientUnitNo") == 0)
        admin_chapters_count = sum(1 for su in suggested_units_structured if su.get("ClientUnitNo") == -1)
        
        chapter_coverage = mapped_to_units / max(total_chapters, 1)
        
        mapped_secs = []
        for su in suggested_units_structured:
            if su.get("ClientUnitNo") is not None and su.get("ClientUnitNo", 0) > 0:
                ch_sec = ts_to_seconds_hms(su.get("Time", ""))
                if ch_sec >= 0:
                    mapped_secs.append(ch_sec)
        
        time_coverage = (max(mapped_secs) - min(mapped_secs)) / duration if mapped_secs and duration > 0 else 0.0
        
        coverage_policy = {
            "total_chapters": total_chapters,
            "mapped_to_client_units": mapped_to_units,
            "ext_chapters": ext_chapters_count,
            "admin_chapters": admin_chapters_count,
            "unmapped": total_chapters - mapped_to_units - ext_chapters_count - admin_chapters_count,
            "chapter_coverage_ratio": round(chapter_coverage, 3),
            "time_coverage_ratio": round(time_coverage, 3),
        }
        
        if chapter_coverage >= 0.6:
            coverage_policy["level"] = "high"
            coverage_policy["recommendation"] = "Unit-first: QA/notes focus on client units"
        elif chapter_coverage >= 0.3:
            coverage_policy["level"] = "partial"
            coverage_policy["recommendation"] = "Mixed: client units + EXT modules"
        else:
            coverage_policy["level"] = "low"
            coverage_policy["recommendation"] = "AI-first: client units barely present"
        
        ext_topics = [su.get("Title", "") for su in suggested_units_structured if su.get("ClientUnitNo") == 0]
        coverage_policy["ext_topics"] = ext_topics
        
        logger.info(f"\n\U0001f4ca COVERAGE POLICY:")
        logger.info(f"   Chapters mapped: {mapped_to_units}/{total_chapters} ({chapter_coverage:.0%})")
        logger.info(f"   EXT: {ext_chapters_count}, ADMIN: {admin_chapters_count}")
        logger.info(f"   Level: {coverage_policy['level']} → {coverage_policy['recommendation']}")
    else:
        coverage_policy = None

    if validated_units and len(validated_units) >= 1 and not suggested_units_structured:
        logger.warning("⚠️ No chapters generated, skipping unit tagging")
    
    # ────────── Log final unit mapping ──────────
    if validated_units and suggested_units_structured:
        mapped = sum(1 for su in suggested_units_structured if su.get("ClientUnitNo") is not None and su.get("ClientUnitNo", 0) > 0)
        total = len(suggested_units_structured)
        if mapped > 0:
            logger.info(f"✅ Final: {mapped}/{total} chapters mapped to units")
    
    # Initialize variables
    enriched_units = None
    unit_diagnostics = None
    
    # ────────── Coverage Guardrail ──────────
    if suggested_units_structured and asr_end_sec > 0:
        cov = chapters_coverage_ratio(suggested_units_structured, asr_end_sec)
        last_ch = suggested_units_structured[-1]["Time"]
        logger.info(f"\U0001f4cf PASS3 coverage check: last_chapter={last_ch}, asr_end={asr_end_ts}, ratio={cov:.2f}")
        
        if asr_end_sec >= 3600 and cov < 0.60:
            logger.warning(f"⚠️ PASS3 chapters end too early (ratio={cov:.2f}). Retrying...")
            retry_hint = f"""
    【強制覆蓋規則】
    - 逐字稿最後時間戳約為：{asr_end_ts}
    - 你輸出的最後一個章節 Time 必須 >= {sec_to_hms(max(0, int(asr_end_sec * 0.85)))}
    - 以下是後段時間戳樣本：
    {", ".join(anchors[-10:] if len(anchors) >= 10 else anchors)}
    """
            chapters_prompt_retry = chapters_prompt + "\n" + retry_hint
            retry_resp = call_llm(
                service_type=config.service_type, client=client,
                system_message="你是細心的章節設計師。請只輸出 JSON。章節必須覆蓋整段逐字稿到後段。",
                user_message=chapters_prompt_retry,
                model=config.openai_model if config.service_type == "openai" else config.azure_model,
                max_tokens=3000, temperature=0.1
            )
            final_text_retry = retry_resp.choices[0].message.content
            data_retry = safe_load_json(final_text_retry)

            suggested_retry: List[Dict[str, Any]] = []
            course_summary_retry: Dict[str, Any] = {}
            if isinstance(data_retry, dict):
                suggested_retry = normalize_suggested_units(data_retry.get("SuggestedUnits"), units=validated_units)
                suggested_retry = clean_all_suggested_units(suggested_retry)
                cs2 = data_retry.get("CourseSummary")
                if isinstance(cs2, dict):
                    course_summary_retry = cs2
            elif isinstance(data_retry, list):
                suggested_retry = normalize_suggested_units(data_retry, units=validated_units)
                suggested_retry = clean_all_suggested_units(suggested_retry)
            
            if suggested_retry:
                suggested_units_structured = suggested_retry
                if course_summary_retry:
                    course_summary = course_summary_retry
                if _opencc and course_summary:
                    for k, v in list(course_summary.items()):
                        if isinstance(v, str):
                            course_summary[k] = to_traditional(v)
                final_text = final_text_retry
                logger.info(f"✅ PASS3 retry succeeded: {len(suggested_units_structured)} chapters")
            else:
                logger.warning("⚠️ PASS3 retry failed; keeping first result")
                    
    # ────────── Build chapters_raw ──────────
    if suggested_units_structured:
        chapters_raw = suggested_units_to_chapters_dict(
            suggested_units_structured, duration_sec=int(duration), bump_limit_sec=120
        )
    else:
        logger.warning("⚠️ PASS 3 JSON parse failed; falling back to text chapter parsing")
        chapters_raw = parse_chapters_from_output(final_text)
        course_summary = parse_summary_from_output(final_text)

    # ────────── Back-calculate Unit timestamps (FIX 4) ──────────
    if validated_units:
        enriched_units, unit_diagnostics = back_calculate_unit_timestamps(
            suggested_units_structured=suggested_units_structured,
            client_units=validated_units
        )
    elif units:
        logger.info("⏭️  Skipping Unit timestamp back-calculation (units rejected)")
        unit_diagnostics = {"units_provided": True, "validation_passed": False, "units_rejected": True}
 
    # ────────── Validate and normalize timestamps ──────────
    chapters = validate_and_normalize_timestamps(chapters_raw, int(duration), video_id="hierarchical_pass3")
    if not chapters:
        logger.error("❌ No valid chapters after timestamp validation, using time-based fallback")
        chapters = create_time_based_fallback(int(duration))

    quality_score = estimate_educational_quality(chapters, structure_text)
    logger.info(f"\U0001f4c8 Educational quality score: {quality_score:.2f}")
    
    # ==================== FIX 5: Build Chapter Windows ====================
    chapter_windows = build_chapter_windows(suggested_units_structured, raw_asr_text, ocr_text)

    # ==================== Build Metadata ====================
    metadata = {
        'generation_method': 'hierarchical_multi_pass_asr_primary',
        'strategy': 'ASR-primary for timing, OCR-supporting for detail',
        'structure_analysis': structure_text,
        'modules_analysis': modules_text,
        'educational_quality_score': quality_score,
        'course_summary': course_summary,
        'content_analysis': course_summary,
        'unit_validation': unit_validation_result,
        'sampler': sampler_stats,
        'token_usage': {
            'original': {'asr_tokens': asr_tokens, 'ocr_tokens': ocr_tokens, 'total_tokens': total_content_tokens},
            'used_per_pass': {'asr_tokens': asr_used, 'ocr_tokens': ocr_used, 'total_tokens': content_used},
            'limits': {'asr_limit': ASR_LIMIT, 'ocr_limit': OCR_LIMIT},
            'coverage': {'asr_coverage': f"{asr_coverage:.1f}%", 'ocr_coverage': f"{ocr_coverage:.1f}%"}
        }
    }
    metadata["_suggested_units_debug"] = suggested_units_structured
    metadata["suggested_units_structured"] = strip_internal_fields(suggested_units_structured)
    metadata["client_units_original"] = units
    metadata["client_units_validated"] = validated_units
    metadata["client_units_with_timestamps"] = enriched_units
    metadata["unit_diagnostics"] = unit_diagnostics
    metadata["coverage_policy"] = coverage_policy
    metadata["unit_descriptors"] = unit_descriptors
    metadata["chapter_windows"] = chapter_windows
    metadata["pass3_raw_json_text"] = final_text
    metadata["quote_validation"] = quote_diagnostics
    metadata["timestamp_snapping"] = snap_diagnostics

    logger.info("\n" + "=" * 60)
    logger.info("✅ HIERARCHICAL GENERATION COMPLETE")
    logger.info(f"\U0001f4ca Chapters: {len(chapters)}, Quality: {quality_score:.2f}")
    logger.info("=" * 60 + "\n")
    
    return final_text, chapters, metadata

def estimate_educational_quality(chapters: Dict[str, str], structure: str) -> float:
    """Simple heuristic to estimate educational quality of chapters"""
    quality_indicators = [
        '講解', '原理', '範例', '練習', '實作', '應用', '總結', '重點',
        '概念', '方法', '技巧', '步驟', '案例', '分析'
    ]
    
    title_text = ' '.join(chapters.values())
    indicator_count = sum(1 for indicator in quality_indicators 
                         if indicator in title_text)
    
    total_titles = len(chapters)
    return min(1.0, indicator_count / max(1, total_titles * 0.7))


def create_time_based_fallback(duration_sec: int) -> Dict[str, str]:
    """Create fallback chapters based on time intervals"""
    fallback_chapters: Dict[str, str] = {}
    interval = 300  # 5 minutes
    for i in range(0, int(duration_sec), interval):
        fallback_chapters[sec_to_hms(i)] = "章節 " + str((i // interval) + 1)
    logger.info(f"Created {len(fallback_chapters)} time-based fallback chapters")
    return fallback_chapters


def generate_chapters_debug(
    raw_asr_text: str,
    ocr_segments: List[Dict],
    duration: float,
    video_id: str,
    video_title: Optional[str] = None,
    section_title: Optional[str] = None,
    units: Optional[List[Dict]] = None,
    run_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    *,
    ocr_context_override: Optional[str] = None,
    force_generation_method: Optional[str] = None,
) -> Tuple[str, Dict[str, str], Dict[str, str], Dict[str, Any]]:
    """Enhanced version with smart routing between hierarchical and single-pass generation"""
    if progress_callback:
        progress_callback("initializing", 0)

    if run_dir is None:
        run_dir = Path(f"/tmp/chapter_generation/{video_id}_{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Starting chapter generation for video {video_id} (duration: {duration}s)")

        config = ChapterConfig()
        if not validate_config(config):
            logger.warning("Configuration validation failed, using time-based fallback")
            fallback = create_time_based_fallback(int(duration))
            fallback = ensure_traditional_chapters(fallback)
            return ("", {}, fallback, {"generation_method": "time_based_fallback", "course_summary": {}})

        if progress_callback:
            progress_callback("processing_inputs", 10)

        if ocr_context_override is not None:
            ocr_context = ocr_context_override
        else:
            ocr_context = build_ocr_context_from_segments(ocr_segments) if ocr_segments else ""

        min_gap_sec, target_range, max_caps = chapter_policy(int(duration))
        
        with open(run_dir / "raw_asr_text.txt", "w", encoding="utf-8") as f:
            f.write(raw_asr_text)
        if ocr_context_override is not None:
            with open(run_dir / "ocr_raw.txt", "w", encoding="utf-8") as f:
                f.write(ocr_context_override)
        else:
            with open(run_dir / "ocr_segments.json", "w", encoding="utf-8") as f:
                json.dump(ocr_segments, f, ensure_ascii=False, indent=2)

        if progress_callback:
            progress_callback("initializing_client", 20)

        service_type = config.service_type
        model = config.openai_model if service_type == "openai" else config.azure_model

        if service_type == "azure":
            client = initialize_client(
                service_type="azure",
                endpoint=config.azure_endpoint,
                key=config.azure_key,
                api_version=config.azure_api_version,
            )
        else:
            client = initialize_client(
                service_type="openai",
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
            )

        use_hierarchical = False
        if force_generation_method == 'hierarchical':
            use_hierarchical = True
        elif force_generation_method == 'single_pass':
            use_hierarchical = False
        else:
            use_hierarchical = should_use_hierarchical(duration, len(raw_asr_text))
        
        logger.info(f"Using generation method: {'hierarchical_multi_pass' if use_hierarchical else 'single_pass'}")

        if use_hierarchical:
            if progress_callback:
                progress_callback("hierarchical_analysis", 30)
            
            raw_llm_text, chapters, metadata = hierarchical_multipass_generation(
                raw_asr_text=raw_asr_text,
                duration=duration,
                ocr_context=ocr_context,
                video_title=video_title,
                section_title=section_title,
                units=units,
                client=client,
                config=config,
                progress_callback=progress_callback
            )
            
            with open(run_dir / "hierarchical_metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            with open(run_dir / "course_structure.txt", "w", encoding="utf-8") as f:
                f.write(metadata.get('structure_analysis', ''))
            with open(run_dir / "learning_modules.txt", "w", encoding="utf-8") as f:
                f.write(metadata.get('modules_analysis', ''))
                
        else:
            if progress_callback:
                progress_callback("single_pass_processing", 30)

            first_ts, last_ts = get_first_last_asr_ts(raw_asr_text, int(duration))
            prompt_template = build_prompt_body(
                "", int(duration), ocr_context, video_title,
                first_ts_override=first_ts,
                last_ts_override=last_ts,
            )
            template_tokens = count_tokens_llama(prompt_template)
            CONTEXT_BUDGET = 110_000  # Safety margin below model's 128k limit

            asr_tokens = count_tokens_llama(raw_asr_text)
            if template_tokens + asr_tokens <= CONTEXT_BUDGET:
                transcript_for_prompt = raw_asr_text
            else:
                max_transcript_tokens = max(0, CONTEXT_BUDGET - template_tokens)
                transcript_for_prompt = truncate_text_by_tokens(raw_asr_text, max_transcript_tokens)

            full_prompt = build_prompt_body(
                transcript_for_prompt, int(duration), ocr_context, video_title,
                first_ts_override=first_ts,
                last_ts_override=last_ts,
            )
                
            with open(run_dir / "full_prompt.txt", "w", encoding="utf-8") as f:
                f.write(full_prompt)

            if progress_callback:
                progress_callback("calling_llm", 50)

            enhanced_system_message = (
                "你是專業的線上課程設計專家，擅長為各種學科創建高品質教育章節結構。"
                "僅輸出章節清單，每行格式: `HH:MM:SS - 標題`（繁體中文）。"
            )

            t0 = time.time()
            resp = call_llm(
                service_type=service_type, client=client,
                system_message=enhanced_system_message,
                user_message=full_prompt,
                model=model,
                max_tokens=2048, temperature=0.2, top_p=0.9,
            )
            dt = time.time() - t0
            logger.info(f"LLM API call completed in {dt:.2f}s")

            raw_llm_text = resp.choices[0].message.content

            chapters_raw = parse_chapters_from_output(raw_llm_text)
            chapters = validate_and_normalize_timestamps(chapters_raw, int(duration), video_id=video_id)
            if not chapters:
                chapters = create_time_based_fallback(int(duration))

            course_summary = parse_summary_from_output(raw_llm_text)
            metadata = {'generation_method': 'single_pass', 'course_summary': course_summary}

        # COMMON POST-PROCESSING
        if progress_callback:
            progress_callback("parsing_response", 70)

        with open(run_dir / "llm_output_raw.txt", "w", encoding="utf-8") as f:
            f.write(raw_llm_text)

        parsed_raw_clean_trad = ensure_traditional_chapters(clean_chapter_titles(chapters))

        with open(run_dir / "parsed_raw_chapters.json", "w", encoding="utf-8") as f:
            json.dump(parsed_raw_clean_trad, f, ensure_ascii=False, indent=2)

        if progress_callback:
            progress_callback("balancing_chapters", 80)

        chapters_final = globally_balance_chapters(
            parsed_raw_clean_trad, int(duration), min_gap_sec, target_range, max_caps
        )
        if not chapters_final:
            raise RuntimeError("No chapters left after balancing")

        with open(run_dir / "chapters_final.json", "w", encoding="utf-8") as f:
            json.dump(chapters_final, f, ensure_ascii=False, indent=2)

        with open(run_dir / "generation_method.txt", "w", encoding="utf-8") as f:
            f.write(metadata.get('generation_method', 'unknown'))

        if progress_callback:
            progress_callback("completed", 100)

        return (raw_llm_text, parsed_raw_clean_trad, chapters_final, metadata)

    except Exception as e:
        logger.error(f"Chapter generation failed: {e}", exc_info=True)
        fallback = ensure_traditional_chapters(create_time_based_fallback(int(duration)))
        fallback_metadata = {
            'generation_method': 'time_based_fallback',
            'educational_quality_score': 0.0,
            'course_summary': {}
        }
        return ("", {}, fallback, fallback_metadata)


def generate_chapters(
    raw_asr_text: str,
    ocr_segments: List[Dict],
    duration: float,
    video_id: str,
    video_title: Optional[str] = None,
    section_title: Optional[str] = None,
    units: Optional[List[Dict]] = None,
    run_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    *,
    ocr_context_override: Optional[str] = None,
    force_generation_method: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Generate chapters and return (chapters_dict, metadata).
    """
    _raw_text, _parsed_raw, final_chapters, metadata = generate_chapters_debug(
        raw_asr_text=raw_asr_text,
        ocr_segments=ocr_segments,
        duration=duration,
        video_id=video_id,
        video_title=video_title,
        section_title=section_title,
        units=units,
        run_dir=run_dir,
        progress_callback=progress_callback,
        ocr_context_override=ocr_context_override,
        force_generation_method=force_generation_method
    )
    return final_chapters, metadata


# ─────────────────────────
# CLI
# ─────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate video chapters from raw ASR and optional OCR.")
    parser.add_argument('--asr-file', type=argparse.FileType('r', encoding='utf-8'), required=True)
    parser.add_argument('--ocr-file', type=argparse.FileType('r', encoding='utf-8'))
    parser.add_argument('--duration', type=float, required=True)
    parser.add_argument('--video-id', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./chapter_debug')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ocr-mode', choices=['none', 'verbatim', 'segments'], default='verbatim')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

    raw_asr_text = args.asr_file.read()
    args.asr_file.close()

    ocr_segments: List[Dict] = []
    ocr_context_override: Optional[str] = None
    if args.ocr_file:
        if args.ocr_mode == 'none':
            try:
                args.ocr_file.close()
            except Exception:
                pass
        elif args.ocr_mode == 'verbatim':
            try:
                ocr_context_override = args.ocr_file.read()
            finally:
                try:
                    args.ocr_file.close()
                except Exception:
                    pass
        else:
            try:
                ocr_segments = load_ocr_segments(args.ocr_file, args.ocr_file.name)
                args.ocr_file.close()
            except Exception as e:
                logger.warning(f"OCR file load failed: {e}")
                try:
                    args.ocr_file.close()
                except Exception:
                    pass
                ocr_segments = []

    run_dir = Path(args.output_dir) / args.video_id
    run_dir.mkdir(parents=True, exist_ok=True)

    def cli_progress_callback(stage: str, percent: int):
        logger.info(f"Progress: {percent}% - {stage}")

    raw_text, parsed_raw, final_chapters, metadata = generate_chapters_debug(
        raw_asr_text=raw_asr_text,
        ocr_segments=ocr_segments,
        duration=args.duration,
        video_id=args.video_id,
        run_dir=run_dir,
        progress_callback=cli_progress_callback,
        ocr_context_override=ocr_context_override,
    )
    
    print("\n" + "="*50)
    print("✅ CHAPTER GENERATION COMPLETE")
    print("="*50)

    if args.debug:
        print("\n--- RAW LLM OUTPUT ---")
        print(raw_text if raw_text else "(empty)")
        print("\n--- PARSED (pre-balance) ---")
        for ts, title in parsed_raw.items():
            print(f"{ts} - {title}")

    print("\n--- FINAL (balanced) ---")
    for ts, title in final_chapters.items():
        print(f"{ts} - {title}")

    output_file = run_dir / "final_chapters.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for timestamp, title in final_chapters.items():
            f.write(f"{timestamp} - {title}\n")

    pre_file = run_dir / "parsed_raw_chapters.txt"
    with open(pre_file, 'w', encoding='utf-8') as f:
        for timestamp, title in parsed_raw.items():
            f.write(f"{timestamp} - {title}\n")

if __name__ == "__main__":
    main()
