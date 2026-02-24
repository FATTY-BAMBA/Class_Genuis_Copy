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
      "Title": "章節標題（繁體中文）",
      "Time": "HH:MM:SS"
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
    (?:[\-\*\u2022]\s*)?
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
    return '\u4e00' <= ch <= '\u9fff'

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
        title = re.sub(r'[。，“”、！？\.!?,]+$', '', title.strip())
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

        out.append({
            "UnitNo": 0,  # will renumber
            "ParentUnitNo": parent,
            "Title": title,
            "Time": ts,
            "ClientUnitNo": client_unit_no,
            "ClientUnitTitle": client_unit_title,
        })

    out.sort(key=lambda x: x["Time"])
    for i, su in enumerate(out, 1):
        su["UnitNo"] = i
    return out

def back_calculate_unit_timestamps(
    suggested_units_structured: List[Dict[str, Any]],
    client_units: Optional[List[Dict[str, Any]]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Back-calculate timestamps for client's Units based on SuggestedUnits mapping.
    Also validates logical order and provides diagnostics.
    
    Args:
        suggested_units_structured: AI-generated chapters with ClientUnitNo mapping
        client_units: Original Units from client (can be None/empty)
        
    Returns:
        Tuple of:
        - enriched_units: Units with timestamps and metadata
        - diagnostics: Validation results and statistics
        
    Example:
        Input client_units:
        [
            {"UnitNo": 1, "Title": "廚具規劃"},
            {"UnitNo": 2, "Title": "天花板大樣圖"},
            {"UnitNo": 3, "Title": "冷氣配置"}
        ]
        
        Input suggested_units_structured:
        [
            {UnitNo: 1, Title: "廚房三角原理", Time: "00:05:10", ClientUnitNo: 1},
            {UnitNo: 2, Title: "廚具尺寸標準", Time: "00:18:30", ClientUnitNo: 1},
            {UnitNo: 3, Title: "大樣圖規範", Time: "00:32:15", ClientUnitNo: 2},
            ...
        ]
        
        Output enriched_units:
        [
            {
                "UnitNo": 1,
                "Title": "廚具規劃",
                "Time": "00:05:10",  # First SuggestedUnit with ClientUnitNo=1
                "EndTime": "00:32:15",
                "Duration": "00:27:05",
                "SuggestedUnitCount": 2,
                "FirstChapter": "廚房工作三角原理與動線設計",
                "LastChapter": "廚具尺寸標準與人體工學考量"
            },
            ...
        ]
    """
    
    # Handle case where no Units provided
    if not client_units:
        logger.info("ℹ️ No client Units provided - skipping Unit timestamp back-calculation")
        return [], {
            "units_provided": False,
            "validation_passed": True,
            "message": "No client Units to process"
        }
    
    if not suggested_units_structured:
        logger.warning("⚠️ No SuggestedUnits generated - cannot back-calculate Unit timestamps")
        return client_units, {
            "units_provided": True,
            "validation_passed": False,
            "error": "No SuggestedUnits available for mapping"
        }
    
    # Build mapping: ClientUnitNo -> list of SuggestedUnits
    unit_chapters_map: Dict[int, List[Dict[str, Any]]] = {}
    unmapped_chapters: List[Dict[str, Any]] = []
    
    for su in suggested_units_structured:
        client_unit_no = su.get("ClientUnitNo")
        
        if client_unit_no is None:
            unmapped_chapters.append(su)
            continue
        
        if client_unit_no not in unit_chapters_map:
            unit_chapters_map[client_unit_no] = []
        
        unit_chapters_map[client_unit_no].append(su)
    
    # Sort chapters within each Unit by time
    for unit_no in unit_chapters_map:
        unit_chapters_map[unit_no].sort(
            key=lambda x: ts_to_seconds_hms(x["Time"])
        )
    
    # Enrich client units with timestamps and metadata
    enriched_units = []
    validation_issues = []
    
    for i, unit in enumerate(client_units):
        unit_no = unit.get("UnitNo")
        enriched_unit = dict(unit)  # Copy original
        
        if unit_no not in unit_chapters_map:
            # Unit not found in video
            logger.warning(
                f"⚠️ Unit {unit_no} ('{unit.get('Title')}') has NO mapped chapters in video!"
            )
            enriched_unit.update({
                "Time": "",
                "EndTime": None,
                "Duration": None,
                "SuggestedUnitCount": 0,
                "FirstChapter": None,
                "LastChapter": None
            })
            validation_issues.append({
                "unit_no": unit_no,
                "issue": "not_found",
                "message": f"Unit '{unit.get('Title')}' not found in video"
            })
            enriched_units.append(enriched_unit)
            continue
        
        # Get chapters for this Unit
        chapters = unit_chapters_map[unit_no]
        first_chapter = chapters[0]
        last_chapter = chapters[-1]
        
        # Calculate start time (first chapter of this Unit)
        start_time = first_chapter["Time"]
        start_sec = ts_to_seconds_hms(start_time)
        
        # Calculate end time (first chapter of NEXT Unit, or None if last)
        end_time = None
        end_sec = None
        duration_sec = None
        
        if i + 1 < len(client_units):
            next_unit_no = client_units[i + 1].get("UnitNo")
            if next_unit_no in unit_chapters_map:
                next_chapters = unit_chapters_map[next_unit_no]
                end_time = next_chapters[0]["Time"]
                end_sec = ts_to_seconds_hms(end_time)
                duration_sec = end_sec - start_sec
        
        enriched_unit.update({
            "Time": start_time,
            "EndTime": end_time,
            "Duration": sec_to_hms(duration_sec) if duration_sec else None,
            "SuggestedUnitCount": len(chapters),
            "FirstChapter": first_chapter["Title"],
            "LastChapter": last_chapter["Title"]
        })
        
        enriched_units.append(enriched_unit)
        
        logger.info(
            f"✅ Unit {unit_no}: {unit['Title']}\n"
            f"   → Starts at: {start_time}\n"
            f"   → Contains: {len(chapters)} chapters\n"
            f"   → First: {first_chapter['Title']}\n"
            f"   → Last: {last_chapter['Title']}"
        )
    
    # Validate logical order
    timestamps_valid = True
    for i in range(len(enriched_units) - 1):
        current = enriched_units[i]
        next_unit = enriched_units[i + 1]
        
        if not current.get("Time") or not next_unit.get("Time"):
            continue
        
        current_sec = ts_to_seconds_hms(current["Time"])
        next_sec = ts_to_seconds_hms(next_unit["Time"])
        
        if current_sec >= next_sec:
            timestamps_valid = False
            validation_issues.append({
                "unit_no": current["UnitNo"],
                "issue": "order_violation",
                "message": f"Unit {current['UnitNo']} ({current['Time']}) should come before Unit {next_unit['UnitNo']} ({next_unit['Time']})"
            })
            logger.error(
                f"❌ ORDER VIOLATION: Unit {current['UnitNo']} ({current['Time']}) >= "
                f"Unit {next_unit['UnitNo']} ({next_unit['Time']})"
            )
    
    # Build diagnostics
    diagnostics = {
        "units_provided": True,
        "total_units": len(client_units),
        "units_found": sum(1 for u in enriched_units if u.get("Time")),
        "units_missing": sum(1 for u in enriched_units if not u.get("Time")),
        "total_suggested_units": len(suggested_units_structured),
        "mapped_suggested_units": sum(len(chapters) for chapters in unit_chapters_map.values()),
        "unmapped_suggested_units": len(unmapped_chapters),
        "timestamps_valid": timestamps_valid,
        "validation_issues": validation_issues
    }
    
    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("📍 UNIT TIMESTAMP BACK-CALCULATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Units found: {diagnostics['units_found']}/{diagnostics['total_units']}")
    logger.info(f"⚠️ Units missing: {diagnostics['units_missing']}/{diagnostics['total_units']}")
    logger.info(f"📊 SuggestedUnits mapped: {diagnostics['mapped_suggested_units']}/{diagnostics['total_suggested_units']}")
    logger.info(f"📊 SuggestedUnits unmapped: {diagnostics['unmapped_suggested_units']}")
    logger.info(f"{'✅' if timestamps_valid else '❌'} Timestamp order: {'Valid' if timestamps_valid else 'INVALID'}")
    
    if validation_issues:
        logger.warning(f"\n⚠️ {len(validation_issues)} validation issues found:")
        for issue in validation_issues:
            logger.warning(f"   - {issue['message']}")
    
    logger.info("=" * 60 + "\n")
    
    return enriched_units, diagnostics

def suggested_units_to_chapters_dict(
    suggested_units: List[Dict[str, Any]],
    *,
    duration_sec: Optional[int] = None,
    bump_limit_sec: int = 120,  # allow a bit more room than 59s
) -> Dict[str, str]:
    """
    Convert SuggestedUnits -> chapters dict, preventing timestamp key collisions
    WITHOUT producing non-HH:MM:SS keys (so later validation won't drop them).

    Strategy:
    - Prefer the original HH:MM:SS.
    - If already used, bump forward by +1..+bump_limit_sec.
    - If still used, bump backward by -1..-bump_limit_sec.
    - If still impossible, drop the duplicate with a warning.
    """

    def bump_ts(ts: str, delta: int) -> Optional[str]:
        base = ts_to_seconds_hms(ts)
        if base < 0:
            return None
        bumped = base + delta
        if duration_sec is not None:
            if bumped < 0 or bumped > duration_sec - 1:
                return None  # <-- don't clamp; reject
        return sec_to_hms(bumped)

    chapters: Dict[str, str] = {}
    used = set()

    for su in suggested_units:
        ts = str(su.get("Time") or "").strip()
        title = str(su.get("Title") or "").strip()
        cu = su.get("ClientUnitNo")
        cut = su.get("ClientUnitTitle")
        prefix = f"[單元{cu}：{cut}] " if cu and cut else (f"[單元{cu}] " if cu else "")

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
                continue  # drop rather than create invalid key

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
    3. Timestamps are distributed across the video (not just first few minutes)
    4. Suspicious clustering is detected and logged
    
    Returns: Cleaned and validated chapters dict
    """
    if not chapters:
        return {}

    def ts_to_seconds(ts: str) -> int:
        """Convert HH:MM:SS to seconds. Return -1 if invalid."""
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
        ts_normalized = _normalize_ts(ts)  # Ensure HH:MM:SS format
        ts_sec = ts_to_seconds(ts_normalized)
        
        # Check 1: Within video duration
        if ts_sec > duration_sec:
            logger.warning(f"⚠️ [{video_id}] Skipping chapter at {ts_normalized} ({ts_sec}s) - exceeds duration ({duration_sec}s)")
            continue
        
        # Check 2: Not negative
        if ts_sec < 0:
            logger.warning(f"⚠️ [{video_id}] Skipping chapter at {ts_normalized} - invalid timestamp")
            continue
        
        validated[ts_normalized] = title
        timestamps_in_seconds.append(ts_sec)
    
    if not validated:
        logger.error(f"❌ [{video_id}] No valid chapters after timestamp validation!")
        return {}
    
    # Check 3: Detect suspicious clustering (all chapters in first 10% of video)
    first_chapter_sec = min(timestamps_in_seconds)
    last_chapter_sec = max(timestamps_in_seconds)
    chapter_span_sec = last_chapter_sec - first_chapter_sec
    chapter_span_percent = (chapter_span_sec / duration_sec) * 100 if duration_sec > 0 else 0
    
    logger.info(f"📊 [{video_id}] Chapter span: {sec_to_hms(int(chapter_span_sec))} ({chapter_span_percent:.1f}% of video)")
    
    # CRITICAL CHECK: If all chapters are in first 10% of video, something is wrong
    if chapter_span_percent < 10 and duration_sec > 600:  # Only check for videos > 10 min
        logger.warning("=" * 70)
        logger.warning(f"⚠️  [{video_id}] SUSPICIOUS CHAPTER CLUSTERING DETECTED!")
        logger.warning(f"⚠️  All {len(validated)} chapters are in first {chapter_span_percent:.1f}% of video")
        logger.warning(f"⚠️  Video duration: {sec_to_hms(int(duration_sec))} ({duration_sec}s)")
        logger.warning(f"⚠️  Chapter range: {sec_to_hms(int(first_chapter_sec))} to {sec_to_hms(int(last_chapter_sec))}")
        logger.warning(f"⚠️  This likely indicates LLM timestamp format confusion")
        logger.warning("=" * 70)
        
        # Log first few chapters for debugging
        logger.warning(f"⚠️  Chapters generated:")
        for ts, title in sorted(validated.items(), key=lambda x: ts_to_seconds(x[0]))[:5]:
            logger.warning(f"     {ts} - {title[:60]}")
        if len(validated) > 5:
            logger.warning(f"     ... and {len(validated) - 5} more")
    
    # Check 4: Detect if too many chapters are within first minute
    chapters_in_first_minute = sum(1 for ts_sec in timestamps_in_seconds if ts_sec < 60)
    if chapters_in_first_minute > len(timestamps_in_seconds) * 0.5:
        logger.warning(f"⚠️ [{video_id}] {chapters_in_first_minute}/{len(timestamps_in_seconds)} chapters in first 60 seconds - possible timestamp error")
    
    # Summary
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
        """Extract [module] tag if present"""
        match = re.match(r'\[([^\]]+)\]', title)
        return match.group(1) if match else ""

    def ts_to_s(ts: str) -> int:
        """
        Convert a timestamp string to seconds.
        Accepts: "MM:SS" or "HH:MM:SS"
        Returns: seconds, or -1 if invalid
        """
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

    # Content-aware deduplication
    dedup = []
    # ✅ UNIVERSAL FIX: Scale merge threshold with video duration
    # This prevents over-aggressive merging for long videos

    adaptive_min_gap = max(120, min_gap_sec // 2)
    
    logger.info(f"📊 Chapter balancing: {len(chapters)} input chapters, merge threshold = {adaptive_min_gap}s (policy min_gap: {min_gap_sec}s)")
    
    for s, ts, title in cands:
        if not dedup:
            dedup.append((s, ts, title))
            continue
            
        time_gap = s - dedup[-1][0]
        
        # Merge if same module and close, or very close regardless
        should_merge = False
        prev_module = extract_module_tag(dedup[-1][2])
        curr_module = extract_module_tag(title)
        
        # ✅ FIXED: Use adaptive threshold instead of hardcoded 120s
        if time_gap < adaptive_min_gap:
            should_merge = True
    
        elif prev_module and curr_module and prev_module == curr_module:
            # Same module tag - use min_gap_sec
            should_merge = time_gap < min_gap_sec * 0.7
        elif time_gap < min_gap_sec // 2:  # Half the min_gap for different modules
            should_merge = True
            
        if should_merge:
            # Keep the better title (prefer tagged, then longer)
            prev_s, prev_ts, prev_title = dedup[-1]
            if curr_module and not prev_module:
                dedup[-1] = (prev_s, prev_ts, title)
            elif len(title) > len(prev_title) * 1.3:
                dedup[-1] = (prev_s, prev_ts, title)
        else:
            dedup.append((s, ts, title))

    t_low, t_high = target_range
    
    # Log the balancing result
    logger.info(f"Chapter balancing: {len(chapters)} → {len(dedup)} (target: {t_low}-{t_high})")
    
    if t_low <= len(dedup) <= t_high:
        return {ts: title for _, ts, title in dedup}

    # Too many chapters: choose representatives from each segment
    if len(dedup) > t_high:
        selected = []
        segment_length = max(1, duration_sec // t_high)
        for i in range(t_high):
            segment_start = i * segment_length
            segment_end = (i + 1) * segment_length if i < t_high - 1 else duration_sec + 1
            segment_chapters = [c for c in dedup if segment_start <= c[0] < segment_end]
            if segment_chapters:
                # Pick the one closest to segment center
                segment_center = (segment_start + segment_end) // 2
                chosen = min(segment_chapters, key=lambda c: abs(c[0] - segment_center))
                selected.append(chosen)
        selected.sort(key=lambda x: x[0])
        return {ts: title for _, ts, title in selected}

    # Not enough chapters: just cap to max_caps
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
    Returns: List[Dict] with keys: start (int), end (int, optional), text (str)
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

    # Try JSONL
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

    # Plain text fallback
    try:
        file_obj.seek(0)
        txt = file_obj.read().strip()
    except Exception:
        txt = ""
    return [{"start": 0, "end": 0, "text": txt}] if txt else []

def build_ocr_context_from_segments(ocr_segments: List[Dict]) -> str:
    """Legacy minimal OCR formatting: timestamped lines with a simple header."""
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
    """Convert a string to Traditional Chinese. Uses OpenCC if available; otherwise minimal mapping."""
    if not text:
        return text
    if _opencc is not None:
        try:
            return _opencc.convert(text)
        except Exception:
            pass
    return ''.join(_S2T_FALLBACK_MAP.get(ch, ch) for ch in text)

def ensure_traditional_chapters(chapters: Dict[str, str]) -> Dict[str, str]:
    """Convert all chapter titles to Traditional Chinese (idempotent if already Traditional)."""
    return {ts: to_traditional(title) for ts, title in chapters.items()}

# ─────────────────────────
# Client Initialization & LLM call
# ─────────────────────────
def initialize_client(service_type: str, **kwargs) -> Any:
    """Initialize the appropriate LLM client"""
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
    """Call LLM API with retry logic"""
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
# Prompt builder (ASR first, OCR second, OCR verbatim supported)
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
    
    # Extract first/last REAL ASR timestamps using the same matcher as the rest of the pipeline.
    # This supports:
    #   00:00:12: text
    #   00:00:12 - text
    #   [00:00:12] text
    #   00:00:12 text
    timestamps: List[str] = []
    for line in transcript.splitlines():
        m = ASR_TS_RE.match(line)
        if m:
            timestamps.append(_normalize_ts(m.group(1)))
    first_ts = first_ts_override or (timestamps[0] if timestamps else "00:00:00")
    last_ts  = last_ts_override  or (timestamps[-1] if timestamps else duration_hms)


    video_title_context = ""
    if video_title:
        # Strip common video extensions
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
    """
    Build educational context from metadata for prompt enhancement.
    
    Returns formatted string with course structure information.
    """
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


# ─────────────────────────
# Hierarchical Multi-Pass Generation (NEW)
# ─────────────────────────

def should_use_hierarchical(duration: float, transcript_length: int) -> bool:
    """Determine if hierarchical multi-pass should be used"""
    # Use hierarchical for longer, content-rich educational videos
    return (duration >= 1800 and  # 30+ minutes
            transcript_length >= 5000 and  # Substantial content
            duration <= 14400)  # Under 4 hours (very long videos might need different handling)

def hierarchical_multipass_generation(
    raw_asr_text: str,
    duration: float,
    ocr_context: str,
    video_title: Optional[str],
    section_title: Optional[str],      # ← ADD
    units: Optional[List[Dict]],       # ← ADD
    client: Any,
    config: ChapterConfig,
    progress_callback: Optional[Callable[[str, int], None]] = None
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    """
    Three-pass hierarchical generation for high-quality educational chapters.
    
    Strategy:
    - ASR (Primary): Provides teaching content, explanations, natural timing, and narrative flow
    - OCR (Supporting): Provides visual structure, precise terminology, and organized summaries
    
    Prioritization:
    - Chapter timing: ASR timestamps (when instructor announces topics)
    - Chapter titles: ASR content enriched with OCR terminology
    - Q&A content: ASR explanations supplemented with OCR structured data

    
    Token Budget:
    - ASR: 100,000 tokens per pass (primary source)
    - OCR: 15,000 tokens per pass (supporting detail)
    - Total: ~115,000 content + ~2,000 instructions = ~117,000 tokens (safe for GPT-4o's 128k context)
    
    Returns: (raw_llm_text, chapters, metadata)
    """
    
    # ==================== Token Budget Initialization ====================
    ASR_LIMIT = 100_000   # ASR transcript limit per pass
    OCR_LIMIT = 15_000    # OCR context limit per pass
    
    asr_tokens = count_tokens_llama(raw_asr_text)
    ocr_tokens = count_tokens_llama(ocr_context) if ocr_context else 0
    total_content_tokens = asr_tokens + ocr_tokens
    
    logger.info("=" * 60)
    logger.info("🎓 HIERARCHICAL MULTI-PASS CHAPTER GENERATION")
    logger.info("   Strategy: ASR-primary (timing) + OCR-supporting (detail)")
    logger.info("=" * 60)
    logger.info(f"📊 Original ASR tokens: {asr_tokens:,} (limit: {ASR_LIMIT:,})")
    logger.info(f"📊 Original OCR tokens: {ocr_tokens:,} (limit: {OCR_LIMIT:,})")
    logger.info(f"📊 Total content tokens: {total_content_tokens:,}")
    logger.info(f"📊 Video duration: {sec_to_hms(int(duration))}")
    
    # Truncate once, reuse in all passes for consistency
    asr_text = truncate_text_by_tokens(raw_asr_text, ASR_LIMIT)
    ocr_text = truncate_text_by_tokens(ocr_context, OCR_LIMIT) if ocr_context else ""

    asr_ts_sorted = extract_asr_timestamps_sorted(raw_asr_text)  # use RAW, not truncated
    asr_end_ts = asr_ts_sorted[-1] if asr_ts_sorted else sec_to_hms(int(duration))
    asr_end_sec = ts_to_seconds_hms(asr_end_ts)
    if asr_end_sec <= 0:
        asr_end_sec = int(duration)
        asr_end_ts = sec_to_hms(asr_end_sec)
    anchors = pick_anchor_timestamps(asr_ts_sorted, k=14)

    logger.info(f"🧾 ASR time coverage: first={asr_ts_sorted[0] if asr_ts_sorted else 'N/A'} "
                f"last={asr_end_ts} (asr_ts_count={len(asr_ts_sorted)})")
    logger.info(f"🧷 Anchor timestamps (spread): {anchors}")

    
    asr_used = count_tokens_llama(asr_text)
    ocr_used = count_tokens_llama(ocr_text)
    content_used = asr_used + ocr_used
    
    asr_coverage = (asr_used / asr_tokens * 100) if asr_tokens > 0 else 100
    ocr_coverage = (ocr_used / ocr_tokens * 100) if ocr_tokens > 0 else 100
    
    logger.info(f"✅ Using per pass:")
    logger.info(f"   • ASR: {asr_used:,} tokens ({asr_coverage:.1f}% of original)")
    logger.info(f"   • OCR: {ocr_used:,} tokens ({ocr_coverage:.1f}% of original)")
    logger.info(f"   • Total: {content_used:,} tokens")
    
    if asr_tokens > ASR_LIMIT:
        logger.warning(f"⚠️ ASR truncated from {asr_tokens:,} to {asr_used:,} tokens")
    if ocr_tokens > OCR_LIMIT:
        logger.warning(f"⚠️ OCR truncated from {ocr_tokens:,} to {ocr_used:,} tokens")
    
    # ==================== PASS 1: Course Structure Analysis ====================
    logger.info("\n" + "-" * 60)
    logger.info("🔍 PASS 1: Course Structure Analysis")
    logger.info("   Goal: Identify learning objectives and overall architecture")
    logger.info("   Approach: Analyze both ASR and OCR equally")
    logger.info("-" * 60)
    
    if progress_callback:
        progress_callback("analyzing_course_structure", 40)
    
    video_info = ""
    if video_title:
        clean_title = re.sub(r'\.(mp4|avi|mov|mkv|webm|flv|m4v)$', '', video_title, flags=re.IGNORECASE)
        video_info = f"課程檔名：{clean_title}\n"
        logger.info(f"📚 Video title: {clean_title}")
    
    # ← ADD THIS LOGGING BLOCK
    if section_title or units:
        logger.info("=" * 60)
        logger.info("📚 EDUCATIONAL METADATA PROVIDED")
        if section_title:
            logger.info(f"   📖 Section: {section_title}")
        if units:
            logger.info(f"   📑 Units: {len(units)} predefined learning units")
            for unit in units:
                logger.info(f"      {unit['UnitNo']}. {unit['Title']}")
        logger.info("=" * 60)
        
    structure_prompt = f"""
作為資深教學設計專家，分析這個{sec_to_hms(int(duration))}教學影片的整體架構：

{video_info}

【核心學習目標】
1. 學生完成本課程後應掌握哪些關鍵能力？
2. 有哪些必須理解的核心理論或概念？
3. 有哪些需要熟練的實用技能？

【知識架構分析】
- 基礎鋪陳：哪些是前提知識或基礎概念？
- 核心教學：最重要的理論/方法/技術是什麼？
- 應用延伸：如何將所學應用於實際場景？
- 總結整合：如何將零散知識系統化？

【教學方法識別】
- 理論講解 vs. 實例演示 vs. 操作練習 的比例分佈
- 是否有問答互動、思考題、重點回顧？

【分析要點】
- 從講師的教學敘述（ASR）理解教學邏輯和重點
- 從投影片內容（OCR）識別主要章節結構和專業術語
- 綜合兩者，建構完整的課程框架

完整逐字稿（講師教學內容與時間軸）：
{asr_text}

視覺輔助內容（投影片/螢幕文字，用於確認主題與術語）：
{ocr_text if ocr_text else "無視覺輔助內容"}
"""
    
    logger.info(f"📤 PASS 1 prompt: ~{count_tokens_llama(structure_prompt):,} tokens")
    logger.info("🤖 Calling LLM for structure analysis...")
    t0 = time.time()
    
    try:
        structure_response = call_llm(
            service_type=config.service_type,
            client=client,
            system_message="你是課程架構分析專家，擅長識別教學影片的整體學習目標和知識體系。你會綜合分析講師講解（ASR）和投影片內容（OCR）來理解課程的完整結構。",
            user_message=structure_prompt,
            model=config.openai_model if config.service_type == "openai" else config.azure_model,
            max_tokens=1200,
            temperature=0.3
        )
        
        elapsed = time.time() - t0
        logger.info(f"✅ PASS 1 completed in {elapsed:.1f}s")
        
        structure_text = (structure_response.choices[0].message.content 
                         if config.service_type == "openai" 
                         else structure_response.choices[0].message.content)
        
        logger.info(f"📝 Structure analysis: {len(structure_text)} characters")
        
    except Exception as e:
        logger.error(f"❌ PASS 1 failed: {e}", exc_info=True)
        raise
    
    # ==================== PASS 2: Learning Modules Identification ====================
    logger.info("\n" + "-" * 60)
    logger.info("📚 PASS 2: Learning Modules Identification")
    logger.info("   Goal: Break down course into 7-12 coherent learning units")
    logger.info("   Approach: ASR-primary (conceptual transitions)")
    logger.info("-" * 60)
    
    if progress_callback:
        progress_callback("identifying_learning_modules", 60)
    
    modules_prompt = f"""
基於課程結構分析：
{structure_text}

現在識別具體的學習模塊（7-12個），每個模塊應滿足：
1. 有明確的學習目標
2. 包含完整的教學閉環（講解→範例→練習）
3. 時長合理（10-30分鐘）
4. 有清晰的開始和結束標記

【模塊邊界識別策略（重要性排序）】

**第一優先：講師的重大主題轉換（ASR - 主要依據）**
- 明確的章節宣告："接下來進入新的章節/部分"、"第一部分/第二部分"
- 重大內容轉換："基礎/理論講完了，現在來看..."、"從概念到實踐"
- 教學方法的重大轉變：理論講解 → 實際操作 → 案例分析
- 講師的總結與過渡："我們剛才講了...，現在來看..."

**第二優先：教學邏輯的結構轉換（ASR內容分析）**
- 從簡單到複雜的明顯層級變化
- 從單一工具/概念到綜合應用
- 從講解到練習的轉換
- 階段性總結後開始新主題

**第三優先：視覺結構變化（OCR - 輔助參考）**
- 投影片的大標題變化（章節編號、大段落標記）
- 顯著的內容類型切換（理論投影片 → 軟體操作界面 → 實例演示）
- 用於確認模塊的主題名稱和專業術語

⚠️ 重點提醒：
- 模塊是大的學習單元，不要被頻繁的小標題變化誤導
- 單個投影片變化不等於模塊邊界
- 優先關注講師的語言信號，投影片用於確認主題

完整逐字稿（主要依據 - 包含講師的主題轉換信號）：
{asr_text}

視覺輔助內容（次要參考 - 用於確認主題名稱）：
{ocr_text if ocr_text else "無視覺輔助內容"}

請輸出格式：
模塊名稱 ~ 起始時間戳(HH:MM:SS) ~ 結束時間戳(HH:MM:SS) ~ 核心學習點 ~ 教學方法
起始時間戳必須是逐字稿中出現過的時間戳之一

範例（注意：時間戳必須是 HH:MM:SS）：
基礎工具操作 ~ 00:05:24 ~ 00:18:45 ~ 介面認識、基本工具使用 ~ 理論講解+實例演示
進階設計技巧 ~ 00:18:45 ~ 00:36:10 ~ 色彩管理、輸出設定 ~ 綜合案例+實際操作
"""
    
    logger.info(f"📤 PASS 2 prompt: ~{count_tokens_llama(modules_prompt):,} tokens")
    logger.info("🤖 Calling LLM for module identification...")
    t0 = time.time()
    
    try:
        modules_response = call_llm(
            service_type=config.service_type,
            client=client,
            system_message="你是課程模塊設計師，擅長將教學內容分解為邏輯連貫的學習單元。你主要依據講師的語言信號（ASR）來識別模塊邊界，因為模塊是基於概念轉換而非視覺變化。投影片（OCR）主要用於確認模塊的主題名稱。",
            user_message=modules_prompt,
            model=config.openai_model if config.service_type == "openai" else config.azure_model,
            max_tokens=1500,
            temperature=0.2
        )
        
        elapsed = time.time() - t0
        logger.info(f"✅ PASS 2 completed in {elapsed:.1f}s")
        
        modules_text = (modules_response.choices[0].message.content 
                       if config.service_type == "openai" 
                       else modules_response.choices[0].message.content)
        
        logger.info(f"📝 Modules analysis: {len(modules_text)} characters")
        
    except Exception as e:
        logger.error(f"❌ PASS 2 failed: {e}", exc_info=True)
        raise

    # ==================== VALIDATE CLIENT UNITS (AFTER PASS 2, BEFORE PASS 3) ====================
    unit_validation_result = None
    validated_units = units  # Default: assume valid
    
    if units and isinstance(units, list) and len(units) > 0:
        logger.info("\n" + "=" * 60)
        logger.info("🔍 VALIDATING CLIENT UNITS (POST-PASS2, PRE-PASS3)")
        logger.info("=" * 60)
        
        try:
            from app.qa_generation import validate_units_relevance, UNIT_VALIDATION_THRESHOLD
            
            bare_units = [
                {"UnitNo": u.get("UnitNo"), "Title": u.get("Title")}
                for u in units
            ]
            
            logger.info(f"Validating {len(bare_units)} client units against video content...")
            
            # Build content dict with the SAME KEYS that validate_units_relevance reads
            # Also extract topic names from modules_text for main_topics list
            _module_topics = []
            for _line in (modules_text or "").splitlines():
                _line_s = _line.strip()
                if "~" in _line_s:
                    _parts = _line_s.split("~")
                    _topic = _parts[0].strip()
                    _topic = re.sub(r'^模塊\d+[：:]\s*', '', _topic)
                    _topic = re.sub(r'^\d+\.\s*', '', _topic)
                    if _topic and len(_topic) >= 2:
                        _module_topics.append(_topic)
                elif "@" in _line_s:
                    _parts = _line_s.split("@")
                    _topic = re.sub(r'^\d+\.\s*', '', _parts[0].strip())
                    if _topic and len(_topic) >= 2:
                        _module_topics.append(_topic)
            
            content_from_passes = {
                "main_topics": _module_topics[:8],
                "key_concepts": [],
                "technical_terms": [],
            }
            
            is_valid, score, reason = validate_units_relevance(
                units=bare_units,
                content_analysis=content_from_passes,
                chapters={},
                video_title=video_title or "",
                threshold=UNIT_VALIDATION_THRESHOLD,
                structure_analysis=structure_text,
                modules_analysis=modules_text,
                section_title=section_title,
            )
            
            unit_validation_result = {
                "is_valid": is_valid,
                "score": score,
                "reason": reason,
                "threshold": UNIT_VALIDATION_THRESHOLD,
                "units_provided": len(units)
            }
            
            if is_valid:
                logger.info(f"✅ CLIENT UNITS VALIDATED AND ACCEPTED")
                logger.info(f"   • Score: {score:.2f} (threshold: {UNIT_VALIDATION_THRESHOLD})")
                logger.info(f"   • Reason: {reason}")
                logger.info(f"   • Will include units in PASS 3 prompt")
                validated_units = units
            else:
                logger.warning(f"❌ CLIENT UNITS REJECTED")
                logger.warning(f"   • Score: {score:.2f} (threshold: {UNIT_VALIDATION_THRESHOLD})")
                logger.warning(f"   • Reason: {reason}")
                logger.warning(f"   • PASS 3 will run WITHOUT client unit context")
                logger.warning(f"   • Will return AI-generated chapters only")
                validated_units = None
            
        except Exception as e:
            logger.error(f"Unit validation error: {e}", exc_info=True)
            logger.warning("Proceeding without validation (fail-open)")
            unit_validation_result = {
                "is_valid": True,
                "score": 1.0,
                "reason": f"Validation error: {str(e)[:50]}",
                "threshold": UNIT_VALIDATION_THRESHOLD,
                "units_provided": len(units),
                "error": str(e)
            }
            validated_units = units
        
        logger.info("=" * 60 + "\n")
    
    # ==================== BUILD EDUCATIONAL CONTEXT (CONDITIONAL) ====================
    # Only include client units in PASS 3 if they passed validation
    if validated_units:
        educational_context = build_educational_context(section_title, validated_units)
        logger.info(f"✅ Educational context included in PASS 3 ({len(validated_units)} validated units)")
    else:
        educational_context = ""
        if units:
            logger.info("ℹ️  Educational context EXCLUDED from PASS 3 (units rejected)")
        else:
            logger.info("ℹ️  No educational context (no units provided)")
    # ==================== PASS 3: Detailed Chapter Generation ====================
    logger.info("\n" + "-" * 60)
    logger.info("📑 PASS 3: Detailed Chapter Generation")
    logger.info("   Goal: Create precise chapter timestamps with titles")
    logger.info("   Approach: Chapters-first, then semantic unit tagging")
    logger.info("-" * 60)
    
    if progress_callback:
        progress_callback("generating_detailed_chapters", 80)
    
    # ────────── STEP 1: Generate chapters freely (no unit constraints) ──────────
    # Let the LLM find natural chapter boundaries from ASR content.
    # Unit tagging happens AFTER chapter generation (STEP 2).
    
    # Build informational unit context (hints only, not constraining)
    unit_context_hint = ""
    if validated_units and len(validated_units) >= 1:
        unit_names = ", ".join(f"{u['UnitNo']}.{u['Title']}" for u in validated_units)
        unit_context_hint = f"""
【參考：客戶預定教學單元】
本課程包含以下教學單元主題：{unit_names}
這些是課程的主要主題，但影片中可能還包含課程介紹、行政事務、休息等非教學內容。
請專注於從逐字稿中找到自然的內容轉換點，不需要將所有章節強制歸入以上單元。
"""
        logger.info(f"📋 Client provided {len(validated_units)} units (used as hints, not constraints)")
    
    chapters_prompt = f"""
【課程整體結構】
{structure_text}

【學習模塊規劃】  
{modules_text}

{educational_context}

{unit_context_hint}

現在為影片生成具體的章節時間點（總計10-20個章節），並提供課程摘要。

【章節設計原則】
1. 每個章節代表一個完整的學習子目標（5-10分鐘）
2. 標記關鍵概念的首次詳細解釋
3. 標記重要範例或案例分析的開始
4. 標記練習題或互動環節
5. 標記重點回顧或總結處
6. 如果影片開頭有課程介紹、行政事務、設備設定等非教學內容，也應該為其建立章節
7. 如果影片中有明顯的休息/中場，可以標記為「休息」或「課間休息」
8. 如果有不同講者（如行政人員）出現，也應建立獨立章節

【章節時間點定位策略（重要性排序）】

**第一優先：ASR語言時間戳（主要依據 - 決定章節開始時間）**
- 講師的明確主題宣告
- 教學轉換信號
- 重要概念的首次詳細解釋開始點

**第二優先：內容邏輯轉換（ASR內容分析）**
- 從理論講解到實際演示的自然轉換點
- 新工具/技術的首次詳細介紹

**第三優先：OCR視覺輔助（補充確認）**
- 確認當前討論的具體主題
- 提供精確的技術術語

完整逐字稿（含精確時間戳）：
{asr_text}

視覺輔助內容：
{ocr_text if ocr_text else "無視覺輔助內容"}

總時長：{sec_to_hms(int(duration))}

【輸出格式要求（重要：只輸出 JSON，不要輸出任何其他文字）】
請輸出一個 JSON 物件，格式如下：

{PASS3_JSON_SCHEMA}

{{ 
  "SuggestedUnits": [
    {{ 
      "UnitNo": 1,
      "ParentUnitNo": null,
      "Title": "章節標題（繁體中文）",
      "Time": "HH:MM:SS",
      "ClientUnitNo": null,
      "ClientUnitTitle": null
    }}
  ],
  "CourseSummary": {{
    "topic": "...",
    "core_content": "...",
    "learning_objectives": "...",
    "target_audience": "...",
    "difficulty": "..."
  }}
}}

規則：
1) Time 必須是逐字稿中存在或非常接近（±60 秒內）的 HH:MM:SS
2) SuggestedUnits 需依 Time 遞增排序
3) ClientUnitNo 和 ClientUnitTitle 先設為 null（稍後會自動標記）
4) 只輸出 JSON，禁止多餘解釋
5) 章節必須覆蓋影片的大部分內容（從開頭到接近結尾）
"""
    
    logger.info(f"📤 PASS 3 (chapters-first) prompt: ~{count_tokens_llama(chapters_prompt):,} tokens")
    logger.info("🤖 Calling LLM for chapter generation (no unit constraints)...")
    t0 = time.time()
    
    try:
        final_response = call_llm(
            service_type=config.service_type,
            client=client,
            system_message=(
                "你是細心的章節設計師。"
                "章節時間以 ASR 時間戳為準，章節標題可用 OCR 補充專業術語。"
                "請只輸出一個 JSON 物件，且 JSON 必須包含 SuggestedUnits 與 CourseSummary。"
                "禁止輸出任何其他文字、禁止 ```。"
            ),
            user_message=chapters_prompt,
            model=config.openai_model if config.service_type == "openai" else config.azure_model,
            max_tokens=3000,
            temperature=0.1
        )
        
        elapsed = time.time() - t0
        logger.info(f"✅ PASS 3 (chapters-first) completed in {elapsed:.1f}s")
        
        final_text = (final_response.choices[0].message.content 
                     if config.service_type == "openai" 
                     else final_response.choices[0].message.content)
        
        logger.info(f"📝 Final output: {len(final_text)} characters")
        
    except Exception as e:
        logger.error(f"❌ PASS 3 failed: {e}", exc_info=True)
        raise
    
    # ────────── Parse chapter results ──────────
    data = safe_load_json(final_text)

    suggested_units_structured: List[Dict[str, Any]] = []
    course_summary: Dict[str, Any] = {}

    if isinstance(data, dict):
        suggested_units_structured = normalize_suggested_units(
            data.get("SuggestedUnits"),
            units=None  # Don't pass units here - we'll tag them in STEP 2
        )
        suggested_units_structured = clean_all_suggested_units(suggested_units_structured)
        cs = data.get("CourseSummary")
        if isinstance(cs, dict):
            course_summary = cs
    elif isinstance(data, list):
        suggested_units_structured = normalize_suggested_units(data, units=None)
        suggested_units_structured = clean_all_suggested_units(suggested_units_structured)

    # Apply Traditional Chinese to course summary
    if _opencc and course_summary:
        for k, v in list(course_summary.items()):
            if isinstance(v, str):
                course_summary[k] = to_traditional(v)

    logger.info(f"📊 Generated {len(suggested_units_structured)} chapters (before unit tagging)")
    
    # ────────── STEP 2: Semantic unit tagging (if client provided validated units) ──────────
    if validated_units and len(validated_units) >= 1 and suggested_units_structured:
        logger.info(f"\n🏷️  STEP 2: Semantic unit tagging for {len(suggested_units_structured)} chapters → {len(validated_units)} units")
        
        # Build a concise chapter list for the tagging prompt
        chapter_lines = []
        for ch in suggested_units_structured:
            chapter_lines.append(f"  {ch['UnitNo']}. [{ch['Time']}] {ch['Title']}")
        chapters_list_text = "\n".join(chapter_lines)
        
        # Build the unit list
        unit_lines = []
        for u in validated_units:
            unit_lines.append(f"  {u['UnitNo']}. {u['Title']}")
        units_list_text = "\n".join(unit_lines)
        
        tagging_prompt = f"""你是教學內容分類專家。以下是一個教學影片的章節列表和客戶定義的教學單元。

【影片章節（已按時間排序）】
{chapters_list_text}

【客戶教學單元】
{units_list_text}

【任務】
為每個章節標記最合適的客戶教學單元編號。

規則：
1. 根據章節標題的「內容主題」判斷它屬於哪個教學單元
2. 如果章節是課程介紹、行政事務、休息、設備設定等「非教學內容」，標記為 0（表示不屬於任何單元）
3. 同一個教學單元的章節不需要連續出現（例如，休息前後可能都在講同一個單元的內容）
4. 一個教學單元可以有多個章節，也可以沒有任何章節（如果影片中沒有涵蓋該單元的內容）
5. 每個章節只能屬於一個教學單元（或標記為 0）

【輸出格式（只輸出 JSON）】
{{
  "tags": [
    {{"chapter": 1, "unit": 0}},
    {{"chapter": 2, "unit": 1}},
    {{"chapter": 3, "unit": 1}},
    {{"chapter": 4, "unit": 2}}
  ]
}}

其中 chapter 是章節編號，unit 是客戶教學單元編號（0 表示非教學內容）。
必須為每個章節都提供標記。只輸出 JSON。
"""
        
        logger.info(f"📤 Tagging prompt: ~{count_tokens_llama(tagging_prompt):,} tokens")
        logger.info("🤖 Calling LLM for semantic unit tagging...")
        t0_tag = time.time()
        
        try:
            tag_response = call_llm(
                service_type=config.service_type,
                client=client,
                system_message=(
                    "你是教學內容分類專家。根據章節的內容主題判斷它屬於哪個教學單元。"
                    "非教學內容（行政、休息、介紹等）標記為 0。"
                    "只輸出 JSON，禁止其他文字。"
                ),
                user_message=tagging_prompt,
                model=config.openai_model if config.service_type == "openai" else config.azure_model,
                max_tokens=1000,
                temperature=0.0
            )
            
            tag_elapsed = time.time() - t0_tag
            logger.info(f"✅ Unit tagging completed in {tag_elapsed:.1f}s")
            
            tag_text = (tag_response.choices[0].message.content
                       if config.service_type == "openai"
                       else tag_response.choices[0].message.content)
            
            tag_data = safe_load_json(tag_text)
            
            # Parse tagging results
            if isinstance(tag_data, dict) and "tags" in tag_data:
                tags_list = tag_data["tags"]
                
                # Build lookup: chapter_no -> unit_no
                valid_unit_nos = {int(u["UnitNo"]) for u in validated_units}
                valid_unit_titles = {int(u["UnitNo"]): str(u.get("Title", "")).strip() for u in validated_units}
                tag_map = {}
                for tag_entry in tags_list:
                    if isinstance(tag_entry, dict):
                        ch_no = tag_entry.get("chapter")
                        u_no = tag_entry.get("unit")
                        if ch_no is not None and u_no is not None:
                            try:
                                ch_no = int(ch_no)
                                u_no = int(u_no)
                            except (ValueError, TypeError):
                                continue
                            if u_no in valid_unit_nos:
                                tag_map[ch_no] = u_no
                            elif u_no == 0:
                                tag_map[ch_no] = None  # Non-teaching content
                
                # Apply tags to chapters
                tagged_count = 0
                untagged_count = 0
                for su in suggested_units_structured:
                    ch_no = su.get("UnitNo")
                    if ch_no in tag_map:
                        mapped_unit = tag_map[ch_no]
                        if mapped_unit is not None:
                            su["ClientUnitNo"] = mapped_unit
                            su["ClientUnitTitle"] = valid_unit_titles.get(mapped_unit, "")
                            tagged_count += 1
                        else:
                            su["ClientUnitNo"] = None
                            su["ClientUnitTitle"] = None
                            untagged_count += 1
                    else:
                        su["ClientUnitNo"] = None
                        su["ClientUnitTitle"] = None
                        untagged_count += 1
                
                # Log results
                logger.info(f"🏷️  Tagging results: {tagged_count} tagged, {untagged_count} untagged (intro/admin/break)")
                
                # Per-unit breakdown
                for vu in validated_units:
                    uno = int(vu["UnitNo"])
                    unit_chapters = [su for su in suggested_units_structured if su.get("ClientUnitNo") == uno]
                    if unit_chapters:
                        first_ts = unit_chapters[0]["Time"]
                        last_ts = unit_chapters[-1]["Time"]
                        logger.info(f"   Unit {uno} ({vu['Title']}): {len(unit_chapters)} chapters, {first_ts} → {last_ts}")
                    else:
                        logger.warning(f"   Unit {uno} ({vu['Title']}): 0 chapters (not found in video)")
                
                # Check for non-contiguous unit assignments (info only)
                unit_time_ranges = {}
                for su in suggested_units_structured:
                    cu = su.get("ClientUnitNo")
                    if cu is not None:
                        ch_sec = ts_to_seconds_hms(su.get("Time", ""))
                        if ch_sec >= 0:
                            if cu not in unit_time_ranges:
                                unit_time_ranges[cu] = {"min": ch_sec, "max": ch_sec}
                            else:
                                unit_time_ranges[cu]["min"] = min(unit_time_ranges[cu]["min"], ch_sec)
                                unit_time_ranges[cu]["max"] = max(unit_time_ranges[cu]["max"], ch_sec)
                
                # Check for overlapping unit ranges (acceptable, log as info)
                sorted_units_ranges = sorted(unit_time_ranges.items(), key=lambda x: x[1]["min"])
                for i in range(len(sorted_units_ranges) - 1):
                    u1_no, u1_range = sorted_units_ranges[i]
                    u2_no, u2_range = sorted_units_ranges[i + 1]
                    if u1_range["max"] > u2_range["min"]:
                        logger.info(
                            f"   ℹ️ Unit {u1_no} and Unit {u2_no} time ranges overlap "
                            f"(Unit {u1_no} ends ~{sec_to_hms(u1_range['max'])}, "
                            f"Unit {u2_no} starts ~{sec_to_hms(u2_range['min'])}). "
                            f"This is acceptable for split/interleaved content."
                        )
                
            else:
                logger.warning("⚠️ Unit tagging response did not contain valid 'tags' array, skipping tagging")
                
        except Exception as e:
            logger.warning(f"⚠️ Unit tagging LLM call failed: {e}. Chapters will have no unit tags.")
    
    elif validated_units and len(validated_units) >= 1 and not suggested_units_structured:
        logger.warning("⚠️ No chapters generated, skipping unit tagging")
    
    # ────────── Log final unit mapping stats ──────────
    if validated_units and suggested_units_structured:
        mapped = sum(1 for su in suggested_units_structured if su.get("ClientUnitNo") is not None)
        total = len(suggested_units_structured)
        if mapped > 0:
            logger.info(f"✅ Final: {mapped}/{total} chapters mapped to units, {total - mapped} untagged")
        else:
            missing = sum(1 for x in suggested_units_structured if x.get("ClientUnitNo") is None)
            if missing:
                logger.warning(
                    f"⚠️ {missing}/{total} SuggestedUnits missing ClientUnitNo "
                    f"(client provided {len(validated_units)} Units)"
                )
    
    # Initialize variables for coverage retry and back-calculation
    enriched_units = None
    unit_diagnostics = None
    
    # ────────── Coverage Guardrail ──────────
    if suggested_units_structured and asr_end_sec > 0:
        cov = chapters_coverage_ratio(suggested_units_structured, asr_end_sec)
        last_ch = suggested_units_structured[-1]["Time"]
        logger.info(f"📏 PASS3 coverage check: last_chapter={last_ch}, asr_end={asr_end_ts}, ratio={cov:.2f}")
        
        if asr_end_sec >= 3600 and cov < 0.60:
            logger.warning(
                f"⚠️ PASS3 chapters end too early (ratio={cov:.2f}). Retrying PASS3 with anchor timestamps..."
            )
            retry_hint = f"""
    【強制覆蓋規則（必須遵守）】
    - 逐字稿最後時間戳約為：{asr_end_ts}
    - 你輸出的最後一個章節 Time 必須 >= {sec_to_hms(max(0, int(asr_end_sec * 0.85)))}（至少覆蓋到後段）
    - 禁止只生成前段章節；必須涵蓋整段教學（包含後半段/後段）
    - 以下是逐字稿中分佈於全程的時間戳樣本（必須用來選章節時間點，且要包含後段時間戳）：
    {", ".join(anchors[-10:] if len(anchors) >= 10 else anchors)}
    """
            chapters_prompt_retry = chapters_prompt + "\n" + retry_hint
            retry_resp = call_llm(
                service_type=config.service_type,
                client=client,
                system_message=(
                    "你是細心的章節設計師。"
                    "請只輸出 JSON，禁止任何其他文字。"
                    "章節 Time 必須對齊 ASR 真實時間戳，且必須覆蓋整段逐字稿到後段。"
                ),
                user_message=chapters_prompt_retry,
                model=config.openai_model if config.service_type == "openai" else config.azure_model,
                max_tokens=3000,
                temperature=0.1
            )
            final_text_retry = (
                retry_resp.choices[0].message.content
                if config.service_type == "openai"
                else retry_resp.choices[0].message.content
            )
            data_retry = safe_load_json(final_text_retry)

            suggested_retry: List[Dict[str, Any]] = []
            course_summary_retry: Dict[str, Any] = {}
            if isinstance(data_retry, dict):
                suggested_retry = normalize_suggested_units(data_retry.get("SuggestedUnits"), units=None)
                suggested_retry = clean_all_suggested_units(suggested_retry)
                cs2 = data_retry.get("CourseSummary")
                if isinstance(cs2, dict):
                    course_summary_retry = cs2
            elif isinstance(data_retry, list):
                suggested_retry = normalize_suggested_units(data_retry, units=None)
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
                logger.info(f"✅ PASS3 retry succeeded: SuggestedUnits={len(suggested_units_structured)}")
                
                # Re-run unit tagging on retry results if we have validated units
                if validated_units and len(validated_units) >= 1:
                    logger.info("🔄 Re-running unit tagging on retry results...")
                    # Note: tagging would use same logic as above; for now back_calculate
                    # will handle the mapping via existing ClientUnitNo fields
            else:
                if data_retry is None:
                    logger.warning("⚠️ PASS3 retry JSON parse failed; keeping first result")
                else:
                    logger.warning("⚠️ PASS3 retry JSON parsed but SuggestedUnits empty/invalid; keeping first result")
                    
    # ────────── Build chapters_raw ──────────
    if suggested_units_structured:
        chapters_raw = suggested_units_to_chapters_dict(
            suggested_units_structured,
            duration_sec=int(duration),
            bump_limit_sec=120
        )
        logger.info(f"📊 Parsed {len(suggested_units_structured)} SuggestedUnits from JSON")
    else:
        logger.warning("⚠️ PASS 3 JSON parse failed; falling back to text chapter parsing")
        chapters_raw = parse_chapters_from_output(final_text)
        course_summary = parse_summary_from_output(final_text)

    # ────────── Back-calculate Unit timestamps ──────────
    if validated_units:
        enriched_units, unit_diagnostics = back_calculate_unit_timestamps(
            suggested_units_structured=suggested_units_structured,
            client_units=validated_units
        )
        logger.info("\n" + "=" * 60)
        logger.info("📍 UNIT TIMESTAMP BACK-CALCULATION RESULTS")
        logger.info("=" * 60)
        if enriched_units:
            for unit in enriched_units:
                if unit.get("Time"):
                    logger.info(
                        f"✅ Unit {unit['UnitNo']}: {unit['Title']}\n"
                        f"   → Starts at: {unit['Time']}\n"
                        f"   → First chapter: {unit.get('FirstChapter', 'N/A')}"
                    )
                else:
                    logger.info(
                        f"⚠️ Unit {unit['UnitNo']}: {unit['Title']}\n"
                        f"   → Not found in video!"
                    )
        logger.info("=" * 60 + "\n")
    elif units:
        logger.info("⏭️  Skipping Unit timestamp back-calculation (units rejected by validation)")
        unit_diagnostics = {
            "units_provided": True,
            "validation_passed": False,
            "units_rejected": True,
            "message": "Units failed validation - not mapped to timestamps"
        }
 
    # ────────── Validate and normalize timestamps ──────────
    chapters = validate_and_normalize_timestamps(
        chapters_raw,
        int(duration),
        video_id="hierarchical_pass3"
    )
    if not chapters:
        logger.error("❌ No valid chapters after timestamp validation, using time-based fallback")
        chapters = create_time_based_fallback(int(duration))

    if course_summary:
        logger.info(f"✅ Successfully extracted course summary with {len(course_summary)} fields:")
        for key, value in course_summary.items():
            display_value = value[:80] + "..." if len(value) > 80 else value
            logger.info(f"   • {key}: {display_value}")
    else:
        logger.warning("⚠️ Course summary extraction failed, using empty dict")
    
    # Calculate educational quality score
    quality_score = estimate_educational_quality(chapters, structure_text)
    logger.info(f"📈 Educational quality score: {quality_score:.2f}")
    

    
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
        'token_usage': {
            'original': {
                'asr_tokens': asr_tokens,
                'ocr_tokens': ocr_tokens,
                'total_tokens': total_content_tokens
            },
            'used_per_pass': {
                'asr_tokens': asr_used,
                'ocr_tokens': ocr_used,
                'total_tokens': content_used
            },
            'limits': {
                'asr_limit': ASR_LIMIT,
                'ocr_limit': OCR_LIMIT
            },
            'coverage': {
                'asr_coverage': f"{asr_coverage:.1f}%",
                'ocr_coverage': f"{ocr_coverage:.1f}%"
            }
        }
    }
    # ✅ CRITICAL: expose structured SuggestedUnits to downstream pipeline (tasks.py)
    metadata["suggested_units_structured"] = suggested_units_structured
    # ✅ NEW: Add enriched units and diagnostics
    metadata["client_units_original"] = units
    metadata["client_units_validated"] = validated_units
    metadata["client_units_with_timestamps"] = enriched_units
    metadata["unit_diagnostics"] = unit_diagnostics

    # ✅ DEBUG: preserve raw PASS3 JSON/text for production debugging
    metadata["pass3_raw_json_text"] = final_text
    logger.info(
        "🧩 PASS3 SuggestedUnits structured: %d (units_provided=%s)",
        len(suggested_units_structured),
        "yes" if units else "no"
    )

    logger.info("\n" + "=" * 60)
    logger.info("✅ HIERARCHICAL GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"📊 Chapters generated: {len(chapters)}")
    logger.info(f"📊 Summary fields: {len(course_summary)}")
    logger.info(f"📊 Quality score: {quality_score:.2f}")
    logger.info(f"📊 Strategy: ASR-primary (timing) + OCR-supporting (detail)")
    logger.info(f"📊 Total content used: {content_used:,} tokens (ASR: {asr_used:,}, OCR: {ocr_used:,})")
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

# ─────────────────────────
# Enhanced Main Function with Smart Routing
# ─────────────────────────

def generate_chapters_debug(
    raw_asr_text: str,
    ocr_segments: List[Dict],
    duration: float,
    video_id: str,
    video_title: Optional[str] = None,  # ADD THIS
    section_title: Optional[str] = None,  # ← ADD
    units: Optional[List[Dict]] = None,    # ← ADD
    run_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    *,
    ocr_context_override: Optional[str] = None,
    # NEW: Add control parameter
    force_generation_method: Optional[str] = None,  # 'hierarchical' or 'single_pass'
) -> Tuple[str, Dict[str, str], Dict[str, str], Dict[str, Any]]:
    """
    Enhanced version with smart routing between hierarchical and single-pass generation
    """
    if progress_callback:
        progress_callback("initializing", 0)

    if run_dir is None:
        run_dir = Path(f"/tmp/chapter_generation/{video_id}_{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Starting chapter generation for video {video_id} (duration: {duration}s)")

        # Load configuration
        config = ChapterConfig()
        if not validate_config(config):
            logger.warning("Configuration validation failed, using time-based fallback")
            fallback = create_time_based_fallback(int(duration))
            fallback = ensure_traditional_chapters(fallback)
            return ("", {}, fallback, {"generation_method": "time_based_fallback", "course_summary": {}})

        if progress_callback:
            progress_callback("processing_inputs", 10)

        # Build OCR context (existing logic)
        if ocr_context_override is not None:
            ocr_context = ocr_context_override
        else:
            ocr_context = build_ocr_context_from_segments(ocr_segments) if ocr_segments else ""

        min_gap_sec, target_range, max_caps = chapter_policy(int(duration))
        
        # Save raw inputs
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

        # Initialize client (existing logic)
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

        # 🎯 NEW: Smart Generation Method Selection
        use_hierarchical = False
        if force_generation_method == 'hierarchical':
            use_hierarchical = True
        elif force_generation_method == 'single_pass':
            use_hierarchical = False
        else:
            # Auto-detect based on content characteristics
            use_hierarchical = should_use_hierarchical(duration, len(raw_asr_text))
        
        logger.info(f"Using generation method: {'hierarchical_multi_pass' if use_hierarchical else 'single_pass'}")

        if use_hierarchical:
            if progress_callback:
                progress_callback("hierarchical_analysis", 30)
            
            # Use hierarchical multi-pass generation
            raw_llm_text, chapters, metadata = hierarchical_multipass_generation(
                raw_asr_text=raw_asr_text,
                duration=duration,
                ocr_context=ocr_context,
                video_title=video_title,  # ADD THIS
                section_title=section_title,      # ← ADD
                units=units,                       # ← ADD
                client=client,
                config=config,
                progress_callback=progress_callback
            )
            
            # Save hierarchical metadata
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
            # Use original single-pass generation
            prompt_template = build_prompt_body(
                "", int(duration), ocr_context, video_title,
                first_ts_override=first_ts,
                last_ts_override=last_ts,
            )
            template_tokens = count_tokens_llama(prompt_template)

            CONTEXT_BUDGET = 128_000

            asr_tokens = count_tokens_llama(raw_asr_text)
            if template_tokens + asr_tokens <= CONTEXT_BUDGET:
                transcript_for_prompt = raw_asr_text
                logger.info(
                    f"✅ Using full ASR (template={template_tokens:,}, asr={asr_tokens:,}, budget={CONTEXT_BUDGET:,})"
                )
            else:
                max_transcript_tokens = max(0, CONTEXT_BUDGET - template_tokens)
                transcript_for_prompt = truncate_text_by_tokens(raw_asr_text, max_transcript_tokens)
                logger.warning(
                    f"⚠️ Truncating ASR (template={template_tokens:,}, asr={asr_tokens:,}, "
                    f"budget={CONTEXT_BUDGET:,}, allowed_asr={max_transcript_tokens:,})"
                )
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
                "自動識別課程領域並使用適當專業術語，專注於學習價值和教育連貫性。"
                "嚴格避免重複模式，創建反映真實教育進程的專業章節標題。"
                "僅輸出章節清單，每行格式: `HH:MM:SS - 標題`（繁體中文）。"
            )

            logger.info(f"Calling {service_type} API for single-pass chapter generation...")
            t0 = time.time()
            resp = call_llm(
                service_type=service_type,
                client=client,
                system_message=enhanced_system_message,
                user_message=full_prompt,
                model=model,
                max_tokens=2048,
                temperature=0.2,
                top_p=0.9,
            )
            dt = time.time() - t0
            logger.info(f"LLM API call completed in {dt:.2f}s")

            if service_type == "azure":
                raw_llm_text = resp.choices[0].message.content
            else:
                raw_llm_text = resp.choices[0].message.content

            # Parse chapters
            chapters_raw = parse_chapters_from_output(raw_llm_text)

            # ✅ NEW: Validate and normalize timestamps
            chapters = validate_and_normalize_timestamps(
                chapters_raw,
                int(duration),
                video_id=video_id
            )
            if not chapters:
                logger.error("❌ No valid chapters after timestamp validation, using time-based fallback")
                chapters = create_time_based_fallback(int(duration))

            # Parse structured summary
            course_summary = parse_summary_from_output(raw_llm_text)
            metadata = {'generation_method': 'single_pass',
                        'course_summary': course_summary,
                        }

        # COMMON POST-PROCESSING (existing logic)
        if progress_callback:
            progress_callback("parsing_response", 70)

        with open(run_dir / "llm_output_raw.txt", "w", encoding="utf-8") as f:
            f.write(raw_llm_text)

        # Apply cleaning and Traditional Chinese conversion
        parsed_raw_clean_trad = ensure_traditional_chapters(clean_chapter_titles(chapters))

        with open(run_dir / "parsed_raw_chapters.json", "w", encoding="utf-8") as f:
            json.dump(parsed_raw_clean_trad, f, ensure_ascii=False, indent=2)

        if progress_callback:
            progress_callback("balancing_chapters", 80)

        # Balance according to policy
        chapters_final = globally_balance_chapters(
            parsed_raw_clean_trad, int(duration), min_gap_sec, target_range, max_caps
        )
        if not chapters_final:
            raise RuntimeError("No chapters left after balancing")

        with open(run_dir / "chapters_final.json", "w", encoding="utf-8") as f:
            json.dump(chapters_final, f, ensure_ascii=False, indent=2)

        # Save generation method info
        with open(run_dir / "generation_method.txt", "w", encoding="utf-8") as f:
            f.write(metadata.get('generation_method', 'unknown'))

        if progress_callback:
            progress_callback("completed", 100)

        # Return 4-tuple: (raw_text, parsed_chapters, final_chapters, metadata)
        return (raw_llm_text, parsed_raw_clean_trad, chapters_final, metadata)  # ← FIXED!

    except Exception as e:
        logger.error(f"Chapter generation failed: {e}", exc_info=True)
        fallback = ensure_traditional_chapters(create_time_based_fallback(int(duration)))
        # Return fallback with empty metadata
        fallback_metadata = {
            'generation_method': 'time_based_fallback',
            'educational_quality_score': 0.0,
            'course_summary': {}
        }
        return ("", {}, fallback, fallback_metadata)  # ← FIXED!
        
# ─────────────────────────
# MAIN FUNCTIONS
# ─────────────────────────

def create_time_based_fallback(duration_sec: int) -> Dict[str, str]:
    """Create fallback chapters based on time intervals"""
    fallback_chapters: Dict[str, str] = {}
    interval = 300  # 5 minutes
    for i in range(0, int(duration_sec), interval):
        fallback_chapters[sec_to_hms(i)] = "章節 " + str((i // interval) + 1)
    logger.info(f"Created {len(fallback_chapters)} time-based fallback chapters")
    return fallback_chapters

def generate_chapters(
    raw_asr_text: str,
    ocr_segments: List[Dict],
    duration: float,
    video_id: str,
    video_title: Optional[str] = None,
    section_title: Optional[str] = None,  # ← ADD
    units: Optional[List[Dict]] = None,    # ← ADD
    run_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    *,
    ocr_context_override: Optional[str] = None,
    force_generation_method: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, Any]]:  # ← FIXED: Return tuple
    """
    Generate chapters and return (chapters_dict, metadata).
    
    For backward compatibility with old code that expects just chapters dict,
    you can use: chapters, _ = generate_chapters(...)
    
    Returns:
        Tuple of (chapters_dict, metadata)
    """
    _raw_text, _parsed_raw, final_chapters, metadata = generate_chapters_debug(  # ← FIXED: Unpack 4 values
        raw_asr_text=raw_asr_text,
        ocr_segments=ocr_segments,
        duration=duration,
        video_id=video_id,
        video_title=video_title,  # ← Make sure this is passed
        section_title=section_title,        # ← ADD
        units=units,                         # ← ADD
        run_dir=run_dir,
        progress_callback=progress_callback,
        ocr_context_override=ocr_context_override,
        force_generation_method=force_generation_method
    )
    return final_chapters, metadata  # ← FIXED: Return tuple

# ─────────────────────────
# CLI
# ─────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate video chapters from raw ASR and optional OCR.")
    parser.add_argument('--asr-file', type=argparse.FileType('r', encoding='utf-8'), required=True,
                        help='Path to file containing raw ASR text with timestamps.')
    parser.add_argument('--ocr-file', type=argparse.FileType('r', encoding='utf-8'),
                        help='Optional path to OCR file. In verbatim mode this is read as raw text.')
    parser.add_argument('--duration', type=float, required=True,
                        help='Duration of the video in seconds.')
    parser.add_argument('--video-id', type=str, required=True,
                        help='Unique identifier for the video (used for output directory).')
    parser.add_argument('--output-dir', type=str, default='./chapter_debug',
                        help='Directory to save debug outputs. Default: ./chapter_debug')
    parser.add_argument('--debug', action='store_true', help='Print RAW LLM output and parsed chapters too.')
    parser.add_argument('--ocr-mode', choices=['none', 'verbatim', 'segments'], default='verbatim',
                        help="How to include OCR: 'none' (omit), 'verbatim' (raw text), or 'segments' (legacy minimal formatting).")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

    # Read ASR
    logger.info(f"Reading ASR text from {args.asr_file.name}...")
    raw_asr_text = args.asr_file.read()
    args.asr_file.close()

    # Read OCR according to the chosen mode
    ocr_segments: List[Dict] = []
    ocr_context_override: Optional[str] = None
    if args.ocr_file:
        if args.ocr_mode == 'none':
            logger.info("OCR mode: none (omit OCR from prompt).")
            try:
                args.ocr_file.close()
            except Exception:
                pass
        elif args.ocr_mode == 'verbatim':
            logger.info(f"OCR mode: verbatim. Reading {args.ocr_file.name} as raw text...")
            try:
                ocr_context_override = args.ocr_file.read()
            finally:
                try:
                    args.ocr_file.close()
                except Exception:
                    pass
            logger.info("OCR loaded verbatim.")
        else:
            logger.info(f"OCR mode: segments. Reading OCR segments from {args.ocr_file.name}...")
            try:
                ocr_segments = load_ocr_segments(args.ocr_file, args.ocr_file.name)
                args.ocr_file.close()
                logger.info(f"Loaded {len(ocr_segments)} OCR segments")
            except Exception as e:
                logger.warning(f"OCR file load failed, proceeding without OCR. Detail: {e}")
                try:
                    args.ocr_file.close()
                except Exception:
                    pass
                ocr_segments = []

    # Output directory
    run_dir = Path(args.output_dir) / args.video_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving debug outputs to: {run_dir}")

    # Simple progress callback
    def cli_progress_callback(stage: str, percent: int):
        logger.info(f"Progress: {percent}% - {stage}")

    # Generate
    logger.info("Starting chapter generation...")

    raw_text, parsed_raw, final_chapters, metadata = generate_chapters_debug(
        raw_asr_text=raw_asr_text,
        ocr_segments=ocr_segments,
        duration=args.duration,
        video_id=args.video_id,
        run_dir=run_dir,
        progress_callback=cli_progress_callback,
        ocr_context_override=ocr_context_override,
    )
    
    # Console output
    print("\n" + "="*50)
    print("✅ CHAPTER GENERATION COMPLETE")
    print("="*50)

    if args.debug:
        print("\n--- RAW LLM OUTPUT (as returned) ---")
        print(raw_text if raw_text else "(empty/raw fallback)")
        print("\n--- PARSED (pre-balance) ---")
        for ts, title in parsed_raw.items():
            print(f"{ts} - {title}")

    print("\n--- FINAL (balanced) ---")
    for ts, title in final_chapters.items():
        print(f"{ts} - {title}")

    # Save final chapters to a clean file
    output_file = run_dir / "final_chapters.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for timestamp, title in final_chapters.items():
            f.write(f"{timestamp} - {title}\n")
    logger.info(f"Final chapters saved to: {output_file}")

    # Also save a pre-balance view for convenience
    pre_file = run_dir / "parsed_raw_chapters.txt"
    with open(pre_file, 'w', encoding='utf-8') as f:
        for timestamp, title in parsed_raw.items():
            f.write(f"{timestamp} - {title}\n")
    logger.info(f"Parsed (pre-balance) chapters saved to: {pre_file}")

if __name__ == "__main__":
    main()
