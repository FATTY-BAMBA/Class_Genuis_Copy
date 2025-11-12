# app/video_chaptering.py
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
    openai_model: str = os.getenv("CHAPTER_OPENAI_MODEL", "gpt-4o")
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

def count_tokens_llama(text: str) -> int:
    """Approximate token counting for mixed Chinese/English (≈1 token per CJK char; 1/4 per other chars)"""
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
                if re.match(r'\d{2}:\d{2}:\d{2}', ts):
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
        p = ts.split(":")
        if len(p) == 2:
            return int(p[0]) * 60 + int(p[1])
        if len(p) == 3:
            return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])
        return 0

    cands = [(ts_to_s(ts), ts, t.strip()) for ts, t in chapters.items() if 0 <= ts_to_s(ts) <= duration_sec]
    cands.sort(key=lambda x: x[0])
    if not cands:
        return {}

    # Content-aware deduplication
    dedup = []
    for s, ts, title in cands:
        if not dedup:
            dedup.append((s, ts, title))
            continue
            
        time_gap = s - dedup[-1][0]
        
        # Merge if same module and close, or very close regardless
        should_merge = False
        prev_module = extract_module_tag(dedup[-1][2])
        curr_module = extract_module_tag(title)
        
        if time_gap < 120:  # Less than 2 minutes - always merge
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
    video_title: Optional[str] = None,  # ADD THIS
) -> str:
    duration_hms = sec_to_hms(int(duration_sec))
    min_gap_sec, (t_low, t_high), max_caps = chapter_policy(int(duration_sec))
    
    # Extract first and last timestamps for concrete examples
    timestamps = []
    for line in transcript.split('\n'):
        if ':' in line and len(line.split(':')) >= 4:
            ts_part = ':'.join(line.split(':')[:3])
            if re.match(r'\d{2}:\d{2}:\d{2}', ts_part):
                timestamps.append(ts_part)
    
    first_ts = timestamps[0] if timestamps else "00:00:00"
    last_ts = timestamps[-1] if timestamps else duration_hms

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
{transcript[:50000]}... [其餘內容已載入]

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
模塊名稱 ~ 預估時間範圍 ~ 核心學習點 ~ 教學方法

範例：
基礎工具操作 ~ 00:00-00:25 ~ Illustrator介面認識、基本工具使用 ~ 理論講解+實例演示
進階設計技巧 ~ 00:25-00:50 ~ Logo設計、色彩管理、輸出設定 ~ 綜合案例+實際操作
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
    
    # ==================== PASS 3: Detailed Chapter Generation ====================
    logger.info("\n" + "-" * 60)
    logger.info("📑 PASS 3: Detailed Chapter Generation")
    logger.info("   Goal: Create 15-30 precise chapter timestamps with titles")
    logger.info("   Approach: ASR-primary (timing) + OCR-supporting (detail)")
    logger.info("-" * 60)
    
    if progress_callback:
        progress_callback("generating_detailed_chapters", 80)
    
    chapters_prompt = f"""
【課程整體結構】
{structure_text}

【學習模塊規劃】  
{modules_text}

現在為每個模塊生成具體的章節時間點（總共15-30個章節），並提供課程摘要。

【章節設計原則】
1. 每個章節代表一個完整的學習子目標（5-10分鐘）
2. 標記關鍵概念的首次詳細解釋
3. 標記重要範例或案例分析的開始
4. 標記練習題或互動環節
5. 標記重點回顧或總結處

【章節時間點定位策略（重要性排序）】

**第一優先：ASR語言時間戳（主要依據 - 決定章節開始時間）**
- 講師的明確主題宣告："接下來我們要講..."、"現在進入..."、"首先..."
- 教學轉換信號："好的，這部分完成了"、"現在來看..."、"我們來示範..."
- 重要概念的首次詳細解釋開始點
- 實例演示的明確開始："我們來實際操作一下..."
- 練習或互動的開始："大家試試看..."

**第二優先：內容邏輯轉換（ASR內容分析）**
- 從理論講解到實際演示的自然轉換點
- 新工具/技術的首次詳細介紹
- 從簡單範例到複雜應用的過渡
- 階段性小結後開始新的子主題

**第三優先：OCR視覺輔助（補充確認 - 提供章節標題細節）**
- 確認當前討論的具體主題（投影片標題提供準確名稱）
- 提供精確的技術術語（當講師說"這個工具"時，OCR顯示"矩形工具"）
- 補充視覺內容描述（圖表標題、代碼片段、操作步驟）
- 當ASR表述不夠明確時，參考螢幕內容來補充細節

【標題命名規範】
- **優先使用講師的自然表述**（來自ASR，更口語化、易懂）
- **結合投影片的專業術語**（來自OCR，提供準確的技術名稱）
- 使用具體、可操作的描述（避免"介紹"、"說明"等模糊詞彙）
- 包含所屬模塊標籤（如：[基礎工具]、[進階技巧]、[實戰案例]）
- 標題格式：[模塊標籤] 動作/對象/目標

【時間點選擇的黃金原則】
⚠️ ASR時間戳記錄了講師實際開始講解新主題的時間 - 這是最自然、最符合學習節奏的章節起點
⚠️ 投影片（OCR）通常在講師宣告主題之後才出現，用於確認內容和提供術語，而非決定時間點
⚠️ 優先選擇講師明確宣告新主題的時間點（從ASR）作為章節開始時間
⚠️ 使用投影片內容（從OCR）來豐富和精確化章節標題

完整逐字稿（含精確時間戳 - 主要用於確定章節時間）：
{asr_text}

視覺輔助內容（投影片/螢幕文字 - 主要用於豐富章節標題）：
{ocr_text if ocr_text else "無視覺輔助內容"}

總時長：{sec_to_hms(int(duration))}

【輸出格式要求】
請嚴格按照以下格式輸出：

第一部分 - 章節列表：
HH:MM:SS - [模塊標籤] 具體章節標題
（每行一個章節，時間來自ASR，標題結合ASR表述與OCR術語）

優質範例（時間點來自講師宣告，標題結合講師表述與投影片術語）：
00:10:03 - [基礎工具介紹] Adobe Illustrator的基本操作和介面設置
00:19:09 - [效果工具應用] 使用效果工具創建文字彎曲效果
00:31:51 - [矩形和鋼筆工具] 使用矩形和鋼筆工具創建基本形狀和線條

（空一行）

第二部分 - 課程摘要：
課程主題：[主要教學領域，如：Adobe Illustrator基礎教學、Python資料分析]
核心內容：[列出6-12個主要教學概念，以頓號分隔，反映整個課程的關鍵知識點]
學習目標：[學生完成後將掌握的具體、可衡量的能力]
適合對象：[目標學員背景，如：設計初學者、有程式基礎的進階學員]
難度級別：[初級/中級/高級]
"""
    
    logger.info(f"📤 PASS 3 prompt: ~{count_tokens_llama(chapters_prompt):,} tokens")
    logger.info("🤖 Calling LLM for final chapter generation...")
    t0 = time.time()
    
    try:
        final_response = call_llm(
            service_type=config.service_type,
            client=client,
            system_message="你是細心的章節設計師，擅長為學習模塊創建精確的時間標記和課程摘要。你的核心原則是：使用ASR時間戳來確定章節開始時間（因為這反映講師的自然教學節奏），使用OCR內容來豐富章節標題（提供準確的專業術語）。請嚴格遵守輸出格式，確保包含章節列表和課程摘要兩部分。",
            user_message=chapters_prompt,
            model=config.openai_model if config.service_type == "openai" else config.azure_model,
            max_tokens=3000,
            temperature=0.1
        )
        
        elapsed = time.time() - t0
        logger.info(f"✅ PASS 3 completed in {elapsed:.1f}s")
        
        final_text = (final_response.choices[0].message.content 
                     if config.service_type == "openai" 
                     else final_response.choices[0].message.content)
        
        logger.info(f"📝 Final output: {len(final_text)} characters")
        
    except Exception as e:
        logger.error(f"❌ PASS 3 failed: {e}", exc_info=True)
        raise
    
    # ==================== Parse Results ====================
    logger.info("\n" + "-" * 60)
    logger.info("🔍 Parsing Generated Content")
    logger.info("-" * 60)
    
    # Parse chapters from final output
    chapters = parse_chapters_from_output(final_text)
    logger.info(f"📊 Parsed {len(chapters)} chapters from LLM output")
    
    if chapters:
        first_chapter = list(chapters.keys())[0]
        last_chapter = list(chapters.keys())[-1]
        logger.info(f"📍 Chapter range: {first_chapter} to {last_chapter}")
    else:
        logger.warning("⚠️ No chapters were parsed from LLM output!")
    
    # Parse structured summary
    course_summary = parse_summary_from_output(final_text)
    
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
    run_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    *,
    ocr_context_override: Optional[str] = None,
    # NEW: Add control parameter
    force_generation_method: Optional[str] = None,  # 'hierarchical' or 'single_pass'
) -> Tuple[str, Dict[str, str], Dict[str, str]]:
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
            return ("", {}, fallback)

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
            
            # Use original single-pass generation
            prompt_template = build_prompt_body("", int(duration), ocr_context)
            template_tokens = count_tokens_llama(prompt_template)
            CONTEXT_BUDGET = 128_000
            max_transcript_tokens = max(0, CONTEXT_BUDGET - template_tokens)
            transcript_for_prompt = truncate_text_by_tokens(raw_asr_text, max_transcript_tokens)
            full_prompt = build_prompt_body(transcript_for_prompt, int(duration), ocr_context, video_title)

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
            chapters = parse_chapters_from_output(raw_llm_text)
            # Parse structured summary
            course_summary = parse_summary_from_output(raw_llm_text)
            metadata = {'generation_method': 'single_pass',
                        'course_summary': course_summary}

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

        return (raw_llm_text, parsed_raw_clean_trad, chapters_final)

    except Exception as e:
        logger.error(f"Chapter generation failed: {e}", exc_info=True)
        fallback = ensure_traditional_chapters(create_time_based_fallback(int(duration)))
        return ("", {}, fallback)
        
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
    run_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    *,
    ocr_context_override: Optional[str] = None,
    # NEW: Add the same parameter here for consistency
    force_generation_method: Optional[str] = None,  # 'hierarchical' or 'single_pass'
) -> Dict[str, str]:
    """
    Backward-compatible wrapper returning only the FINAL balanced chapters.
    If you want raw model output as well, call `generate_chapters_debug` instead.
    """
    _raw_text, _parsed_raw, final_chapters = generate_chapters_debug(
        raw_asr_text=raw_asr_text,
        ocr_segments=ocr_segments,
        duration=duration,
        video_id=video_id,
        run_dir=run_dir,
        progress_callback=progress_callback,
        ocr_context_override=ocr_context_override,
        force_generation_method=force_generation_method  # Pass through the new parameter
    )
    return final_chapters

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
    raw_text, parsed_raw, final_chapters = generate_chapters_debug(
        raw_asr_text=raw_asr_text,
        ocr_segments=ocr_segments,
        duration=args.duration,
        video_id=args.video_id,
        run_dir=run_dir,
        progress_callback=cli_progress_callback,
        ocr_context_override=ocr_context_override,  # raw OCR passes straight through
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
