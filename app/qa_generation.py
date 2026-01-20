# app/qa_generation.py
"""
Module for generating educational content (MCQs and Lecture Notes) from pre-processed ASR and OCR segments.
Designed for enhancing student learning and review of educational materials.

This version:
- ASR-first prompts for MCQs and Lecture Notes (OCR is auxiliary; conflict -> ASR wins)
- Bloom-structured MCQs; detailed, past-tense lecture notes; strict JSON outputs
- Simplified->Traditional conversion safety net (OpenCC if available, else fallback map)
- NEW: Post-processing helpers (shuffle options, regenerate explanations, enforce difficulty)
- NEW: Post-processing **controlled by function parameters**, not environment variables
"""

import hashlib
import json
import logging
import os
import re
import time
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from pydantic import ValidationError


# Azure AI Inference imports
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# OpenAI import
from openai import OpenAI

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# ==================== PROGRESS ====================
STAGES = {
    "initializing": 5,
    "processing_inputs": 15,
    "initializing_client": 25,
    "generating_topics_summary": 40,
    "generating_mcqs": 55,
    "generating_notes": 80,
    "processing_results": 92,
    "completed": 100,
}

def report(stage: str, progress_callback: Optional[Callable[[str, int], None]]):
    if progress_callback:
        progress_callback(stage, STAGES.get(stage, 0))

# ==================== CONFIGURATION ====================
@dataclass
class EducationalContentConfig:
    """Configuration for educational content generation service"""
    service_type: str = os.getenv("EDU_SERVICE_TYPE", "openai")
    openai_model: str = os.getenv("EDU_OPENAI_MODEL", "gpt-4o-mini")
    azure_model: str = os.getenv("EDU_AZURE_MODEL", "Meta-Llama-3.1-8B-Instruct")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    azure_endpoint: Optional[str] = os.getenv("AZURE_AI_ENDPOINT")
    azure_key: Optional[str] = os.getenv("AZURE_AI_KEY")
    azure_api_version: str = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
    max_questions: int = int(os.getenv("MAX_QUESTIONS", "10"))
    max_notes_pages: int = int(os.getenv("MAX_NOTES_PAGES", "5"))
    enable_cache: bool = os.getenv("EDU_ENABLE_CACHE", "true").lower() == "true"
    force_traditional: bool = os.getenv("EDU_FORCE_TRADITIONAL", "true").lower() == "true"

def validate_config(config: EducationalContentConfig) -> bool:
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

def get_content_hash(transcript: str, ocr_context: str, content_type: str) -> str:
    """Generate hash for content to enable caching"""
    content = f"{transcript}{ocr_context}{content_type}"
    return hashlib.md5(content.encode()).hexdigest()

# ==================== DATA STRUCTURES ====================
@dataclass
class MCQ:
    question: str
    options: List[str]
    correct_answer: str
    explanation: str
    difficulty: str
    topic: str
    tags: List[str] = field(default_factory=list)  # NEW
    course_type: str = "general"  # NEW

@dataclass
class LectureNoteSection:
    title: str
    content: str
    key_points: List[str]
    examples: List[str]

@dataclass
class EducationalContentResult:
    mcqs: List[MCQ]
    lecture_notes: List[LectureNoteSection]
    summary: str
    topics: List[Dict] = field(default_factory=list)          # ← NEW
    key_takeaways: List[str] = field(default_factory=list)    # ← NEW
# ==================== PYDANTIC MODELS FOR VALIDATION ====================
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class ValidatedLectureSection(BaseModel):
    """Pydantic model for validated lecture sections"""
    title: str = Field(..., min_length=1, max_length=300)
    content: str = Field(..., min_length=10, max_length=10000)
    key_points: List[str] = Field(default_factory=list, max_length=10)
    examples: List[str] = Field(default_factory=list, max_length=5)
    
    @field_validator('key_points', 'examples', mode='before')
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if not isinstance(v, list):
            return []
        # Filter out empty strings and ensure all items are strings
        return [str(item).strip() for item in v if item and str(item).strip()]

class ValidatedLectureNotes(BaseModel):
    """Pydantic model for complete lecture notes response"""
    sections: List[ValidatedLectureSection] = Field(..., min_length=1, max_length=20)
    summary: str = Field(..., min_length=10, max_length=1000)
    
    class Config:
        extra = 'ignore'  # Ignore unexpected fields
        str_strip_whitespace = True  # Auto strip whitespace
        
# ==================== UTILITIES ====================
def sec_to_hms(sec: int) -> str:
    """Convert seconds to HH:MM:SS format"""
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def count_tokens_llama(text: str) -> int:
    """Approximate token counting for Llama-like tokenization"""
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    non_chinese_len = len(text) - chinese_chars
    return chinese_chars + max(1, non_chinese_len // 4)

def truncate_text_by_tokens(text: str, max_tokens: int = 120_000) -> str:
    """Truncate to approx max_tokens, preserving whole sentences"""
    if max_tokens <= 0:
        return ""
    tokens = count_tokens_llama(text)
    if tokens <= max_tokens:
        return text
    logger.warning(f"Truncating content from {tokens:,} tokens to {max_tokens:,} tokens")
    sentences = re.split(r'(?<=[。！？.!?])', text)
    truncated, current = [], 0
    for sentence in sentences:
        t = count_tokens_llama(sentence)
        if current + t > max_tokens:
            break
        truncated.append(sentence)
        current += t
    return "".join(truncated)

def format_chapters_for_prompt(chapters: Dict[str, str]) -> List[Dict]:
    """Convert chapter dict to list format for prompts"""
    formatted = []
    for ts, title in chapters.items():
        formatted.append({
            "ts": ts,
            "title": title
        })
    return formatted

def build_ocr_context_from_segments(ocr_segments: List[Dict]) -> str:
    """Convert OCR segments into a descriptive context string (sentence-level bullets)."""
    if not ocr_segments:
        return ""
    context_lines = ["# 從投影片與螢幕捕捉到的相關文字："]
    SENT_SPLIT = re.compile(r"[。；;！？!?]\s*|\n+")
    for seg in ocr_segments:
        start = int(seg.get('start', 0))
        text = (seg.get('text') or "").strip()
        if not text:
            continue
        timestamp = sec_to_hms(start)
        context_lines.append(f"*   於 {timestamp} 左右捕捉到:")
        for sent in filter(None, (s.strip() for s in SENT_SPLIT.split(text))):
            context_lines.append(f"    - 「{sent}」")
    return "\n".join(context_lines)

def ocr_segments_to_raw_text(ocr_segments: List[Dict]) -> str:
    """Flatten OCR segments to raw lines (optionally timestamp-prefixed)."""
    lines: List[str] = []
    for seg in ocr_segments or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = seg.get("start")
        ts = f"[{sec_to_hms(int(start))}] " if isinstance(start, (int, float)) else ""
        lines.append(f"{ts}{text}")
    return "\n".join(lines)

# ---------- Simplified -> Traditional conversion ----------
# ==================== Initialize OpenCC (REQUIRED) ====================
def _init_opencc():
    """
    Initialize OpenCC converter. This is REQUIRED for proper Traditional Chinese output.
    Falls back to limited character mapping with warning if OpenCC not installed.
    """
    try:
        from opencc import OpenCC
        converter = OpenCC('s2t')
        logger.info("OpenCC initialized successfully for S->T conversion")
        return converter
    except ImportError:
        logger.error(
            "OpenCC is not installed but is required for proper Traditional Chinese conversion. "
            "Please install it with: pip install opencc-python-reimplemented"
        )
        # Return None to use fallback, but log warning on every conversion
        return None
    except Exception as e:
        logger.error(f"Failed to initialize OpenCC: {e}")
        return None

# Initialize on module load
_OPENCC = _init_opencc()

# ==================== Comprehensive Fallback Mapping ====================
# Extended character mapping for when OpenCC is unavailable
# This covers common educational and technical terms
_S2T_FALLBACK = str.maketrans({
    # Basic common characters
    "后": "後", "里": "裡", "台": "臺", "万": "萬", "与": "與", "书": "書", 
    "体": "體", "价": "價", "优": "優", "儿": "兒", "动": "動", "华": "華", 
    "发": "發", "复": "復", "国": "國", "广": "廣", "汉": "漢", "会": "會", 
    "纪": "紀", "简": "簡", "经": "經", "历": "歷", "马": "馬", "门": "門", 
    "面": "麵", "内": "內", "气": "氣", "权": "權", "确": "確", "实": "實", 
    "术": "術", "云": "雲", "众": "眾", "为": "為", "从": "從", "冲": "衝",
    
    # Educational and learning terms
    "练": "練", "习": "習", "题": "題", "设": "設", "识": "識", "导": "導",
    "统": "統", "议": "議", "论": "論", "验": "驗", "类": "類", "证": "證",
    "释": "釋", "译": "譯", "编": "編", "课": "課", "讲": "講", "义": "義",
    
    # Technical and programming terms
    "库": "庫", "码": "碼", "执": "執", "态": "態", "储": "儲", "载": "載",
    "输": "輸", "进": "進", "选": "選", "错": "錯", "数": "數", "据": "據",
    "构": "構", "节": "節", "块": "塊", "链": "鏈", "队": "隊", "栈": "棧",
    
    # Common verbs and actions
    "说": "說", "读": "讀", "写": "寫", "问": "問", "应": "應", "见": "見",
    "开": "開", "关": "關", "买": "買", "卖": "賣", "听": "聽", "观": "觀",
    "记": "記", "认": "認", "让": "讓", "谈": "談", "请": "請", "转": "轉",
    
    # Analysis and evaluation terms
    "评": "評", "测": "測", "试": "試", "检": "檢", "查": "查", "审": "審",
    "对": "對", "错": "錯", "难": "難", "题": "題", "答": "答", "总": "總",
    
    # Additional common characters in educational content
    "师": "師", "学": "學", "声": "聲", "战": "戰", "钟": "鐘", "级": "級",
    "这": "這", "爱": "愛", "时": "時", "间": "間", "现": "現", "电": "電",
    "视": "視", "频": "頻", "网": "網", "络": "絡", "线": "線", "连": "連",
    "图": "圖", "画": "畫", "场": "場", "报": "報", "纸": "紙", "张": "張",
})

# ==================== Conversion Function ====================
def to_traditional(text: str) -> str:
    """
    Convert Simplified Chinese to Traditional Chinese.
    
    Priority:
    1. Use OpenCC if available (recommended)
    2. Fall back to character mapping with warning
    
    Args:
        text: Input text potentially containing Simplified Chinese
    
    Returns:
        Text converted to Traditional Chinese
    """
    if not text:
        return text
    
    # Try OpenCC first (recommended path)
    if _OPENCC is not None:
        try:
            return _OPENCC.convert(text)
        except Exception as e:
            logger.warning(f"OpenCC conversion failed: {e}, using fallback")
    
    # Fallback path - warn on first use in session
    if not hasattr(to_traditional, '_fallback_warned'):
        logger.warning(
            "Using limited character mapping for S->T conversion. "
            "For best results, install OpenCC: pip install opencc-python-reimplemented"
        )
        to_traditional._fallback_warned = True
    
    # Apply fallback character mapping
    return text.translate(_S2T_FALLBACK)

# ==================== Validation Function (Optional) ====================
def validate_traditional_conversion() -> bool:
    """
    Validate that Traditional Chinese conversion is working properly.
    Can be called during initialization to ensure system is ready.
    
    Returns:
        True if OpenCC is available and working, False otherwise
    """
    test_pairs = [
        ("学习", "學習"),
        ("编程", "編程"),
        ("问题", "問題"),
        ("这个", "這個"),
    ]
    
    if _OPENCC is None:
        logger.warning("OpenCC not available - using fallback conversion")
        return False
    
    try:
        for simplified, expected in test_pairs:
            result = to_traditional(simplified)
            if result != expected:
                logger.warning(f"Conversion test failed: {simplified} -> {result} (expected {expected})")
                return False
        logger.info("Traditional Chinese conversion validated successfully")
        return True
    except Exception as e:
        logger.error(f"Conversion validation failed: {e}")
        return False

# ==================== PROMPT BUILDERS (Topics and Summary, ASR-first) ====================
def build_topics_summary_prompt(transcript: str, 
                                video_title: Optional[str] = None,  # ADD THIS
                                context: Optional[Dict[str, str]] = None) -> str:
    """
    Build prompt for LLM to extract meaningful topics and global summary from ASR transcript.
    
    Args:
        transcript: The lecture transcript text
        context: Optional context about the lecture (course name, instructor, etc.)
    
    Returns:
        Formatted prompt string for the LLM
    """
    
    context_info = ""
    if context:
        context_items = [f"- {k}: {v}" for k, v in context.items()]
        context_info = f"""
# 課程背景資訊
{chr(10).join(context_items)}
"""
    video_title_info = ""
    if video_title:
        clean_title = re.sub(r'\.(mp4|avi|mov|mkv|webm|flv|m4v)$', '', video_title, flags=re.IGNORECASE)
        video_title_info = f"""
# 課程檔名
原始檔名：{clean_title}
請參考檔名理解課程的主題範圍和重點。
"""

    prompt = f"""
# 角色定位
你是一位資深的課程分析專家，專精於教學設計和知識結構化。你的任務是分析講座逐字稿，
提取核心主題並生成高質量的課程摘要。

{context_info}
{video_title_info}

# 分析指令

## 1. 深度理解
- 仔細閱讀逐字稿，理解講座的整體脈絡
- 識別教學目標、核心概念和邏輯流程
- 注意講者的重點和強調內容

## 2. 主題提取
識別 **5-8 個**最重要的教學主題，每個主題應該：
- 代表一個完整、有意義的學習單元
- 具有明確的教學價值
- 有足夠的內容深度（約佔15-30分鐘的講座時間）

## 3. 內容過濾
- 排除：行政公告、個人閒聊、技術問題
- 合併：重複或零散但相關的內容
- 保留：所有具教學價值的核心內容

## 4. 摘要撰寫
- 簡潔但全面地總結課程
- 使用清晰、專業的語言
- 捕捉講座精髓和學習價值

# 輸出格式（務必嚴格遵守）

```json
{{
  "topics": [
    {{
      "id": "01",
      "title": "主題名稱（具體且描述性）",
      "summary": "該主題的說明，2-3句話，涵蓋核心概念、學習要點和應用場景",
      "keywords": ["關鍵詞1", "關鍵詞2", "關鍵詞3"]
    }}
  ],
  "global_summary": "整個講座的綜合摘要（3-5句話），說明：1) 課程目標 2) 主要內容 3) 學習成果",
  "key_takeaways": [
    "核心要點1",
    "核心要點2",
    "核心要點3"
  ]
}}
```

# 主題品質標準

1. **教學相關性**: 每個主題必須具有教育價值
2. **具體明確**: 使用精確的主題名稱
   - ✅ 好："Python列表切片與索引操作"
   - ❌ 差："Python基礎"
3. **邏輯連貫**: 主題順序應反映知識遞進關係
4. **適當粒度**: 不要過於細碎或寬泛
5. **實用導向**: 強調可應用的知識和技能

# 輸入資料

## ASR 逐字稿內容：
{transcript}

# 重要提醒
- 時間戳和章節標記僅供參考，不要完全依賴
- 關注講者的教學意圖，而非表面內容
- 保持客觀中立，避免主觀評價
- 確保輸出為有效的JSON格式
"""
    
    return prompt


# ==================== EDUCATIONAL METADATA HELPERS ====================
def build_educational_metadata_context(
    section_title: Optional[str],
    units: Optional[List[Dict]]
) -> str:
    """
    Build educational context from SectionTitle and Units metadata.
    Similar to chapter_generation.py approach but for Q&A context.
    
    Args:
        section_title: Course section title (e.g., "室內設計實務 廚具規劃")
        units: List of units with UnitNo and Title
        
    Returns:
        Formatted context string for prompt enhancement
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
        context_parts.append("本課程包含以下教學單元：")
        for unit in units:
            unit_no = unit.get("UnitNo", "")
            unit_title = unit.get("Title", "")
            context_parts.append(f"   {unit_no}. {unit_title}")
        context_parts.append("")
        
        context_parts.append("## Q&A 設計指引")
        context_parts.append("✅ 題目應涵蓋各個教學單元的核心知識點")
        context_parts.append("✅ 在可能的情況下，為每個單元設計相應題目")
        context_parts.append(f"✅ 建議分配：每單元 {max(1, 10 // len(units))} 題左右")
        context_parts.append("✅ 題目標籤 (tags) 中可標註相關單元編號")
        context_parts.append("")
        
        context_parts.append("## 講義筆記指引")
        context_parts.append("✅ 講義章節應對應教學單元結構")
        context_parts.append("✅ 為每個單元提供清晰的學習要點整理")
        context_parts.append("✅ 章節標題建議格式：[單元N：單元名稱] 具體內容")
    
    return "\n".join(context_parts)


def extract_unit_tags(units: Optional[List[Dict]]) -> List[str]:
    """
    Extract unit titles as potential tags for MCQs.
    
    Args:
        units: List of units with UnitNo and Title
        
    Returns:
        List of unit titles suitable for tagging
    """
    if not units:
        return []
    
    return [f"單元{unit['UnitNo']}：{unit['Title']}" for unit in units]

def parse_topics_summary_response(response_text: str) -> tuple[List[Dict], str, List[str]]:
    """
    Parse topics, summary, and key takeaways from LLM response.
    
    Args:
        response_text: Raw LLM response containing JSON
    
    Returns:
        Tuple of (topics_list, global_summary, key_takeaways)
        Returns empty structures if parsing fails
    """
    # Use module-level logger instead of print
    logger = logging.getLogger(__name__)
    
    # Parse JSON from response
    data = _safe_load_json(response_text)  # Use your existing function
    if not data:
        logger.warning("Failed to parse topics/summary response JSON")
        return [], "", []
    
    # Extract with defensive parsing
    topics = []
    for i, topic_data in enumerate(data.get('topics', [])):
        if not isinstance(topic_data, dict):
            continue
            
        # Ensure required fields with sensible defaults
        topic_id = str(topic_data.get('id', f"{i+1:02d}")).strip()
        title = str(topic_data.get('title', f"主題 {i+1}")).strip()
        summary = str(topic_data.get('summary', '')).strip()
        
        # Handle keywords - ensure it's a list of strings
        keywords = topic_data.get('keywords', [])
        if isinstance(keywords, str):
            # Split comma-separated keywords: "word1, word2" → ["word1", "word2"]
            keywords = [k.strip() for k in keywords.split(',') if k.strip()]
        elif not isinstance(keywords, list):
            keywords = []
        else:
            # Ensure all keywords are strings
            keywords = [str(k).strip() for k in keywords if k]
        
        # Only add topics with meaningful content
        if len(title) > 3 and len(summary) > 10:  # Basic validation
            topics.append({
                "id": topic_id,
                "title": title,
                "summary": summary,
                "keywords": keywords
            })
        else:
            logger.debug(f"Skipping topic {topic_id}: insufficient content")
    
    # Extract global summary
    global_summary = str(data.get('global_summary', '')).strip()
    if not global_summary:
        # Create a fallback summary from the first few topics
        if topics:
            topic_titles = [t['title'] for t in topics[:3]]
            global_summary = f"本講座涵蓋{len(topics)}個主要主題，包括{'、'.join(topic_titles)}{'等' if len(topics) > 3 else ''}重要內容。"
        else:
            global_summary = "無法從內容生成摘要。"
    
    # Extract key takeaways
    key_takeaways = []
    raw_takeaways = data.get('key_takeaways', [])
    
    if isinstance(raw_takeaways, str):
        # Handle string input - split by newlines or bullets
        lines = [line.strip() for line in raw_takeaways.split('\n') if line.strip()]
        for line in lines:
            # Remove common bullet markers: •, -, *, numbers, etc.
            clean_line = re.sub(r'^[\s•\-*\d\.\)]+', '', line).strip()
            if clean_line:
                key_takeaways.append(clean_line)
    elif isinstance(raw_takeaways, list):
        for item in raw_takeaways:
            if isinstance(item, str) and item.strip():
                key_takeaways.append(item.strip())
            elif isinstance(item, (int, float)):
                key_takeaways.append(str(item))
    
    # Ensure we have at least some takeaways
    if not key_takeaways and topics:
        key_takeaways = [f"掌握{t['title']}的核心概念" for t in topics[:3]]
    
    logger.info(f"Parsed {len(topics)} topics, summary: {len(global_summary)} chars, {len(key_takeaways)} takeaways")
    return topics, global_summary, key_takeaways

def parse_modules_to_topics(modules_analysis: str) -> List[Dict]:
    """
    Parse modules_analysis from chapter generation into topics format.
    
    Args:
        modules_analysis: Text like "模塊1：基礎工具操作 ~ 00:00-00:25 ~ 介面、工具 ~ 理論+演示"
    
    Returns:
        List of topic dicts compatible with topics_list format
    """
    topics = []
    
    if not modules_analysis or not modules_analysis.strip():
        return topics
    
    # Parse each line
    lines = modules_analysis.strip().split('\n')
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-'):
            continue
        
        # Try to parse format: "模塊名稱 ~ 時間範圍 ~ 核心學習點 ~ 教學方法"
        parts = [p.strip() for p in line.split('~')]
        
        if len(parts) >= 3:
            module_name = parts[0]
            time_range = parts[1] if len(parts) > 1 else ""
            learning_points = parts[2] if len(parts) > 2 else ""
            teaching_method = parts[3] if len(parts) > 3 else ""
            
            # Clean module name (remove "模塊1：" prefix if present)
            module_name = re.sub(r'^模塊\d+[：:]\s*', '', module_name)
            
            # Extract keywords from learning points
            keywords = [kw.strip() for kw in learning_points.split('、') if kw.strip()][:5]
            
            # Build summary
            summary_parts = []
            if learning_points:
                summary_parts.append(learning_points)
            if teaching_method:
                summary_parts.append(f"教學方式：{teaching_method}")
            summary = '，'.join(summary_parts)
            
            topics.append({
                "id": str(i).zfill(2),
                "title": module_name,
                "summary": summary,
                "keywords": keywords,
                "time_range": time_range  # Extra info for context
            })
    
    # If parsing failed, try simpler format
    if not topics:
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-'):
                # Use the whole line as title
                topics.append({
                    "id": str(i).zfill(2),
                    "title": line[:100],  # Limit length
                    "summary": "從章節模塊提取的教學主題",
                    "keywords": []
                })
    
    logger.info(f"Parsed {len(topics)} topics from modules_analysis")
    return topics

# ==================== SUGGESTED UNITS EXTRACTION ====================

def extract_suggested_units_from_chapters(
    chapters_dict: Dict[str, str],
    max_units: int = 5
) -> List[Dict]:
    """
    Extract suggested units from generated chapters.
    
    Strategy: Takes the first N chapters and converts them to unit format.
    Works well when chapters represent major learning segments.
    
    Args:
        chapters_dict: Chapter dictionary {"00:05:30": "[單元1：廚具規劃] 內容"} 
                      or {"00:05:30": "章節標題"}
        max_units: Maximum number of suggested units (default: 5)
        
    Returns:
        List of suggested units in client API format:
        [{"UnitNo": 1, "Title": "單元標題", "Time": "00:05:30"}, ...]
    """
    suggested = []
    unit_counter = 1
    
    # Take first N chapters as unit representatives
    for timestamp, title in list(chapters_dict.items())[:max_units]:
        # Extract clean title (remove unit tags if present like [單元1：xxx])
        clean_title = re.sub(r'^\[單元\d+[：:][^\]]+\]\s*', '', title)
        
        # Also remove other common prefixes
        clean_title = re.sub(r'^\[.*?\]\s*', '', clean_title)
        clean_title = clean_title.strip()
        
        # Limit title length for readability
        if len(clean_title) > 50:
            clean_title = clean_title[:47] + "..."
        
        suggested.append({
            "UnitNo": unit_counter,
            "Title": clean_title,
            "Time": timestamp  # Already in HH:mm:ss format from chapters
        })
        unit_counter += 1
    
    logger.info(f"Extracted {len(suggested)} suggested units from {len(chapters_dict)} chapters")
    return suggested


def extract_suggested_units_from_topics(
    topics_list: List[Dict],
    chapters_dict: Optional[Dict[str, str]] = None
) -> List[Dict]:
    """
    Convert topics to suggested units format with optional timestamp matching.
    
    Strategy: Uses topic analysis for unit titles, tries to find matching 
    chapter timestamps based on topic keywords.
    
    Args:
        topics_list: Topics from parse_modules_to_topics() or LLM extraction
                    Format: [{"id": "01", "title": "主題", "summary": "...", "keywords": [...]}]
        chapters_dict: Optional chapter timestamps for time matching
        
    Returns:
        List of suggested units in client API format:
        [{"UnitNo": 1, "Title": "單元標題", "Time": "00:05:30"}, ...]
    """
    suggested = []
    
    for i, topic in enumerate(topics_list[:5], 1):  # Limit to 5 units
        title = topic.get('title', f'主題 {i}')
        
        # Limit title length
        if len(title) > 50:
            title = title[:47] + "..."
        
        # Try to find matching timestamp from chapters
        time = ""
        if chapters_dict:
            # Strategy 1: Look for chapters that mention this topic
            # Extract first 3 significant words from topic title as keywords
            topic_keywords = [w for w in title.split() if len(w) > 1][:3]
            
            for chapter_ts, chapter_title in chapters_dict.items():
                # Check if any topic keyword appears in chapter title
                if any(keyword in chapter_title for keyword in topic_keywords):
                    time = chapter_ts
                    logger.debug(f"Matched topic '{title}' to chapter at {time}")
                    break
        
        # Strategy 2: Fallback to time_range if available in topic
        if not time and topic.get('time_range'):
            time_range = topic['time_range']
            # Extract start time from "00:00-00:25" format
            if '-' in time_range:
                time = time_range.split('-')[0].strip()
            else:
                time = time_range
        
        suggested.append({
            "UnitNo": i,
            "Title": title,
            "Time": time  # Empty string if no match found
        })
    
    logger.info(f"Extracted {len(suggested)} suggested units from {len(topics_list)} topics")
    return suggested
    
def validate_topics_output(data: Dict) -> tuple[bool, List[str]]:
    """
    Validate the structure of parsed topics/summary data.
    Returns (is_valid, list_of_errors)
    """
    errors = []
    
    if not isinstance(data, dict):
        return False, ["Data is not a dictionary"]
    
    # Check required top-level fields
    if 'topics' not in data:
        errors.append("Missing 'topics' field")
    elif not isinstance(data['topics'], list):
        errors.append("'topics' should be a list")
    
    if 'global_summary' not in data:
        errors.append("Missing 'global_summary' field")
    elif not isinstance(data['global_summary'], str):
        errors.append("'global_summary' should be a string")
    
    # Validate individual topics
    if isinstance(data.get('topics'), list):
        for i, topic in enumerate(data['topics']):
            if not isinstance(topic, dict):
                errors.append(f"Topic {i} is not a dictionary")
                continue
                
            if 'title' not in topic:
                errors.append(f"Topic {i} missing 'title'")
            elif not isinstance(topic['title'], str):
                errors.append(f"Topic {i} title is not a string")
                
            if 'summary' not in topic:
                errors.append(f"Topic {i} missing 'summary'")
            elif not isinstance(topic['summary'], str):
                errors.append(f"Topic {i} summary is not a string")
    
    return len(errors) == 0, errors
                                    
# ==================== PROMPT BUILDERS (V2, ASR-first) ====================

def build_mcq_prompt_v2(
    transcript: str,
    *,
    ocr_context: str = "",
    video_title: Optional[str] = None,  # ← ADD THIS PARAMETER
    num_questions: int = 10,
    chapters: Optional[List[Dict]] = None,
    global_summary: str = "",
    hierarchical_metadata: Optional[Dict] = None,
    section_title: Optional[str] = None,      # ← NEW
    units: Optional[List[Dict]] = None        # ← NEW
) -> str:
    """ASR-first MCQ prompt with Bloom structuring, global context, and practical constraints.
       Schema preserved: {"mcqs":[{question, options[A..D], correct_answer, explanation, difficulty, topic, tags, course_type}]}.
    """
    base = num_questions // 3
    rem  = num_questions % 3
    recall_n      = base + (1 if rem >= 1 else 0)
    application_n = base + (1 if rem >= 2 else 0)
    analysis_n    = base

    # Build educational metadata context (must NOT depend on chapters)
    edu_metadata_context = build_educational_metadata_context(section_title, units)

    # Log if metadata provided (also should NOT depend on chapters)
    if section_title or units:
        logger.info("=" * 60)
        logger.info("📚 EDUCATIONAL METADATA FOR MCQ GENERATION")
        if section_title:
            logger.info(f"   📖 Section: {section_title}")
        if units:
            logger.info(f"   📑 Units: {len(units)} predefined units")
            for unit in units:
                logger.info(f"      {unit['UnitNo']}. {unit['Title']}")
        logger.info("=" * 60)

    chap_lines = []
    if chapters:
        # Enhanced question distribution if units provided
        if units and len(units) > 0:
            questions_per_unit = max(1, num_questions // len(units))
            logger.info(f"📊 Suggested distribution: ~{questions_per_unit} questions per unit")
        
        for c in chapters[:18]:
            ts = c.get("ts") or c.get("timestamp") or ""
            title = c.get("title") or ""
            if ts or title:
                chap_lines.append(f"- {ts}：{title}")
    video_title_context = ""
    if video_title:
        # Strip common video extensions
        clean_title = re.sub(r'\.(mp4|avi|mov|mkv|webm|flv|m4v)$', '', video_title, flags=re.IGNORECASE)
        video_title_context = f"📚 此影片檔名為：「{clean_title}」，請參考檔名資訊設計相關題目。\n\n"
                
    global_ctx = []
    if global_summary.strip():
        global_ctx.append(f"- 摘要：{global_summary.strip()}")

    if hierarchical_metadata:
        # Add structured educational context
        course_summary = hierarchical_metadata.get('course_summary', {})
        if course_summary:
            global_ctx.extend([
                f"- 核心主題：{course_summary.get('topic', '')}",
                f"- 關鍵技術：{course_summary.get('core_content', '')}",
                f"- 學習目標：{course_summary.get('learning_objectives', '')}",
                f"- 目標學員：{course_summary.get('target_audience', '')}",
                f"- 難度級別：{course_summary.get('difficulty', '')}"
            ])
        
        # Add structure analysis (PASS 1 insights)
        structure_analysis = hierarchical_metadata.get('structure_analysis', '')
        if structure_analysis:
            # Extract key points from structure (limit length)
            structure_summary = structure_analysis[:500] + "..." if len(structure_analysis) > 500 else structure_analysis
            global_ctx.append(f"- 課程結構分析：{structure_summary}")
        
        # Add module analysis for question distribution guidance
        modules_analysis = hierarchical_metadata.get('modules_analysis', '')
        if modules_analysis:
            global_ctx.append(f"- 教學模組劃分：\n{modules_analysis}")
        
        # Add quality score for context
        quality_score = hierarchical_metadata.get('educational_quality_score', 0)
        if quality_score > 0:
            quality_label = "高" if quality_score > 0.7 else "中" if quality_score > 0.4 else "基礎"
            global_ctx.append(f"- 教育品質評分：{quality_score:.2f} ({quality_label})")
            
    if chap_lines:
        global_ctx.append("- 章節：\n" + "\n".join(chap_lines))
    global_ctx_block = "\n".join(global_ctx) if global_ctx else "（無）"

    ocr_block = ""
    if ocr_context.strip():
        ocr_block = f"## 螢幕文字（OCR，僅作輔助參考）\n{ocr_context}\n\n"
        
    # NEW: Enhanced question distribution logic based on metadata
    if hierarchical_metadata and hierarchical_metadata.get('educational_quality_score', 0) > 0.7:
        # High-quality content: shift toward application/analysis
        if hierarchical_metadata['educational_quality_score'] > 0.8:
            recall_n = max(2, recall_n - 1)
            analysis_n = analysis_n + 1
        # Adjust based on difficulty level
        difficulty = hierarchical_metadata.get('course_summary', {}).get('difficulty', '')
        if difficulty == '高級':
            recall_n = max(1, recall_n - 2)
            analysis_n = analysis_n + 2
            
    # ADD THE LOGGING RIGHT HERE (after distribution logic, before prompt construction)
    if hierarchical_metadata and hierarchical_metadata.get('educational_quality_score', 0) > 0.7:
        original_recall = base + (1 if rem >= 1 else 0)
        original_analysis = base
        if recall_n != original_recall or analysis_n != original_analysis:
            logger.info(f"Adjusted question distribution based on metadata: "
                       f"Recall {original_recall}→{recall_n}, "
                       f"Application {application_n}→{application_n}, "
                       f"Analysis {original_analysis}→{analysis_n}")

    
    # --- KEY ENHANCEMENT: Revised Prompt (WITH ADDITIONS FOR TAGS AND COURSE_TYPE) --- 
    prompt = f"""
{video_title_context}{edu_metadata_context}
你是一位資深的教學設計專家，負責為「{global_summary.splitlines()[0] if global_summary else "各種科目"}」課程設計高品質的多選題（MCQ）。請嚴格依照下列規則出題，並**僅**輸出 JSON。

### 核心原則
- **問題必須基於對逐字稿的整體理解**，而非孤立的單句。首先分析整段文本的 5-8 個核心主題與教學目標，再據此設計題目。
- **測試深度理解**：問題應促使學生應用、分析、評估所學，而不僅是回憶事實。

### 資料來源優先序
1) **ASR 逐字稿（主要依據）**：所有題目必須基於此內容。
2) **OCR 螢幕文字（輔助參考）**：可用於生成有關視覺內容（如軟體界面、圖表、代碼）的題目。若與 ASR 衝突，以 ASR 為準。

### 全域脈絡（Global Context）
{global_ctx_block}

### 出題結構（Bloom's 分類法；合計 {num_questions} 題）
- **Recall（記憶）{recall_n} 題**：測驗關鍵術語、概念、步驟的名稱。*Example: 「Adobe Premiere 中剪輯影片的快捷鍵是什麼？」*
- **Application（應用）{application_n} 題**：測驗在特定情境下運用所學知識的能力。
  - *編程課程：必須包含「預測代碼輸出」或「找出代碼錯誤」的題目。請提供完整代碼片段。*
  - *設計/行銷課程：測驗工具操作（e.g., 「要達成XX效果，下一步該點選哪個工具？」）或策略應用（e.g., 「對於一款新產品，應優先採用哪種行銷策略？」）。*
- **Analysis（分析）{analysis_n} 題**：測驗比較、對照、解釋概念和推理的能力。*Example: 「為什麼講師建議使用 A 方法而不是 B 方法？」、「這個設計原則背後的目的是什麼？」*

### 題目品質指引
- **選項設計**：生成 4 個具備「迷惑性」的選項。錯誤選項必須基於**常見的學生錯誤、實務上的誤解或容易混淆的概念**。避免無關或明顯錯誤的玩笑式選項。
- **難度比例**：30% easy / 40% medium / 30% hard。
- **解釋說明**：每題的解釋必須包含「為何正確」以及「常見的錯誤選擇及其原因」。
- **主題標籤**：`topic` 字段應標明該題測驗的具體知識點（e.g., `Python列表索引`, `色彩理論`, `Facebook廣告受眾設定`）。
- **標籤生成**：為每題生成 3-5 個相關標籤（tags），涵蓋核心概念、技術、應用場景。標籤應具體且有助於分類和搜索。
- **課程類型判斷**：根據題目內容自動判斷並標記課程類型（course_type）。

### 課程類型分類指南
- **設計**：涉及視覺設計、UI/UX、色彩理論、排版、創意軟體（Photoshop、Illustrator、Figma等）
- **程式**：涉及編程語言、演算法、資料結構、軟體開發、API、框架、資料庫
- **數學**：涉及數學運算、公式、定理、統計、微積分、幾何、代數
- **語言**：涉及語言學習、文法、詞彙、寫作、翻譯、口語表達
- **商業**：涉及管理、行銷、財務、經濟、策略、創業、商業模式
- **科學**：涉及物理、化學、生物、地球科學、實驗方法、科學理論
- **其他**：不屬於以上類別的課程內容

### 輸出格式（僅 JSON）
```json
{{
  "mcqs": [
    {{
      "question": "問題（繁體中文）",
      "options": ["選項A", "選項B", "選項C", "選項D"],
      "correct_answer": "A|B|C|D",
      "explanation": "為何正確＋常見誤解",
      "difficulty": "easy|medium|hard",
      "topic": "主題/概念",
      "tags": ["標籤1", "標籤2", "標籤3"],
      "course_type": "設計|程式|數學|語言|商業|科學|其他"
    }}
  ]
}}

### 輸入資料
## ASR 逐字稿（主要依據）
{transcript}

{ocr_block}
"""
    return prompt


def build_lecture_notes_prompt_v2(
    transcript: str,
    *,
    ocr_context: str = "",
    num_pages: int = 5,
    chapters: Optional[List[Dict]] = None,
    topics: Optional[List[Dict]] = None,
    video_title: Optional[str] = None,
    global_summary: str = "",
    hierarchical_metadata: Optional[Dict] = None,
    section_title: Optional[str] = None,
    units: Optional[List[Dict]] = None
) -> str:
    """ASR-first lecture notes prompt. Transforms transcripts into structured, hierarchical study guides.
       Schema: sections[{title, content, key_points[], examples[]}], summary, key_terms[]
    """
    edu_metadata_context = build_educational_metadata_context(section_title, units)

    # Log if metadata provided
    if section_title or units:
        logger.info("=" * 60)
        logger.info("📚 EDUCATIONAL METADATA FOR LECTURE NOTES")
        if section_title:
            logger.info(f"   📖 Section: {section_title}")
        if units:
            logger.info(f"   📑 Units: {len(units)} predefined units")
            for unit in units:
                logger.info(f"      {unit.get('UnitNo')}. {unit.get('Title')}")
        logger.info("=" * 60)

    # Topics snippet
    topics_snippet = ""
    if topics:
        lines = []
        for i, t in enumerate(topics, 1):
            tid = t.get("id", str(i).zfill(2))
            title = t.get("title", "")
            summ = t.get("summary", "")
            lines.append(f"{tid}. {title}：{summ}")
        topics_snippet = "\n".join(lines)

    # Chapters snippet
    chap_lines = []
    if chapters:
        for c in chapters[:18]:
            ts = c.get("ts") or c.get("timestamp") or ""
            title = c.get("title", "")
            if ts or title:
                chap_lines.append(f"- {ts}：{title}")

    # Global context block
    global_ctx = []
    if (video_title or "").strip():
        global_ctx.append(f"- 影片/課程標題：{(video_title or '').strip()}")
    if global_summary.strip():
        global_ctx.append(f"- 摘要：{global_summary.strip()}")

    if hierarchical_metadata:
        course_summary = hierarchical_metadata.get("course_summary", {}) or {}
        if course_summary:
            global_ctx.extend([
                f"- 核心主題：{course_summary.get('topic', '')}",
                f"- 關鍵技術：{course_summary.get('core_content', '')}",
                f"- 學習目標：{course_summary.get('learning_objectives', '')}",
                f"- 目標學員：{course_summary.get('target_audience', '')}",
                f"- 難度級別：{course_summary.get('difficulty', '')}",
            ])

        structure_analysis = hierarchical_metadata.get("structure_analysis", "") or ""
        if structure_analysis:
            structure_summary = structure_analysis[:500] + "..." if len(structure_analysis) > 500 else structure_analysis
            global_ctx.append(f"- 課程架構：{structure_summary}")

        modules_analysis = hierarchical_metadata.get("modules_analysis", "") or ""
        if modules_analysis:
            global_ctx.append(f"- 模組架構：\n{modules_analysis}")

    if chap_lines:
        global_ctx.append("- 章節：\n" + "\n".join(chap_lines))
    if topics_snippet:
        global_ctx.append("- 主題大綱：\n" + topics_snippet)

    global_ctx_block = "\n".join(global_ctx) if global_ctx else "（無）"

    # OCR block
    ocr_block = ""
    if ocr_context.strip():
        ocr_block = f"## 螢幕文字（OCR，僅作輔助參考）\n{ocr_context}\n\n"

    min_words = num_pages * 400
    max_words = (num_pages + 1) * 350

    prompt = f"""
{edu_metadata_context}
你是一位資深的課程編輯和教學設計專家。你的任務是將逐字稿**轉化、提煉、重構**成「可複習、可照做」的終極講義。請**只輸出 JSON**。

### 核心原則（必須遵守）
1) **ASR-first**：以 ASR 逐字稿為主要依據；OCR 僅輔助描述畫面內容。衝突時以 ASR 為準。
2) **重構，勿抄寫**：刪除贅詞與離題內容，可重排順序以提升學習敘事。
3) **可掃讀**：標題層級清楚、列表化、步驟化；60 秒內能定位主題。
4) **可操作**：每節都要產出「可以照做」的步驟與例子，不要只有概述。

### 全域脈絡（Global Context）
{global_ctx_block}

### 章節/Units 使用規則（非常重要）
- **Units/章節只能用來決定分段與排序**。
- **禁止**把 Units 標題改寫成「本節介紹…」就結束。
- 每一節都必須包含：步驟、注意事項、陷阱、技巧、應用情境、例子（含程式碼若適用）。

---

## ✅ 每個 section.content 必須照下面模板輸出（標題不可改名，不可省略）

### 課程目標與概述
（2-4 行，描述本節學什麼、為什麼重要）

### 核心概念講解
- **術語/概念**：定義（務必清楚）
- **術語/概念**：定義

### 操作指南與實例
1) 步驟…
2) 步驟…
- 若是編程/前端/資料處理（HTML/JS/CSV/JSON/Chart.js/FileReader/事件監聽等），本小節**必須至少包含 1 個可執行的 code block**（```html / ```javascript / ```python）。

### ❌ 常見錯誤與陷阱
- 錯誤：… → 後果：…
- 錯誤：… → 後果：…

### ✅ 最佳實踐與技巧
- 技巧：… → 原因/效果：…
- 技巧：… → 原因/效果：…

### 💡 真實應用場景
- 情境：… → 如何應用：…

---

## ✅ 輸出硬性規則（違反視為失敗）
1) 每個 section 的 content 必須包含上述 **6 個小節標題**（不可省略）。
2) 若逐字稿沒有講到某小節：可根據上下文**合理推斷**並保持簡短；真的無法推斷就寫「（本節無）」但**不可刪標題**。
3) **程式/前端相關內容**：`### 操作指南與實例` 必須至少 1 個可執行程式碼區塊（禁止只有偽碼）。
4) ❌/✅/💡 的最低要求：
   - ❌ 至少 2 條（錯在哪 + 後果）
   - ✅ 至少 2 條（可操作技巧）
   - 💡 至少 1 條（真實情境）
5) `key_points`：每節 2–3 條，必須可用來出題（避免空泛）。
6) `examples`：每節 3–5 條具體例子：
   - 編程課程：至少 1 條含 code（可與 content 重複或補充）
   - 非編程課程：至少 1 條提供實務案例

---

### 輸出格式（嚴格 JSON；只輸出 JSON）
```json
{{
  "sections": [
    {{
      "title": "層級化標題（例：'1.2 事件監聽：click 與回呼函式'）",
      "content": "必須包含 6 個小節標題的 Markdown 內容",
      "key_points": ["2-3 條可複習考點"],
      "examples": ["3-5 條具體例子（編程需至少 1 條含 code）"]
    }}
  ],
  "summary": "全文過去式總結：3-5 個收穫 + 後續行動建議",
  "key_terms": [
    {{ "term": "術語1", "definition": "清晰定義" }},
    {{ "term": "術語2", "definition": "清晰定義" }}
  ]
}}
```
字數建議: {min_words}–{max_words}（軟限制）。品質和清晰度優先於嚴格遵守字數。

### 輸入資料
## ASR 逐字稿（主要依據）
{transcript}

{ocr_block}
"""
    return prompt

# ==================== SYSTEM MESSAGES (ASR-first) ====================
MCQ_SYSTEM_MESSAGE = (
    "你是一位專業的教學設計專家。你的核心任務是基於對「ASR 逐字稿」的整體理解，為學生設計能測試深度知識應用的高品質多選題。"
    "「OCR 文字」僅作輔助視覺參考。出題時須遵循 Bloom 分類法結構，並確保錯誤選項基於常見誤解。"
    "請嚴格遵守指定的 JSON 輸出格式，且僅輸出 JSON，不做任何其他說明。"
)

NOTES_SYSTEM_MESSAGE = (
    "你是一位專業的課程編輯和教學設計專家。你的任務是將原始逐字稿提煉、重構為結構清晰、極具學習價值的專業講義。"
    "專注於深度理解與邏輯重組，而非簡單抄寫。以『ASR 逐字稿』為核心依據；『OCR 文字』僅作輔助視覺參考，衝突時以 ASR 為準。"
    "請嚴格遵守指定的 JSON 輸出格式，且僅輸出 JSON，不做任何其他說明。"
)

TOPICS_SUMMARY_SYSTEM_MESSAGE = (
    "你是一位專業的教育內容分析助手。"
    "請從教學內容中提取主要主題和概念，並提供簡潔的摘要。"
    "請嚴格輸出指定的 JSON 架構，且僅輸出 JSON。"
)
# ==================== CLIENT INITIALIZATION ====================
def initialize_client(service_type: str, **kwargs) -> Any:
    """Initialize the appropriate LLM client"""
    if service_type == "azure":
        return ChatCompletionsClient(
            endpoint=kwargs["endpoint"],
            credential=AzureKeyCredential(kwargs["key"]),
            api_version=kwargs.get("api_version", "2024-05-01-preview")
        )
    elif service_type == "openai":
        return OpenAI(
            api_key=kwargs["api_key"],
            base_url=kwargs.get("base_url", "https://api.openai.com/v1/")
        )
    else:
        raise ValueError(f"Unknown service type: {service_type}")

# ==================== LLM API CALLS ====================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def call_llm(
    service_type: str,
    client: Any,
    system_message: str,
    user_message: str,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    top_p: float = 0.9,
    force_json: bool = False  # NEW PARAMETER
) -> Any:
    """Call LLM API with retry logic and optional JSON format enforcement"""
    if service_type == "azure":
        # Azure doesn't support response_format yet
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
        # Build kwargs
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        # Add JSON format enforcement if requested
        if force_json and model in ["gpt-4o-mini", "gpt-4-turbo-preview", "gpt-3.5-turbo-1106"]:
            kwargs["response_format"] = {"type": "json_object"}
            logger.info("Using JSON format enforcement for OpenAI API")
        
        response = client.chat.completions.create(**kwargs)
        return response
    else:
        raise ValueError(f"Unknown service type: {service_type}")

def extract_text_from_response(resp, service_type: str) -> str:
    """Handle Azure/OpenAI response shape differences safely."""
    try:
        if service_type == "azure":
            choice_list = getattr(resp, "choices", None)
            if not choice_list:
                return ""
            choice0 = choice_list[0]
            msg = getattr(choice0, "message", None)
            if msg and getattr(msg, "content", None):
                return msg.content
            if isinstance(choice0, dict):
                msg = choice0.get("message") or {}
                return msg.get("content", "") or ""
            return ""
        else:
            return resp.choices[0].message.content
    except Exception:
        logger.exception("Failed to extract content from LLM response")
        return ""

# ==================== SAFE JSON HELPERS ====================
def _safe_load_json(text: str) -> Optional[dict]:
    """Extract JSON from fenced block if present and gently repair common issues."""
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL) or re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = (m.group(1) if m else text).strip()
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        blob2 = (blob
                 .replace("“", '\"').replace("”", '\"')
                 .replace("’", "'").replace("\u0000", ""))
        try:
            return json.loads(blob2)
        except Exception:
            logger.error("JSON parse failed. First 2k chars:\n%s", blob[:2000])
            return None

def _coerce_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def _norm_mcq(d: dict) -> dict:
    """Normalize MCQ schema fields/types."""
    d = dict(d)
    if "correct_option" in d and "correct_answer" not in d:
        d["correct_answer"] = d.pop("correct_option")
    d["options"] = [str(o) for o in _coerce_list(d.get("options"))][:4]
    diff = str(d.get("difficulty", "medium")).lower()
    d["difficulty"] = diff if diff in {"easy", "medium", "hard"} else "medium"
    return d

# ==================== RESPONSE PARSING ====================
def parse_mcq_response(response_text: str, force_traditional: bool = True) -> List[MCQ]:
    """Parse MCQ response from LLM and convert to Traditional if requested"""
    data = _safe_load_json(response_text)
    if not data:
        return []
    
    mcqs: List[MCQ] = []
    for mcq_data in data.get('mcqs', []):
        d = _norm_mcq(mcq_data)
        
        # Extract basic fields
        q = d.get('question', '')
        opts = d.get('options', [])
        exp = d.get('explanation', '')
        topic = d.get('topic', '')
        
        # NEW: Extract tags
        tags = d.get('tags', [])
        if not isinstance(tags, list):
            tags = [tags] if tags else []
        tags = [str(tag).strip() for tag in tags if tag][:5]  # Limit to 5 tags
        
        # NEW: Extract course_type
        course_type = str(d.get('course_type', '其他')).strip()
        valid_types = ['設計', '程式', '數學', '語言', '商業', '科學', '其他']
        if course_type not in valid_types:
            course_type = '其他'
        
        # Apply Traditional Chinese conversion
        if force_traditional:
            q = to_traditional(q)
            opts = [to_traditional(o) for o in opts]
            exp = to_traditional(exp)
            topic = to_traditional(topic)
            tags = [to_traditional(tag) for tag in tags]
            # course_type is already in Traditional
        
        mcqs.append(MCQ(
            question=q,
            options=opts,
            correct_answer=d.get('correct_answer', ''),
            explanation=exp,
            difficulty=d.get('difficulty', 'medium'),
            topic=topic,
            tags=tags,  # NEW
            course_type=course_type  # NEW
        ))
    
    return mcqs

def parse_lecture_notes_response(response_text: str, force_traditional: bool = True) -> Tuple[List[LectureNoteSection], str]:
    """Parse lecture notes response from LLM and convert to Traditional if requested"""
    data = _safe_load_json(response_text)
    if not data:
        return [], ''
    sections: List[LectureNoteSection] = []
    for section_data in data.get('sections', []):
        title = section_data.get('title', '')
        content = section_data.get('content', '')
        key_points = section_data.get('key_points', [])
        examples = section_data.get('examples', [])
        if force_traditional:
            title = to_traditional(title)
            content = to_traditional(content)
            key_points = [to_traditional(x) for x in key_points]
            examples = [to_traditional(x) for x in examples]
        sections.append(LectureNoteSection(
            title=title,
            content=content,
            key_points=key_points,
            examples=examples
        ))
    summary = data.get('summary', '')
    if force_traditional:
        summary = to_traditional(summary)
    return sections, summary
    
def parse_lecture_notes_with_validation(
    response_text: str,
    force_traditional: bool = True,
    run_dir: Optional[Path] = None
) -> Tuple[List[LectureNoteSection], str]:
    """
    Parse lecture notes using Pydantic validation for guaranteed structure.
    Never fails - always returns valid data.
    """
    sections: List[LectureNoteSection] = []
    summary: str = ""
    
    try:
        # Step 1: Extract JSON from response
        json_text = response_text.strip()
        
        # Handle markdown code blocks
        if "```json" in json_text:
            match = re.search(r"```json\s*(.*?)\s*```", json_text, re.DOTALL)
            if match:
                json_text = match.group(1)
        elif "```" in json_text:
            match = re.search(r"```\s*(.*?)\s*```", json_text, re.DOTALL)
            if match:
                json_text = match.group(1)
        
        # Step 2: Parse JSON
        data = json.loads(json_text)
        
        # Step 3: Validate with Pydantic
        validated = ValidatedLectureNotes(**data)
        
        # Step 4: Convert to your dataclass format
        for section in validated.sections:
            title = section.title
            content = section.content
            key_points = section.key_points
            examples = section.examples
            
            # Apply Traditional Chinese conversion if needed
            if force_traditional:
                title = to_traditional(title)
                content = to_traditional(content)
                key_points = [to_traditional(kp) for kp in key_points]
                examples = [to_traditional(ex) for ex in examples]
            
            sections.append(LectureNoteSection(
                title=title,
                content=content,
                key_points=key_points,
                examples=examples
            ))
        
        summary = validated.summary
        if force_traditional:
            summary = to_traditional(summary)
            
        logger.info(f"Successfully validated {len(sections)} lecture sections")
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed at position {e.pos}: {e.msg}")
        if run_dir:
            error_file = run_dir / "notes_json_error.txt"
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"JSON Error: {e}\n\nRaw response:\n{response_text}")
        
        # Create fallback content from raw text
        sections, summary = _create_fallback_notes(response_text, force_traditional)
        
    except ValidationError as e:
        logger.error(f"Pydantic validation failed: {e}")
        if run_dir:
            error_file = run_dir / "notes_validation_error.txt"
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"Validation Error: {e}\n\nRaw response:\n{response_text}")
        
        # Try to use partial data if available
        try:
            if isinstance(data, dict):
                sections, summary = _extract_partial_notes(data, force_traditional)
            else:
                sections, summary = _create_fallback_notes(response_text, force_traditional)
        except:
            sections, summary = _create_fallback_notes(response_text, force_traditional)
    
    except Exception as e:
        logger.error(f"Unexpected error in lecture notes parsing: {e}")
        sections, summary = _create_fallback_notes(response_text, force_traditional)
    
    # Ensure we always have at least one section
    if not sections:
        sections = [LectureNoteSection(
            title="講義內容",
            content="無法生成完整講義內容",
            key_points=["請參考課程錄影"],
            examples=[]
        )]
    
    if not summary:
        summary = "講義摘要生成中發生錯誤"
    
    return sections, summary

def _extract_partial_notes(
    data: dict,
    force_traditional: bool
) -> Tuple[List[LectureNoteSection], str]:
    """Extract whatever valid data we can from partial response"""
    sections = []
    
    # Try to extract sections
    if 'sections' in data and isinstance(data['sections'], list):
        for section_data in data['sections'][:10]:  # Limit to 10 sections
            if not isinstance(section_data, dict):
                continue
                
            try:
                # Use defaults for missing fields
                title = str(section_data.get('title', '未命名章節'))[:300]
                content = str(section_data.get('content', ''))[:10000]
                key_points = section_data.get('key_points', [])
                examples = section_data.get('examples', [])
                
                # Ensure lists
                if not isinstance(key_points, list):
                    key_points = [str(key_points)] if key_points else []
                if not isinstance(examples, list):
                    examples = [str(examples)] if examples else []
                
                # Limit list sizes
                key_points = [str(kp) for kp in key_points[:10] if kp]
                examples = [str(ex) for ex in examples[:5] if ex]
                
                if force_traditional:
                    title = to_traditional(title)
                    content = to_traditional(content)
                    key_points = [to_traditional(kp) for kp in key_points]
                    examples = [to_traditional(ex) for ex in examples]
                
                if title or content:
                    sections.append(LectureNoteSection(
                        title=title,
                        content=content,
                        key_points=key_points,
                        examples=examples
                    ))
            except Exception as e:
                logger.debug(f"Failed to extract section: {e}")
                continue
    
    # Extract summary
    summary = str(data.get('summary', ''))[:1000]
    if force_traditional:
        summary = to_traditional(summary)
    
    return sections, summary

def _create_fallback_notes(
    response_text: str,
    force_traditional: bool
) -> Tuple[List[LectureNoteSection], str]:
    """Create minimal notes from raw text when all parsing fails"""
    # Try to extract any meaningful content from the response
    content = response_text[:5000]  # Use first 5000 chars
    
    if force_traditional:
        content = to_traditional(content)
    
    sections = [
        LectureNoteSection(
            title="講義內容（自動生成）" if force_traditional else "Lecture Notes (Auto-generated)",
            content=content,
            key_points=["系統自動提取的內容，可能不完整"],
            examples=[]
        )
    ]
    
    summary = "由於格式錯誤，系統自動生成了基本講義內容" if force_traditional else "Basic notes auto-generated due to format error"
    
    return sections, summary
    
# ==================== ANSWER DISTRIBUTION & POST-PROCESSING ====================
def regenerate_explanation_with_llm(
    mcq: MCQ,
    *,
    service_type: str,
    client: Any,
    model: str,
    force_traditional: bool = True
) -> None:
    """Regenerate explanation text based on the (possibly shuffled) correct answer/option set."""
    labels = ["A", "B", "C", "D"]
    lines = [f"{labels[i]}. {opt}" for i, opt in enumerate(mcq.options[:4])]
    options_block = "\n".join(lines)

    sys = "你是一位出題老師，負責為測驗題提供解析（繁體中文）。請簡潔並指出常見誤解。"
    prompt = f"""
請根據以下題目與選項，提供中文解析說明為何正確答案是 {mcq.correct_answer}，並簡要說明其他選項為何不正確。
開頭請使用：「正確答案是 {mcq.correct_answer}，因為…」

題目：
{mcq.question}

選項：
{options_block}
""".strip()

    resp = call_llm(
        service_type=service_type,
        client=client,
        system_message=sys,
        user_message=prompt,
        model=model,
        max_tokens=400,
        temperature=0.2,
        top_p=0.9
    )
    text = extract_text_from_response(resp, service_type) or ""
    mcq.explanation = to_traditional(text.strip()) if force_traditional else text.strip()

def shuffle_mcq_options(
    mcqs: List[MCQ],
    *,
    seed: Optional[int] = None,
    regenerate_explanations: bool = False,
    regeneration_cb: Optional[Callable[[MCQ], None]] = None
) -> None:
    """Shuffle options for each MCQ, update correct_answer letter accordingly.
       If regenerate_explanations is True and the correct letter moved,
       call regeneration_cb(mcq) to rebuild the explanation.
    """
    rng = random.Random(seed) if seed is not None else random
    for mcq in mcqs:
        if not mcq.options:
            continue
        letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
        old_idx = letter_to_idx.get(mcq.correct_answer, 0)
        correct_text = mcq.options[old_idx] if old_idx < len(mcq.options) else mcq.options[0]
        rng.shuffle(mcq.options)
        mcq.options = mcq.options[:4]
        new_idx = next((i for i, t in enumerate(mcq.options) if t == correct_text), 0)
        new_letter = "ABCD"[new_idx]
        moved = (new_letter != mcq.correct_answer)
        mcq.correct_answer = new_letter
        if regenerate_explanations and moved and regeneration_cb:
            try:
                regeneration_cb(mcq)
            except Exception as e:
                logger.warning(f"Explanation regeneration failed: {e}")
        else:
            mcq.explanation = re.sub(r"正確答案是\s+[A-D]", f"正確答案是 {mcq.correct_answer}", mcq.explanation or "")

def enforce_difficulty_distribution(
    mcqs: List[MCQ],
    target_ratio: Tuple[float, float, float] = (0.3, 0.4, 0.3)
) -> List[MCQ]:
    """Reassign difficulty labels to approximate target ratios using a simple proxy (explanation length)."""
    total = len(mcqs) or 1
    target_easy   = round(target_ratio[0] * total)
    target_medium = round(target_ratio[1] * total)
    target_hard   = total - target_easy - target_medium
    ranked = sorted(mcqs, key=lambda q: len(q.explanation or ""))
    for idx, q in enumerate(ranked):
        if idx < target_easy:
            q.difficulty = "easy"
        elif idx < target_easy + target_medium:
            q.difficulty = "medium"
        else:
            q.difficulty = "hard"
    return mcqs

def postprocess_mcqs(
    mcqs: List[MCQ],
    *,
    shuffle: bool,
    regenerate_explanations: bool,
    enforce_difficulty: bool,
    seed: Optional[int],
    service_type: Optional[str],
    client: Optional[Any],
    model: Optional[str],
    force_traditional: bool
) -> List[MCQ]:
    """Apply optional post-processing steps controlled by function parameters."""
    if enforce_difficulty:
        enforce_difficulty_distribution(mcqs)
    if shuffle:
        def regen_cb(m: MCQ):
            if regenerate_explanations and service_type and client and model:
                regenerate_explanation_with_llm(
                    m,
                    service_type=service_type,
                    client=client,
                    model=model,
                    force_traditional=force_traditional
                )
        shuffle_mcq_options(
            mcqs,
            seed=seed,
            regenerate_explanations=regenerate_explanations,
            regeneration_cb=regen_cb if (regenerate_explanations and service_type and client and model) else None
        )
    return mcqs

# ==================== OUTPUT ADAPTERS (Hook points for legacy formats) ====================
def result_to_simple_json(result: EducationalContentResult) -> dict:
    return {
        "mcqs": [
            {
                "question": m.question,
                "options": m.options,
                "correct_answer": m.correct_answer,
                "explanation": m.explanation,
                "difficulty": m.difficulty,
                "topic": m.topic,
            }
            for m in result.mcqs
        ],
        "lecture_notes": [
            {
                "title": s.title,
                "content": s.content,
                "key_points": s.key_points,
                "examples": s.examples,
            }
            for s in result.lecture_notes
        ],
        "summary": result.summary,
    }

def result_to_markdown(result: EducationalContentResult) -> str:
    lines = ["# 測驗題 (MCQs)", ""]
    for i, m in enumerate(result.mcqs, 1):
        lines.append(f"## Q{i}. {m.question}")
        for idx, opt in enumerate(m.options[:4]):
            lines.append(f"- {'ABCD'[idx]}. {opt}")
        lines.append(f"**正確答案**：{m.correct_answer}")
        if m.explanation:
            lines.append(f"**解析**：{m.explanation}")
        lines.append(f"**難度**：{m.difficulty}　**主題**：{m.topic}")
        lines.append("")
    lines.append("# 講義筆記")
    for s in result.lecture_notes:
        lines.append(f"## {s.title}")
        lines.append(s.content or "")
        if s.key_points:
            lines.append("**重點：**")
            for k in s.key_points:
                lines.append(f"- {k}")
        if s.examples:
            lines.append("**範例：**")
            for ex in s.examples:
                lines.append(f"- {ex}")
        lines.append("")
    lines.append("## 摘要")
    lines.append(result.summary or "")
    return "\n".join(lines)

def _mcqs_as_items(mcqs: List[MCQ]) -> List[dict]:
    items = []
    for i, m in enumerate(mcqs, 1):
        items.append({
            "QuestionId": f"Q{str(i).zfill(3)}",
            "QuestionText": m.question,
            "Options": [
                {"Label": "A", "Text": m.options[0] if len(m.options) > 0 else ""},
                {"Label": "B", "Text": m.options[1] if len(m.options) > 1 else ""},
                {"Label": "C", "Text": m.options[2] if len(m.options) > 2 else ""},
                {"Label": "D", "Text": m.options[3] if len(m.options) > 3 else ""},
            ],
            "CorrectAnswer": m.correct_answer,
            "Explanation": m.explanation,
            "Difficulty": {"easy": "簡單", "medium": "中等", "hard": "困難"}.get(m.difficulty, m.difficulty),
            "Topic": m.topic,
        })
    return items

def result_to_pipeline_like(
    result: EducationalContentResult,
    *,
    num_questions: int,
    num_pages: int,
    meta: Optional[dict] = None
) -> dict:
    meta = meta or {}
    return {
        "success": True,
        "qa_and_notes": {
            "questions": _mcqs_as_items(result.mcqs),
            "lecture_notes": {
                "sections": [
                    {
                        "title": s.title,
                        "content": s.content,
                        "key_points": s.key_points,
                        "examples": s.examples
                    } for s in result.lecture_notes
                ],
                "summary": result.summary
            }
        },
        "summary": {
            "questions_generated": len(result.mcqs),
            "lecture_notes_pages": num_pages,
        },
        "pipeline_info": {**meta}
    }


# ==================== LEGACY CLIENT ADAPTER ====================

def result_to_legacy_client_format(
    result: EducationalContentResult,
    *,
    id: str,
    team_id: str,
    section_no: int,
    created_at: str,
    chapters: Optional[List[Dict]] = None,
    original_units: Optional[List[Dict]] = None,      # ← NEW
    suggested_units: Optional[List[Dict]] = None      # ← NEW
) -> dict:
    """
    Convert to client's expected API format with proper Options structure,
    Tags, CourseType fields, and Units/SuggestedUnits.
    
    Args:
        result: Educational content result with MCQs and notes
        id: Video/section ID
        team_id: Team identifier
        section_no: Section number
        created_at: ISO timestamp
        chapters: Optional chapter list for response
        original_units: Original units from client (pass-through)
        suggested_units: AI-generated suggested units
    
    Returns:
        Dictionary in client API format ready for webhook POST
    """
    
    # Difficulty mapping to Chinese
    difficulty_map = {
        "easy": "簡單",
        "medium": "中等",
        "hard": "困難"
    }
    
    # Transform questions to client format
    client_questions = []
    for i, mcq in enumerate(result.mcqs, start=1):
        # Ensure we have tags (fallback if needed)
        tags = mcq.tags if hasattr(mcq, 'tags') and mcq.tags else []
        if not tags and mcq.topic:
            # Fallback: extract from topic if no tags
            tags = [mcq.topic]
        
        # Ensure we have course_type
        course_type = mcq.course_type if hasattr(mcq, 'course_type') else '其他'
        
        client_questions.append({
            "QuestionId": f"Q{str(i).zfill(3)}",
            "QuestionText": mcq.question,
            "Options": [
                {"Label": label, "Text": text}
                for label, text in zip(["A", "B", "C", "D"], mcq.options)
            ],
            "CorrectAnswer": mcq.correct_answer,
            "Explanation": mcq.explanation,
            "Tags": tags[:5],  # Limit to 5 tags
            "Difficulty": difficulty_map.get(mcq.difficulty, mcq.difficulty),
            "CourseType": course_type
        })
    
    # Build lecture notes markdown
    markdown_lines = []
    for section in result.lecture_notes:
        markdown_lines.append(f"## {section.title}")
        markdown_lines.append(section.content)
        if section.key_points:
            markdown_lines.append("### 重點")
            for point in section.key_points:
                markdown_lines.append(f"- {point}")
        if section.examples:
            markdown_lines.append("### 範例")
            for example in section.examples:
                markdown_lines.append(f"- {example}")
        markdown_lines.append("")
    
    markdown_lines.append("## 總結")
    markdown_lines.append(result.summary)
    
    # Build response with Units fields
    response = {
        "Id": id,
        "TeamId": team_id,
        "SectionNo": section_no,
        "CreatedAt": created_at,
        "Questions": client_questions,
        "CourseNote": "\n".join(markdown_lines).strip(),
        "Units": original_units or [],           # ← NEW: Original units (pass-through)
        "SuggestedUnits": suggested_units or []  # ← NEW: AI-generated suggestions
    }
    
    # Add chapters if provided (optional - may be removed in future)
    if chapters:
        response["chapters"] = chapters
    
    return response

# ==================== ASR PREPROCESSING ====================
def preprocess_asr_text(raw_asr_text: str, min_chunk_duration: int = 60, max_gap: int = 10) -> str:
    """Preprocess raw ASR text by combining lines into meaningful chunks."""
    lines = raw_asr_text.strip().split('\n')
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_start_time: Optional[int] = None
    current_end_time: Optional[int] = None
    for line in lines:
        if not line.strip():
            continue
        if ':' in line and len(line.split(':', 1)) > 1:
            time_part, content = line.split(':', 1)
            time_part = time_part.strip()
            content = content.strip()
            if not content:
                continue
            try:
                if time_part.count(':') == 2:  # HH:MM:SS
                    h, m, s = map(int, time_part.split(':'))
                    timestamp_sec = h * 3600 + m * 60 + s
                elif time_part.count(':') == 1:  # MM:SS
                    m, s = map(int, time_part.split(':'))
                    timestamp_sec = m * 60 + s
                else:
                    continue
            except ValueError:
                continue
            if current_start_time is None:
                current_start_time = timestamp_sec
                current_end_time = timestamp_sec
                current_chunk = [content]
            elif timestamp_sec - current_end_time > max_gap:
                if current_chunk and current_end_time - current_start_time >= min_chunk_duration:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(f"[{sec_to_hms(current_start_time)}-{sec_to_hms(current_end_time)}] {chunk_text}")
                current_start_time = timestamp_sec
                current_end_time = timestamp_sec
                current_chunk = [content]
            else:
                current_chunk.append(content)
                current_end_time = timestamp_sec
        else:
            if current_chunk:
                current_chunk.append(line.strip())
    if current_chunk and current_end_time is not None and current_start_time is not None:
        dur = current_end_time - current_start_time
        if dur >= min_chunk_duration or not chunks:
            chunk_text = ' '.join(current_chunk)
            chunks.append(f"[{sec_to_hms(current_start_time)}-{sec_to_hms(current_end_time)}] {chunk_text}")
    if not chunks:
        cleaned_lines = []
        for line in lines:
            if ':' in line and len(line.split(':', 1)) > 1:
                _, content = line.split(':', 1)
                if content.strip():
                    cleaned_lines.append(content.strip())
        return ' '.join(cleaned_lines)
    return '\n\n'.join(chunks)

# ==================== MAIN FUNCTIONS ====================
def initialize_and_get_client(config: EducationalContentConfig):
    service_type = config.service_type
    model = config.openai_model if service_type == "openai" else config.azure_model
    if service_type == "azure":
        client = initialize_client(
            service_type="azure",
            endpoint=config.azure_endpoint,
            key=config.azure_key,
            api_version=config.azure_api_version
        )
    else:
        client = initialize_client(
            service_type="openai",
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
    return service_type, model, client

from typing import Union

def generate_educational_content(
    raw_asr_text: str,
    ocr_segments: Union[List[Dict], str],
    video_id: str,
    video_title: Optional[str] = None,
    run_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    *,
    # NEW PARAMETERS
    chapters: Optional[Dict[str, str]] = None,  # {"00:10:17": "[課程導入] 課程開始"}
    course_summary: Optional[Dict[str, str]] = None,  # DEPRECATED: Use hierarchical_metadata instead
    hierarchical_metadata: Optional[Dict] = None,  # ← ADD THIS! Full metadata from chapter generation
    section_title: Optional[str] = None,      # ← ADD THIS
    units: Optional[List[Dict]] = None,       # ← ADD THIS
    # Existing parameters
    shuffle_options: bool = False,
    regenerate_explanations: bool = False,
    enforce_difficulty: bool = True,
    shuffle_seed: Optional[int] = None,
    ocr_text_override: Optional[str] = None,
) -> EducationalContentResult:
    """
    Main function to generate educational content from pre-processed segments.
    
    Args:
        hierarchical_metadata: Full metadata from video_chaptering.hierarchical_multipass_generation()
                              Contains: structure_analysis, modules_analysis, course_summary, 
                              educational_quality_score, token_usage
    """
    
    report("initializing", progress_callback)

    if run_dir is None:
        run_dir = Path(f"/tmp/educational_content/{video_id}_{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Starting educational content generation for video {video_id}")
        
        # Log educational metadata if provided
        if section_title or units:
            logger.info("=" * 60)
            logger.info("📚 EDUCATIONAL METADATA RECEIVED")
            if section_title:
                logger.info(f"   📖 Section Title: {section_title}")
            if units:
                logger.info(f"   📑 Units ({len(units)}):")
                for unit in units:
                    logger.info(f"      {unit['UnitNo']}. {unit['Title']}")
            logger.info("=" * 60)

        config = EducationalContentConfig()

        if not validate_config(config):
            raise RuntimeError("Configuration validation failed")

        report("processing_inputs", progress_callback)

        transcript = (raw_asr_text)
        if ocr_text_override is not None:
            ocr_context = ocr_text_override
        elif isinstance(ocr_segments, str):
            ocr_context = ocr_segments
        else:
            ocr_context = "\n".join(
                (seg.get("text") or "").strip()
                for seg in (ocr_segments or [])
                if (seg.get("text") or "").strip()
            )
        
        logger.info(f"ASR-first policy active. Generating {config.max_questions} MCQs and {config.max_notes_pages}p notes.")
        logger.info(f"Preprocessed transcript chars: {len(transcript)}, OCR context chars: {len(ocr_context)}")

        # Save input files
        with open(run_dir / "raw_asr_text.txt", "w", encoding="utf-8") as f:
            f.write(raw_asr_text)
        with open(run_dir / "preprocessed_transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript)
        with open(run_dir / "ocr_segments.json", "w", encoding="utf-8") as f:
            json.dump(ocr_segments, f, ensure_ascii=False, indent=2)
        with open(run_dir / "ocr_context.txt", "w", encoding="utf-8") as f:
            f.write(ocr_context)

        report("initializing_client", progress_callback)
        service_type, model, client = initialize_and_get_client(config)

        # Centralize context budgets per model
        MODEL_BUDGETS = {
            "gpt-4o-mini": 128_000,
            "Meta-Llama-3.1-8B-Instruct": 128_000,
        }
        ctx_budget = MODEL_BUDGETS.get(model, 100_000)

        # ========================================================================
        # Topic Extraction - Use metadata if available, else extract
        # ========================================================================

        # Priority 1: Use full hierarchical_metadata (best)
        if hierarchical_metadata and hierarchical_metadata.get('modules_analysis'):
            logger.info("=" * 60)
            logger.info("📊 Using FULL hierarchical metadata from chapter generation")
            logger.info("   ✅ Skipping LLM call - using existing analysis")
            logger.info("=" * 60)
    
            # Extract rich context from metadata
            structure_analysis = hierarchical_metadata.get('structure_analysis', '')
            modules_analysis = hierarchical_metadata.get('modules_analysis', '')
            course_summary_data = hierarchical_metadata.get('course_summary', {})
            quality_score = hierarchical_metadata.get('educational_quality_score', 0)
    
            # Parse modules into topics
            topics_list = parse_modules_to_topics(modules_analysis)
            
            # Build global summary from multiple sources
            summary_parts = []
            if course_summary_data.get('topic'):
                summary_parts.append(course_summary_data['topic'])
            if course_summary_data.get('core_content'):
                summary_parts.append(f"核心內容包括{course_summary_data['core_content']}")
            if course_summary_data.get('learning_objectives'):
                summary_parts.append(course_summary_data['learning_objectives'])
            
            global_summary = "。".join(summary_parts) if summary_parts else "教學課程內容"
    
            # Extract key takeaways
            key_takeaways = []
            if course_summary_data.get('learning_objectives'):
                key_takeaways.append(course_summary_data['learning_objectives'])
            if structure_analysis and '學習目標' in structure_analysis:
                goals_match = re.search(r'學習目標[：:](.*?)(?:\n|$)', structure_analysis)
                if goals_match:
                    key_takeaways.append(goals_match.group(1).strip())
            
            if not key_takeaways:
                key_takeaways = [f"掌握{t['title']}" for t in topics_list[:3]]
            
            logger.info(f"✅ Extracted from metadata:")
            logger.info(f"   • Topics: {len(topics_list)}")
            logger.info(f"   • Quality score: {quality_score}")
            logger.info(f"   • Summary: {global_summary[:100]}...")

            # Create topics_output for file saving later
            topics_output = f"""Topics extracted from hierarchical metadata (no LLM call needed):

Source: hierarchical_metadata from 3-pass chapter generation
Method: Parsed from modules_analysis
Quality Score: {quality_score:.2f}

Topics ({len(topics_list)}):
{chr(10).join(f"{i+1}. {t['title']} - {t['summary']}" for i, t in enumerate(topics_list))}

Course Summary:
{global_summary}

Key Takeaways:
{chr(10).join(f"- {kt}" for kt in key_takeaways)}

Structure Analysis (excerpt):
{structure_analysis[:500]}...

Modules Analysis:
{modules_analysis}
"""
            
            # Save extracted info for debugging
            with open(run_dir / "metadata_extracted_topics.json", "w", encoding="utf-8") as f:
                json.dump({
                    "topics": topics_list,
                    "global_summary": global_summary,
                    "key_takeaways": key_takeaways,
                    "source": "hierarchical_metadata"
                }, f, ensure_ascii=False, indent=2)

        # Priority 2: Fallback to course_summary only (limited)
        elif course_summary:
            logger.info("=" * 60)
            logger.info("📊 Using LIMITED course_summary (backward compatibility)")
            logger.info("   ⚠️  Consider passing full hierarchical_metadata for better results")
            logger.info("=" * 60)
    
            # Convert course_summary to expected format
            topics_list = []
            if course_summary.get('core_content'):
                core_items = course_summary['core_content'].split('、')
                for i, item in enumerate(core_items[:5], 1):
                    topics_list.append({
                        "id": str(i).zfill(2),
                        "title": item.strip(),
                        "summary": f"課程重點：{item}",
                        "keywords": []
                    })
            
            global_summary = f"{course_summary.get('topic', '')}課程，{course_summary.get('core_content', '')}。{course_summary.get('learning_objectives', '')}"
            key_takeaways = [course_summary.get('learning_objectives', '')] if course_summary.get('learning_objectives') else []
    
            logger.info(f"✅ Extracted from course_summary: {len(topics_list)} topics")
    
            # Create topics_output for file saving later
            topics_output = f"""Topics extracted from course_summary (legacy mode):

Source: course_summary (limited metadata)
Warning: Consider using full hierarchical_metadata for better results

Topics ({len(topics_list)}):  
{chr(10).join(f"{i+1}. {t['title']}" for i, t in enumerate(topics_list))}

Course Summary:
{global_summary}

Key Takeaways:
{chr(10).join(f"- {kt}" for kt in key_takeaways if kt)}
"""
        # Priority 3: Extract topics ourselves (slowest)
        else:
            report("generating_topics_summary", progress_callback)
            logger.info("=" * 60)
            logger.info("📊 NO metadata provided - extracting topics from transcript")
            logger.info("   ⏱️  This will take extra time (~5-10 seconds)")
            logger.info("=" * 60)
            
            # Calculate budget for topic extraction
            topics_prompt_template_tokens = count_tokens_llama(
                build_topics_summary_prompt(transcript="", context=None)
            )
            topics_budget = max(2_000, ctx_budget - topics_prompt_template_tokens)
            topics_transcript = truncate_text_by_tokens(transcript, topics_budget)
    
            # Build context (optional)
            topics_context = {
                "視頻ID": video_id,
                "內容類型": "教學視頻"
            }
         
            # Generate topics prompt
            topics_prompt = build_topics_summary_prompt(
                transcript=topics_transcript,
                video_title=video_title,
                context=topics_context
            )
            logger.info(f"Topics extraction prompt approx tokens: {count_tokens_llama(topics_prompt):,}")
            
            # Call LLM for topics extraction
            topics_response = call_llm(
                service_type=service_type,
                client=client,
                system_message=TOPICS_SUMMARY_SYSTEM_MESSAGE,
                user_message=topics_prompt,
                model=model,
                max_tokens=2048,
                temperature=0.15,
                top_p=0.9
            )
            # Parse the response
            topics_output = extract_text_from_response(topics_response, service_type)
            topics_list, global_summary, key_takeaways = parse_topics_summary_response(topics_output)

            # Log extraction results
            logger.info(f"✅ Extracted via LLM: {len(topics_list)} topics")
                
            # Save topics to file for debugging
            with open(run_dir / "extracted_topics.json", "w", encoding="utf-8") as f:
                json.dump({
                    "topics": topics_list,
                    "global_summary": global_summary,
                    "key_takeaways": key_takeaways,
                    "source": "llm_extraction"
                }, f, ensure_ascii=False, indent=2)
                

        # ========================================================================
        # UPDATED: MCQ Generation with Topics and Summary
        # ========================================================================
        report("generating_mcqs", progress_callback)
        # Format chapters if provided
        formatted_chapters = format_chapters_for_prompt(chapters) if chapters else None

        mcq_prompt_template_tokens = count_tokens_llama(build_mcq_prompt_v2(
            transcript="",
            ocr_context=ocr_context,
            num_questions=config.max_questions,
            chapters=formatted_chapters,
            global_summary=global_summary, 
        ))
        mcq_budget = max(2_000, ctx_budget - mcq_prompt_template_tokens)
        mcq_transcript = truncate_text_by_tokens(transcript, mcq_budget)
        
        final_mcq_prompt = build_mcq_prompt_v2(
            transcript=mcq_transcript,
            ocr_context=ocr_context,
            video_title=video_title,  # ← ADD THIS
            num_questions=config.max_questions,
            chapters=formatted_chapters,
            global_summary=global_summary, 
            hierarchical_metadata=hierarchical_metadata,
            section_title=section_title,      # ← ADD
            units=units                        # ← ADD
        )

        logger.info(f"MCQ prompt approx tokens: {count_tokens_llama(final_mcq_prompt):,}")
        logger.info(f"📚 Generating {config.max_questions} MCQs with ASR-first policy, chapters, and topic context")

        mcq_response = call_llm(
            service_type=service_type,
            client=client,
            system_message=MCQ_SYSTEM_MESSAGE,
            user_message=final_mcq_prompt,
            model=model,
            max_tokens=4096,
            temperature=0.2,
            top_p=0.9
        )
        mcq_output = extract_text_from_response(mcq_response, service_type)
        mcqs = parse_mcq_response(mcq_output, force_traditional=config.force_traditional)

        # Post-processing
        mcqs = postprocess_mcqs(
            mcqs,
            shuffle=shuffle_options,
            regenerate_explanations=regenerate_explanations,
            enforce_difficulty=enforce_difficulty,
            seed=shuffle_seed,
            service_type=service_type,
            client=client,
            model=model,
            force_traditional=config.force_traditional
        )

        # ========================================================================
        # UPDATED: Lecture Notes Generation with Topics and Summary
        # ========================================================================
        report("generating_notes", progress_callback)

        # Build the notes prompt (same as before)
        notes_prompt_template_tokens = count_tokens_llama(build_lecture_notes_prompt_v2(
            transcript="",
            ocr_context=ocr_context,
            num_pages=config.max_notes_pages,
            chapters=None,
            topics=topics_list,
            global_summary=global_summary,
        ))
        notes_budget = max(2_000, ctx_budget - notes_prompt_template_tokens)
        notes_transcript = truncate_text_by_tokens(transcript, notes_budget)

        notes_prompt = build_lecture_notes_prompt_v2(
            transcript=notes_transcript,
            ocr_context=ocr_context,
            video_title=video_title,  # ← ADD THIS
            num_pages=config.max_notes_pages,
            chapters=formatted_chapters,
            topics=topics_list,
            global_summary=global_summary,
            hierarchical_metadata=hierarchical_metadata,
            section_title=section_title,      # ← ADD
            units=units                        # ← ADD
        )
        logger.info(f"📘 Generating {config.max_notes_pages} pages of lecture notes with validation")

        # Call LLM with JSON format enforcement (if using OpenAI)
        notes_response = call_llm(
            service_type=service_type,
            client=client,
            system_message=NOTES_SYSTEM_MESSAGE,
            user_message=notes_prompt,
            model=model,
            max_tokens=8096,
            temperature=0.2,
            top_p=0.9,
            force_json=(service_type == "openai")  # Enable JSON format for OpenAI
        )
        notes_output = extract_text_from_response(notes_response, service_type)

        # Save raw response for debugging
        with open(run_dir / "notes_response.txt", "w", encoding="utf-8") as f:
            f.write(notes_output)
            
        # Use the new validation-based parsing
        lecture_sections, summary = parse_lecture_notes_with_validation(
            notes_output,
            force_traditional=config.force_traditional,
            run_dir=run_dir
        )
        logger.info(f"✅ Generated {len(lecture_sections)} validated lecture sections")

        # ========================================================================
        # Rest of the function remains the same
        # ========================================================================
        report("processing_results", progress_callback)

        result = EducationalContentResult(
            mcqs=mcqs,
            lecture_notes=lecture_sections,
            summary=summary,
            topics=topics_list,           
            key_takeaways=key_takeaways
        )

        # Optional caching
        if config.enable_cache:
            cache_dir = run_dir / "cache"
            cache_dir.mkdir(exist_ok=True)
            mcq_key = get_content_hash(mcq_transcript, ocr_context, "mcq")
            notes_key = get_content_hash(notes_transcript, ocr_context, "notes")
            with open(cache_dir / f"{mcq_key}.json", "w", encoding="utf-8") as f:
                json.dump([vars(x) for x in mcqs], f, ensure_ascii=False, indent=2)
            with open(cache_dir / f"{notes_key}.json", "w", encoding="utf-8") as f:
                json.dump({"sections": [vars(s) for s in lecture_sections], "summary": summary}, f, ensure_ascii=False, indent=2)

        # Persist raw LLM outputs & final results
        with open(run_dir / "mcq_response.txt", "w", encoding="utf-8") as f:
            f.write(mcq_output)
        with open(run_dir / "notes_response.txt", "w", encoding="utf-8") as f:
            f.write(notes_output)
        with open(run_dir / "topics_response.txt", "w", encoding="utf-8") as f:
            f.write(topics_output)  # ← SAVE TOPICS RESPONSE TOO
        with open(run_dir / "final_result.json", "w", encoding="utf-8") as f:
            json.dump({
                "mcqs": [vars(mcq) for mcq in mcqs],
                "lecture_notes": [vars(section) for section in lecture_sections],
                "summary": summary,
                "topics": topics_list,  # ← INCLUDE TOPICS IN FINAL OUTPUT
                "key_takeaways": key_takeaways  # ← INCLUDE KEY TAKEAWAYS TOO
            }, f, ensure_ascii=False, indent=2)

        report("completed", progress_callback)
        logger.info(f"Successfully generated {len(mcqs)} MCQs and {len(lecture_sections)} lecture note sections")
        return result

    except Exception as e:
        logger.error(f"Educational content generation failed: {e}", exc_info=True)
        raise

# ---- Adapter for tasks.py compatibility ----

def process_text_for_qa_and_notes(
    *,
    # Prefer raw_asr_text (matches new tasks.py). Fallback to audio_segments if not provided.
    raw_asr_text: str = "",
    audio_segments: Optional[List[Dict]] = None,
    ocr_segments: Optional[List[Dict]] = None,
    video_title: Optional[str] = None,  # ← ADD THIS
    chapters: Optional[Dict[str, str]] = None,  # ← ADD THIS
    hierarchical_metadata: Optional[Dict] = None,
    section_title: Optional[str] = None,      # ← ADD
    units: Optional[List[Dict]] = None,       # ← ADD
    num_questions: int = 10,
    num_pages: int = 3,
    id: str = "",
    team_id: str = "",
    section_no: int = 0,
    created_at: str = "",
) -> EducationalContentResult:
    """
    Adapter so tasks.py (and other callers) can invoke the generator.
    - If raw_asr_text is provided, we use it directly (ASR-first).
    - Else, we reconstruct a raw ASR string from audio_segments ("HH:MM:SS: text" per line).
    Returns the raw EducationalContentResult object (not the pipeline format).
    """
    ocr_segments = ocr_segments or []

    # 1) Choose ASR source
    if raw_asr_text and raw_asr_text.strip():
        asr_text_for_prompt = raw_asr_text
    else:
        audio_segments = audio_segments or []
        def _line(seg: Dict) -> str:
            ts = sec_to_hms(int(seg.get("start", 0)))
            txt = (seg.get("text") or "").strip()
            return f"{ts}: {txt}" if txt else ""
        asr_text_for_prompt = "\n".join(filter(None, (_line(s) for s in audio_segments)))

    # 2) Honor per-call counts by temporarily overriding env
    old_max_q = os.environ.get("MAX_QUESTIONS")
    old_max_p = os.environ.get("MAX_NOTES_PAGES")
    os.environ["MAX_QUESTIONS"] = str(num_questions)
    os.environ["MAX_NOTES_PAGES"] = str(num_pages)

    try:
        # Just return the raw result object, not the pipeline format
        result = generate_educational_content(
            raw_asr_text=asr_text_for_prompt,   # ← ASR-first (raw string)
            ocr_segments=ocr_segments,          # ← simple OCR (list or string)
            video_id=id or "video",
            video_title=video_title,  # ← ADD THIS
            chapters=chapters,  # ← ADD THIS
            hierarchical_metadata=hierarchical_metadata,
            section_title=section_title,      # ← ADD
            units=units,                       # ← ADD
            run_dir=None,
            progress_callback=None,
            shuffle_options=False,
            regenerate_explanations=False,
            enforce_difficulty=True,
            shuffle_seed=None,
            ocr_text_override=None,
        )
        
        return result  # ← Return the EducationalContentResult object directly

    finally:
        if old_max_q is not None:
            os.environ["MAX_QUESTIONS"] = old_max_q
        else:
            os.environ.pop("MAX_QUESTIONS", None)
        if old_max_p is not None:
            os.environ["MAX_NOTES_PAGES"] = old_max_p
        else:
            os.environ.pop("MAX_NOTES_PAGES", None)

# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_transcript = """
    00:00:03: 今天我們來學習微積分的基本概念。首先，導數表示函數在某一點的瞬時變化率。
    00:01:05: 積分則是導數的逆運算，用來計算面積和累積量。
    """
    sample_ocr = [
        {"start": 0, "end": 10, "text": "導數定義: f'(x) = lim(h→0) [f(x+h)-f(x)]/h"},
        {"start": 60, "end": 70, "text": "積分符號: ∫ f(x) dx"}
    ]

    # Example: turn on shuffling + difficulty enforcement for this run
    result = generate_educational_content(
        raw_asr_text=sample_transcript,
        ocr_segments=sample_ocr,
        video_id="calc_101",
        shuffle_options=True,
        regenerate_explanations=False,
        enforce_difficulty=True,
        shuffle_seed=42
    )

    print(f"Generated {len(result.mcqs)} MCQs and {len(result.lecture_notes)} lecture sections")
    print("Summary:", (result.summary or "")[:120], "…")
