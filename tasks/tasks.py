# tasks/tasks.py

import os
import sys
import json
import uuid
import time
import subprocess
import logging
import re
import codecs
from pathlib import Path
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

# Use the ONE shared Celery instance
from app.celery_app import celery

from app.runpod_controller import get_pod_status, start_pod
from tasks.file_maintenance import clean_old_files
from app.chapter_generation import generate_chapters  # (aka app/video_chaptering.py wrapper)

from .cleaning import *  # re-export tasks so autodiscover finds them
from app.qa_generation import process_text_for_qa_and_notes, result_to_legacy_client_format

# ---------- Optional NumPy patch for old code paths ----------
import numpy as np
if not hasattr(np, "int"):
    np.int = int

# ---------- Env / paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, ".."))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, "..", ".env"))

CLIENT_UPLOAD_API = os.getenv("CLIENT_UPLOAD_API")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RUNPOD_POD_ID = os.getenv("RUNPOD_POD_ID")
DISABLE_RUNPOD_CHECK = os.getenv("DISABLE_RUNPOD_CHECK", "false").lower() == "true"

# Window size for legacy QA chunking (seconds). Default 5 minutes.
QA_WINDOW_SEC = int(os.getenv("QA_WINDOW_SEC", "300"))
WIN_LABEL = f"{QA_WINDOW_SEC // 60}min"

# ---------- OCR strategy (defaults favor aligner-based flow) ----------
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"
# When true, we build QA audio windows from aligner micro-windows instead of 5-min windows
USE_ALIGNED_WINDOWS_ONLY = os.getenv("USE_ALIGNED_WINDOWS_ONLY", "true").lower() == "true"
# If you still want the old 5-min artifacts for debugging, set this to true
WRITE_FIVEMIN_ARTIFACTS = os.getenv("WRITE_FIVEMIN_ARTIFACTS", "false").lower() == "true"

# ---------- Logging ----------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------- Persistence ----------
PERSIST_BASE = os.getenv("PERSIST_BASE", "/workspace")
try:
    os.makedirs(PERSIST_BASE, exist_ok=True)
    _t = os.path.join(PERSIST_BASE, ".write_test")
    with open(_t, "w") as f:
        f.write("ok")
    os.remove(_t)
except Exception:
    logger.warning("⚠️ PERSIST_BASE not writable; falling back to /app")
    PERSIST_BASE = "/app"

CACHE_DIR         = os.path.join(PERSIST_BASE, "video_cache")
RUNS_BASE         = os.path.join(PERSIST_BASE, "runs")
UPLOAD_FOLDER     = os.path.join(PERSIST_BASE, "uploads")
LOGS_DIR          = os.path.join(PERSIST_BASE, "logs")
SENT_PAYLOADS_DIR = os.path.join(PERSIST_BASE, "sent_payloads")
for d in (CACHE_DIR, RUNS_BASE, UPLOAD_FOLDER, LOGS_DIR, SENT_PAYLOADS_DIR):
    os.makedirs(d, exist_ok=True)

logger.info(f"📦 Persistence root: {PERSIST_BASE}")
logger.info(f"🗂  Folders → cache:{CACHE_DIR} runs:{RUNS_BASE} uploads:{UPLOAD_FOLDER} logs:{LOGS_DIR}")

# ==================== Helpers ====================

def download_video(play_url, filename, max_retries=3, timeout=1800):
    """
    Download video with comprehensive logging and validation.
    Uses aria2c for better performance and reliability.
    """
    local_path = os.path.join(UPLOAD_FOLDER, filename)
    
    logger.info("=" * 60)
    logger.info("📥 VIDEO DOWNLOAD STARTING")
    logger.info(f"🔗 Source URL: {play_url}")
    logger.info(f"💾 Target path: {local_path}")
    logger.info(f"📁 Upload folder: {UPLOAD_FOLDER}")
    logger.info("=" * 60)
    
    # Ensure upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    for attempt in range(1, max_retries + 1):
        logger.info(f"🌐 Download Attempt {attempt}/{max_retries}")
        
        try:
            # Check if aria2c is available, fallback to curl
            download_tool = "aria2c"
            try:
                subprocess.run(["which", "aria2c"], check=True, capture_output=True)
                # Use aria2c with optimized settings
                cmd = [
                    "aria2c",
                    "-x", "4",  # 4 parallel connections
                    "-s", "4",  # Split file into 4 segments
                    "-k", "1M",  # 1MB chunk size
                    "--file-allocation=none",  # Faster for SSDs
                    "--console-log-level=warn",  # Less verbose
                    "--summary-interval=10",  # Progress every 10 seconds
                    "-d", UPLOAD_FOLDER,  # Download directory
                    "-o", filename,  # Output filename
                    play_url
                ]
                logger.info(f"📥 Using aria2c for faster download")
            except subprocess.CalledProcessError:
                # Fallback to curl
                download_tool = "curl"
                cmd = ["curl", "-L", "-o", local_path, "--progress-bar", play_url]
                logger.info(f"📥 Using curl (aria2c not found)")
            
            # Start download
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0:
                download_time = time.time() - start_time
                
                # Verify file exists
                if os.path.exists(local_path):
                    file_size = os.path.getsize(local_path)
                    file_size_mb = file_size / (1024 * 1024)
                    
                    # Check if file is not empty
                    if file_size == 0:
                        logger.error(f"❌ Downloaded file is empty (0 bytes)")
                        if attempt < max_retries:
                            time.sleep(attempt * 5)
                            continue
                        raise RuntimeError("Downloaded file is empty")
                    
                    logger.info(f"✅ Download SUCCESSFUL")
                    logger.info(f"📊 File size: {file_size_mb:.2f} MB ({file_size:,} bytes)")
                    logger.info(f"⏱️ Download time: {download_time:.1f} seconds")
                    logger.info(f"📍 File location: {local_path}")
                    
                    # Verify it's a valid video file
                    try:
                        verify_cmd = subprocess.run(
                            ["ffprobe", "-v", "error", "-show_format", "-show_streams", local_path],
                            capture_output=True, text=True, timeout=10
                        )
                        if verify_cmd.returncode == 0:
                            logger.info(f"✅ Video file VERIFIED as valid media")
                            
                            # Extract video info
                            duration_cmd = subprocess.run(
                                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                 "-of", "default=noprint_wrappers=1:nokey=1", local_path],
                                capture_output=True, text=True
                            )
                            if duration_cmd.returncode == 0:
                                duration = float(duration_cmd.stdout.strip())
                                logger.info(f"📹 Video duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
                        else:
                            logger.warning(f"⚠️ File may not be a valid video: {verify_cmd.stderr}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not verify video with ffprobe: {e}")
                    
                    logger.info("=" * 60)
                    return local_path
                else:
                    logger.error(f"❌ File not found after download: {local_path}")
            else:
                logger.error(f"❌ Download failed with {download_tool}")
                logger.error(f"❌ Error output: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Download timeout after {timeout} seconds")
        except Exception as e:
            logger.error(f"❌ Download error: {e}", exc_info=True)
        
        # Retry logic
        if attempt < max_retries:
            wait_time = attempt * 5
            logger.info(f"⏳ Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
            
            # Clean up partial download if exists
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                    logger.info(f"🗑️ Cleaned up partial download")
                except:
                    pass
    
    # All retries exhausted
    logger.error("=" * 60)
    logger.error(f"❌ DOWNLOAD FAILED after {max_retries} attempts")
    logger.error("=" * 60)
    raise RuntimeError(f"Failed to download video after {max_retries} attempts")

def post_to_client_api(payload):
    if not CLIENT_UPLOAD_API:
        logger.warning("⚠️ CLIENT_UPLOAD_API not set. Skipping POST.")
        return
    try:
        pretty = json.dumps(payload, indent=2, ensure_ascii=False)
        logger.info(f"📦 Client_Final_API_Sent:\n{pretty}")
        r = requests.post(CLIENT_UPLOAD_API, headers={"Content-Type": "application/json"}, json=payload)
        logger.info(f"📤 POST status: {r.status_code}")
        r.raise_for_status()
    except Exception as e:
        logger.error(f"❌ POST failed: {e}", exc_info=True)

def transform_chapters_to_units(chapters: dict) -> list:
    """
    Transform chapters from dict format to Units array format.
    
    Args:
        chapters: Dict with timestamp keys and title values
                 {"00:07:54": "[合成技術概述] 影像合成的基本概念", ...}
    
    Returns:
        List of unit dicts in client API format
        [{"UnitNo": 1, "Title": "...", "Time": "00:07:54"}, ...]
    """
    if not chapters:
        logger.warning("No chapters to transform to Units format")
        return []
    
    units = []
    # Sort chapters by timestamp to ensure correct UnitNo ordering
    sorted_chapters = sorted(chapters.items(), key=lambda x: x[0])
    
    logger.info(f"Transforming {len(sorted_chapters)} chapters to Units format")
    
    for idx, (timestamp, title) in enumerate(sorted_chapters, start=1):
        # Clean up title: remove module tag brackets and reformat
        # "[合成技術概述] 影像合成的基本概念" -> "合成技術概述 - 影像合成的基本概念"
        clean_title = title.strip()
        
        if clean_title.startswith('[') and ']' in clean_title:
            bracket_end = clean_title.index(']')
            module_tag = clean_title[1:bracket_end].strip()
            content = clean_title[bracket_end+1:].strip()
            clean_title = f"{module_tag} - {content}" if content else module_tag
        
        units.append({
            "UnitNo": idx,
            "Title": clean_title,
            "Time": timestamp
        })
    
    logger.info(f"✅ Transformed to {len(units)} Units")
    return units

# ---------- ASR (chapter_llama → SingleVideo) ----------

ASR_LINE_RE = re.compile(r"^\s*(\d{2}):(\d{2}):(\d{2})\s*:\s*(.+?)\s*$")

def _hms_to_seconds(h, m, s) -> float:
    return int(h) * 3600 + int(m) * 60 + float(s)

def sec_to_hms(sec: int) -> str:
    """Convert seconds to HH:MM:SS format."""
    if sec < 0:
        sec = 0
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _simple_window_segments(raw_segs, window_sec=300):
    """
    Legacy 5-min windows (fallback / optional artifacts).
    raw_segs: list of {"start": float, "end": float, "text": str}
    Returns: list of {"start": s, "end": e, "text": "..."} windows.
    """
    if not raw_segs:
        return []
    windows = []
    vid_start = raw_segs[0]["start"]
    vid_end = max(s["end"] for s in raw_segs)
    w_start = vid_start
    while w_start < vid_end:
        w_end = min(w_start + window_sec, vid_end)
        buf = [seg["text"] for seg in raw_segs if seg["end"] > w_start and seg["start"] < w_end]
        text = " ".join(buf).strip()
        if text:
            windows.append({"start": w_start, "end": w_end, "text": text})
        w_start = w_end
    return windows

# ---------- OCR / Aligner imports (guarded) ----------
try:
    from chapter_llama.tools.extract.ocr_processor import OCRProcessor
except Exception:
    OCRProcessor = None

try:
    from chapter_llama.tools.extract.ocr_asr_aligner import (
        parse_existing_asr as _aligner_parse_existing_asr,
        get_ocr_context_for_asr as _aligner_get_context_markdown,
    )
except Exception:
    _aligner_parse_existing_asr = None
    _aligner_get_context_markdown = None

def _concat_asr_text_in_window(segs_raw, w_start: float, w_end: float) -> str:
    """Join ASR texts that overlap [w_start, w_end)."""
    buf = []
    for seg in segs_raw:
        if seg["end"] > w_start and seg["start"] < w_end:
            t = seg.get("text", "").strip()
            if t:
                buf.append(t)
    return " ".join(buf).strip()

def _ocr_segments_from_aligner(video_path: str, asr_file: str):
    """
    Returns:
      - ocr_filtered: [{"start": float, "end": float, "text": str}, ...]  (can be [])
      - ocr_raw: [{"start": int, "end": int, "texts": [..]}, ...]         (can be [])
      - ocr_context_md: str                                               (can be "")
      - seg_pairs: List[(start:int, end:int)]                              (ALWAYS computed if ASR exists)
    """
    # 1) Always derive seg_pairs from ASR
    if _aligner_parse_existing_asr is None:
        logger.warning("⚠️ Aligner parse unavailable; cannot build aligned windows.")
        return [], [], "", []

    asr_items = _aligner_parse_existing_asr(Path(asr_file))  # [{"start_sec", "text"}]
    if not asr_items:
        return [], [], "", []

    n = int(os.getenv("OCR_SAMPLE_EVERY_N", "10"))
    sampled = asr_items[::max(1, n)]
    seg_pairs = [(max(0, x["start_sec"] - 5), x["start_sec"] + 15) for x in sampled]

    # 2) Try OCR; if unavailable/failed, just return empty OCR arrays but keep seg_pairs
    ocr_filtered, ocr_raw, ocr_context_md = [], [], ""
    if OCRProcessor is not None:
        try:
            proc = OCRProcessor()
            items = proc.get_text_for_many_segments(video_path=Path(video_path), segments=seg_pairs)
            ocr_raw = items
            for it in items:
                joined = " ".join(it.get("texts", []) or []).strip()
                if joined:
                    ocr_filtered.append({"start": float(it["start"]), "end": float(it["end"]), "text": joined})
            # Pretty markdown optional
            if _aligner_get_context_markdown is not None:
                try:
                    ocr_context_md = _aligner_get_context_markdown(
                        video_path=Path(video_path),
                        asr_file_path=Path(asr_file),
                        ocr_processor=proc
                    )
                except Exception as _e:
                    logger.warning("⚠️ Failed to build OCR context markdown: %s", _e)
        except Exception as e:
            logger.warning("⚠️ OCR processing failed; continuing with ASR-only: %s", e)

    return ocr_filtered, ocr_raw, ocr_context_md, seg_pairs

def chapter_llama_asr_processing_fn(video_path: str, window_sec: int = QA_WINDOW_SEC, do_ocr: bool = True):
    """
    ASR via SingleVideo with comprehensive GPU detection and video validation logging.
    
    Behavior (no legacy fallback):
      - Try to build ASR-anchored micro-windows via the aligner parser (seg_pairs).
      - Build QA audio windows by concatenating ASR text overlapping each seg_pair.
      - Optionally run OCR on those micro-windows (if do_ocr=True & OCR available).
      - If aligner seg_pairs cannot be built OR produce no non-empty windows,
        proceed with ASR-only by using per-line ASR segments (segs_raw) as the QA audio windows.
      - Never compute or use legacy 5-minute windows.
    """
    
    # ============= STAGE 1: VIDEO VALIDATION & GPU DETECTION =============
    logger.info("=" * 60)
    logger.info("🎯 STARTING ASR PROCESSING")
    logger.info("=" * 60)
    
    # Check video file exists and is valid
    if not os.path.exists(video_path):
        logger.error(f"❌ Video file NOT FOUND: {video_path}")
        return {
            "success": False,
            "audio_segments": [],
            "ocr_segments": [],
            "method": "chapter_llama_asr",
            "error": "Video file not found"
        }
    
    # Get video file info
    video_size = os.path.getsize(video_path)
    video_size_mb = video_size / (1024 * 1024)
    logger.info(f"📹 Processing video: {video_path}")
    logger.info(f"📊 Video size: {video_size_mb:.2f} MB ({video_size:,} bytes)")
    
    # Verify it's a valid video file using ffprobe
    try:
        verify_cmd = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams", video_path],
            capture_output=True, text=True, timeout=10
        )
        if verify_cmd.returncode == 0:
            logger.info(f"✅ Video file VERIFIED as valid media")
        else:
            logger.warning(f"⚠️ Video validation warning: {verify_cmd.stderr}")
    except Exception as e:
        logger.warning(f"⚠️ Could not verify video with ffprobe: {e}")
    
    # GPU Status Check
    logger.info("-" * 60)
    logger.info("🖥️ GPU/CUDA STATUS CHECK")
    logger.info("-" * 60)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        logger.info(f"🎮 CUDA Available: {cuda_available}")
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_memory_gb = gpu_props.total_memory / (1024**3)
            
            logger.info(f"🎮 GPU Count: {gpu_count}")
            logger.info(f"🎮 GPU Name: {gpu_name}")
            logger.info(f"🎮 GPU Memory: {gpu_memory_gb:.2f} GB")
            logger.info(f"🎮 CUDA Version: {torch.version.cuda}")
            
            # Check current GPU memory usage
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            free = gpu_memory_gb - reserved
            
            logger.info(f"📊 GPU Memory Allocated: {allocated:.2f} GB")
            logger.info(f"📊 GPU Memory Reserved: {reserved:.2f} GB")
            logger.info(f"📊 GPU Memory Free: {free:.2f} GB")
            
            # Test GPU is actually working
            try:
                test_tensor = torch.randn(100, 100).cuda()
                logger.info("✅ GPU compute test: PASSED")
            except Exception as gpu_test_error:
                logger.error(f"❌ GPU compute test FAILED: {gpu_test_error}")
                cuda_available = False
        else:
            logger.warning("⚠️ CUDA NOT AVAILABLE - Running on CPU (will be SLOW!)")
            logger.warning("⚠️ This will significantly increase processing time")
            
        # Check environment variables
        whisper_device = os.getenv("WHISPER_DEVICE", "cuda")
        cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "not set")
        logger.info(f"🎤 WHISPER_DEVICE env: {whisper_device}")
        logger.info(f"🎤 CUDA_VISIBLE_DEVICES env: {cuda_visible}")
        
        # Warn if mismatch
        if whisper_device == "cuda" and not cuda_available:
            logger.warning("⚠️ WHISPER_DEVICE set to 'cuda' but no GPU available!")
            
    except ImportError as e:
        logger.error(f"❌ PyTorch not properly installed: {e}")
        cuda_available = False
    except Exception as e:
        logger.error(f"❌ Error checking GPU status: {e}")
        cuda_available = False
    
    logger.info("=" * 60)
    
    # ============= STAGE 2: ASR PROCESSING =============
    try:
        # Load ASR from chapter_llama
        logger.info("🚀 Loading chapter_llama SingleVideo processor...")
        try:
            from chapter_llama.src.data.single_video import SingleVideo
        except ImportError:
            from chapter_llama.src.data.single_video import SingleVideo
        
        t0 = time.time()
        
        logger.info(f"📝 Initializing SingleVideo for: {Path(video_path).name}")
        sv = SingleVideo(Path(video_path))
        vid_id = next(iter(sv))
        logger.info(f"📝 Video ID: {vid_id}")
        
        # Start ASR transcription
        logger.info("🎤 Starting ASR transcription (this may take several minutes)...")
        logger.info(f"🎤 Using device: {'GPU' if cuda_available else 'CPU'}")
        
        asr_start = time.time()
        asr_text = sv.get_asr(vid_id)             # "HH:MM:SS: text"
        asr_elapsed = time.time() - asr_start
        
        duration = float(sv.get_duration(vid_id))  # seconds
        
        logger.info(f"✅ ASR transcription COMPLETE in {asr_elapsed:.1f} seconds")
        logger.info(f"📊 Video duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"📊 ASR text length: {len(asr_text):,} characters")
        logger.info(f"📊 ASR lines: {len(asr_text.splitlines())} lines")
        
        # Sample first few lines for verification
        sample_lines = asr_text.splitlines()[:3]
        logger.info("📝 Sample ASR output (first 3 lines):")
        for line in sample_lines:
            logger.info(f"   {line[:100]}...")  # First 100 chars of each line
        
        # Ensure raw ASR artifacts exist for downstream consumers
        asr_cache_dir = Path("outputs/inference") / Path(video_path).stem
        asr_cache_dir.mkdir(parents=True, exist_ok=True)
        asr_file = asr_cache_dir / "asr.txt"
        dur_file = asr_cache_dir / "duration.txt"
        
        logger.info(f"💾 Saving ASR artifacts to: {asr_cache_dir}")
        
        if not asr_file.exists():
            with codecs.open(asr_file, "w", encoding="utf-8") as f:
                f.write(asr_text if asr_text.endswith("\n") else asr_text + "\n")
            logger.info(f"✅ Saved ASR text to: {asr_file}")
        
        if not dur_file.exists():
            dur_file.write_text(str(duration), encoding="utf-8")
            logger.info(f"✅ Saved duration to: {dur_file}")
        
        # Try to surface VAD coverage metrics saved by SingleVideo/ASRProcessor
        metrics_path = asr_cache_dir / "asr_metrics.json"
        speech_duration = removed_duration = speech_ratio = removed_ratio = None
        
        if metrics_path.exists():
            try:
                m = json.loads(metrics_path.read_text(encoding="utf-8"))
                speech_duration = float(m.get("speech_duration", 0.0))
                removed_duration = float(m.get("removed_duration", 0.0))
                duration_f = float(m.get("duration", duration) or duration)
                speech_ratio = float(m.get("speech_ratio", (speech_duration / duration_f) if duration_f > 0 else 0.0))
                removed_ratio = float(m.get("removed_ratio", 1.0 - speech_ratio))
                
                logger.info("-" * 60)
                logger.info("🧪 VAD (Voice Activity Detection) Summary:")
                logger.info(
                    f"   Speech kept: {speech_ratio * 100.0:.1f}% ({speech_duration:.1f}s)"
                )
                logger.info(
                    f"   Silence removed: {removed_ratio * 100.0:.1f}% ({removed_duration:.1f}s)"
                )
                logger.info(f"   Total duration: {duration_f:.1f}s")
                logger.info("-" * 60)
            except Exception as _e:
                logger.warning(f"⚠️ Failed to read asr_metrics.json: {_e}")
        
        # ============= STAGE 3: PARSE ASR TEXT =============
        logger.info("📝 Parsing ASR text into segments...")
        
        # Parse raw ASR lines -> per-line segments (start), then infer end
        segs_raw = []
        for line in asr_text.splitlines():
            m = ASR_LINE_RE.match(line)
            if not m:
                continue
            hh, mm, ss, text = m.groups()
            start = _hms_to_seconds(hh, mm, ss)
            text = text.strip()
            if text:
                segs_raw.append({"start": float(start), "text": text})
        
        if not segs_raw:
            logger.error("❌ No ASR lines could be parsed")
            logger.error("❌ VAD removed all audio - video likely has no valid speech/audio")
            return {
                "success": False,
                "audio_segments": [],
                "ocr_segments": [],
                "method": "chapter_llama_asr",
                "error": "no_valid_audio",
                "error_message": "Video has no valid audio or speech content"
            }
        
        logger.info(f"✅ Parsed {len(segs_raw)} ASR segments")
        
        # Infer end times for segments
        for i in range(len(segs_raw)):
            segs_raw[i]["end"] = float(segs_raw[i+1]["start"]) if i < len(segs_raw) - 1 else duration
        
        # ============= STAGE 4: BUILD ALIGNED WINDOWS (OCR OPTIONAL) =============
        logger.info("-" * 60)
        logger.info("🔧 Building aligned windows for QA...")
        
        ocr_raw = []
        ocr_filtered = []
        ocr_context_md = ""
        audio_windows_for_qa = []
        
        # Try to derive aligned windows (seg_pairs) from ASR timestamps via aligner parser
        seg_pairs = []
        if _aligner_parse_existing_asr is not None:
            try:
                logger.info("📐 Using aligner to build micro-windows...")
                asr_items = _aligner_parse_existing_asr(asr_file)  # [{"start_sec", "text"}]
                
                if asr_items:
                    n = int(os.getenv("OCR_SAMPLE_EVERY_N", "10"))
                    sampled = asr_items[::max(1, n)]
                    seg_pairs = [(max(0, x["start_sec"] - 5), x["start_sec"] + 15) for x in sampled]
                    logger.info(f"✅ Built {len(seg_pairs)} aligned segment pairs")
                else:
                    logger.warning("⚠️ No items returned from aligner parse")
            except Exception as e:
                logger.warning(f"⚠️ Aligner parse failed: {e}")
        else:
            logger.warning("⚠️ Aligner parse function not available")
        
        # Build QA audio windows from seg_pairs (if any)
        if seg_pairs:
            logger.info("📊 Building QA audio windows from aligned segments...")
            for (ws, we) in seg_pairs:
                text = _concat_asr_text_in_window(segs_raw, ws, we)
                if text:
                    audio_windows_for_qa.append({"start": float(ws), "end": float(we), "text": text})
            logger.info(f"✅ Created {len(audio_windows_for_qa)} audio windows")
            
            # Optionally run OCR on those micro-windows
            if do_ocr and OCRProcessor is not None:
                logger.info("🔍 Running OCR processing on video frames...")
                ocr_start = time.time()
                try:
                    proc = OCRProcessor()
                    items = proc.get_text_for_many_segments(video_path=Path(video_path), segments=seg_pairs)
                    ocr_raw = items
                    
                    for it in items:
                        joined = " ".join(it.get("texts", []) or []).strip()
                        if joined:
                            ocr_filtered.append({"start": float(it["start"]), "end": float(it["end"]), "text": joined})
                    
                    ocr_elapsed = time.time() - ocr_start
                    logger.info(f"✅ OCR complete in {ocr_elapsed:.1f}s: {len(ocr_filtered)} segments with text")
                    
                    if _aligner_get_context_markdown is not None:
                        try:
                            ocr_context_md = _aligner_get_context_markdown(
                                video_path=Path(video_path),
                                asr_file_path=asr_file,
                                ocr_processor=proc
                            )
                            logger.info("✅ Generated OCR context markdown")
                        except Exception as _e:
                            logger.warning(f"⚠️ Failed to build OCR context markdown: {_e}")
                except Exception as e:
                    logger.warning(f"⚠️ OCR processing failed: {e}")
            elif do_ocr:
                logger.info("ℹ️ OCR requested but OCRProcessor not available")
        else:
            logger.info("ℹ️ No aligned seg_pairs; using ASR-only segments")
        
        # If aligned windows produced nothing, fall back to ASR-only per-line segments
        if not audio_windows_for_qa:
            logger.info("📊 Using ASR-only segments as audio windows...")
            audio_windows_for_qa = [
                {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
                for seg in segs_raw
                if seg.get("text")
            ]
            logger.info(f"✅ Created {len(audio_windows_for_qa)} ASR-only windows")
        
        # ============= STAGE 5: FINAL SUMMARY =============
        elapsed = time.time() - t0
        
        logger.info("=" * 60)
        logger.info("✅ ASR PROCESSING COMPLETE")
        logger.info(f"⏱️ Total processing time: {elapsed:.1f} seconds")
        logger.info(f"📊 Audio windows: {len(audio_windows_for_qa)}")
        logger.info(f"📊 OCR segments: {len(ocr_filtered)}")
        logger.info(f"🎮 GPU used: {'Yes' if cuda_available else 'No (CPU only)'}")
        logger.info(f"📁 Cache directory: {asr_cache_dir}")
        logger.info("=" * 60)
        
        return {
            "success": True,
            "audio_segments": audio_windows_for_qa,
            "audio_segments_used": audio_windows_for_qa,
            "audio_segments_raw": segs_raw,
            "ocr_segments": ocr_filtered,
            "ocr_segments_filtered": ocr_filtered,
            "ocr_segments_raw": ocr_raw,
            "ocr_context_markdown": ocr_context_md,
            "processing_time": elapsed,
            "method": "chapter_llama_asr_aligned" if seg_pairs else "chapter_llama_asr_asr_only",
            "asr_cache_dir": str(asr_cache_dir),
            "asr_file": str(asr_file),
            "duration": duration,
            "gpu_used": cuda_available,
            "speech_duration": speech_duration,
            "removed_duration": removed_duration,
            "speech_ratio": speech_ratio,
            "removed_ratio": removed_ratio,
        }
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ ASR PROCESSING FAILED")
        logger.error(f"Error: {e}", exc_info=True)
        logger.error("=" * 60)
        
        return {
            "success": False,
            "audio_segments": [],
            "ocr_segments": [],
            "method": "chapter_llama_asr",
            "error": str(e)
        }
  

# =============== Debug helper for prompt snapshot (local) ===============

def combine_for_prompt(audio_segments, ocr_segments) -> str:
    """
    Build a compact, human/LLM-friendly string from aligned audio + OCR (for debug snapshots).
    - Audio: one line per segment -> "[HH:MM:SS-HH:MM:SS] text"
    - OCR: simple flat lines with timestamps (if present)
    """
    def _hms(x: float) -> str:
        return sec_to_hms(int(x or 0))

    # audio block
    lines = []
    for seg in (audio_segments or []):
        s = _hms(seg.get("start", 0.0))
        e = _hms(seg.get("end", seg.get("start", 0.0)))
        t = (seg.get("text") or "").strip()
        if t:
            lines.append(f"[{s}-{e}] {t}")

    # ocr block
    ocr_lines = []
    for seg in (ocr_segments or []):
        t = (seg.get("text") or "").strip()
        if not t:
            continue
        st = seg.get("start")
        ts = f"[{sec_to_hms(int(st))}] " if isinstance(st, (int, float)) else ""
        ocr_lines.append(f"{ts}{t}")

    out = "\n".join(lines)
    if ocr_lines:
        out += "\n\n--- OCR (auxiliary) ---\n" + "\n".join(ocr_lines)
    return out

# ==================== Celery Tasks (named to match task_routes) ====================

@celery.task(name="tasks.generate_qa_and_notes")
def generate_qa_and_notes(
    processing_result, 
    video_info, 
    raw_asr_text,
    chapters_dict=None,           
    hierarchical_metadata=None,
    section_title=None,      # ← ADD THIS
    units=None                # ← ADD THIS
):
    
    """
    Generate Q&A and notes using *raw ASR* (ASR-first) + simple OCR context,
    with optional chapter metadata for enhanced quality.
    """
    from app.qa_generation import process_text_for_qa_and_notes, result_to_legacy_client_format

    if not processing_result.get("success"):
        logger.error("❌ Cannot generate Q&A: video processing failed")
        return None

    audio_segments = processing_result.get("audio_segments_used", processing_result.get("audio_segments", []))
    ocr_segments = processing_result.get("ocr_segments_filtered", processing_result.get("ocr_segments", []))
    method = processing_result.get("method", "unknown")

    # ← ADD LOGGING FOR METADATA
    if hierarchical_metadata:
        logger.info("✅ Received hierarchical_metadata from chapter generation")
        logger.info(f"   • Educational quality score: {hierarchical_metadata.get('educational_quality_score', 0):.2f}")
    else:
        logger.warning("⚠️  No hierarchical_metadata provided - using limited context")
    
    logger.info("📚 Generating Q&A (ASR-first; chapters come from video_chaptering)")
    logger.info(f"📊 Input: raw_asr_len={len(raw_asr_text)}, ocr_segments={len(ocr_segments)}")

    # ← LOG EDUCATIONAL METADATA FOR Q&A
    if section_title or units:
        logger.info("📚 Using educational metadata for Q&A generation")
        if section_title:
            logger.info(f"   • Section: {section_title}")
        if units:
            logger.info(f"   • Units: {len(units)} learning units")
    try:
        # Get the raw EducationalContentResult object
        qa_result_obj = process_text_for_qa_and_notes(
            raw_asr_text=raw_asr_text,
            ocr_segments=ocr_segments,
            video_title=video_info.get("OriginalFilename"),
            chapters=chapters_dict,                    
            hierarchical_metadata=hierarchical_metadata,
            section_title=section_title,        # ← ADD THIS
            units=units,                         # ← ADD THIS
            num_questions=10,
            num_pages=3,
            id=video_info["Id"],
            team_id=video_info["TeamId"],
            section_no=video_info["SectionNo"],
            created_at=video_info.get("CreatedAt", datetime.now(timezone.utc).isoformat()),
        )

        # Convert it to the legacy format the client expects
        legacy_payload = result_to_legacy_client_format(
            result=qa_result_obj,  # This is now the EducationalContentResult object
            id=video_info["Id"],
            team_id=video_info["TeamId"],
            section_no=video_info["SectionNo"],
            created_at=video_info.get("CreatedAt", datetime.now(timezone.utc).isoformat()),
            chapters=[]  # Will be filled later with actual chapters from chaptering
        )

        # Add processing metadata
        legacy_payload["processing_metadata"] = {
            "processing_method": method,
            "optimized_processing_time": processing_result.get("processing_time", 0),
            "cache_used": processing_result.get("cache_used", False),
            "fallback_used": processing_result.get("fallback_used", False),
            "audio_blocks_processed": len(audio_segments),
            "ocr_segments_processed": len(ocr_segments),
            "duration": processing_result.get("duration"),
            "speech_duration": processing_result.get("speech_duration"),
            "removed_duration": processing_result.get("removed_duration"),
            "speech_ratio": processing_result.get("speech_ratio"),
            "removed_ratio": processing_result.get("removed_ratio"),
        }

        logger.info("✅ Q&A generated and converted to legacy client format successfully")
        return legacy_payload

    except Exception as e:
        logger.error(f"❌ Q&A generation failed: {e}", exc_info=True)
        return None

def generate_from_saved_segments(run_dir, video_info, num_questions=10, num_pages=3):
    """
    Offline generator using saved artifacts. Prefers *raw ASR* if available,
    falls back to reconstructing from segments as a last resort.
    """
    from app.qa_generation import process_text_for_qa_and_notes

    audio_paths = [
        os.path.join(run_dir, "audio_segments.aligned.json"),  # preferred
        os.path.join(run_dir, f"audio_{WIN_LABEL}.json"),
        os.path.join(run_dir, "audio_segments.used.json"),
        os.path.join(run_dir, "audio_segments.json"),
    ]
    ocr_paths = [
        os.path.join(run_dir, "ocr_filtered.json"),
        os.path.join(run_dir, "ocr_segments.filtered.json"),
        os.path.join(run_dir, "ocr_segments.json"),
    ]

    # Try to find a raw ASR dump
    raw_asr_candidates = [
        os.path.join(run_dir, "raw_asr_text.txt"),
        os.path.join(run_dir, "..", "raw_asr_text.txt"),
    ]
    raw_asr_text = ""
    for p in raw_asr_candidates:
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    raw_asr_text = f.read()
                break
        except Exception:
            pass

    audio_file = next((p for p in audio_paths if os.path.exists(p)), None)
    ocr_file   = next((p for p in ocr_paths if os.path.exists(p)), None)

    if not audio_file or not ocr_file:
        raise FileNotFoundError(f"Missing segments. audio={audio_file}, ocr={ocr_file}")

    with open(audio_file, "r", encoding="utf-8") as f:
        audio_segments = json.load(f)
    with open(ocr_file, "r", encoding="utf-8") as f:
        ocr_segments = json.load(f)

    # LAST-RESORT fallback if raw ASR text wasn't found
    if not raw_asr_text:
        logger.warning("⚠️ raw_asr_text not found in artifacts; reconstructing from audio segments")
        def _fmt(ts):
            h = int(ts // 3600); m = int((ts % 3600) // 60); s = int(ts % 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        lines = []
        for seg in audio_segments:
            t = (seg.get("text") or "").strip()
            if not t:
                continue
            ts = _fmt(float(seg.get("start", 0)))
            lines.append(f"{ts}: {t}")
        raw_asr_text = "\n".join(lines)

    if not video_info.get("CreatedAt"):
        video_info["CreatedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    result = process_text_for_qa_and_notes(
        raw_asr_text=raw_asr_text,   # ← ASR-first
        ocr_segments=ocr_segments,    # ← simple OCR
        num_questions=num_questions,
        num_pages=num_pages,
        id=video_info["Id"],
        team_id=video_info["TeamId"],
        section_no=video_info["SectionNo"],
        created_at=video_info["CreatedAt"],
    )

    # Convert to legacy format for consistency with online processing
    legacy_payload = result_to_legacy_client_format(
        result=result,
        id=video_info["Id"],
        team_id=video_info["TeamId"],
        section_no=video_info["SectionNo"],
        created_at=video_info["CreatedAt"],
        chapters=[]  # Offline mode might not have chapters
    )

    out_path = os.path.join(run_dir, "qa_and_notes.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(legacy_payload, f, ensure_ascii=False, indent=2)  # Save legacy format

    return out_path

@celery.task(name="tasks.generate_from_artifacts")
def generate_from_artifacts(run_dir, video_info, num_questions=10, num_pages=3):
    return generate_from_saved_segments(run_dir, video_info, num_questions, num_pages)

@celery.task(name="tasks.cleanup_files")
def cleanup_files_task(_result, file_path):
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"🗑️ Deleted temp file: {file_path}")
        clean_old_files(LOGS_DIR, max_age_hours=2)
        clean_old_files(SENT_PAYLOADS_DIR, max_age_hours=2)
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}", exc_info=True)

@celery.task(name="tasks.clean_old_uploads")
def clean_old_uploads(max_age_hours=8):
    uploads_dir = UPLOAD_FOLDER
    cache_dir = CACHE_DIR
    now = time.time()
    age_seconds = max_age_hours * 3600
    files_cleaned = 0

    if os.path.exists(uploads_dir):
        for root, dirs, files in os.walk(uploads_dir):
            for name in files:
                file_path = os.path.join(root, name)
                if now - os.path.getmtime(file_path) > age_seconds:
                    try:
                        os.remove(file_path)
                        files_cleaned += 1
                        logger.info(f"🗑️ [sweeper] Deleted old file: {file_path}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to delete {file_path}: {e}")
            for name in dirs:
                dir_path = os.path.join(root, name)
                if name.endswith("_segments") and now - os.path.getmtime(dir_path) > age_seconds:
                    try:
                        import shutil
                        shutil.rmtree(dir_path)
                        files_cleaned += 1
                        logger.info(f"🗑️ [sweeper] Deleted old segment folder: {dir_path}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to delete {dir_path}: {e}")

    cache_age_seconds = 48 * 3600
    if os.path.exists(cache_dir):
        for root, _dirs, files in os.walk(cache_dir):
            for name in files:
                file_path = os.path.join(root, name)
                if now - os.path.getmtime(file_path) > cache_age_seconds:
                    try:
                        os.remove(file_path)
                        files_cleaned += 1
                        logger.info(f"🗑️ [sweeper] Deleted old cache file: {file_path}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to delete cache {file_path}: {e}")

    logger.info(f"🗑️ [sweeper] Cleaned {files_cleaned} old files")

@celery.task(
    name="tasks.process_video_task",
    bind=True, soft_time_limit=7200, time_limit=7500, max_retries=5, default_retry_delay=1200
)

def process_video_task(self, play_url_or_path, video_info, num_questions=10, num_pages=3):
    file_path = None
    processing_start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("🚀 NEW VIDEO PROCESSING JOB STARTED")
    logger.info("=" * 80)
    logger.info(f"📋 Video Info: {json.dumps(video_info, indent=2)}")
    logger.info(f"🔗 Input URL/Path: {play_url_or_path}")
    logger.info(f"⏰ Start Time: {datetime.now().isoformat()}")
    logger.info("=" * 80)
    
    try:
        # Stage 1: Download/Locate Video
        logger.info("\n" + "="*60)
        logger.info("📥 STAGE 1: VIDEO ACQUISITION")
        logger.info("="*60)
        
        if isinstance(play_url_or_path, str) and play_url_or_path.startswith("file://"):
            file_path = play_url_or_path.replace("file://", "")
            logger.info(f"📂 Using local file: {file_path}")
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"✅ Local file exists: {size_mb:.2f} MB")
            else:
                logger.error(f"❌ Local file NOT FOUND: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
        elif isinstance(play_url_or_path, str):
            logger.info("🌐 Downloading video from URL...")
            file_path = download_video(play_url_or_path, f"{uuid.uuid4()}.mp4")
        else:
            raise ValueError("Invalid input: must be URL or file:// path")
        
        # Stage 2: ASR Processing
        logger.info("\n" + "="*60)
        logger.info("🎤 STAGE 2: ASR TRANSCRIPTION")
        logger.info("="*60)
        
        processing_result = chapter_llama_asr_processing_fn(
            file_path,
            window_sec=QA_WINDOW_SEC,
            do_ocr=ENABLE_OCR
        )
        
        if not processing_result.get("success"):
            error_code = processing_result.get("error")
            error_msg = processing_result.get("error_message", error_code)
            
            logger.error("❌ ASR PROCESSING FAILED")
            logger.error(f"Error: {error_msg}")
            
            # If it's a "no valid audio" error, notify client and don't retry
            if error_code == "no_valid_audio":
                logger.error("❌ Video has no usable audio - notifying client and skipping retry")
                
                error_payload = {
                    "success": False,
                    "Id": video_info["Id"],
                    "TeamId": video_info["TeamId"],
                    "SectionNo": video_info["SectionNo"],
                    "error": "no_valid_audio",
                    "error_message": "Video has no valid audio or speech content",
                    "processing_time": time.time() - processing_start_time,
                    "video_info": video_info
                }
                
                # Notify client
                post_to_client_api(error_payload)
                
                # Save error report
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                error_dir = os.path.join(RUNS_BASE, f"{video_info['Id']}_{ts}")
                os.makedirs(error_dir, exist_ok=True)
                
                error_file = os.path.join(error_dir, "error_report.json")
                with open(error_file, "w", encoding="utf-8") as f:
                    json.dump(error_payload, f, indent=2, ensure_ascii=False)
                logger.info(f"💾 Saved error report to: {error_file}")
                
                # Don't retry - this is a permanent failure
                return error_payload
            
            # For other errors, retry as normal
            raise RuntimeError(f"ASR processing failed: {error_msg}")

        # ---- Video Chaptering ----
        logger.info("📑 Generating video chapters...")

        # Create a run directory up front and reuse it for chaptering + artifacts
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{video_info['Id']}_{ts}"
        run_dir = os.path.join(RUNS_BASE, base)
        os.makedirs(run_dir, exist_ok=True)

        audio_used    = processing_result.get("audio_segments_used", processing_result.get("audio_segments", []))
        ocr_filtered  = processing_result.get("ocr_segments_filtered", processing_result.get("ocr_segments", []))
        duration      = processing_result.get("duration")
        asr_file_path = processing_result.get("asr_file")  # path to raw ASR text file

        # ---- Persist VAD metrics & log summary (if present)
        vad_metrics = {
            "duration":         processing_result.get("duration"),
            "speech_duration":  processing_result.get("speech_duration"),
            "removed_duration": processing_result.get("removed_duration"),
            "speech_ratio":     processing_result.get("speech_ratio"),
            "removed_ratio":    processing_result.get("removed_ratio"),
        }
        try:
            with open(os.path.join(run_dir, "asr_vad_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(vad_metrics, f, indent=2, ensure_ascii=False)
        except Exception as _e:
            logger.warning("⚠️ Failed to write asr_vad_metrics.json: %s", _e)

        try:
            dur = float(vad_metrics["duration"] or 0.0)
            sp  = float(vad_metrics["speech_duration"] or 0.0)
            rm  = float(vad_metrics["removed_duration"] or max(0.0, dur - sp))
            if dur > 0:
                pct_kept    = (sp / dur) * 100.0
                pct_removed = 100.0 - pct_kept
                logger.info(
                    "🧪 VAD summary (run): kept %.1f%% (%.1fs), removed %.1f%% (%.1fs) of %.1fs total",
                    pct_kept, sp, pct_removed, rm, dur
                )
        except Exception:
            pass

        # Read the raw ASR text from the file that was already saved
        # ---- NEW: safe ASR-text load ------------------------------------------
        raw_asr_text = ""
        asr_file_path = processing_result.get("asr_file") 

        if asr_file_path and Path(asr_file_path).is_file():
            logger.info(f"📝 ASR file claimed: {asr_file_path}")
            logger.info(f"📄 ASR file size: {Path(asr_file_path).stat().st_size:,} bytes")

            # peek at first 3 lines for quick sanity check
            with open(asr_file_path, encoding="utf-8") as f:
                raw_asr_text = f.read()
                peek = "\n".join(raw_asr_text.splitlines()[:3])
                logger.info("🧪 ASR peek (first 3 lines):\n%s", peek or "<empty>")
        else:
            logger.warning("⚠️ No usable asr_file; falling back to segments")
            audio_segs = processing_result.get("audio_segments", [])
            raw_asr_text = "\n".join(
                f"{sec_to_hms(int(s.get('start', 0)))}: {s.get('text','')}".strip()
                for s in audio_segs if s.get("text")
            )
            if not raw_asr_text:
                raise RuntimeError("ASR stage failed and no segments available for fallback")
        # -----------------------------------------------------------------------

        # ← NEW: EXTRACT EDUCATIONAL METADATA FROM video_info
        section_title = video_info.get("SectionTitle")
        units = video_info.get("Units", [])

        # Log what we received
        if section_title or units:
            logger.info("=" * 60)
            logger.info("📚 EDUCATIONAL METADATA RECEIVED FROM API")
            logger.info("=" * 60)
            if section_title:
                logger.info(f"📖 Section Title: {section_title}")
            if units:
                logger.info(f"📑 Units ({len(units)}):")
                for unit in units:
                    logger.info(f"   {unit['UnitNo']}. {unit['Title']}")
            logger.info("=" * 60)
        else:
            logger.info("ℹ️  No educational metadata provided - using standard processing")

        # ← MODIFIED: Pass educational metadata to chapter generation
        chaptering_result = generate_chapters(
            raw_asr_text=raw_asr_text,
            ocr_segments=ocr_filtered,
            video_title=video_info.get("OriginalFilename"),
            section_title=section_title,
            units=units,
            duration=duration,
            video_id=video_info["Id"],
            run_dir=Path(run_dir)
        )
        
        # ← ADD THIS: Extract chapters and metadata from chaptering_result
        logger.info("📊 Extracting chapter metadata for Q&A generation...")
        
        # chaptering_result structure: (chapters_dict, metadata)
        if isinstance(chaptering_result, tuple) and len(chaptering_result) == 2:
            chapters_dict, chapter_metadata = chaptering_result
            logger.info(f"✅ Received {len(chapters_dict)} chapters and metadata")
            logger.info(f"   • Metadata keys: {list(chapter_metadata.keys())}")
        else:
            # Fallback for old format (just dict of chapters)
            chapters_dict = chaptering_result if isinstance(chaptering_result, dict) else {}
            chapter_metadata = None
            logger.warning("⚠️  Chapter generation returned old format (no metadata)")

        # ---- Q&A + notes ----
        logger.info("📚 Generating Q&A and lecture notes (ASR-first)...")
        
        # ← LOG EDUCATIONAL METADATA FOR Q&A
        if section_title or units:
            logger.info("📚 Passing educational metadata to Q&A generation")
            if section_title:
                logger.info(f"   • Section: {section_title}")
            if units:
                logger.info(f"   • Units: {len(units)} learning units to structure Q&A")
        
        qa_result = generate_qa_and_notes(
            processing_result, 
            video_info, 
            raw_asr_text,
            chapters_dict=chapters_dict,           
            hierarchical_metadata=chapter_metadata,
            section_title=section_title,
            units=units
        )

        # ========== NEW: GENERATE SUGGESTED UNITS ==========
        logger.info("=" * 60)
        logger.info("🎓 GENERATING SUGGESTED UNITS FOR CLIENT")
        logger.info("=" * 60)

        # Import the helper functions
        from app.qa_generation import (
            parse_modules_to_topics,
            extract_suggested_units_from_chapters,
            extract_suggested_units_from_topics
        )

        # Extract topics from hierarchical metadata (if available)
        topics_list = []
        if chapter_metadata and chapter_metadata.get('modules_analysis'):
            try:
                topics_list = parse_modules_to_topics(
                    chapter_metadata.get('modules_analysis', '')
                )
                logger.info(f"✅ Extracted {len(topics_list)} topics from chapter metadata")
            except Exception as e:
                logger.warning(f"⚠️ Failed to extract topics from metadata: {e}")

        # Generate suggested units using the best available method
        suggested_units = []

        if topics_list and len(topics_list) >= 3:
            # Method 1: Use topics (best for educational quality)
            try:
                suggested_units = extract_suggested_units_from_topics(
                    topics_list=topics_list,
                    chapters_dict=chapters_dict
                )
                logger.info(f"📚 Generated {len(suggested_units)} units from topics analysis")
                logger.info(f"   Method: Topic-based (highest educational quality)")
            except Exception as e:
                logger.warning(f"⚠️ Topic-based unit extraction failed: {e}")

        if not suggested_units and chapters_dict and len(chapters_dict) >= 3:
            # Method 2: Fall back to chapters (still good)
            try:
                suggested_units = extract_suggested_units_from_chapters(
                    chapters_dict=chapters_dict,
                    max_units=5
                )
                logger.info(f"📚 Generated {len(suggested_units)} units from chapters")
                logger.info(f"   Method: Chapter-based (fallback)")
            except Exception as e:
                logger.warning(f"⚠️ Chapter-based unit extraction failed: {e}")

        if not suggested_units:
            # Method 3: Minimum fallback (empty list)
            logger.warning("⚠️ Insufficient data for unit generation - using empty list")
            logger.warning("   This will still work, but client won't get AI suggestions")

        # Log what we're providing
        logger.info("-" * 60)
        logger.info("📋 UNITS SUMMARY")
        logger.info(f"   • Original Units (from API): {len(units)}")
        logger.info(f"   • Suggested Units (AI-generated): {len(suggested_units)}")

        if suggested_units:
            logger.info("   • Suggested Units:")
            for unit in suggested_units:
                logger.info(f"      {unit['UnitNo']}. {unit['Title']} @ {unit['Time']}")
        logger.info("=" * 60)
        if qa_result:
            # ========== METADATA COLLECTION (Don't add to client payload) ==========
            total_processing_time = time.time() - processing_start_time
    
            audio_used    = processing_result.get("audio_segments_used", processing_result.get("audio_segments", []))
            audio_raw     = processing_result.get("audio_segments_raw", [])
            ocr_filtered  = processing_result.get("ocr_segments_filtered", processing_result.get("ocr_segments", []))
            ocr_raw       = processing_result.get("ocr_segments_raw", [])
            ocr_ctx_md    = processing_result.get("ocr_context_markdown", "")

            # ========== PERSIST ARTIFACTS TO WORKSPACE ==========
            # Persist ASR/OCR artifacts (aligned-first)
            with open(os.path.join(run_dir, "audio_segments.raw.json"), "w", encoding="utf-8") as f:
                json.dump(audio_raw, f, indent=2, ensure_ascii=False)

            with open(os.path.join(run_dir, "audio_segments.aligned.json"), "w", encoding="utf-8") as f:
                json.dump(audio_used, f, indent=2, ensure_ascii=False)

            if WRITE_FIVEMIN_ARTIFACTS:
                with open(os.path.join(run_dir, f"audio_{WIN_LABEL}.json"), "w", encoding="utf-8") as f:
                    json.dump(audio_used, f, indent=2, ensure_ascii=False)
                with open(os.path.join(run_dir, "audio_segments.used.json"), "w", encoding="utf-8") as f:
                    json.dump(audio_used, f, indent=2, ensure_ascii=False)

            with open(os.path.join(run_dir, "ocr_segments.raw.json"), "w", encoding="utf-8") as f:
                json.dump(ocr_raw, f, indent=2, ensure_ascii=False)
            with open(os.path.join(run_dir, "ocr_segments.filtered.json"), "w", encoding="utf-8") as f:
                json.dump(ocr_filtered, f, indent=2, ensure_ascii=False)
            with open(os.path.join(run_dir, "ocr_filtered.json"), "w", encoding="utf-8") as f:
                json.dump(ocr_filtered, f, indent=2, ensure_ascii=False)

            if ocr_ctx_md:
                with open(os.path.join(run_dir, "ocr_context.md"), "w", encoding="utf-8") as f:
                    f.write(ocr_ctx_md)

            # Keep a combined snapshot purely for debug/audit
            try:
                combined_text = combine_for_prompt(audio_used, ocr_filtered)
                with open(os.path.join(run_dir, "combined_text_for_gpt.txt"), "w", encoding="utf-8") as f:
                    f.write(combined_text)
            except Exception as _e:
                logger.warning("⚠️ Failed writing combined_text_for_gpt.txt: %s", _e)

            # ========== MERGE CHAPTERS INTO PAYLOAD ==========
            if not chapters_dict:
                logger.warning("⚠️ No chapters returned; using empty dict")
                chapters_dict = {}
    
            qa_result["chapters"] = chapters_dict

            # Save chapters separately
            with open(os.path.join(run_dir, "chapters.json"), "w", encoding="utf-8") as f:
                json.dump(chapters_dict, f, indent=2, ensure_ascii=False)
    
            # Save metadata separately for debugging
            if chapter_metadata:
                with open(os.path.join(run_dir, "chapter_metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(chapter_metadata, f, indent=2, ensure_ascii=False)
                logger.info(f"💾 Saved chapter metadata to: {run_dir}/chapter_metadata.json")

            # ========== SAVE COMPREHENSIVE WORKSPACE ARTIFACT ==========
            workspace_artifact = {
                **qa_result,
                "Units": units_from_chapters or [],                    # Generated chapters (transformed)
                "SuggestedUnits": suggested_units_from_api or [],      # From incoming API
                "AIGeneratedSuggestedUnits": suggested_units or [],    # Your AI suggestions
                "total_processing_time": total_processing_time,
                "processing_metadata": {
                    "processing_method": processing_result.get("method"),
                    "optimized_processing_time": processing_result.get("processing_time", 0),
                    "cache_used": processing_result.get("cache_used", False),
                    "fallback_used": processing_result.get("fallback_used", False),
                    "audio_blocks_processed": len(audio_used),
                    "ocr_segments_processed": len(ocr_filtered),
                    "duration": processing_result.get("duration"),
                    "speech_duration": processing_result.get("speech_duration"),
                    "removed_duration": processing_result.get("removed_duration"),
                    "speech_ratio": processing_result.get("speech_ratio"),
                    "removed_ratio": processing_result.get("removed_ratio"),
                },
                "chapter_metadata": chapter_metadata,
            }
               
            # Save full workspace result
            workspace_path = os.path.join(run_dir, "full_processing_result.json")
            with open(workspace_path, "w", encoding="utf-8") as f:
                json.dump(workspace_artifact, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved full workspace artifact to {workspace_path}")

            # ========== TRANSFORM CHAPTERS FOR CLIENT ==========
            logger.info("🔄 Transforming chapters to client format...")

            # Transform chapters to Units format (safe - returns [] if None)
            units_from_chapters = transform_chapters_to_units(qa_result.get("chapters", {})) or []

            # Prepare SuggestedUnits from incoming API (safe - defaults to [])
            suggested_units_from_api = []
            if units and isinstance(units, list):
                logger.info(f"📚 Including {len(units)} SuggestedUnits from incoming API")
                for idx, unit in enumerate(units, start=1):
                    if isinstance(unit, dict):
                        suggested_units_from_api.append({
                            "UnitNo": unit.get("UnitNo", idx),
                            "Title": unit.get("Title", ""),
                            "Time": unit.get("Time", "")
                        })
            else:
                logger.info("ℹ️  No SuggestedUnits in incoming API")
                
            # Safe defaults for all unit types
            units_from_chapters = units_from_chapters or []
            suggested_units_from_api = suggested_units_from_api or []

            # ========== BUILD CLEAN CLIENT PAYLOAD ==========
            client_payload = {
                "Id": qa_result["Id"],
                "TeamId": qa_result["TeamId"],
                "SectionNo": qa_result["SectionNo"],
                "CreatedAt": qa_result["CreatedAt"],
                "Questions": qa_result["Questions"],
                "CourseNote": qa_result["CourseNote"],
                "Units": units_from_chapters,              # Generated chapters in client format
                "SuggestedUnits": suggested_units_from_api  # Original units from API
            }

            logger.info(f"📦 Client payload summary:")
            logger.info(f"   • Units: {len(units_from_chapters)}")
            logger.info(f"   • SuggestedUnits: {len(suggested_units_from_api)}")
            logger.info(f"   • Questions: {len(client_payload['Questions'])}")  

            # Save clean client payload
            client_path = os.path.join(run_dir, "client_payload.json")
            with open(client_path, "w", encoding="utf-8") as f:
                json.dump(client_payload, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved clean client payload to {client_path}")

            # Save legacy QA output (for backward compatibility)
            output_path = os.path.join(run_dir, "qa_and_notes.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(workspace_artifact, f, indent=2, ensure_ascii=False)  # Save full version
            logger.info(f"💾 Saved legacy output to {output_path}")

            transcript_path = os.path.join(run_dir, "merged_transcript.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                for seg in audio_used:
                    s = seg.get("start", 0); e = seg.get("end", 0); t = seg.get("text", "")
                    f.write(f"[{s:.1f} - {e:.1f}] {t}\n")
            logger.info(f"💾 Saved transcript to {transcript_path}")

            # ========== SEND CLEAN PAYLOAD TO CLIENT API ==========
            post_to_client_api(client_payload)  # ← Only send clean data
            logger.info(f"✅ Complete pipeline finished in {total_processing_time:.1f}s")
        else:
            logger.error("❌ Q&A generation failed")
            # Optionally post failure
            post_to_client_api({
                "success": False,
                "error": "Q&A generation failed",
                "video_info": video_info,
                "processing_time": time.time() - processing_start_time,
            })

    except Exception as e:
        logger.error(f"❌ Error during task, retrying: {e}", exc_info=True)
        raise self.retry(exc=e)
    finally:
        if file_path and os.path.exists(file_path):
            cleanup_files_task.delay(None, file_path)
