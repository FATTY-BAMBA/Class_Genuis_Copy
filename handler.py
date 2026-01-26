# handler.py - RunPod Serverless Handler for Class Genius
# This wraps your existing pipeline for serverless execution

import os
import sys
import json
import uuid
import time
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

import runpod

# ==================== Setup Logging ====================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== Environment Setup ====================
# Ensure paths are set up for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
# sys.path.insert(0, os.path.join(BASE_DIR, "app"))  # REMOVED - breaks package imports

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ==================== Import Your Existing Pipeline ====================
# These imports happen after path setup

# First, create a mock Celery for serverless mode (tasks.py imports celery)
# This prevents import errors when Celery/Redis aren't available
try:
    from app.celery_app import celery
except ImportError:
    # Create a mock celery decorator for serverless mode
    class MockCelery:
        @staticmethod
        def task(*args, **kwargs):
            def decorator(func):
                func.run = func  # Allow .run access
                func.delay = func  # Mock .delay() to run synchronously
                func.apply_async = lambda *a, **kw: func(*a, **kw)
                return func
            if args and callable(args[0]):
                return decorator(args[0])
            return decorator
    
    # Inject mock into app.celery_app module
    import sys
    from types import ModuleType
    mock_module = ModuleType('app.celery_app')
    mock_module.celery = MockCelery()
    sys.modules['app.celery_app'] = mock_module
    logger.info("📦 Running in serverless mode (Celery mocked)")

try:
    from tasks.tasks import (
        download_video,
        chapter_llama_asr_processing_fn,
        generate_qa_and_notes as generate_qa_and_notes_task,  # This is a Celery task
        post_to_client_api,
        transform_chapters_to_units,
        fill_unit_times_from_suggested_units,
        clean_client_title,
        sec_to_hms,
        combine_for_prompt,
        UPLOAD_FOLDER,
        RUNS_BASE,
        QA_WINDOW_SEC,
        ENABLE_OCR,
    )
    from app.chapter_generation import generate_chapters
    from app.qa_generation import (
        process_text_for_qa_and_notes,
        result_to_legacy_client_format,
        parse_modules_to_topics,
        extract_suggested_units_from_chapters,
        extract_suggested_units_from_topics,
    )
    
    # Unwrap Celery task to get the underlying function
    # This ensures we call it synchronously, not as a Celery task
    if hasattr(generate_qa_and_notes_task, 'run'):
        # Celery task - get the underlying function
        generate_qa_and_notes = generate_qa_and_notes_task.run
    else:
        # Already a regular function
        generate_qa_and_notes = generate_qa_and_notes_task
    
    IMPORTS_OK = True
    logger.info("✅ All pipeline imports successful")
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)
    logger.error(f"❌ Import error: {e}")


# ==================== Helper Functions ====================

def validate_input(job_input: dict) -> tuple[bool, str]:
    """Validate the job input has required fields."""
    
    # Check for old format (fields at root level)
    is_old_format = "PlayUrl" in job_input and "video_info" not in job_input
    
    if is_old_format:
        # Old format validation
        if not job_input.get("PlayUrl"):
            return False, "Missing required field: PlayUrl"
        required_fields = ["Id", "TeamId", "SectionNo"]
        for field in required_fields:
            if field not in job_input:
                return False, f"Missing required field: {field}"
    else:
        # New format validation
        video_url = job_input.get("video_url") or job_input.get("play_url") or job_input.get("PlayUrl")
        if not video_url:
            return False, "Missing required field: video_url or play_url"
        
        video_info = job_input.get("video_info") or job_input.get("VideoInfo")
        if not video_info:
            return False, "Missing required field: video_info"
        
        required_fields = ["Id", "TeamId", "SectionNo"]
        for field in required_fields:
            if field not in video_info:
                return False, f"Missing required field in video_info: {field}"
    
    return True, ""
def normalize_input(job_input: dict) -> dict:
    """Normalize input field names to match internal expectations."""
    
    # Check if old format (fields at root level)
    is_old_format = "PlayUrl" in job_input and "video_info" not in job_input
    
    if is_old_format:
        # Convert old format to new format
        logger.info("📋 Detected OLD input format - converting...")
        video_url = job_input.get("PlayUrl")
        normalized_video_info = {
            "Id": job_input.get("Id"),
            "TeamId": job_input.get("TeamId"),
            "SectionNo": job_input.get("SectionNo"),
            "SectionTitle": job_input.get("SectionTitle", ""),
            "Units": job_input.get("Units", []),
            "CreatedAt": job_input.get("CreatedAt", datetime.now(timezone.utc).isoformat()),
            "OriginalFilename": job_input.get("OriginalFilename", "video.mp4"),
        }
    else:
        # New format
        logger.info("📋 Detected NEW input format")
        video_url = (
            job_input.get("video_url") or 
            job_input.get("play_url") or 
            job_input.get("PlayUrl")
        )
        
        video_info = job_input.get("video_info") or job_input.get("VideoInfo") or {}
        
        normalized_video_info = {
            "Id": video_info.get("Id"),
            "TeamId": video_info.get("TeamId"),
            "SectionNo": video_info.get("SectionNo"),
            "CreatedAt": video_info.get("CreatedAt", datetime.now(timezone.utc).isoformat()),
            "OriginalFilename": video_info.get("OriginalFilename", "video.mp4"),
            "SectionTitle": video_info.get("SectionTitle"),
            "Units": video_info.get("Units", []),
        }
    
    return {
        "video_url": video_url,
        "video_info": normalized_video_info,
        "num_questions": job_input.get("num_questions", 10),
        "num_pages": job_input.get("num_pages", 3),
        "webhook_url": job_input.get("webhook_url"),
        "skip_client_post": job_input.get("skip_client_post", False),
    }


def send_webhook(webhook_url: str, payload: dict):
    """Send results to a webhook URL if provided."""
    if not webhook_url:
        return
    
    try:
        import requests
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        logger.info(f"📤 Webhook sent: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Webhook failed: {e}")


# ==================== Main Handler ====================

def handler(job: dict) -> dict:
    """
    RunPod Serverless Handler for Class Genius Video Processing.
    
    Input Schema:
    {
        "input": {
            "video_url": "https://...",  # or "play_url" or "PlayUrl"
            "video_info": {
                "Id": "video-123",
                "TeamId": "team-456", 
                "SectionNo": 1,
                "SectionTitle": "Optional title",
                "Units": [{"UnitNo": 1, "Title": "..."}],
                "OriginalFilename": "lecture.mp4"
            },
            "num_questions": 10,  # optional, default 10
            "num_pages": 3,       # optional, default 3
            "webhook_url": "https://...",  # optional
            "skip_client_post": false  # optional, for testing
        }
    }
    
    Output Schema:
    {
        "status": "success" | "error",
        "output": { ... client payload ... },
        "processing_time": 123.4,
        "error": "..." (if status == "error")
    }
    """
    
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})
    
    logger.info("=" * 80)
    logger.info(f"🚀 RUNPOD SERVERLESS JOB STARTED: {job_id}")
    logger.info("=" * 80)
    
    start_time = time.time()
    file_path = None
    webhook_url = job_input.get("webhook_url")
    
    try:
        # Check imports
        if not IMPORTS_OK:
            raise RuntimeError(f"Pipeline imports failed: {IMPORT_ERROR}")
        
        # Validate input
        is_valid, error_msg = validate_input(job_input)
        if not is_valid:
            logger.error(f"❌ Input validation failed: {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "error_type": "validation_error"
            }
        
        # Normalize input
        normalized = normalize_input(job_input)
        video_url = normalized["video_url"]
        video_info = normalized["video_info"]
        num_questions = normalized["num_questions"]
        num_pages = normalized["num_pages"]
        webhook_url = normalized["webhook_url"]
        skip_client_post = normalized["skip_client_post"]
        
        logger.info(f"📋 Video Info: {json.dumps(video_info, indent=2)}")
        logger.info(f"🔗 Video URL: {video_url}")
        
        # ==================== STAGE 1: Download Video ====================
        logger.info("\n" + "="*60)
        logger.info("📥 STAGE 1: VIDEO ACQUISITION")
        logger.info("="*60)
        
        if video_url.startswith("file://"):
            file_path = video_url.replace("file://", "")
            logger.info(f"📂 Using local file: {file_path}")
        else:
            file_path = download_video(video_url, f"{uuid.uuid4()}.mp4")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")
        
        # ==================== STAGE 2: ASR Processing ====================
        logger.info("\n" + "="*60)
        logger.info("🎤 STAGE 2: ASR TRANSCRIPTION")
        logger.info("="*60)
        
        processing_result = chapter_llama_asr_processing_fn(
            file_path,
            window_sec=QA_WINDOW_SEC,
            do_ocr=ENABLE_OCR
        )
        
        if not processing_result.get("success"):
            error_code = processing_result.get("error", "unknown")
            error_msg = processing_result.get("error_message", error_code)
            
            # Handle no-audio case specially
            if error_code == "no_valid_audio":
                logger.error("❌ Video has no usable audio")
                result = {
                    "status": "error",
                    "error": "no_valid_audio",
                    "error_message": "Video has no valid audio or speech content",
                    "video_info": video_info,
                    "processing_time": time.time() - start_time
                }
                
                if not skip_client_post:
                    post_to_client_api(result)
                send_webhook(webhook_url, result)
                
                return result
            
            raise RuntimeError(f"ASR processing failed: {error_msg}")
        
        # ==================== STAGE 3: Chapter Generation ====================
        logger.info("\n" + "="*60)
        logger.info("📑 STAGE 3: CHAPTER GENERATION")
        logger.info("="*60)
        
        # Create run directory
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(RUNS_BASE, f"{video_info['Id']}_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Get ASR text
        asr_file_path = processing_result.get("asr_file")
        if asr_file_path and Path(asr_file_path).is_file():
            with open(asr_file_path, encoding="utf-8") as f:
                raw_asr_text = f.read()
        else:
            # Fallback to segments
            audio_segs = processing_result.get("audio_segments", [])
            raw_asr_text = "\n".join(
                f"{sec_to_hms(int(s.get('start', 0)))}: {s.get('text','')}"
                for s in audio_segs if s.get("text")
            )
        
        ocr_filtered = processing_result.get("ocr_segments_filtered", [])
        duration = processing_result.get("duration")
        
        # Extract educational metadata
        section_title = video_info.get("SectionTitle")
        units = video_info.get("Units", [])
        
        # Generate chapters
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
        
        # Extract chapters and metadata
        if isinstance(chaptering_result, tuple) and len(chaptering_result) == 2:
            chapters_dict, chapter_metadata = chaptering_result
        else:
            chapters_dict = chaptering_result if isinstance(chaptering_result, dict) else {}
            chapter_metadata = None
        
        # ==================== STAGE 4: Q&A Generation ====================
        logger.info("\n" + "="*60)
        logger.info("📚 STAGE 4: Q&A GENERATION")
        logger.info("="*60)
        
        qa_result = generate_qa_and_notes(
            processing_result,
            video_info,
            raw_asr_text,
            chapters_dict=chapters_dict,
            hierarchical_metadata=chapter_metadata,
            section_title=section_title,
            units=units,
            num_questions=num_questions,
            num_pages=num_pages
        )
        
        if not qa_result:
            raise RuntimeError("Q&A generation failed")
        
        # ==================== STAGE 5: Build Output ====================
        logger.info("\n" + "="*60)
        logger.info("📦 STAGE 5: BUILD OUTPUT")
        logger.info("="*60)
        
        # Transform chapters to units
        navigation_units = transform_chapters_to_units(chapters_dict or {}) or []
        
        # Process units from API
        units_from_api = []
        for idx, unit in enumerate(units or [], start=1):
            if isinstance(unit, dict):
                units_from_api.append({
                    "UnitNo": unit.get("UnitNo", idx),
                    "Title": clean_client_title(unit.get("Title", "")),
                    "Time": unit.get("Time", "")
                })
        
        # Fill unit times from chapter metadata if available
        if chapter_metadata and chapter_metadata.get("client_units_with_timestamps"):
            enriched_units = chapter_metadata["client_units_with_timestamps"]
            units_from_api = [
                {
                    "UnitNo": unit.get("UnitNo"),
                    "Title": clean_client_title(unit.get("Title", "")),
                    "Time": unit.get("Time", "")
                }
                for unit in enriched_units
            ]
        elif chapter_metadata and chapter_metadata.get("suggested_units_structured"):
            units_from_api, _ = fill_unit_times_from_suggested_units(
                units_from_api,
                chapter_metadata["suggested_units_structured"],
                only_fill_if_empty=True
            )
        
        # Build client payload
        client_payload = {
            "Id": qa_result["Id"],
            "TeamId": qa_result["TeamId"],
            "SectionNo": qa_result["SectionNo"],
            "CreatedAt": qa_result["CreatedAt"],
            "Questions": qa_result["Questions"],
            "CourseNote": qa_result["CourseNote"],
            "Units": units_from_api,
            "SuggestedUnits": navigation_units
        }
        
        total_time = time.time() - start_time
        
        # Save artifacts
        with open(os.path.join(run_dir, "client_payload.json"), "w", encoding="utf-8") as f:
            json.dump(client_payload, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Processing complete in {total_time:.1f}s")
        logger.info(f"📊 Questions: {len(client_payload['Questions'])}")
        logger.info(f"📊 Units: {len(client_payload['Units'])}")
        logger.info(f"📊 SuggestedUnits: {len(client_payload['SuggestedUnits'])}")
        
        # Send to client API
        if not skip_client_post:
            post_to_client_api(client_payload)

        # Send webhook if provided
        send_webhook(webhook_url, client_payload)
        return client_payload
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error("=" * 60)
        logger.error(f"❌ JOB FAILED: {e}")
        logger.error(traceback.format_exc())
        logger.error("=" * 60)
        
        error_result = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "processing_time": error_time
        }
        
        send_webhook(webhook_url, error_result)
        
        return error_result
        
    finally:
        # Cleanup downloaded file
        if file_path and os.path.exists(file_path) and not file_path.startswith("file://"):
            try:
                os.remove(file_path)
                logger.info(f"🗑️ Cleaned up: {file_path}")
            except Exception as e:
                logger.warning(f"⚠️ Cleanup failed: {e}")


# ==================== Health Check Handler ====================

def health_check(job: dict) -> dict:
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "imports_ok": IMPORTS_OK,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ==================== Start Serverless Worker ====================

if __name__ == "__main__":
    logger.info("🚀 Starting RunPod Serverless Worker")
    logger.info(f"📦 Imports OK: {IMPORTS_OK}")
    
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True  # For long-running jobs
    })
