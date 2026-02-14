# handler.py - RunPod Serverless Handler for Class Genius
# This wraps your existing pipeline for serverless execution
# FULL PARITY with tasks.py process_video_task

import os
import sys
import json
import uuid
import time
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from collections import OrderedDict, defaultdict

import runpod

# ==================== Setup Logging ====================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== Environment Setup ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv()

# ==================== Import Your Existing Pipeline ====================
try:
    from app.celery_app import celery
except ImportError:
    class MockCelery:
        @staticmethod
        def task(*args, **kwargs):
            def decorator(func):
                func.run = func
                func.delay = func
                func.apply_async = lambda *a, **kw: func(*a, **kw)
                return func
            if args and callable(args[0]):
                return decorator(args[0])
            return decorator

    from types import ModuleType
    mock_module = ModuleType('app.celery_app')
    mock_module.celery = MockCelery()
    sys.modules['app.celery_app'] = mock_module
    logger.info("📦 Running in serverless mode (Celery mocked)")

try:
    from tasks.tasks import (
        download_video,
        chapter_llama_asr_processing_fn,
        generate_qa_and_notes as generate_qa_and_notes_task,
        post_to_client_api,
        transform_chapters_to_units,
        fill_unit_times_from_suggested_units,
        clean_client_title,
        sec_to_hms,
        combine_for_prompt,
        _hms_to_sec,
        UPLOAD_FOLDER,
        RUNS_BASE,
        QA_WINDOW_SEC,
        ENABLE_OCR,
        WRITE_FIVEMIN_ARTIFACTS,
        WIN_LABEL,
    )
    from app.chapter_generation import generate_chapters
    from app.qa_generation import (
        process_text_for_qa_and_notes,
        result_to_legacy_client_format,
        parse_modules_to_topics,
        extract_suggested_units_from_chapters,
        extract_suggested_units_from_topics,
    )
    from app.s3_storage import upload_video_artifacts
    from app.rag_chunking import chunk_and_embed_video

    if hasattr(generate_qa_and_notes_task, 'run'):
        generate_qa_and_notes = generate_qa_and_notes_task.run
    else:
        generate_qa_and_notes = generate_qa_and_notes_task

    IMPORTS_OK = True
    logger.info("✅ All pipeline imports successful")
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)
    logger.error(f"❌ Import error: {e}")


# ==================== Helper Functions ====================

def validate_input(job_input: dict) -> tuple[bool, str]:
    is_old_format = "PlayUrl" in job_input and "video_info" not in job_input
    if is_old_format:
        if not job_input.get("PlayUrl"):
            return False, "Missing required field: PlayUrl"
        for field in ["Id", "TeamId", "SectionNo"]:
            if field not in job_input:
                return False, f"Missing required field: {field}"
    else:
        video_url = job_input.get("video_url") or job_input.get("play_url") or job_input.get("PlayUrl")
        if not video_url:
            return False, "Missing required field: video_url or play_url"
        video_info = job_input.get("video_info") or job_input.get("VideoInfo")
        if not video_info:
            return False, "Missing required field: video_info"
        for field in ["Id", "TeamId", "SectionNo"]:
            if field not in video_info:
                return False, f"Missing required field in video_info: {field}"
    return True, ""


def normalize_input(job_input: dict) -> dict:
    is_old_format = "PlayUrl" in job_input and "video_info" not in job_input
    if is_old_format:
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
    if not webhook_url:
        return
    try:
        import requests
        response = requests.post(webhook_url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
        logger.info(f"📤 Webhook sent: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Webhook failed: {e}")


# ==================== Main Handler ====================

def handler(job: dict) -> dict:
    """
    RunPod Serverless Handler for Class Genius Video Processing.
    Full parity with tasks.py process_video_task.
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
        if not IMPORTS_OK:
            raise RuntimeError(f"Pipeline imports failed: {IMPORT_ERROR}")

        is_valid, error_msg = validate_input(job_input)
        if not is_valid:
            logger.error(f"❌ Input validation failed: {error_msg}")
            return {"status": "error", "error": error_msg, "error_type": "validation_error"}

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
        logger.info("\n" + "=" * 60)
        logger.info("📥 STAGE 1: VIDEO ACQUISITION")
        logger.info("=" * 60)

        if video_url.startswith("file://"):
            file_path = video_url.replace("file://", "")
            logger.info(f"📂 Using local file: {file_path}")
        else:
            file_path = download_video(video_url, f"{uuid.uuid4()}.mp4")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")

        # ==================== STAGE 2: ASR Processing ====================
        logger.info("\n" + "=" * 60)
        logger.info("🎤 STAGE 2: ASR TRANSCRIPTION")
        logger.info("=" * 60)

        processing_result = chapter_llama_asr_processing_fn(
            file_path, window_sec=QA_WINDOW_SEC, do_ocr=ENABLE_OCR
        )

        if not processing_result.get("success"):
            error_code = processing_result.get("error", "unknown")
            error_msg = processing_result.get("error_message", error_code)

            if error_code == "no_valid_audio":
                logger.error("❌ Video has no usable audio")
                result = {
                    "status": "error", "error": "no_valid_audio",
                    "error_message": "Video has no valid audio or speech content",
                    "video_info": video_info, "processing_time": time.time() - start_time
                }
                if not skip_client_post:
                    post_to_client_api(result)
                send_webhook(webhook_url, result)
                return result

            raise RuntimeError(f"ASR processing failed: {error_msg}")

        # ==================== STAGE 3: Chapter Generation ====================
        logger.info("\n" + "=" * 60)
        logger.info("📑 STAGE 3: CHAPTER GENERATION")
        logger.info("=" * 60)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(RUNS_BASE, f"{video_info['Id']}_{ts}")
        os.makedirs(run_dir, exist_ok=True)

        # ---- Persist VAD metrics ----
        vad_metrics = {
            "duration": processing_result.get("duration"),
            "speech_duration": processing_result.get("speech_duration"),
            "removed_duration": processing_result.get("removed_duration"),
            "speech_ratio": processing_result.get("speech_ratio"),
            "removed_ratio": processing_result.get("removed_ratio"),
        }
        try:
            with open(os.path.join(run_dir, "asr_vad_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(vad_metrics, f, indent=2, ensure_ascii=False)
        except Exception as _e:
            logger.warning("⚠️ Failed to write asr_vad_metrics.json: %s", _e)

        try:
            dur = float(vad_metrics["duration"] or 0.0)
            sp = float(vad_metrics["speech_duration"] or 0.0)
            rm = float(vad_metrics["removed_duration"] or max(0.0, dur - sp))
            if dur > 0:
                pct_kept = (sp / dur) * 100.0
                pct_removed = 100.0 - pct_kept
                logger.info("🧪 VAD summary: kept %.1f%% (%.1fs), removed %.1f%% (%.1fs) of %.1fs total",
                            pct_kept, sp, pct_removed, rm, dur)
        except Exception:
            pass

        # ---- Safe ASR-text load ----
        asr_file_path = processing_result.get("asr_file")
        if asr_file_path and Path(asr_file_path).is_file():
            logger.info(f"📝 ASR file: {asr_file_path} ({Path(asr_file_path).stat().st_size:,} bytes)")
            with open(asr_file_path, encoding="utf-8") as f:
                raw_asr_text = f.read()
                peek = "\n".join(raw_asr_text.splitlines()[:3])
                logger.info("🧪 ASR peek:\n%s", peek or "<empty>")
        else:
            logger.warning("⚠️ No usable asr_file; falling back to segments")
            audio_segs = processing_result.get("audio_segments", [])
            raw_asr_text = "\n".join(
                f"{sec_to_hms(int(s.get('start', 0)))}: {s.get('text', '')}".strip()
                for s in audio_segs if s.get("text")
            )
            if not raw_asr_text:
                raise RuntimeError("ASR stage failed and no segments available for fallback")

        ocr_filtered = processing_result.get("ocr_segments_filtered", [])
        duration = processing_result.get("duration")

        # ---- Extract educational metadata ----
        section_title = video_info.get("SectionTitle")
        units = video_info.get("Units", [])

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

        chaptering_result = generate_chapters(
            raw_asr_text=raw_asr_text, ocr_segments=ocr_filtered,
            video_title=video_info.get("OriginalFilename"),
            section_title=section_title, units=units, duration=duration,
            video_id=video_info["Id"], run_dir=Path(run_dir)
        )

        if isinstance(chaptering_result, tuple) and len(chaptering_result) == 2:
            chapters_dict, chapter_metadata = chaptering_result
            logger.info(f"✅ Received {len(chapters_dict)} chapters and metadata")
            logger.info(f"   • Metadata keys: {list(chapter_metadata.keys())}")
        else:
            chapters_dict = chaptering_result if isinstance(chaptering_result, dict) else {}
            chapter_metadata = None
            logger.warning("⚠️  Chapter generation returned old format (no metadata)")

        # ==================== STAGE 4: Q&A Generation ====================
        logger.info("\n" + "=" * 60)
        logger.info("📚 STAGE 4: Q&A GENERATION")
        logger.info("=" * 60)

        if section_title or units:
            logger.info("📚 Passing educational metadata to Q&A generation")
            if section_title:
                logger.info(f"   • Section: {section_title}")
            if units:
                logger.info(f"   • Units: {len(units)} learning units")

        qa_result = generate_qa_and_notes(
            processing_result, video_info, raw_asr_text,
            chapters_dict=chapters_dict, hierarchical_metadata=chapter_metadata,
            section_title=section_title, units=units,
            num_questions=num_questions, num_pages=num_pages
        )

        if not qa_result:
            raise RuntimeError("Q&A generation failed")

        # ==================== STAGE 5: Build Output (FULL PARITY with tasks.py) ====================
        logger.info("\n" + "=" * 60)
        logger.info("📦 STAGE 5: BUILD OUTPUT")
        logger.info("=" * 60)

        # ========== GENERATE SUGGESTED UNITS ==========
        logger.info("=" * 60)
        logger.info("🎓 GENERATING TEACHING SUGGESTIONS (workspace only)")
        logger.info("=" * 60)

        topics_list = []
        if chapter_metadata and chapter_metadata.get('modules_analysis'):
            try:
                topics_list = parse_modules_to_topics(chapter_metadata.get('modules_analysis', ''))
                logger.info(f"✅ Extracted {len(topics_list)} topics from chapter metadata")
            except Exception as e:
                logger.warning(f"⚠️ Failed to extract topics from metadata: {e}")

        suggested_units = []
        if topics_list and len(topics_list) >= 3:
            try:
                suggested_units = extract_suggested_units_from_topics(
                    topics_list=topics_list, chapters_dict=chapters_dict)
                logger.info(f"📚 Generated {len(suggested_units)} units from topics analysis")
                logger.info(f"   Method: Topic-based (highest educational quality)")
            except Exception as e:
                logger.warning(f"⚠️ Topic-based unit extraction failed: {e}")

        if not suggested_units and chapters_dict and len(chapters_dict) >= 3:
            try:
                suggested_units = extract_suggested_units_from_chapters(
                    chapters_dict=chapters_dict, max_units=5)
                logger.info(f"📚 Generated {len(suggested_units)} units from chapters")
                logger.info(f"   Method: Chapter-based (fallback)")
            except Exception as e:
                logger.warning(f"⚠️ Chapter-based unit extraction failed: {e}")

        if not suggested_units:
            logger.warning("⚠️ Insufficient data for unit generation - using empty list")

        logger.info("-" * 60)
        logger.info("📋 UNITS SUMMARY")
        logger.info(f"   • Original Units (from API): {len(units)}")
        logger.info(f"   • Suggested Units (AI-generated): {len(suggested_units)}")
        if suggested_units:
            for unit in suggested_units:
                logger.info(f"      {unit['UnitNo']}. {unit['Title']} @ {unit['Time']}")
        logger.info("=" * 60)

        # ========== METADATA COLLECTION ==========
        total_processing_time = time.time() - start_time
        audio_used = processing_result.get("audio_segments_used", processing_result.get("audio_segments", []))
        audio_raw = processing_result.get("audio_segments_raw", [])
        ocr_filtered = processing_result.get("ocr_segments_filtered", processing_result.get("ocr_segments", []))
        ocr_raw = processing_result.get("ocr_segments_raw", [])
        ocr_ctx_md = processing_result.get("ocr_context_markdown", "")

        # ========== PERSIST ARTIFACTS TO WORKSPACE ==========
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

        with open(os.path.join(run_dir, "chapters.json"), "w", encoding="utf-8") as f:
            json.dump(chapters_dict, f, indent=2, ensure_ascii=False)

        if chapter_metadata:
            with open(os.path.join(run_dir, "chapter_metadata.json"), "w", encoding="utf-8") as f:
                json.dump(chapter_metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved chapter metadata to: {run_dir}/chapter_metadata.json")

        # ========== TRANSFORM CHAPTERS FOR CLIENT ==========
        logger.info("🔄 Transforming chapters to client format...")
        navigation_units = transform_chapters_to_units(chapters_dict or {}) or []

        units_from_api = []
        if units and isinstance(units, list):
            logger.info(f"📚 Including {len(units)} Units from incoming API")
            for idx, unit in enumerate(units, start=1):
                if isinstance(unit, dict):
                    units_from_api.append({
                        "UnitNo": unit.get("UnitNo", idx),
                        "Title": clean_client_title(unit.get("Title", "")),
                        "Time": unit.get("Time", "")
                    })
        else:
            logger.info("ℹ️  No Units provided in incoming API")

        navigation_units = navigation_units or []
        units_from_api = units_from_api or []

        suggested_structured = None
        if chapter_metadata and isinstance(chapter_metadata.get("suggested_units_structured"), list):
            suggested_structured = chapter_metadata["suggested_units_structured"]

        # ========== 🔍 DIAGNOSTIC LOGGING FOR UNIT MAPPING ==========
        logger.info("=" * 60)
        logger.info("🔍 DIAGNOSTIC: UNIT MAPPING DATA")
        logger.info("=" * 60)

        logger.info(f"📥 Units from API: {len(units_from_api)}")
        for unit in units_from_api:
            logger.info(f"   • Unit {unit['UnitNo']}: {unit['Title'][:50]} | Time: '{unit.get('Time', 'EMPTY')}'")

        if suggested_structured:
            logger.info(f"📊 Structured SuggestedUnits: {len(suggested_structured)}")
            mapping = defaultdict(list)
            unmapped = []
            for su in suggested_structured:
                client_unit_no = su.get("ClientUnitNo")
                if client_unit_no:
                    mapping[client_unit_no].append({
                        "UnitNo": su.get("UnitNo"), "Title": su.get("Title", "")[:50], "Time": su.get("Time")
                    })
                else:
                    unmapped.append({
                        "UnitNo": su.get("UnitNo"), "Title": su.get("Title", "")[:50], "Time": su.get("Time")
                    })

            logger.info("📍 Mapping Summary:")
            for client_unit_no in sorted(mapping.keys()):
                chapters_mapped = mapping[client_unit_no]
                logger.info(f"   ClientUnit {client_unit_no} → {len(chapters_mapped)} chapters")
                logger.info(f"      First chapter: {chapters_mapped[0]['Title']} @ {chapters_mapped[0]['Time']}")
                if len(chapters_mapped) > 1:
                    logger.info(f"      Last chapter: {chapters_mapped[-1]['Title']} @ {chapters_mapped[-1]['Time']}")
            if unmapped:
                logger.info(f"⚠️ {len(unmapped)} SuggestedUnits have NO ClientUnitNo:")
                for item in unmapped[:3]:
                    logger.info(f"      • {item['Title']} @ {item['Time']}")
        else:
            logger.info("❌ No structured SuggestedUnits in metadata!")
            if chapter_metadata:
                logger.info(f"   Available metadata keys: {list(chapter_metadata.keys())}")
            else:
                logger.info("   No chapter_metadata at all!")

        if chapter_metadata and chapter_metadata.get("client_units_with_timestamps"):
            enriched = chapter_metadata["client_units_with_timestamps"]
            logger.info(f"✅ Found enriched units in metadata: {len(enriched)} units")
            for unit in enriched:
                logger.info(f"   • Unit {unit['UnitNo']}: {unit['Title'][:50]} | Time: '{unit.get('Time', 'EMPTY')}'")
        else:
            logger.info("⚠️ No enriched units in metadata - will use legacy fill method")

        logger.info("=" * 60)

        # ========== ✅ CHECK VALIDATION BEFORE USING ENRICHED UNITS ==========
        unit_validation = chapter_metadata.get("unit_validation", {}) if chapter_metadata else {}
        units_were_accepted = unit_validation.get("is_valid", False)

        if units_were_accepted and chapter_metadata and chapter_metadata.get("client_units_with_timestamps"):
            enriched_units = chapter_metadata["client_units_with_timestamps"]
            unit_diagnostics = chapter_metadata.get("unit_diagnostics", {})
            logger.info("=" * 60)
            logger.info("✅ USING VALIDATED ENRICHED UNITS FROM CHAPTER GENERATION")
            logger.info("=" * 60)
            units_from_api = []
            for unit in enriched_units:
                units_from_api.append({
                    "UnitNo": unit.get("UnitNo"),
                    "Title": clean_client_title(unit.get("Title", "")),
                    "Time": unit.get("Time", "")
                })
            unit_time_stats = {
                "filled_count": unit_diagnostics.get("units_found", 0),
                "skipped_existing_time": 0,
                "unmatched_units": unit_diagnostics.get("units_missing", 0),
                "unmatched_suggested": unit_diagnostics.get("unmapped_suggested_units", 0),
                "invalid_suggested_times": 0,
            }
            logger.info(f"✅ Enriched {len(units_from_api)} validated Units with timestamps")
            logger.info("   Method: Back-calculation from validated chapter generation")

        elif chapter_metadata and "unit_validation" in chapter_metadata:
            logger.warning("=" * 60)
            logger.warning("⚠️ UNITS REJECTED - RETURNING ORIGINALS WITHOUT TIMESTAMPS")
            logger.warning("=" * 60)
            logger.warning(f"   • Validation score: {unit_validation.get('score', 0):.2f}")
            logger.warning(f"   • Reason: {unit_validation.get('reason', '')}")
            units_from_api = []
            for unit in units:
                if isinstance(unit, dict):
                    units_from_api.append({
                        "UnitNo": unit.get("UnitNo"),
                        "Title": clean_client_title(unit.get("Title", "")),
                        "Time": ""
                    })
            unit_time_stats = {
                "filled_count": 0, "skipped_existing_time": 0,
                "unmatched_units": len(units_from_api), "unmatched_suggested": 0,
                "invalid_suggested_times": 0,
            }
            logger.warning(f"⚠️ Returned {len(units_from_api)} units WITHOUT timestamps (validation failed)")

        elif suggested_structured:
            logger.warning("⚠️ No validation metadata available; using legacy fill method")
            units_from_api, unit_time_stats = fill_unit_times_from_suggested_units(
                units_from_api, suggested_structured, only_fill_if_empty=True)
            logger.info("🧩 Using structured SuggestedUnits for Units Time fill (legacy)")
        else:
            unit_time_stats = {
                "filled_count": 0, "skipped_existing_time": 0,
                "unmatched_units": 0, "unmatched_suggested": 0,
                "invalid_suggested_times": 0,
            }
            logger.info("ℹ️ No structured SuggestedUnits available; skipping Units Time fill")

        logger.info(
            "🕒 Unit Time Fill Results: filled=%d skipped_existing=%d unmatched_units=%d unmatched_suggested=%d invalid_suggested_times=%d",
            unit_time_stats["filled_count"], unit_time_stats["skipped_existing_time"],
            unit_time_stats["unmatched_units"], unit_time_stats["unmatched_suggested"],
            unit_time_stats["invalid_suggested_times"],
        )

        # ========== 📊 LOG FINAL UNIT STATE ==========
        logger.info("=" * 60)
        logger.info("📋 FINAL UNITS STATE (For Client)")
        logger.info("=" * 60)
        for unit in units_from_api:
            time_status = "✅" if unit.get("Time") else "❌"
            logger.info(f"   {time_status} Unit {unit['UnitNo']}: {unit['Title'][:50]} | Time: '{unit.get('Time', 'EMPTY')}'")
        logger.info("=" * 60)

        # ========== SAVE COMPREHENSIVE WORKSPACE ARTIFACT ==========
        workspace_artifact = {
            **qa_result,
            "Units": units_from_api or [],
            "SuggestedUnits": navigation_units or [],
            "TeachingSuggestions": suggested_units or [],
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

        # ========== 📊 LOG WORKSPACE ARTIFACT SUMMARY ==========
        logger.info("=" * 60)
        logger.info("📦 WORKSPACE ARTIFACT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"   • Questions: {len(workspace_artifact.get('Questions', []))}")
        logger.info(f"   • CourseNote pages: {workspace_artifact.get('CourseNote', '').count('---') + 1}")
        logger.info(f"   • Units (from API): {len(workspace_artifact.get('Units', []))}")
        logger.info(f"   • SuggestedUnits (navigation): {len(workspace_artifact.get('SuggestedUnits', []))}")
        logger.info(f"   • TeachingSuggestions: {len(workspace_artifact.get('TeachingSuggestions', []))}")

        units_with_time = sum(1 for u in workspace_artifact.get('Units', []) if u.get('Time'))
        units_without_time = len(workspace_artifact.get('Units', [])) - units_with_time
        logger.info(f"   • Units with timestamps: {units_with_time}/{len(workspace_artifact.get('Units', []))}")
        if units_without_time > 0:
            logger.warning(f"   ⚠️ Units WITHOUT timestamps: {units_without_time}")
            for unit in workspace_artifact.get('Units', []):
                if not unit.get('Time'):
                    logger.warning(f"      - Unit {unit['UnitNo']}: {unit['Title'][:50]}")
        logger.info("=" * 60)

        workspace_path = os.path.join(run_dir, "full_processing_result.json")
        with open(workspace_path, "w", encoding="utf-8") as f:
            json.dump(workspace_artifact, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved full workspace artifact to {workspace_path}")

        # ========== BUILD CLEAN CLIENT PAYLOAD ==========
        ordered_units_from_api = [
            OrderedDict([
                ("UnitNo", u.get("UnitNo") if isinstance(u, dict) else u["UnitNo"]),
                ("Title", u.get("Title") if isinstance(u, dict) else u["Title"]),
                ("Time", u.get("Time", "") if isinstance(u, dict) else u.get("Time", ""))
            ]) for u in units_from_api
        ]
        ordered_navigation_units = [
            OrderedDict([
                ("UnitNo", u.get("UnitNo")), ("Title", u.get("Title")), ("Time", u.get("Time", ""))
            ]) for u in navigation_units
        ]

        client_payload = OrderedDict([
            ("Id", qa_result["Id"]),
            ("TeamId", qa_result["TeamId"]),
            ("SectionNo", qa_result["SectionNo"]),
            ("CreatedAt", qa_result["CreatedAt"]),
            ("Questions", qa_result["Questions"]),
            ("CourseNote", qa_result["CourseNote"]),
            ("Units", ordered_units_from_api),
            ("SuggestedUnits", ordered_navigation_units)
        ])

        # ========== 📤 CLIENT PAYLOAD SUMMARY ==========
        logger.info("=" * 60)
        logger.info("📤 CLIENT PAYLOAD SUMMARY (What's being sent)")
        logger.info("=" * 60)
        logger.info(f"   • Questions: {len(client_payload['Questions'])}")
        logger.info(f"   • CourseNote: {'Present' if client_payload.get('CourseNote') else 'Empty'}")
        logger.info(f"   • Units (from API): {len(client_payload['Units'])}")
        logger.info(f"   • SuggestedUnits (navigation): {len(client_payload['SuggestedUnits'])}")
        logger.info("")
        logger.info("📍 Client Units Detail:")
        for unit in client_payload['Units']:
            time_status = "✅" if unit.get('Time') else "❌ MISSING"
            logger.info(f"   {time_status} Unit {unit['UnitNo']}: {unit['Title'][:50]} @ {unit.get('Time', 'NO TIME')}")

        missing_times = sum(1 for u in client_payload['Units'] if not u.get('Time'))
        if missing_times > 0:
            logger.error(f"❌ CRITICAL: {missing_times}/{len(client_payload['Units'])} Units have NO timestamps!")
            logger.error("   This may indicate chapter generation mapping failure")
        else:
            logger.info(f"✅ All {len(client_payload['Units'])} Units have timestamps")
        logger.info("=" * 60)

        client_payload_json = json.dumps(client_payload, ensure_ascii=False, sort_keys=False)

        with open(os.path.join(run_dir, "client_payload.json"), "w", encoding="utf-8") as f:
            f.write(client_payload_json)
        logger.info(f"💾 Saved clean client payload to {run_dir}/client_payload.json")

        with open(os.path.join(run_dir, "qa_and_notes.json"), "w", encoding="utf-8") as f:
            json.dump(workspace_artifact, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved legacy output to {run_dir}/qa_and_notes.json")

        transcript_path = os.path.join(run_dir, "merged_transcript.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            for seg in audio_used:
                s = seg.get("start", 0); e = seg.get("end", 0); t = seg.get("text", "")
                f.write(f"[{s:.1f} - {e:.1f}] {t}\n")
        logger.info(f"💾 Saved transcript to {transcript_path}")

        # ---- Guardrail: SuggestedUnits must exactly reflect chapters_dict ----
        chapters_sorted = sorted((chapters_dict or {}).items(), key=lambda x: _hms_to_sec(x[0]) or 10**12)
        nav_sorted = [(u.get("Time"), u.get("Title")) for u in (navigation_units or [])]

        if len(nav_sorted) != len(chapters_sorted):
            logger.warning("⚠️ navigation_units count %d != chapters_dict count %d",
                           len(nav_sorted), len(chapters_sorted))

        chapters_times = [ts for ts, _ in chapters_sorted]
        nav_times = [ts for ts, _ in nav_sorted]

        if chapters_times and nav_times and chapters_times != nav_times:
            logger.error("❌ navigation_units times do not match chapters_dict times. Refusing to send.")
            logger.error("   chapters_times=%s", chapters_times)
            logger.error("   nav_times=%s", nav_times)
            raise RuntimeError("Refusing to send mismatched navigation SuggestedUnits")

        # ========== UPLOAD TO S3 ==========
        try:
            upload_video_artifacts(
                run_dir=run_dir, video_info=video_info,
                processing_result=processing_result, chapters_dict=chapters_dict,
                chapter_metadata=chapter_metadata, client_payload=client_payload,
                raw_asr_text=raw_asr_text,
            )
        except Exception as e:
            logger.warning(f"⚠️ S3 upload failed (non-fatal): {e}")

        # ========== RAG: CHUNK + EMBED + PINECONE ==========
        try:
            chunk_and_embed_video(
                transcript_text=raw_asr_text, chapters_dict=chapters_dict,
                team_id=video_info["TeamId"], video_id=video_info["Id"],
                section_title=video_info.get("SectionTitle", ""),
                section_no=video_info.get("SectionNo"),
                video_duration=processing_result.get("duration"),
            )
        except Exception as e:
            logger.warning(f"⚠️ RAG pipeline failed (non-fatal): {e}")

        # ========== SEND CLEAN PAYLOAD TO CLIENT API ==========
        logger.info(f"✅ Complete pipeline finished in {total_processing_time:.1f}s")

        if not skip_client_post:
            post_to_client_api(client_payload)

        send_webhook(webhook_url, client_payload)
        return client_payload_json

    except Exception as e:
        error_time = time.time() - start_time
        logger.error("=" * 60)
        logger.error(f"❌ JOB FAILED: {e}")
        logger.error(traceback.format_exc())
        logger.error("=" * 60)
        error_result = {
            "status": "error", "error": str(e), "error_type": type(e).__name__,
            "traceback": traceback.format_exc(), "processing_time": error_time
        }
        send_webhook(webhook_url, error_result)
        return error_result

    finally:
        if file_path and os.path.exists(file_path) and not file_path.startswith("file://"):
            try:
                os.remove(file_path)
                logger.info(f"🗑️ Cleaned up: {file_path}")
            except Exception as e:
                logger.warning(f"⚠️ Cleanup failed: {e}")


# ==================== Health Check Handler ====================

def health_check(job: dict) -> dict:
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
        "return_aggregate_stream": True
    })
