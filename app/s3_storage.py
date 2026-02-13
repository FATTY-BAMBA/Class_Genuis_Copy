# app/s3_storage.py
"""
S3 Storage Module for Class Genius Pipeline
============================================
Uploads processed video artifacts to S3 after each pipeline run.

Drop-in integration: Call `upload_video_artifacts()` from tasks.py 
right after all files are saved to run_dir and before post_to_client_api().

Required env vars:
    S3_BUCKET           - Bucket name (e.g., "classgenius-edu-data")
    AWS_ACCESS_KEY_ID   - IAM credentials
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION  - e.g., "ap-northeast-1"

Optional:
    S3_ENABLED          - "true"/"false" (default: "true")
    S3_PREFIX           - Optional prefix (default: "")

If S3_BUCKET is not set, all uploads silently no-op (safe for local dev).
"""

import os
import json
import hashlib
import logging
import time
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------- Lazy boto3 import (avoids crash if not installed) ----------
_s3_client = None

def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        try:
            import boto3
            _s3_client = boto3.client("s3")
            logger.info("✅ S3 client initialized")
        except ImportError:
            logger.warning("⚠️ boto3 not installed. S3 uploads disabled. pip install boto3")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to create S3 client: {e}")
            return None
    return _s3_client


def _is_s3_enabled():
    """Check if S3 is configured and enabled."""
    bucket = os.getenv("S3_BUCKET", "")
    enabled = os.getenv("S3_ENABLED", "true").lower() == "true"
    return bool(bucket) and enabled


def _s3_key(team_id, video_id, category, filename):
    """Build S3 key: {prefix}/raw|processed/{team_id}/{video_id}/{filename}"""
    prefix = os.getenv("S3_PREFIX", "").strip("/")
    parts = [prefix, category, str(team_id), str(video_id), filename]
    return "/".join(p for p in parts if p)


def _upload_file(local_path, s3_key):
    """Upload a single file to S3. Returns True on success."""
    client = _get_s3_client()
    if not client:
        return False
    
    bucket = os.getenv("S3_BUCKET")
    try:
        client.upload_file(str(local_path), bucket, s3_key)
        size_kb = os.path.getsize(local_path) / 1024
        logger.info(f"   ☁️  Uploaded {s3_key} ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        logger.error(f"   ❌ Failed to upload {s3_key}: {e}")
        return False


def _upload_json(data, s3_key):
    """Upload a dict/list as JSON to S3."""
    client = _get_s3_client()
    if not client:
        return False
    
    bucket = os.getenv("S3_BUCKET")
    try:
        body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        client.put_object(Bucket=bucket, Key=s3_key, Body=body, ContentType="application/json")
        logger.info(f"   ☁️  Uploaded {s3_key} ({len(body)/1024:.1f} KB)")
        return True
    except Exception as e:
        logger.error(f"   ❌ Failed to upload {s3_key}: {e}")
        return False


def _hash_content(text):
    """SHA-256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ==================== Main Entry Point ====================

def upload_video_artifacts(
    run_dir,
    video_info,
    processing_result,
    chapters_dict=None,
    chapter_metadata=None,
    client_payload=None,
    raw_asr_text="",
):
    """
    Upload all artifacts for one processed video to S3.
    
    Call this from process_video_task() AFTER all local files are saved
    and BEFORE post_to_client_api().
    
    This function is safe to call even if S3 is not configured —
    it will log a warning and return immediately.
    
    Args:
        run_dir:            Path to /workspace/runs/{video_id}_{timestamp}/
        video_info:         Original API request payload (dict)
        processing_result:  ASR processing result (dict)
        chapters_dict:      Chapter timestamps dict (optional)
        chapter_metadata:   Chapter generation metadata (optional)
        client_payload:     Final client payload (optional)
        raw_asr_text:       Raw ASR text string
    
    Returns:
        dict with upload summary, or None if S3 is disabled
    """
    if not _is_s3_enabled():
        logger.info("ℹ️  S3 not configured. Skipping upload. Set S3_BUCKET to enable.")
        return None
    
    t0 = time.time()
    team_id = video_info.get("TeamId", "unknown")
    video_id = video_info.get("Id", "unknown")
    
    logger.info("=" * 60)
    logger.info("☁️  S3 UPLOAD STARTING")
    logger.info(f"   Team: {team_id} | Video: {video_id}")
    logger.info("=" * 60)
    
    uploaded = 0
    failed = 0
    run_path = Path(run_dir)
    
    # ========== 1. RAW DATA (immutable source of truth) ==========
    
    # 1a. Raw transcript
    transcript_file = run_path / "merged_transcript.txt"
    if transcript_file.is_file():
        key = _s3_key(team_id, video_id, "raw", "transcript.txt")
        if _upload_file(transcript_file, key):
            uploaded += 1
        else:
            failed += 1
    
    # 1b. Input metadata (the original API request — critical for reprocessing)
    key = _s3_key(team_id, video_id, "raw", "input_metadata.json")
    if _upload_json(video_info, key):
        uploaded += 1
    else:
        failed += 1
    
    # 1c. Raw ASR segments (preserves exact Whisper output)
    raw_segs_file = run_path / "audio_segments.raw.json"
    if raw_segs_file.is_file():
        key = _s3_key(team_id, video_id, "raw", "asr_segments.json")
        if _upload_file(raw_segs_file, key):
            uploaded += 1
        else:
            failed += 1
    
    # ========== 2. PROCESSED DATA (regeneratable from raw) ==========
    
    # 2a. Chapters
    if chapters_dict:
        key = _s3_key(team_id, video_id, "processed", "chapters.json")
        if _upload_json(chapters_dict, key):
            uploaded += 1
        else:
            failed += 1
    
    # 2b. Chapter metadata (quality scores, course summary, learning objectives)
    if chapter_metadata:
        key = _s3_key(team_id, video_id, "processed", "chapter_metadata.json")
        if _upload_json(chapter_metadata, key):
            uploaded += 1
        else:
            failed += 1
    
    # 2c. Questions (extracted from client_payload)
    if client_payload and client_payload.get("Questions"):
        key = _s3_key(team_id, video_id, "processed", "questions.json")
        if _upload_json(client_payload["Questions"], key):
            uploaded += 1
        else:
            failed += 1
    
    # 2d. Course notes
    if client_payload and client_payload.get("CourseNote"):
        key = _s3_key(team_id, video_id, "processed", "course_notes.md")
        client = _get_s3_client()
        if client:
            try:
                bucket = os.getenv("S3_BUCKET")
                body = client_payload["CourseNote"].encode("utf-8")
                client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="text/markdown; charset=utf-8")
                logger.info(f"   ☁️  Uploaded {key} ({len(body)/1024:.1f} KB)")
                uploaded += 1
            except Exception as e:
                logger.error(f"   ❌ Failed to upload {key}: {e}")
                failed += 1
    
    # 2e. Course summary (extracted from chapter_metadata if available)
    if chapter_metadata and chapter_metadata.get("course_summary"):
        key = _s3_key(team_id, video_id, "processed", "course_summary.json")
        if _upload_json(chapter_metadata["course_summary"], key):
            uploaded += 1
        else:
            failed += 1
    
    # ========== 3. PROCESSING METADATA (for pipeline tracking) ==========
    
    processing_meta = {
        "video_id": video_id,
        "team_id": team_id,
        "section_no": video_info.get("SectionNo"),
        "section_title": video_info.get("SectionTitle", ""),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "1.0.0",
        "asr_model": "whisper-large-v2",
        "llm_model": "gpt-4o",
        "embedding_model": None,
        "chunk_strategy": None,
        "video_duration_seconds": processing_result.get("duration"),
        "asr_text_length": len(raw_asr_text),
        "asr_method": processing_result.get("method", "unknown"),
        "gpu_used": processing_result.get("gpu_used", False),
        "speech_ratio": processing_result.get("speech_ratio"),
        "chapters_generated": len(chapters_dict) if chapters_dict else 0,
        "questions_generated": len(client_payload.get("Questions", [])) if client_payload else 0,
        "educational_quality_score": (
            chapter_metadata.get("educational_quality_score") if chapter_metadata else None
        ),
        "content_hash": _hash_content(raw_asr_text) if raw_asr_text else None,
        "units_from_api": len(video_info.get("Units", [])),
    }
    
    key = _s3_key(team_id, video_id, "processed", "processing_meta.json")
    if _upload_json(processing_meta, key):
        uploaded += 1
    else:
        failed += 1
    
    # ========== 4. UPDATE TEAM MANIFEST ==========
    _update_manifest(team_id, video_id, processing_meta)
    
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"☁️  S3 UPLOAD COMPLETE in {elapsed:.1f}s")
    logger.info(f"   ✅ Uploaded: {uploaded} | ❌ Failed: {failed}")
    logger.info("=" * 60)
    
    return {
        "uploaded": uploaded,
        "failed": failed,
        "elapsed": elapsed,
        "team_id": team_id,
        "video_id": video_id,
    }


def _update_manifest(team_id, video_id, processing_meta):
    """
    Read-modify-write the per-team manifest in S3.
    
    Manifest lives at: manifests/{team_id}.json
    """
    client = _get_s3_client()
    if not client:
        return
    
    bucket = os.getenv("S3_BUCKET")
    prefix = os.getenv("S3_PREFIX", "").strip("/")
    manifest_key = f"{prefix}/manifests/{team_id}.json" if prefix else f"manifests/{team_id}.json"
    
    # Try to read existing manifest
    manifest = {"team_id": str(team_id), "videos": {}, "last_updated": ""}
    try:
        response = client.get_object(Bucket=bucket, Key=manifest_key)
        manifest = json.loads(response["Body"].read().decode("utf-8"))
    except client.exceptions.NoSuchKey:
        logger.info(f"   📝 Creating new manifest for team {team_id}")
    except Exception as e:
        logger.warning(f"   ⚠️ Could not read manifest for team {team_id}: {e}")
    
    # Update with this video
    manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
    manifest["videos"][str(video_id)] = {
        "section_no": processing_meta.get("section_no"),
        "section_title": processing_meta.get("section_title", ""),
        "processed_at": processing_meta.get("processed_at"),
        "pipeline_version": processing_meta.get("pipeline_version"),
        "video_duration_seconds": processing_meta.get("video_duration_seconds"),
        "chapters_generated": processing_meta.get("chapters_generated", 0),
        "questions_generated": processing_meta.get("questions_generated", 0),
        "has_chunks": False,
        "has_embeddings": False,
        "status": "processed",
    }
    
    # Write back
    try:
        body = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
        client.put_object(Bucket=bucket, Key=manifest_key, Body=body, ContentType="application/json")
        logger.info(f"   ☁️  Updated manifest: {manifest_key} ({len(manifest['videos'])} videos)")
    except Exception as e:
        logger.error(f"   ❌ Failed to update manifest: {e}")


# ==================== Backfill Utility ====================

def backfill_from_runs_dir(runs_base="/workspace/runs"):
    """
    One-time utility to upload all existing run directories to S3.
    
    Run this manually on your RunPod worker:
        python -c "from app.s3_storage import backfill_from_runs_dir; backfill_from_runs_dir()"
    
    It reads the saved artifacts from each /workspace/runs/{videoId}_{timestamp}/ 
    directory and uploads them to S3.
    """
    if not _is_s3_enabled():
        print("S3 not configured. Set S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
        return
    
    runs_path = Path(runs_base)
    if not runs_path.exists():
        print(f"Runs directory not found: {runs_base}")
        return
    
    dirs = sorted([d for d in runs_path.iterdir() if d.is_dir()])
    print(f"Found {len(dirs)} run directories to backfill")
    
    success = 0
    errors = 0
    
    for run_dir in dirs:
        # Try to extract video_info from saved files
        try:
            # Try client_payload first (has Id, TeamId, SectionNo)
            client_file = run_dir / "client_payload.json"
            workspace_file = run_dir / "full_processing_result.json"
            
            payload = None
            if client_file.is_file():
                with open(client_file, encoding="utf-8") as f:
                    payload = json.load(f)
            elif workspace_file.is_file():
                with open(workspace_file, encoding="utf-8") as f:
                    payload = json.load(f)
            
            if not payload or "Id" not in payload:
                print(f"  ⚠️  Skipping {run_dir.name}: no Id found")
                errors += 1
                continue
            
            video_info = {
                "Id": payload["Id"],
                "TeamId": payload.get("TeamId", "unknown"),
                "SectionNo": payload.get("SectionNo"),
                "SectionTitle": payload.get("SectionTitle", ""),
                "Units": payload.get("Units", []),
            }
            
            # Build minimal processing_result
            processing_result = {
                "duration": payload.get("processing_metadata", {}).get("duration"),
                "method": payload.get("processing_metadata", {}).get("processing_method", "backfill"),
                "gpu_used": payload.get("processing_metadata", {}).get("gpu_used"),
                "speech_ratio": payload.get("processing_metadata", {}).get("speech_ratio"),
            }
            
            # Read raw ASR text
            raw_asr_text = ""
            transcript_file = run_dir / "merged_transcript.txt"
            if transcript_file.is_file():
                raw_asr_text = transcript_file.read_text(encoding="utf-8")
            
            # Read chapters
            chapters_dict = None
            chapters_file = run_dir / "chapters.json"
            if chapters_file.is_file():
                with open(chapters_file, encoding="utf-8") as f:
                    chapters_dict = json.load(f)
            elif payload.get("chapters"):
                chapters_dict = payload["chapters"]
            
            # Read chapter metadata
            chapter_metadata = None
            meta_file = run_dir / "chapter_metadata.json"
            if meta_file.is_file():
                with open(meta_file, encoding="utf-8") as f:
                    chapter_metadata = json.load(f)
            
            print(f"  📤 Uploading video {video_info['Id']} (team {video_info['TeamId']})...")
            
            result = upload_video_artifacts(
                run_dir=str(run_dir),
                video_info=video_info,
                processing_result=processing_result,
                chapters_dict=chapters_dict,
                chapter_metadata=chapter_metadata,
                client_payload=payload,
                raw_asr_text=raw_asr_text,
            )
            
            if result and result["failed"] == 0:
                success += 1
            else:
                errors += 1
                
        except Exception as e:
            print(f"  ❌ Error processing {run_dir.name}: {e}")
            errors += 1
    
    print(f"\nBackfill complete: {success} succeeded, {errors} failed out of {len(dirs)} total")
