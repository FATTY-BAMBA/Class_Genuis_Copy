# app/rag_chunking.py
"""
RAG Chunking Pipeline for Class Genius
========================================
Reads transcripts + chapters from S3, creates semantic chunks using
chapter boundaries, generates embeddings, and upserts to Pinecone.

Can run:
  1. Inline after S3 upload (add to process_video_task)
  2. As a standalone batch job for existing S3 data
  3. As a one-time backfill for all videos in S3
  
Required env vars:
    S3_BUCKET                - Your S3 bucket
    AWS_ACCESS_KEY_ID        - IAM credentials
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION

    PINECONE_API_KEY         - Pinecone API key
    PINECONE_INDEX_NAME      - Index name (e.g., "classgenius-edu")

    OPENAI_API_KEY           - For embeddings (text-embedding-3-small)

Optional:
    CHUNK_TARGET_TOKENS      - Target chunk size (default: 400)
    CHUNK_MAX_TOKENS         - Max chunk size (default: 600)
    CHUNK_OVERLAP_TOKENS     - Overlap between chunks (default: 50)
    EMBEDDING_MODEL          - OpenAI model (default: "text-embedding-3-small")
    EMBEDDING_DIMENSIONS     - Vector dimensions (default: 1536)
    RAG_ENABLED              - "true"/"false" (default: "false")
"""

import os
import json
import re
import logging
import time
import hashlib
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------- Config ----------
CHUNK_TARGET_TOKENS = int(os.getenv("CHUNK_TARGET_TOKENS", "400"))
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "600"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))


def _is_rag_enabled():
    return os.getenv("RAG_ENABLED", "false").lower() == "true"


# ==================== Token Estimation ====================

def _estimate_tokens(text):
    """
    Rough token estimation for Chinese/mixed text.
    Chinese: ~1.5 tokens per character
    English/punctuation: ~0.25 tokens per word
    Good enough for chunking — we don't need exact counts.
    """
    if not text:
        return 0
    # Count CJK characters
    cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))
    # Remaining is roughly English/numbers/punctuation
    non_cjk = len(text) - cjk_chars
    return int(cjk_chars * 1.5 + non_cjk * 0.3)


# ==================== Transcript Parsing ====================

def _parse_transcript(transcript_text):
    """
    Parse timestamped transcript into segments.
    
    Handles two formats:
      1. "[start - end] text"     (merged_transcript.txt format)
      2. "HH:MM:SS: text"        (raw ASR format)
    
    Returns: list of {"start": float, "end": float, "text": str}
    """
    segments = []
    
    for line in transcript_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        
        # Format 1: [0.0 - 15.2] text
        m1 = re.match(r'\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s*(.*)', line)
        if m1:
            segments.append({
                "start": float(m1.group(1)),
                "end": float(m1.group(2)),
                "text": m1.group(3).strip()
            })
            continue
        
        # Format 2: HH:MM:SS: text
        m2 = re.match(r'(\d{2}):(\d{2}):(\d{2}):\s*(.*)', line)
        if m2:
            h, m, s = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            start = h * 3600 + m * 60 + s
            segments.append({
                "start": float(start),
                "end": float(start),  # Will be inferred
                "text": m2.group(4).strip()
            })
            continue
    
    # Infer end times for format 2 (where end == start)
    for i in range(len(segments)):
        if segments[i]["end"] == segments[i]["start"]:
            if i < len(segments) - 1:
                segments[i]["end"] = segments[i + 1]["start"]
            else:
                segments[i]["end"] = segments[i]["start"] + 30  # last segment
    
    return segments


def _hms_to_sec(hms):
    """Convert "HH:MM:SS" to seconds."""
    try:
        parts = hms.strip().split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return 0
    except Exception:
        return 0


def _sec_to_hms(sec):
    """Convert seconds to "HH:MM:SS"."""
    h = int(sec) // 3600
    m = (int(sec) % 3600) // 60
    s = int(sec) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ==================== Chapter-Based Chunking ====================

def chunk_transcript_by_chapters(transcript_text, chapters_dict, video_duration=None):
    """
    Core chunking logic: split transcript using chapter boundaries,
    then sub-chunk within chapters if they exceed CHUNK_MAX_TOKENS.
    
    Args:
        transcript_text: Raw transcript string (timestamped)
        chapters_dict:   {"HH:MM:SS": "Chapter Title", ...}
        video_duration:  Total video duration in seconds (optional)
    
    Returns:
        list of chunk dicts:
        [{
            "chunk_id": "ch0_0",
            "text": "...",
            "chapter_title": "...",
            "chapter_index": 0,
            "chunk_index": 0,
            "start_time": 120.0,
            "end_time": 450.0,
            "start_hms": "00:02:00",
            "end_hms": "00:07:30",
            "token_estimate": 380,
        }, ...]
    """
    segments = _parse_transcript(transcript_text)
    if not segments:
        logger.warning("No segments parsed from transcript")
        return []
    
    # Sort chapters by timestamp
    chapter_boundaries = []
    for ts, title in sorted(chapters_dict.items(), key=lambda x: _hms_to_sec(x[0])):
        chapter_boundaries.append({
            "start_sec": _hms_to_sec(ts),
            "title": title,
            "timestamp": ts,
        })
    
    if not chapter_boundaries:
        # No chapters — treat entire transcript as one chapter
        logger.warning("No chapters provided. Chunking entire transcript as single chapter.")
        chapter_boundaries = [{"start_sec": 0, "title": "Full Transcript", "timestamp": "00:00:00"}]
    
    # Determine end time for each chapter (start of next chapter, or video end)
    max_time = video_duration or (segments[-1]["end"] if segments else 0)
    for i, ch in enumerate(chapter_boundaries):
        if i < len(chapter_boundaries) - 1:
            ch["end_sec"] = chapter_boundaries[i + 1]["start_sec"]
        else:
            ch["end_sec"] = max_time
    
    # Assign segments to chapters
    all_chunks = []
    
    for ch_idx, chapter in enumerate(chapter_boundaries):
        ch_start = chapter["start_sec"]
        ch_end = chapter["end_sec"]
        ch_title = chapter["title"]
        
        # Collect segments that fall within this chapter's time range
        chapter_segments = [
            seg for seg in segments
            if seg["start"] >= ch_start - 1 and seg["start"] < ch_end
        ]
        
        if not chapter_segments:
            continue
        
        # Concatenate text for this chapter
        chapter_text = " ".join(seg["text"] for seg in chapter_segments if seg.get("text"))
        chapter_tokens = _estimate_tokens(chapter_text)
        
        if chapter_tokens <= CHUNK_MAX_TOKENS:
            # Chapter fits in one chunk
            all_chunks.append({
                "chunk_id": f"ch{ch_idx}_0",
                "text": chapter_text,
                "chapter_title": ch_title,
                "chapter_index": ch_idx,
                "chunk_index": 0,
                "start_time": chapter_segments[0]["start"],
                "end_time": chapter_segments[-1]["end"],
                "start_hms": _sec_to_hms(chapter_segments[0]["start"]),
                "end_hms": _sec_to_hms(chapter_segments[-1]["end"]),
                "token_estimate": chapter_tokens,
            })
        else:
            # Sub-chunk within this chapter
            sub_chunks = _sub_chunk_segments(
                chapter_segments, ch_idx, ch_title,
                target_tokens=CHUNK_TARGET_TOKENS,
                max_tokens=CHUNK_MAX_TOKENS,
                overlap_tokens=CHUNK_OVERLAP_TOKENS,
            )
            all_chunks.extend(sub_chunks)
    
    logger.info(
        f"📦 Chunked transcript: {len(chapter_boundaries)} chapters → {len(all_chunks)} chunks "
        f"(avg {sum(c['token_estimate'] for c in all_chunks) // max(len(all_chunks), 1)} tokens/chunk)"
    )
    
    return all_chunks


def _sub_chunk_segments(segments, chapter_index, chapter_title, 
                        target_tokens, max_tokens, overlap_tokens):
    """
    Split a long chapter's segments into sub-chunks of ~target_tokens,
    with overlap_tokens of context carried forward.
    """
    chunks = []
    current_texts = []
    current_tokens = 0
    current_start = segments[0]["start"]
    chunk_idx = 0
    
    for seg in segments:
        seg_text = seg.get("text", "").strip()
        if not seg_text:
            continue
        
        seg_tokens = _estimate_tokens(seg_text)
        
        # If adding this segment would exceed max, flush current chunk
        if current_tokens + seg_tokens > max_tokens and current_texts:
            chunk_text = " ".join(current_texts)
            chunks.append({
                "chunk_id": f"ch{chapter_index}_{chunk_idx}",
                "text": chunk_text,
                "chapter_title": chapter_title,
                "chapter_index": chapter_index,
                "chunk_index": chunk_idx,
                "start_time": current_start,
                "end_time": seg["start"],
                "start_hms": _sec_to_hms(current_start),
                "end_hms": _sec_to_hms(seg["start"]),
                "token_estimate": _estimate_tokens(chunk_text),
            })
            chunk_idx += 1
            
            # Carry overlap: keep last ~overlap_tokens worth of text
            overlap_texts = []
            overlap_count = 0
            for t in reversed(current_texts):
                t_tokens = _estimate_tokens(t)
                if overlap_count + t_tokens > overlap_tokens:
                    break
                overlap_texts.insert(0, t)
                overlap_count += t_tokens
            
            current_texts = overlap_texts
            current_tokens = overlap_count
            current_start = seg["start"]
        
        current_texts.append(seg_text)
        current_tokens += seg_tokens
    
    # Flush remaining
    if current_texts:
        chunk_text = " ".join(current_texts)
        chunks.append({
            "chunk_id": f"ch{chapter_index}_{chunk_idx}",
            "text": chunk_text,
            "chapter_title": chapter_title,
            "chapter_index": chapter_index,
            "chunk_index": chunk_idx,
            "start_time": current_start,
            "end_time": segments[-1]["end"],
            "start_hms": _sec_to_hms(current_start),
            "end_hms": _sec_to_hms(segments[-1]["end"]),
            "token_estimate": _estimate_tokens(chunk_text),
        })
    
    return chunks


# ==================== Embeddings ====================

def generate_embeddings(chunks, batch_size=100):
    """
    Generate embeddings for chunks using OpenAI's embedding API.
    
    Returns: list of embedding vectors (same order as chunks)
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed")
        return None
    
    client = OpenAI()  # Uses OPENAI_API_KEY env var
    
    texts = [c["text"] for c in chunks]
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
                dimensions=EMBEDDING_DIMENSIONS,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            logger.info(f"   🧮 Embedded batch {i // batch_size + 1} ({len(batch)} chunks)")
        except Exception as e:
            logger.error(f"   ❌ Embedding failed for batch {i // batch_size + 1}: {e}")
            return None
    
    return all_embeddings


# ==================== Pinecone ====================

def upsert_to_pinecone(chunks, embeddings, team_id, video_id, 
                       section_title="", section_no=None):
    """
    Upsert chunk embeddings to Pinecone with rich metadata.
    
    Each vector ID format: {team_id}_{video_id}_ch{chapter_idx}_{chunk_idx}
    """
    try:
        from pinecone import Pinecone
    except ImportError:
        logger.error("pinecone package not installed. pip install pinecone")
        return False
    
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "classgenius-edu")
    
    if not api_key:
        logger.warning("⚠️ PINECONE_API_KEY not set. Skipping upsert.")
        return False
    
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vector_id = f"{team_id}_{video_id}_{chunk['chunk_id']}"
        
        metadata = {
            "team_id": str(team_id),
            "video_id": str(video_id),
            "section_title": section_title or "",
            "section_no": section_no or 0,
            "chapter_title": chunk["chapter_title"],
            "chapter_index": chunk["chapter_index"],
            "chunk_index": chunk["chunk_index"],
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "start_hms": chunk["start_hms"],
            "end_hms": chunk["end_hms"],
            "token_count": chunk["token_estimate"],
            "text": chunk["text"][:1000],  # Pinecone metadata limit
        }
        
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": metadata,
        })
    
    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            index.upsert(vectors=batch)
            logger.info(f"   📌 Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")
        except Exception as e:
            logger.error(f"   ❌ Pinecone upsert failed: {e}")
            return False
    
    logger.info(f"   ✅ Upserted {len(vectors)} vectors to Pinecone index '{index_name}'")
    return True


# ==================== S3 Integration ====================

def _save_chunks_to_s3(chunks, embeddings, team_id, video_id):
    """Save chunks.json to S3 for caching/audit."""
    try:
        from app.s3_storage import _upload_json, _s3_key, _is_s3_enabled
    except ImportError:
        return
    
    if not _is_s3_enabled():
        return
    
    # Save chunks (without embeddings — those go to Pinecone)
    chunks_data = {
        "video_id": video_id,
        "team_id": team_id,
        "chunked_at": datetime.now(timezone.utc).isoformat(),
        "chunk_strategy": "chapter_based",
        "chunk_target_tokens": CHUNK_TARGET_TOKENS,
        "chunk_max_tokens": CHUNK_MAX_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "total_chunks": len(chunks),
        "chunks": chunks,  # Full chunk data (text, timestamps, metadata)
    }
    
    key = _s3_key(team_id, video_id, "processed", "chunks/chunks.json")
    _upload_json(chunks_data, key)


def _update_manifest_rag(team_id, video_id):
    """Update manifest to mark video as chunked + embedded."""
    try:
        from app.s3_storage import _get_s3_client
    except ImportError:
        return
    
    client = _get_s3_client()
    if not client:
        return
    
    bucket = os.getenv("S3_BUCKET")
    prefix = os.getenv("S3_PREFIX", "").strip("/")
    manifest_key = f"{prefix}/manifests/{team_id}.json" if prefix else f"manifests/{team_id}.json"
    
    try:
        response = client.get_object(Bucket=bucket, Key=manifest_key)
        manifest = json.loads(response["Body"].read().decode("utf-8"))
    except Exception:
        return  # Manifest doesn't exist yet
    
    vid_str = str(video_id)
    if vid_str in manifest.get("videos", {}):
        manifest["videos"][vid_str]["has_chunks"] = True
        manifest["videos"][vid_str]["has_embeddings"] = True
        manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        body = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
        client.put_object(Bucket=bucket, Key=manifest_key, Body=body, ContentType="application/json")
        logger.info(f"   ☁️ Updated manifest: {vid_str} → has_chunks=true, has_embeddings=true")


# ==================== Main Entry Points ====================

def chunk_and_embed_video(
    transcript_text,
    chapters_dict,
    team_id,
    video_id,
    section_title="",
    section_no=None,
    video_duration=None,
    skip_pinecone=False,
):
    """
    Full RAG pipeline for one video:
      1. Chunk transcript by chapters
      2. Generate embeddings
      3. Save chunks to S3
      4. Upsert to Pinecone
      5. Update manifest
    
    Call this from process_video_task() after upload_video_artifacts(),
    or run standalone for batch processing.
    
    Args:
        transcript_text: Raw timestamped transcript
        chapters_dict:   {"HH:MM:SS": "title", ...}
        team_id:         Team identifier
        video_id:        Video identifier
        section_title:   Course section title
        section_no:      Section number
        video_duration:  Video duration in seconds
        skip_pinecone:   If True, only chunk + save to S3 (no Pinecone)
    
    Returns:
        dict with results, or None if RAG is disabled
    """
    if not _is_rag_enabled():
        logger.info("ℹ️ RAG not enabled. Set RAG_ENABLED=true to activate.")
        return None
    
    t0 = time.time()
    
    logger.info("=" * 60)
    logger.info("🧩 RAG CHUNKING PIPELINE STARTING")
    logger.info(f"   Team: {team_id} | Video: {video_id}")
    logger.info("=" * 60)
    
    # Step 1: Chunk
    chunks = chunk_transcript_by_chapters(
        transcript_text, chapters_dict, video_duration
    )
    
    if not chunks:
        logger.error("❌ No chunks generated")
        return {"success": False, "error": "no_chunks"}
    
    logger.info(f"✅ Generated {len(chunks)} chunks")
    
    # Step 2: Embed
    logger.info(f"🧮 Generating embeddings ({EMBEDDING_MODEL})...")
    embeddings = generate_embeddings(chunks)
    
    if not embeddings:
        logger.error("❌ Embedding generation failed")
        return {"success": False, "error": "embedding_failed", "chunks": len(chunks)}
    
    logger.info(f"✅ Generated {len(embeddings)} embeddings")
    
    # Step 3: Save chunks to S3
    _save_chunks_to_s3(chunks, embeddings, team_id, video_id)
    
    # Step 4: Upsert to Pinecone
    pinecone_success = False
    if not skip_pinecone:
        logger.info("📌 Upserting to Pinecone...")
        pinecone_success = upsert_to_pinecone(
            chunks, embeddings, team_id, video_id,
            section_title=section_title, section_no=section_no
        )
    else:
        logger.info("ℹ️ Skipping Pinecone upsert (skip_pinecone=True)")
    
    # Step 5: Update manifest
    _update_manifest_rag(team_id, video_id)
    
    elapsed = time.time() - t0
    
    result = {
        "success": True,
        "chunks": len(chunks),
        "embeddings": len(embeddings),
        "pinecone_upserted": pinecone_success,
        "elapsed": elapsed,
        "avg_tokens_per_chunk": sum(c["token_estimate"] for c in chunks) // max(len(chunks), 1),
    }
    
    logger.info("=" * 60)
    logger.info(f"🧩 RAG PIPELINE COMPLETE in {elapsed:.1f}s")
    logger.info(f"   Chunks: {result['chunks']} | Embeddings: {result['embeddings']}")
    logger.info(f"   Pinecone: {'✅' if pinecone_success else '⏭️ skipped'}")
    logger.info(f"   Avg tokens/chunk: {result['avg_tokens_per_chunk']}")
    logger.info("=" * 60)
    
    return result


# ==================== Batch Processing ====================

def batch_chunk_from_s3():
    """
    Process all videos in S3 that don't have chunks yet.
    Reads manifests to find unprocessed videos.
    
    Run manually:
        python -c "from app.rag_chunking import batch_chunk_from_s3; batch_chunk_from_s3()"
    """
    try:
        from app.s3_storage import _get_s3_client
    except ImportError:
        print("S3 storage module not available")
        return
    
    client = _get_s3_client()
    if not client:
        print("S3 client not available")
        return
    
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        print("S3_BUCKET not set")
        return
    
    # List all manifests
    prefix = os.getenv("S3_PREFIX", "").strip("/")
    manifest_prefix = f"{prefix}/manifests/" if prefix else "manifests/"
    
    try:
        response = client.list_objects_v2(Bucket=bucket, Prefix=manifest_prefix)
    except Exception as e:
        print(f"Failed to list manifests: {e}")
        return
    
    if "Contents" not in response:
        print("No manifests found")
        return
    
    total = 0
    processed = 0
    skipped = 0
    errors = 0
    
    for obj in response["Contents"]:
        key = obj["Key"]
        if not key.endswith(".json"):
            continue
        
        try:
            manifest_response = client.get_object(Bucket=bucket, Key=key)
            manifest = json.loads(manifest_response["Body"].read().decode("utf-8"))
        except Exception as e:
            print(f"  ❌ Failed to read {key}: {e}")
            continue
        
        team_id = manifest.get("team_id", "unknown")
        
        for video_id, video_meta in manifest.get("videos", {}).items():
            total += 1
            
            if video_meta.get("has_chunks"):
                skipped += 1
                continue
            
            print(f"  📦 Processing video {video_id} (team {team_id})...")
            
            # Read transcript from S3
            raw_prefix = f"{prefix}/" if prefix else ""
            transcript_key = f"{raw_prefix}raw/{team_id}/{video_id}/transcript.txt"
            chapters_key = f"{raw_prefix}processed/{team_id}/{video_id}/chapters.json"
            
            try:
                transcript_resp = client.get_object(Bucket=bucket, Key=transcript_key)
                transcript_text = transcript_resp["Body"].read().decode("utf-8")
                
                chapters_resp = client.get_object(Bucket=bucket, Key=chapters_key)
                chapters_dict = json.loads(chapters_resp["Body"].read().decode("utf-8"))
            except Exception as e:
                print(f"  ❌ Failed to read data for {video_id}: {e}")
                errors += 1
                continue
            
            result = chunk_and_embed_video(
                transcript_text=transcript_text,
                chapters_dict=chapters_dict,
                team_id=team_id,
                video_id=video_id,
                section_title=video_meta.get("section_title", ""),
                section_no=video_meta.get("section_no"),
            )
            
            if result and result.get("success"):
                processed += 1
            else:
                errors += 1
    
    print(f"\nBatch complete: {processed} processed, {skipped} skipped (already done), {errors} errors, {total} total")
