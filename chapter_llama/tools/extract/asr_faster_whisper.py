from pathlib import Path
import re
import logging
import torch
from faster_whisper import WhisperModel
from chapter_llama.src.data.chapters import sec_to_hms

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DEFAULT_COMPUTE = "float16" if _DEVICE == "cuda" else "int8"

_DURATION_RE = re.compile(r"Processing audio with duration\s+(\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)")
_VAD_REMOVED_RE = re.compile(r"VAD filter removed\s+(\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)\s+of audio")

def _hms_to_seconds(hms: str) -> float:
    hh, mm, ss = hms.split(":")
    return int(hh)*3600 + int(mm)*60 + float(ss)

def _has_enough_cjk(text: str, min_cjk: int = 2) -> bool:
    return sum(0x4E00 <= ord(ch) <= 0x9FFF for ch in text) >= min_cjk

def _basic_cleanup(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r"^[\s:：,\-–—·.。!！?？]+", "", t)
    return t

def _merge_intervals(intervals, total_duration=None, eps=0.05):
    if not intervals:
        return []
    clamped = []
    if total_duration and total_duration > 0:
        for s, e in intervals:
            s = max(0.0, min(float(s), total_duration))
            e = max(0.0, min(float(e), total_duration))
            if e > s:
                clamped.append((s, e))
    else:
        for s, e in intervals:
            try:
                s, e = float(s), float(e)
                if e > s:
                    clamped.append((s, e))
            except Exception:
                pass
    if not clamped:
        return []
    clamped.sort()
    merged = []
    for s, e in clamped:
        if not merged:
            merged.append([s, e]); continue
        ms, me = merged[-1]
        if s <= me + eps:
            merged[-1][1] = max(me, e)
        else:
            merged.append([s, e])
    return merged

def _sum_intervals(intervals):
    return float(sum((e - s) for s, e in intervals)) if intervals else 0.0

class _VADLogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.duration_hms = None
        self.removed_hms = None
    def emit(self, record: logging.LogRecord):
        try:
            msg = record.getMessage()
        except Exception:
            return
        m1 = _DURATION_RE.search(msg)
        if m1:
            self.duration_hms = m1.group(1)
        m2 = _VAD_REMOVED_RE.search(msg)
        if m2:
            self.removed_hms = m2.group(1)

class ASRProcessor:
    """
    Returns transcript lines and exposes metrics where:
      - Top-level fields are **canonical** (prefer VAD)
      - 'debug' holds 'vad', 'raw', 'kept' for inspection
    """

    def __init__(
        self,
        model_name: str = "large-v2",
        compute_type: str = _DEFAULT_COMPUTE,
        language: str = "zh",
        use_vad: bool = True,
    ):
        self.language = language
        self.use_vad = use_vad
        self.model = WhisperModel(model_name, device=_DEVICE, compute_type=compute_type)

        self._cc = None
        try:
            from opencc import OpenCC  # type: ignore
            self._cc = OpenCC("s2t")
        except Exception:
            self._cc = None

        self.last_metrics = None  # filled after get_asr

    def get_asr(self, audio_file, return_duration: bool = True):
        audio_path = Path(audio_file)
        assert audio_path.exists(), f"File {audio_file} does not exist"

        # Capture VAD lines from faster_whisper logger during transcribe
        vad_logger = logging.getLogger("faster_whisper")
        handler = _VADLogCapture()
        vad_logger.addHandler(handler)

        import sys
        print("=" * 60, flush=True)
        print("🔄 WHISPER: Starting transcription...", flush=True)
        print("⏳ WHISPER: Loading large-v2 model (this takes 30-60 seconds)...", flush=True)
        print("=" * 60, flush=True)
        sys.stdout.flush()

        try:
            segments, info = self.model.transcribe(
                str(audio_path),
                language=self.language,
                task="transcribe",
                beam_size=5,
                temperature=0.0,
                length_penalty=0.5,
                condition_on_previous_text=False,
                vad_filter=self.use_vad,
                vad_parameters={"min_silence_duration_ms": 150},
                no_speech_threshold=0.70,
            )
        finally:
            vad_logger.removeHandler(handler)

        # Base duration
        total_duration = float(getattr(info, "duration", 0.0) or 0.0)
        vad_duration = _hms_to_seconds(handler.duration_hms) if handler.duration_hms else None
        if total_duration <= 0 and vad_duration is not None:
            total_duration = vad_duration
        if total_duration <= 0:
            total_duration = 1.0
        dur = total_duration

        # Build intervals: all (raw) vs kept (after text filters)
        asr_lines = []
        prev_text = None
        all_intervals = []
        kept_intervals = []
        sum_raw = sum_kept = 0.0

        raw_segments = []
        for seg in segments:
            try:
                s = float(seg.start); e = float(seg.end)
                if e > s:
                    raw_segments.append((s, e, seg.text or ""))
            except Exception:
                continue

        for s, e, raw_text in raw_segments:
            all_intervals.append((s, e))
            sum_raw += (e - s)

            t = _basic_cleanup(raw_text)
            if not t:
                continue
            if not (_has_enough_cjk(t) or len(t) >= 4):
                continue
            if self._cc is not None:
                t = self._cc.convert(t)
            if prev_text and t == prev_text:
                continue
            prev_text = t

            kept_intervals.append((s, e))
            sum_kept += (e - s)
            asr_lines.append(f"{sec_to_hms(s)}: {t}")

        asr_text = "\n".join(asr_lines) + ("\n" if asr_lines else "")

        # Overlap-safe unions
        merged_all  = _merge_intervals(all_intervals,  total_duration=total_duration, eps=0.05)
        merged_kept = _merge_intervals(kept_intervals, total_duration=total_duration, eps=0.05)
        union_all   = _sum_intervals(merged_all)
        union_kept  = _sum_intervals(merged_kept)

        # VAD parsed (exactly matches console)
        vad_removed = _hms_to_seconds(handler.removed_hms) if handler.removed_hms else None
        vad_metrics = None
        if vad_removed is not None:
            vad_speech = max(0.0, dur - vad_removed)
            vad_metrics = {
                "duration": vad_duration if vad_duration is not None else dur,
                "removed_duration": vad_removed,
                "removed_ratio": vad_removed / dur,
                "speech_duration": vad_speech,
                "speech_ratio": vad_speech / dur,
                "source": "logger",
            }

        # ---- Canonical selection (VAD-first) ----
        if vad_metrics is not None:
            canonical_speech = vad_metrics["speech_duration"]
            canonical_removed = vad_metrics["removed_duration"]
        elif union_kept > 0:
            canonical_speech = union_kept
            canonical_removed = max(0.0, dur - union_kept)
        else:
            canonical_speech = union_all
            canonical_removed = max(0.0, dur - union_all)

        # Build final metrics with canonical on top
        self.last_metrics = {
            # Canonical (what you should persist/use in UI)
            "duration": dur,
            "speech_duration": canonical_speech,
            "removed_duration": canonical_removed,
            "speech_ratio": canonical_speech / dur,
            "removed_ratio": canonical_removed / dur,
            "source": "logger" if vad_metrics is not None else ("kept" if union_kept > 0 else "raw"),

            # Debug-only (keep for investigation; do not use for user-facing ratios)
            "debug": {
                "vad": vad_metrics,  # may be None
                "raw": {
                    "segments": len(all_intervals),
                    "sum_duration": sum_raw,
                    "union_duration": union_all,
                    "speech_ratio": union_all / dur,
                    "removed_duration": max(0.0, dur - union_all),
                    "removed_ratio": max(0.0, dur - union_all) / dur,
                },
                "kept": {
                    "segments": len(kept_intervals),
                    "sum_duration": sum_kept,
                    "union_duration": union_kept,
                    "speech_ratio": union_kept / dur,
                    "removed_duration": max(0.0, dur - union_kept),
                    "removed_ratio": max(0.0, dur - union_kept) / dur,
                },
            },
        }

        return (asr_text, dur) if return_duration else asr_text
