# chapter_llama/tools/extract/ocr_processor.py
import os
import time
import math
import re
import difflib
from pathlib import Path
from typing import List, Tuple, Iterable, Dict, Optional

import cv2
import numpy as np
from difflib import SequenceMatcher
import easyocr  # Changed from paddleocr

# ---------------- Env knobs (all optional) ----------------
# Balanced for accuracy without big slowdown
_OCR_FPS             = float(os.getenv("OCR_FPS", "3.0"))
_OCR_DOWNSCALE_W     = int(os.getenv("OCR_DOWNSCALE_W", "960"))      # less aggressive than 640
_OCR_CONF            = float(os.getenv("OCR_CONF", "0.75"))          # base threshold
_OCR_BATCH           = int(os.getenv("OCR_BATCH", "16"))
_OCR_LANGS           = os.getenv("OCR_LANGS", "ch_tra,en").split(",")  # ch_tra for Traditional, ch_sim for Simplified
_OCR_GPU             = os.getenv("OCR_GPU", "true").lower() == "true"
_OCR_MAX_FRAMES      = int(os.getenv("OCR_MAX_FRAMES", "50"))
_OCR_LOG_EVERY_N     = int(os.getenv("OCR_LOG_EVERY_N", "50"))

# Keep only the top-K sharpest frames per window
_OCR_TOPK_SHARPEST   = int(os.getenv("OCR_TOPK_SHARPEST", "6"))

# Optional ROI to drop right sidebar / chrome (e.g., YouTube recommendations)
# Keep only x in [0, ROI_X_MAX] of the frame. Set ROI_X_MAX=1.0 to disable.
_ROI_X_MAX           = float(os.getenv("OCR_ROI_X_MAX", "0.82"))

# Reading-order grouping tolerances
_Y_TOL_RATIO         = float(os.getenv("OCR_Y_TOL_RATIO", "0.015"))  # line baseline tolerance (fraction of H)
_X_GAP_RATIO         = float(os.getenv("OCR_X_GAP_RATIO", "0.02"))   # gap to insert a space/new token (fraction of W)

# Temporal fuzzy dedup (0..1); higher = stricter dedup
_DEDUP_CUTOFF        = float(os.getenv("OCR_DEDUP_CUTOFF", "0.93"))
_DEDUP_MAX_MEMORY    = int(os.getenv("OCR_DEDUP_MAX_MEMORY", "5000"))  # cap global seen lines

# Small tech lexicon to gently fix common Latin terms (cheap & safe)
_TECH_TERMS = {
    "Jupyter", "Anaconda", "Google", "Python", "Assistant",
    "Untitled", "Login", "Distribution", "Manage", "Trusted",
    "Packages", "Environments", "Navigator", "Colab", "Drive"
}

# --------- Post-OCR cleanup toggles (fast + topic-agnostic) ---------
_DEDUP_INTRA_ENABLED   = os.getenv("DEDUP_INTRA_ENABLED", "true").lower() == "true"
_MERGE_SIMILAR_CUTOFF  = float(os.getenv("MERGE_SIMILAR_CUTOFF", "0.90"))
_MAX_LINES_FOR_MERGE   = int(os.getenv("MAX_LINES_FOR_MERGE", "80"))
_LOWINFO_MIN_LEN       = int(os.getenv("LOWINFO_MIN_LEN", "2"))
_LOWINFO_REPEAT_RATIO  = float(os.getenv("LOWINFO_REPEAT_RATIO", "0.60"))
_LOWINFO_MIN_ENTROPY   = float(os.getenv("LOWINFO_MIN_ENTROPY", "1.20"))
_UI_NOISE_ENABLE       = os.getenv("UI_NOISE_ENABLE", "true").lower() == "true"

# ---------------- Utils ----------------
def _sec_to_hms(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def _downscale(frame: np.ndarray, target_w: int) -> np.ndarray:
    if target_w <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= target_w:
        return frame
    scale = target_w / float(w)
    new_w, new_h = target_w, max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

def _ascii_ratio(s: str) -> float:
    """How much of the alphabetic content is ASCII (Latin)."""
    letters = sum(c.isalpha() for c in s)
    ascii_letters = sum((c.isalpha() and ord(c) < 128) for c in s)
    return 0.0 if letters == 0 else ascii_letters / letters

def _cjk_count(s: str) -> int:
    return sum('\u4e00' <= c <= '\u9fff' for c in s)

def _crop_from_poly(frame: np.ndarray, poly, pad: int = 2) -> np.ndarray:
    """Crop a rectangle that covers the given polygon (with small padding)."""
    try:
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
    except Exception:
        return np.zeros((0, 0, 3), dtype=frame.dtype)
    h, w = frame.shape[:2]
    x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    x2, y2 = min(w, max(xs) + pad), min(h, max(ys) + pad)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=frame.dtype)
    return frame[y1:y2, x1:x2]

def _sharpness_score(rgb: np.ndarray) -> float:
    """Laplacian variance as a blur/sharpness metric."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()

def _unique_filtered_texts(texts: Iterable[str]) -> List[str]:
    """
    Dedup + quick denylist + length filter, language-agnostic (line-level).
    (Hardened: drops non-strings.)
    """
    denylist = {
        'user', '使用者', 'user:', '使用者:', 'username',
        'ta', '助教', 'ta:', '助教:', '聯成小幫手', '小幫手', '幫手',
        'file', '檔案', 'edit', '編輯', 'view', '檢視',
        'settings', '設定', 'options', '選項',
        'loading...', '載入中...', 'please wait', '請稍候',
        'welcome', '歡迎', 'thank you', '謝謝', 'logout', '登出',
        'next', '下一步', 'previous', '上一步', 'submit', '提交'
    }
    seen = set()
    out: List[str] = []
    for t in texts:
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not t:
            continue
        tl = t.lower()
        if len(t) < 2 or tl in denylist:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _postfix_terms(texts: List[str]) -> List[str]:
    """Gently snap near-misses to known tech terms (Latin-heavy tokens only)."""
    fixed = []
    for t in texts:
        if not isinstance(t, str):
            continue
        if _ascii_ratio(t) >= 0.7:
            hit = difflib.get_close_matches(t, _TECH_TERMS, n=1, cutoff=0.82)
            if hit:
                fixed.append(hit[0])
                continue
        fixed.append(t)
    return fixed

# --- Hardened fuzzy compare: always operate on strings, fail-safe on errors
def _fuzzy_eq(a, b, cutoff: float = _DEDUP_CUTOFF) -> bool:
    try:
        sa = a if isinstance(a, str) else ("" if a is None else str(a))
        sb = b if isinstance(b, str) else ("" if b is None else str(b))
        if not sa or not sb:
            return False
        return SequenceMatcher(None, sa, sb).ratio() >= cutoff
    except Exception:
        return False

def _temporal_dedup(prev_lines_global: List[str], new_lines: List[str], cutoff: float = _DEDUP_CUTOFF) -> Tuple[List[str], List[str]]:
    """
    Keep new lines that aren't fuzzy-duplicates of the global memory,
    and update the memory. (Hardened against non-strings.)
    """
    prev_lines_global = [g for g in prev_lines_global if isinstance(g, str)]
    new_lines = [(ln if isinstance(ln, str) else ("" if ln is None else str(ln))) for ln in new_lines]

    kept: List[str] = []
    updated: List[str] = list(prev_lines_global)
    for ln in new_lines:
        if not ln:
            continue
        if any(_fuzzy_eq(ln, g, cutoff) for g in prev_lines_global):
            continue
        kept.append(ln)
        updated.append(ln)
        if len(updated) > _DEDUP_MAX_MEMORY:
            updated = updated[-_DEDUP_MAX_MEMORY:]
    return kept, updated

# ---------------- Topic-agnostic cleanup helpers ----------------
_CODE_HINTS = ("def ", "class ", "import ", "from ", "for ", "while ", "if ", "elif ", "else:", "return ")
_CODE_CHARS = set("_()[]{}=:.<>+-*/%|&!^,@'\"\\")
_URL_RE = re.compile(r"https?://|(?:\w[\w\.-]+)\.(?:com|org|edu|net|io|gov|tw|jp)(?:/|\b)")
_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
_NUM_UNIT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|px|cm|mm|in|s|ms|GB|MB|KB|dpi|fps|Hz)\b", re.I)
_DATE_RE = re.compile(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[:：]\d{2}(?:[:：]\d{2})?)\b")
_CJK_DUP_RE = re.compile(r"^([\u4e00-\u9fffA-Za-z0-9\-\_]+)(?:\s+\1){1,}$")
_PUNCT_RUN_RE = re.compile(r"[?？!！、,…．。]{2,}")

# Common UI words across apps (Jupyter/Anaconda/Office/Adobe/AutoCAD)
_UI_TOKENS = {
    "help","settings","file","edit","view","run","kernel","tools","navigator","account","sign in","login",
    "checkpoint","last checkpoint","trusted","packages","environments","console","output","undo","redo",
    "merge cell","split cell","clear output","selected cell","restart","interrupt","shutdown","assistant",
    "home","insert","format","window","workspace","ribbon","toolbar","properties","layers","model","layout"
}

def _looks_like_code(t: str) -> bool:
    if not isinstance(t, str):
        return False
    t2 = t.strip()
    if any(t2.startswith(h) for h in _CODE_HINTS):
        return True
    sym_hits = sum(ch in _CODE_CHARS for ch in t2)
    return sym_hits >= max(2, len(t2) // 10)

def _normalize_line(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = _PUNCT_RUN_RE.sub(lambda m: m.group(0)[0], t)
    m = _CJK_DUP_RE.match(t)
    if m:
        t = m.group(1)
    t = re.sub(r"\s{2,}", " ", t).strip(" \t-–—")
    return t.strip()

def _shannon_entropy(t: str) -> float:
    if not t or not isinstance(t, str):
        return 0.0
    freq: Dict[str, int] = {}
    for ch in t:
        freq[ch] = freq.get(ch, 0) + 1
    N = len(t)
    return -sum((c/N) * math.log(c/N + 1e-12, 2) for c in freq.values())

def _repeat_ratio(t: str) -> float:
    if not t or not isinstance(t, str):
        return 1.0
    tokens = t.split()
    if len(tokens) <= 1:
        return 1.0
    unique = len(set(tokens))
    return 1.0 - unique / len(tokens)

def _is_ui_noise(t: str) -> bool:
    if not _UI_NOISE_ENABLE or not isinstance(t, str):
        return False
    lo = t.lower()
    return any(tok in lo for tok in _UI_TOKENS)

def _is_low_information(t: str) -> bool:
    if not isinstance(t, str) or not t or len(t) < _LOWINFO_MIN_LEN:
        return True
    if _looks_like_code(t):
        return False
    if _URL_RE.search(t) or _EMAIL_RE.search(t) or _NUM_UNIT_RE.search(t) or _DATE_RE.search(t):
        return False
    if len(t) >= 25:
        return False
    if _repeat_ratio(t) >= _LOWINFO_REPEAT_RATIO:
        return True
    if _shannon_entropy(t) < _LOWINFO_MIN_ENTROPY:
        return True
    if _is_ui_noise(t):
        return True
    return False

def _merge_similar_lines(lines: List[str], cutoff: float = _MERGE_SIMILAR_CUTOFF) -> List[str]:
    out: List[str] = []
    kept: List[str] = []
    for t in lines:
        if not isinstance(t, str):
            continue
        t_norm = t.lower()
        if any(SequenceMatcher(None, t_norm, k).ratio() >= cutoff for k in kept):
            continue
        kept.append(t_norm)
        out.append(t)
    return out

def _clean_lines_topic_agnostic(lines: List[str]) -> List[str]:
    # Normalize and KEEP ONLY strings
    lines = [_normalize_line(t) for t in lines if isinstance(t, str) and t.strip()]
    lines = [t for t in lines if not _is_low_information(t)]
    if _DEDUP_INTRA_ENABLED and (0 < _MAX_LINES_FOR_MERGE >= len(lines)):
        lines = _merge_similar_lines(lines, cutoff=_MERGE_SIMILAR_CUTOFF)
    return lines

# ---------------- Main class ----------------
class OCRProcessor:
    """
    OCR processor using EasyOCR for mixed Chinese + English slides/screens.
    Produces line-preserving, de-duplicated text per time window.
    """

    def __init__(self):
        t0 = time.time()
        print("🔄 初始化 EasyOCR Reader ...")

        # Initialize EasyOCR with languages from environment
        # Default: Traditional Chinese + English
        self.reader = easyocr.Reader(_OCR_LANGS, gpu=_OCR_GPU)
        
        print(f"✅ EasyOCR ready (gpu={_OCR_GPU}, langs={_OCR_LANGS}) in {time.time() - t0:.1f}s")

    # ----------- Public (compat) single-window API -----------
    def get_text_from_segment(self, video_path: Path, start_sec: int, end_sec: int) -> str:
        """
        Keeps your original signature & output format (pretty bullets).
        """
        items = self.get_text_for_many_segments(
            video_path=video_path,
            segments=[(start_sec, end_sec)],
            fps=_OCR_FPS,
            downscale_w=_OCR_DOWNSCALE_W,
        )
        if not items:
            return ""
        seg = items[0]
        texts = seg.get("texts", []) if isinstance(seg, dict) else []
        if not texts:
            return ""
        ts = _sec_to_hms(int(start_sec))
        formatted = [f"*   於 {ts} 左右捕捉到:"]
        formatted += [f"    - 「{t}」" for t in texts if isinstance(t, str)]
        return "\n".join(formatted)

    # ----------- New: multi-window fast path -----------
    def get_text_for_many_segments(
        self,
        video_path: Path,
        segments: List[Tuple[int, int]],
        fps: float = _OCR_FPS,
        downscale_w: int = _OCR_DOWNSCALE_W,
    ) -> List[Dict]:
        """
        Extract OCR texts for many (start,end) windows efficiently.

        Returns:
          [
            {"start": s, "end": e, "texts": ["...", "..."]},
            ...
          ]
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        results: List[Dict] = []
        global_seen_lines: List[str] = []  # temporal memory across windows

        try:
            for idx, (s, e) in enumerate(segments, 1):
                frames = self._extract_frames_optimized(cap, s, e, fps=fps, downscale_w=downscale_w)

                # 1) tokens with geometry/conf
                tokens = self._run_ocr_easyocr(frames)  # List[Dict]
                if not tokens:
                    results.append({"start": s, "end": e, "texts": []})
                    continue

                # 2) group tokens into lines (reading order)
                lines = self._group_tokens_into_lines(tokens, y_tol_ratio=_Y_TOL_RATIO, x_gap_ratio=_X_GAP_RATIO)

                # 3) normalize + denylist + short filter
                lines = _unique_filtered_texts(lines)
                lines = _postfix_terms(lines)

                # 3.5) topic-agnostic cleanup (spam + duplicates + UI noise)
                lines = _clean_lines_topic_agnostic(lines)

                # 4) fuzzy temporal de-dup (across windows) — hardened
                lines, global_seen_lines = _temporal_dedup(global_seen_lines, lines, cutoff=_DEDUP_CUTOFF)

                results.append({"start": s, "end": e, "texts": lines})

                if idx % _OCR_LOG_EVERY_N == 0:
                    print(f"🖼️ OCR progress: {idx}/{len(segments)} windows processed")
        finally:
            cap.release()

        return results

    # ----------- OPTIMIZED Frame Extraction -----------
    def _extract_frames_optimized(
        self,
        cap: cv2.VideoCapture,
        start_sec: int,
        end_sec: int,
        fps: float,
        downscale_w: int,
    ) -> List[np.ndarray]:
        """
        Extract consecutive frames without repeated seeking,
        then keep only the top-K sharpest to save time + improve quality.
        """
        start_ms = max(0, int(start_sec * 1000))
        end_ms = max(start_ms, int(end_sec * 1000))

        # Seek to start position ONCE
        cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)

        # Get video FPS to calculate frame interval
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0  # fallback

        # Read every Nth frame to approximate target OCR fps
        frame_interval = max(1, int(round(video_fps / max(0.1, fps))))

        frames: List[np.ndarray] = []
        count = 0

        while cap.get(cv2.CAP_PROP_POS_MSEC) <= end_ms and len(frames) < _OCR_MAX_FRAMES:
            ok, bgr = cap.read()
            if not ok:
                break
            if count % frame_interval == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = _downscale(rgb, downscale_w)
                frames.append(rgb)
            count += 1

        # Keep only the sharpest K frames
        if frames:
            scored = [(f, _sharpness_score(f)) for f in frames]
            scored.sort(key=lambda x: x[1], reverse=True)
            frames = [f for f, _ in scored[:_OCR_TOPK_SHARPEST]]

        return frames

    # ----------- EasyOCR Processing (structured) -----------
    def _run_ocr_easyocr(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Process frames using EasyOCR.

        Returns token dicts:
          {"text": str, "conf": float, "box": [(x,y)...], "w": int, "h": int}
        """
        out: List[Dict] = []
        if not frames:
            return out

        for frame in frames:
            try:
                # EasyOCR returns: [(bbox, text, confidence), ...]
                # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                results = self.reader.readtext(frame)
                
                if not results:
                    continue

                H, W = frame.shape[:2]

                for bbox, txt, conf in results:
                    try:
                        txt = str(txt).strip()
                        conf = float(conf)
                        
                        if not txt:
                            continue

                        # Optional ROI filter: drop right sidebar area
                        if _ROI_X_MAX < 0.999:
                            try:
                                xs = [p[0] for p in bbox]
                                cx = sum(xs) / len(xs)
                                if (cx / max(1.0, W)) > _ROI_X_MAX:
                                    continue
                            except Exception:
                                pass

                        # Dynamic thresholds
                        is_latin = _ascii_ratio(txt) >= 0.7
                        keep = (conf >= 0.60) if is_latin else (
                            conf >= _OCR_CONF or
                            (conf >= 0.55 and _cjk_count(txt) >= 4)
                        )
                        
                        if keep:
                            # Normalize trivial typos (Latin)
                            txt = _postfix_terms([txt])[0]
                            out.append({
                                "text": txt,
                                "conf": conf,
                                "box": bbox,
                                "w": W, "h": H
                            })

                    except Exception as e:
                        print(f"⚠️ OCR token processing error: {e}")
                        continue

            except Exception as e:
                print(f"⚠️ OCR frame processing error: {e}")
                continue

        return out

    # ----------- Token -> line grouping -----------
    def _group_tokens_into_lines(
        self,
        tokens: List[Dict],
        y_tol_ratio: float = _Y_TOL_RATIO,
        x_gap_ratio: float = _X_GAP_RATIO
    ) -> List[str]:
        """
        Convert token boxes into reading-order lines (top->bottom, left->right),
        joining tokens on the same baseline and within small horizontal gaps.
        """
        if not tokens:
            return []

        items = []
        for t in tokens:
            poly = t.get("box")
            if not poly:
                continue
            try:
                xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                conf = float(t.get("conf", 0.0))
                W = int(t.get("w", 1)); H = int(t.get("h", 1))
                text = t.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
            except Exception:
                continue
            items.append((cy, cx, x1, x2, y1, y2, text, conf, W, H))

        # sort top→bottom then left→right
        items.sort(key=lambda r: (r[0], r[1]))

        lines: List[str] = []
        current = []
        last_cy: Optional[float] = None

        for cy, cx, x1, x2, y1, y2, text, conf, W, H in items:
            y_tol = max(1.0, H * y_tol_ratio)
            if last_cy is None or abs(cy - last_cy) <= y_tol:
                current.append((x1, x2, text))
                last_cy = cy if last_cy is None else (last_cy + cy) / 2.0
            else:
                lines.extend(self._flush_line_group(current, W, x_gap_ratio))
                current = [(x1, x2, text)]
                last_cy = cy

        if current:
            lines.extend(self._flush_line_group(current, items[-1][8], x_gap_ratio))

        # Remove trivial junk again after merge
        lines = _unique_filtered_texts(lines)
        return lines

    @staticmethod
    def _flush_line_group(group: List[Tuple[float, float, str]], W: int, x_gap_ratio: float) -> List[str]:
        if not group:
            return []
        group.sort(key=lambda x: x[0])  # by x1
        merged: List[str] = []
        buf = ""
        prev_x2: Optional[float] = None
        for lx1, lx2, t in group:
            if not isinstance(t, str):
                continue
            if prev_x2 is not None and (lx1 - prev_x2) > (W * x_gap_ratio):
                if buf.strip():
                    merged.append(buf.strip())
                buf = t
            else:
                buf = (buf + " " + t).strip() if buf else t
            prev_x2 = lx2
        if buf.strip():
            merged.append(buf.strip())
        return merged

    # ----------- Compatibility shims -----------
    def _extract_frames(
        self,
        cap: cv2.VideoCapture,
        start_sec: int,
        end_sec: int,
        fps: float,
        downscale_w: int,
    ) -> List[np.ndarray]:
        """Original method kept for compatibility"""
        return self._extract_frames_optimized(cap, start_sec, end_sec, fps, downscale_w)

    def _run_ocr(self, frames: List[np.ndarray]) -> List[str]:
        """Original method kept for compatibility (now line-level)"""
        tokens = self._run_ocr_easyocr(frames)
        lines = self._group_tokens_into_lines(tokens, y_tol_ratio=_Y_TOL_RATIO, x_gap_ratio=_X_GAP_RATIO)
        lines = _postfix_terms(_unique_filtered_texts(lines))
        return lines
