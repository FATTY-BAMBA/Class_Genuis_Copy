# syntax=docker/dockerfile:1
# =========================================================
# Base: PyTorch 2.2.2 + CUDA 11.8 + cuDNN 8 (Docker Hub)
# Using -devel variant for packages that need compilation (pycairo, etc.)
# =========================================================
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel

# -------------------- Environment --------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    TZ=Asia/Taipei \
    HF_HOME=/workspace/models \
    WHISPER_CACHE=/workspace/models \
    CTRANSLATE2_CACHE=/workspace/models \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    GLOG_minloglevel=2 \
    GLOG_logtostderr=0 \
    FLAGS_fraction_of_gpu_memory_to_use=0.9

# -------------------- System deps --------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      # Build tools for packages that need compilation
      pkg-config libcairo2-dev \
      # FFmpeg development headers for ctranslate2/faster-whisper
      libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
      libavfilter-dev libswscale-dev libswresample-dev \
      # Runtime dependencies
      ffmpeg redis-server redis-tools \
      libsndfile1 libgl1 libgomp1 libglib2.0-0 \
      libsm6 libxext6 libxrender1 libcairo2 \
      curl aria2 netcat-openbsd procps net-tools lsof patchelf \
      # tiny safety net if any wheel expects system BLAS:
      libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Pin pip for reproducibility
RUN python -m pip install --upgrade pip==24.0 setuptools wheel

# -------------------- Workdir & app files --------------------
WORKDIR /app

# Copy dependency files first for layer caching
COPY requirements.txt constraints.txt /app/

# Then your source
COPY . .

# -------------------- Python deps --------------------
# Keep NumPy 1.x FIRST to avoid accidental upgrades to 2.x
RUN python -m pip install --no-cache-dir numpy==1.26.4

# Your project requirements (Torch already in base, so keep torch* commented in requirements.txt)
RUN python -m pip install --no-cache-dir \
    -r /app/requirements.txt -c /app/constraints.txt

# Whisper stack (use newer faster-whisper that supports modern PyAV with wheels)
RUN python -m pip install --no-cache-dir \
    ctranslate2==3.24.0 faster-whisper==1.0.3

# Tokenizers (devel image has build tools if needed), then Transformers
RUN python -m pip install --no-cache-dir "tokenizers>=0.14,<0.15"

RUN python -m pip install --no-cache-dir \
    "transformers==4.36.2" -c /app/constraints.txt

# Paddle + OCR (+ VisualDL)
RUN python -m pip install --no-cache-dir \
      -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html \
      paddlepaddle-gpu==2.5.1 && \
    python -m pip install --no-cache-dir \
      paddleocr==2.7.0 visualdl==2.5.3 -c /app/constraints.txt

# -------------------- Optional: legacy numpy.int shim --------------------
RUN python - <<'PY'
import sys, pathlib
site = pathlib.Path(next(p for p in sys.path if p.endswith("site-packages")))
(site/'numpy_patch.py').write_text("import numpy as np; np.int = int if not hasattr(np, 'int') else np.int")
with (site/'sitecustomize.py').open('a') as f: f.write("\ntry: import numpy_patch\nexcept: pass\n")
PY

# -------------------- Non-root user & dirs --------------------
RUN useradd -ms /bin/bash appuser && \
    mkdir -p /app/uploads /app/segments /workspace/logs /workspace/models /workspace/uploads && \
    chown -R appuser:appuser /app /workspace && \
    chmod +x /app/start.sh

USER appuser

# -------------------- Ports, healthcheck, entrypoint --------------------
EXPOSE 5000 8888

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD curl -f http://localhost:5000/healthz || exit 1

CMD ["./start.sh"]
