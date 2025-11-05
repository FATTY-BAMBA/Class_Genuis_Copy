# syntax=docker/dockerfile:1.7
# CUDA 11.8 + cuDNN 8 (Ubuntu 20.04)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    TZ=Asia/Taipei \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# ---- System deps (builder) ----
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common build-essential cmake git curl ca-certificates wget \
      libcairo2-dev libjpeg-dev libgif-dev pkg-config libopenblas-dev libssl-dev patchelf \
      python3.10 python3.10-dev python3.10-venv && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
# Copy dependency manifests first for better caching
COPY requirements.txt constraints.txt /build/

# Optional: filter a few packages you’ll pin separately
RUN grep -v "ctranslate2\|faster-whisper\|tokenizers\|transformers\|numpy" requirements.txt > requirements_filtered.txt || cp requirements.txt requirements_filtered.txt

# ---- Python build tools & pinned low-level deps ----
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip==24.0 setuptools wheel && \
    python -m pip install --no-cache-dir \
      "packaging>=20.0" \
      Cython==3.0.10 \
      pybind11==2.12.0 \
      meson==1.2.3 meson-python==0.15.0 ninja==1.11.1

# ---- NumPy first (1.x), then PyTorch CUDA 11.8 ----
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir numpy==1.26.4 && \
    python -m pip install \
      torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 \
      --extra-index-url https://download.pytorch.org/whl/cu118

# ---- Rest of Python deps (respect constraints) ----
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir -r requirements_filtered.txt -c constraints.txt

# ---- Whisper stack (pinned) ----
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir \
      faster-whisper==0.10.1 \
      ctranslate2==3.24.0 \
      transformers==4.36.2 -c constraints.txt

# ---- Polygon3 ----
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir "Polygon3==3.0.9.1"

# ---- Paddle + OCR ----
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir \
      paddlepaddle-gpu==2.5.1 \
      -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html && \
    python -m pip install --no-cache-dir paddleocr==2.7.0 -c constraints.txt

# ---- VisualDL ----
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir visualdl==2.5.3

# (Optional) If you keep the NumPy guardrail:
# RUN python -m pip install --no-cache-dir --force-reinstall numpy==1.26.4

# ------------------ Runtime ------------------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 AS final

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
    FLAGS_fraction_of_gpu_memory_to_use=0.9 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/lib:${LD_LIBRARY_PATH}

# Minimal runtime libs
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common \
      python3.10 python3.10-distutils \
      ffmpeg redis-server redis-tools \
      libsndfile1 libgl1 libgomp1 libglib2.0-0 libsm6 libxext6 libxrender1 libcairo2 \
      curl aria2 netcat-openbsd procps net-tools lsof patchelf \
      # optional tiny safety net if any wheel expects system BLAS:
      libopenblas0 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 && ln -sf /usr/bin/python3 /usr/bin/python \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

# Copy Python runtime from builder
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Make sure loader cache sees copied libs
RUN ldconfig

WORKDIR /app
COPY . .

# Small shim for legacy numpy.int usage
RUN python - <<'PY'
import pathlib
site = pathlib.Path('/usr/local/lib/python3.10/dist-packages')
site.mkdir(parents=True, exist_ok=True)
(site/'numpy_patch.py').write_text("import numpy as np; np.int = int if not hasattr(np, 'int') else np.int")
with (site/'sitecustomize.py').open('a') as f: f.write("\ntry: import numpy_patch\nexcept: pass\n")
PY

# Non-root user and dirs
RUN useradd -ms /bin/bash appuser && \
    mkdir -p /app/uploads /app/segments /workspace/logs /workspace/models /workspace/uploads && \
    chown -R appuser:appuser /app /workspace && \
    chmod +x /app/start.sh
USER appuser

EXPOSE 5000 8888

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:5000/healthz || exit 1

CMD ["./start.sh"]
