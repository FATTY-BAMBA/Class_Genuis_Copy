# syntax=docker/dockerfile:1.4
# =========================================================
# Class_Genius Educational Video Processing Pipeline
# Base: NVIDIA CUDA 11.8 + cuDNN 8.7
# Python 3.11 + PyTorch 2.1.2 + EasyOCR 1.7.0
# =========================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# -------------------- Install Python 3.11 --------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# -------------------- Environment Variables --------------------
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
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# -------------------- System Dependencies --------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      # Build tools
      pkg-config \
      # Video processing (FFmpeg)
      libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
      libavfilter-dev libswscale-dev libswresample-dev \
      ffmpeg \
      # Redis
      redis-server redis-tools \
      # Graphics and rendering
      libcairo2-dev libcairo2 \
      libsndfile1 libgl1 libgomp1 libglib2.0-0 \
      libsm6 libxext6 libxrender1 \
      # Utilities
      curl aria2 netcat-openbsd procps net-tools lsof patchelf \
      # Math libraries
      libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel
RUN python -m pip install --no-cache-dir --upgrade pip==24.0 setuptools wheel

# -------------------- Workdir & App Files --------------------
WORKDIR /app

COPY requirements.txt constraints.txt /app/
COPY . .

# ================================================================================
# CRITICAL: Install packages in specific order to avoid dependency conflicts
# ================================================================================

# -------------------- 1. PyTorch (MUST BE FIRST) --------------------
# PyTorch 2.1.2 with CUDA 11.8 support
RUN pip3 install --no-cache-dir --force-reinstall \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# -------------------- 2. NumPy Lock (AFTER PyTorch) --------------------
# Lock NumPy to 1.26.4 to prevent upgrades to 2.x
RUN python -m pip install --no-cache-dir --force-reinstall numpy==1.26.4

# -------------------- 3. Core Web Framework --------------------
RUN python -m pip install --no-cache-dir \
    Flask==2.3.3 \
    gunicorn==23.0.0 \
    celery \
    redis \
    async-timeout \
    tenacity \
    python-dotenv==1.0.1 \
    requests==2.32.3 \
    pycairo \
    opencc-python-reimplemented==0.1.7 \
    pydantic

# -------------------- 4. AI/LLM APIs --------------------
RUN python -m pip install --no-cache-dir \
    openai==1.55.3 \
    azure-ai-inference \
    google-generativeai==0.8.3

# -------------------- 5. Whisper Stack (Audio Transcription) --------------------
# PyAV (video/audio I/O)
RUN python -m pip install --no-cache-dir --only-binary=:all: av==12.3.0

# CTranslate2 (optimized inference)
RUN python -m pip install --no-cache-dir ctranslate2==3.24.0

# Faster-Whisper (main transcription engine)
RUN python -m pip install --no-cache-dir --no-deps faster-whisper==0.10.1

# Whisper dependencies
RUN python -m pip install --no-cache-dir \
    onnxruntime \
    "huggingface-hub>=0.13" \
    "tokenizers>=0.14,<0.15" \
    transformers==4.36.2

# -------------------- 6. EasyOCR (Screen Text Extraction) --------------------
# Version 1.7.0: Last version compatible with cuDNN 8.7
# Provides excellent Traditional Chinese + English OCR with GPU acceleration
RUN python -m pip install --no-cache-dir easyocr==1.7.0

# CRITICAL: EasyOCR may upgrade numpy, force it back to 1.26.4
RUN python -m pip install --no-cache-dir --force-reinstall numpy==1.26.4

# -------------------- 7. Computer Vision & Image Processing --------------------
RUN python -m pip install --no-cache-dir \
    opencv-python-headless==4.7.0.72 \
    Pillow==9.5.0 \
    imgaug==0.4.0

# -------------------- 8. ML & Scientific Computing --------------------
RUN python -m pip install --no-cache-dir \
    scipy==1.10.1 \
    sentencepiece \
    torchmetrics \
    lightning \
    peft \
    optimum \
    evaluate \
    llvmlite==0.43.0 \
    numba==0.60.0

# -------------------- 9. Document Processing --------------------
RUN python -m pip install --no-cache-dir \
    openpyxl==3.1.5 \
    pdf2docx==0.5.8 \
    "PyMuPDF>=1.23.0" \
    premailer==3.10.0

# -------------------- 10. Utilities & Miscellaneous --------------------
RUN python -m pip install --no-cache-dir \
    attrdict==2.0.1 \
    beautifulsoup4==4.13.4 \
    fire==0.7.1 \
    fonttools==4.51.0 \
    lmdb==1.7.3 \
    rapidfuzz==3.13.0 \
    "protobuf<4,>=3.20.0" \
    tqdm==4.67.1 \
    Cython==0.29.36 \
    hf_transfer \
    gradio \
    loralib

# -------------------- FINAL: Lock NumPy --------------------
# Ensure NumPy 1.26.4 after all package installations
RUN python -m pip install --no-cache-dir --force-reinstall numpy==1.26.4

# -------------------- Verify Critical Installations --------------------
RUN echo "🔍 Verifying package installations..." && \
    echo "================================================================================" && \
    python -c "import torch; print('✅ PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, '| cuDNN:', torch.backends.cudnn.version())" && \
    python -c "import numpy; print('✅ NumPy:', numpy.__version__)" && \
    python -c "import flask; print('✅ Flask:', flask.__version__)" && \
    python -c "import celery; print('✅ Celery:', celery.__version__)" && \
    python -c "import tenacity; print('✅ Tenacity: installed')" && \
    python -c "import av; print('✅ PyAV:', av.__version__)" && \
    python -c "import ctranslate2; print('✅ CTranslate2:', ctranslate2.__version__)" && \
    python -c "import faster_whisper; print('✅ Faster-Whisper:', faster_whisper.__version__)" && \
    python -c "import easyocr; print('✅ EasyOCR:', easyocr.__version__)" && \
    python -c "import cv2; print('✅ OpenCV:', cv2.__version__)" && \
    echo "================================================================================" && \
    echo "🎉 All packages verified successfully!"

# -------------------- GPU Verification --------------------
RUN echo "🎮 GPU Configuration:" && \
    python -c "import torch; print('  CUDA Available:', torch.cuda.is_available())" && \
    python -c "import torch; print('  cuDNN Enabled:', torch.backends.cudnn.enabled)" && \
    echo "================================================================================"

# -------------------- NumPy Legacy Compatibility Shim --------------------
# Some older packages may use deprecated numpy.int
RUN python -c "import sys, pathlib, site; \
    site_dir = pathlib.Path(site.getsitepackages()[0]); \
    (site_dir/'numpy_patch.py').write_text('import numpy as np; np.int = int if not hasattr(np, \"int\") else np.int'); \
    sitecustomize = site_dir/'sitecustomize.py'; \
    sitecustomize.write_text((sitecustomize.read_text() if sitecustomize.exists() else '') + '\ntry: import numpy_patch\nexcept: pass\n')"

# -------------------- User & Directory Setup --------------------
RUN useradd -ms /bin/bash appuser && \
    mkdir -p /app/uploads /app/segments /workspace/logs /workspace/models /workspace/uploads && \
    chown -R appuser:appuser /app /workspace && \
    chmod +x /app/start.sh

USER appuser

EXPOSE 5000 8888

CMD ["./start.sh"]
