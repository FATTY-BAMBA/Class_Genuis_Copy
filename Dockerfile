# syntax=docker/dockerfile:1.4
# =========================================================
# Base: NVIDIA CUDA 11.8 + cuDNN 8.7
# =========================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# -------------------- Install Python 3.11 --------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

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
      pkg-config libcairo2-dev \
      libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
      libavfilter-dev libswscale-dev libswresample-dev \
      ffmpeg redis-server redis-tools \
      libsndfile1 libgl1 libgomp1 libglib2.0-0 \
      libsm6 libxext6 libxrender1 libcairo2 \
      curl aria2 netcat-openbsd procps net-tools lsof patchelf \
      libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip==24.0 setuptools wheel

# -------------------- Workdir & app files --------------------
WORKDIR /app

COPY requirements.txt constraints.txt /app/
COPY . .

# -------------------- Install PyTorch 2.1.2 (CUDA 11.8) - MUST BE FIRST --------------------
RUN pip3 install --no-cache-dir --force-reinstall \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# CRITICAL: Force NumPy 1.26.4 AFTER PyTorch
RUN python -m pip install --no-cache-dir --force-reinstall numpy==1.26.4

# -------------------- Core App Dependencies --------------------
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
    pydantic \
    openai==1.55.3 \
    azure-ai-inference

# -------------------- Whisper stack (CUDA 11.8 + cuDNN 8) --------------------
RUN python -m pip install --no-cache-dir --only-binary=:all: av==12.3.0

RUN python -m pip install --no-cache-dir ctranslate2==3.24.0

RUN python -m pip install --no-cache-dir --no-deps faster-whisper==0.10.1

RUN python -m pip install --no-cache-dir \
    onnxruntime \
    "huggingface-hub>=0.13"

RUN python -m pip install --no-cache-dir "tokenizers>=0.14,<0.15"

RUN python -m pip install --no-cache-dir transformers==4.36.2

# -------------------- EasyOCR --------------------
RUN python -m pip install --no-cache-dir easyocr==1.7.1

# CRITICAL: EasyOCR upgrades numpy to 2.x, force it back to 1.26.4
RUN python -m pip install --no-cache-dir --force-reinstall numpy==1.26.4

# -------------------- PaddleOCR (CUDA 11.8 compatible) --------------------
# Install PaddlePaddle GPU version 3.2.0 for CUDA 11.8 to ensure Python 3.11 compatibility.
RUN python -m pip install --no-cache-dir \
    paddlepaddle-gpu==3.2.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Install PaddleOCR and its dependencies
RUN python -m pip install --no-cache-dir \
    "paddleocr[all]"

# Force NumPy back to 1.26.4 after PaddleOCR install
RUN python -m pip install --no-cache-dir --force-reinstall numpy==1.26.4

# -------------------- Additional dependencies --------------------
RUN python -m pip install --no-cache-dir \
    sentencepiece \
    torchmetrics \
    lightning \
    peft \
    optimum \
    evaluate \
    gradio \
    loralib \
    Cython==0.29.36 \
    google-generativeai==0.8.3 \
    hf_transfer \
    opencv-python-headless==4.7.0.72 \
    Pillow==9.5.0 \
    attrdict==2.0.1 \
    beautifulsoup4==4.13.4 \
    fire==0.7.1 \
    fonttools==4.51.0 \
    imgaug==0.4.0 \
    lmdb==1.7.3 \
    openpyxl==3.1.5 \
    pdf2docx==0.5.8 \
    premailer==3.10.0 \
    "PyMuPDF>=1.23.0" \
    rapidfuzz==3.13.0 \
    visualdl==2.5.3 \
    "protobuf<4,>=3.20.0" \
    llvmlite==0.43.0 \
    numba==0.60.0 \
    tqdm==4.67.1 \
    scipy==1.10.1

# Final NumPy lock (ensure 1.26.4 after all installs)
RUN python -m pip install --no-cache-dir --force-reinstall numpy==1.26.4

# -------------------- Verify installations --------------------
# **FIXED ERROR HERE:** Ensuring the entire command is complete and correct.
RUN python -c "import torch; print('✅ PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'cuDNN:', torch.backends.cudnn.version())" && \
    python -c "import numpy; print('✅ NumPy:', numpy.__version__)" && \
    python -c "import flask; print('✅ Flask:', flask.__version__)" && \
    python -c "import celery; print('✅ Celery:', celery.__version__)" && \
    python -c "import tenacity; print('✅ Tenacity: installed')" && \
    python -c "import av; print('✅ PyAV:', av.__version__)" && \
    python -c "import ctranslate2; print('✅ ctranslate2:', ctranslate2.__version__)" && \
    python -c "import easyocr; print('✅ EasyOCR:', easyocr.__version__)" && \
    python -c "import faster_whisper; print('✅ faster-whisper:', faster_whisper.__version__)" && \
    python -c "import paddle; print('✅ PaddlePaddle:', paddle.__version__, 'CUDA:', paddle.device.is_compiled_with_cuda())" && \
    python -c "from paddleocr import PaddleOCR; print('✅ PaddleOCR: installed')"

# -------------------- Optional: legacy numpy.int shim --------------------
RUN python -c "import sys, pathlib, site; \
    site_dir = pathlib.Path(site.getsitepackages()[0]); \
    (site_dir/'numpy_patch.py').write_text('import numpy as np; np.int = int if not hasattr(np, \"int\") else np.int'); \
    sitecustomize = site_dir/'sitecustomize.py'; \
    sitecustomize.write_text((sitecustomize.read_text() if sitecustomize.exists() else '') + '\ntry: import numpy_patch\nexcept: pass\n')"

# -------------------- Non-root user & dirs --------------------
RUN useradd -ms /bin/bash appuser && \
    mkdir -p /app/uploads /app/segments /workspace/logs /workspace/models /workspace/uploads && \
    chown -R appuser:appuser /app /workspace && \
    chmod +x /app/start.sh

USER appuser

EXPOSE 5000 8888

CMD ["./start.sh"]
