#!/usr/bin/env bash
set -Eeuo pipefail

# Fix library paths for ctranslate2
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/site-packages/ctranslate2.libs:${LD_LIBRARY_PATH:-}"

# Ensure libraries are accessible
if [ -d "/usr/local/lib/python3.10/site-packages/ctranslate2.libs" ]; then
    echo "Setting up ctranslate2 library links..."
    for lib in /usr/local/lib/python3.10/site-packages/ctranslate2.libs/*.so*; do
        if [ -f "$lib" ]; then
            libname=$(basename "$lib")
            if [ ! -f "/usr/local/lib/$libname" ]; then
                ln -sf "$lib" "/usr/local/lib/$libname" 2>/dev/null || true
            fi
        fi
    done
    ldconfig 2>/dev/null || true
fi

# Test critical imports at runtime
echo "Testing critical Python imports..."
python -c "
import sys

# Test ctranslate2
try:
    import ctranslate2
    print('✓ ctranslate2 import successful')
except ImportError as e:
    print(f'✗ ctranslate2 import failed: {e}')
    sys.exit(1)

# Test faster_whisper
try:
    import faster_whisper
    print('✓ faster_whisper import successful')
except ImportError as e:
    print(f'✗ faster_whisper import failed: {e}')
    sys.exit(1)

# Test other critical imports
try:
    import torch
    print(f'✓ torch {torch.__version__} import successful')
    if torch.cuda.is_available():
        print(f'✓ CUDA is available')
    else:
        print('⚠ CUDA not available - will use CPU')
except ImportError as e:
    print(f'✗ torch import failed: {e}')
"

# Check if the imports succeeded
if [ $? -ne 0 ]; then
    echo "❌ Critical imports failed. Container cannot start properly."
    echo "This is likely due to missing privileges. Ensure docker-compose.yml has:"
    echo "  privileged: true"
    echo "  security_opt:"
    echo "    - seccomp:unconfined"
    exit 1
fi

echo "✅ All critical imports successful"

# Clean up stale PID files from previous runs
echo "Cleaning up stale PID files..."
rm -f /tmp/gunicorn.pid
rm -f /tmp/celery.pid

### ---------- Environment Variables ----------
export PYTHONUNBUFFERED=1
export CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-LAZY}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export WHISPER_DEVICE=${WHISPER_DEVICE:-cuda}
export WHISPER_IMPL=${WHISPER_IMPL:-fast}
export WHISPER_MODEL=${WHISPER_MODEL:-large-v2}
export PYTHONPATH="/app:${PYTHONPATH:-}"

# Detect RunPod → prefer local Redis inside container
if [ "${RUNPOD_POD_ID:-}" != "" ]; then
  echo "Detected RunPod environment - using local Redis"
  export REDIS_URL="redis://localhost:6379/0"
else
  export REDIS_URL=${REDIS_URL:-rediss://red-d332fu3uibrs73a90n5g:R8Nfl8YuJhfRc9djsmWPEnr7tJOx4prK@virginia-keyvalue.render.com:6379/0}
fi

export REDIS_WAIT_SEC=${REDIS_WAIT_SEC:-30}
export CELERY_QUEUES=${CELERY_QUEUES:-video_processing,qa_generation,maintenance,monitoring,celery}
export CELERY_LOGLEVEL=${CELERY_LOGLEVEL:-INFO}
export CELERY_POOL=${CELERY_POOL:-solo}
export CELERY_CONCURRENCY=${CELERY_CONCURRENCY:-1}
export CELERY_PREFETCH=${CELERY_PREFETCH:-1}
export CELERY_MAX_TASKS_PER_CHILD=${CELERY_MAX_TASKS_PER_CHILD:-0}
export CELERY_LEAN=${CELERY_LEAN:-false}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

# Suppress PaddlePaddle noise
export GLOG_minloglevel=2
export GLOG_logtostderr=0

# Create log directory
mkdir -p /workspace/logs
CELERY_LOG=/workspace/logs/celery.worker.log
GUNICORN_LOG=/workspace/logs/gunicorn.log

echo "========= CUDA / GPU ========="
if command -v nvidia-smi >/dev/null 2>&1; then 
    nvidia-smi || true
else 
    echo "nvidia-smi not found"
fi

python - <<'PY'
import torch, os
print(f"torch {torch.__version__} | cuda_available={torch.cuda.is_available()} | device_env={os.environ.get('WHISPER_DEVICE','auto')}")
if torch.cuda.is_available():
    print("cuda device count:", torch.cuda.device_count())
    print("cuda name:", torch.cuda.get_device_name(0))
PY
echo "==============================="

### ---------- Redis Setup ----------
MASKED_REDIS_URL="${REDIS_URL//:[^@]*@/:*****@}"
echo "REDIS_URL detected: ${MASKED_REDIS_URL}"
REDIS_STARTED_LOCALLY=0

# Start local redis if URL is localhost
if [[ "${REDIS_URL}" =~ ^redis://(127\.0\.0\.1|localhost)(:|/|$) ]]; then
  echo "Starting local Redis..."
  redis-server --daemonize yes --save "" --appendonly no --loglevel warning
  REDIS_STARTED_LOCALLY=1
  sleep 2
else
  echo "Using external Redis at: ${MASKED_REDIS_URL}"
fi

# Wait for Redis
echo -n "Waiting for Redis "
REDIS_CONNECTED=0
for (( i=1; i<=REDIS_WAIT_SEC; i++ )); do
  if redis-cli -u "${REDIS_URL}" ping >/dev/null 2>&1; then 
    echo " ✓"
    REDIS_CONNECTED=1
    break
  fi
  echo -n "."
  sleep 1
done

if [ "${REDIS_CONNECTED}" = "0" ]; then
  echo
  echo "ERROR: Redis not reachable after ${REDIS_WAIT_SEC}s"
  if [ "${RUNPOD_POD_ID:-}" != "" ] && [ "${REDIS_STARTED_LOCALLY}" = "0" ]; then
    echo "Attempting local Redis fallback..."
    redis-server --daemonize yes --save "" --appendonly no --loglevel warning
    export REDIS_URL="redis://localhost:6379/0"
    REDIS_STARTED_LOCALLY=1
    sleep 2
    if ! redis-cli -u "${REDIS_URL}" ping >/dev/null 2>&1; then
      echo "Failed to start local Redis"
      exit 1
    fi
  else
    exit 1
  fi
fi

### ---------- Cleanup handlers ----------
cleanup() {
  echo
  echo "Shutting down..."
  
  # Stop Gunicorn
  if [[ -f /tmp/gunicorn.pid ]]; then
    PID=$(cat /tmp/gunicorn.pid 2>/dev/null || true)
    if [[ -n "${PID}" ]] && kill -0 "${PID}" 2>/dev/null; then
      echo "Stopping Gunicorn (PID: ${PID})..."
      kill -TERM "${PID}" 2>/dev/null || true
      sleep 2
    fi
  fi
  
  # Stop Celery
  if [[ -f /tmp/celery.pid ]]; then
    PID=$(cat /tmp/celery.pid 2>/dev/null || true)
    if [[ -n "${PID}" ]] && kill -0 "${PID}" 2>/dev/null; then
      echo "Stopping Celery (PID: ${PID})..."
      kill -TERM "${PID}" 2>/dev/null || true
      sleep 2
    fi
  fi
  
  # Stop local Redis if we started it
  if [[ "${REDIS_STARTED_LOCALLY}" = "1" ]]; then
    echo "Stopping local Redis..."
    redis-cli -u "${REDIS_URL}" shutdown 2>/dev/null || true
  fi
  
  echo "Cleanup complete"
}
trap cleanup EXIT INT TERM

### ---------- Pre-flight checks ----------
echo "Running pre-flight checks..."

# Test Python imports
echo "Testing Python imports..."
python -c "
try:
    from app import app
    print('  ✓ Flask app imports successfully')
except Exception as e:
    print(f'  ✗ Flask app import failed: {e}')
    exit(1)

try:
    from tasks import tasks
    print('  ✓ Celery tasks import successfully')
except Exception as e:
    print(f'  ✗ Celery tasks import failed: {e}')
    exit(1)
" || {
    echo "Python import checks failed. Exiting."
    exit 1
}

# Test Redis connection from Python
echo "Testing Redis connection from Python..."
python -c "
import redis
import os
try:
    r = redis.from_url(os.environ.get('REDIS_URL'))
    r.ping()
    print('  ✓ Redis connection successful')
except Exception as e:
    print(f'  ✗ Redis connection failed: {e}')
    exit(1)
" || {
    echo "Redis connection test failed. Exiting."
    exit 1
}

### ---------- Celery options ----------
CELERY_COMMON_OPTS=(
  --loglevel="${CELERY_LOGLEVEL}"
  -P "${CELERY_POOL}"
  --concurrency="${CELERY_CONCURRENCY}"
  --prefetch-multiplier="${CELERY_PREFETCH}"
  --max-tasks-per-child="${CELERY_MAX_TASKS_PER_CHILD}"
  -Q "${CELERY_QUEUES}"
  --hostname="worker@%h"
)

if [[ "${CELERY_LEAN}" == "true" ]]; then
  CELERY_COMMON_OPTS+=( --without-gossip --without-mingle --without-heartbeat )
fi

### ---------- Start Celery Worker ----------
echo "Starting Celery worker..."
touch "${CELERY_LOG}"
chmod 666 "${CELERY_LOG}"

# Start Celery with tee (logs to both stdout and file)
celery -A tasks.tasks:celery worker "${CELERY_COMMON_OPTS[@]}" \
  --pidfile=/tmp/celery.pid 2>&1 | tee "${CELERY_LOG}" &
CELERY_BG_PID=$!

# Wait for Celery PID file with extended timeout
echo "Waiting for Celery to write PID file..."
CELERY_STARTED=0
for i in {1..15}; do
  if [ -f /tmp/celery.pid ]; then
    CELERY_PID=$(cat /tmp/celery.pid 2>/dev/null || echo "")
    if [ -n "${CELERY_PID}" ] && kill -0 "${CELERY_PID}" 2>/dev/null; then
      echo "Celery started successfully (PID: ${CELERY_PID})"
      CELERY_STARTED=1
      break
    fi
  fi
  
  # Also check if background process is still alive
  if ! kill -0 "${CELERY_BG_PID}" 2>/dev/null; then
    echo "Celery background process died unexpectedly"
    break
  fi
  
  echo "  Waiting... ($i/15)"
  sleep 2
done

if [ "${CELERY_STARTED}" = "0" ]; then
  echo "ERROR: Celery failed to start properly"
  echo "Celery log output (last 100 lines):"
  echo "-----------------------------------"
  tail -n 100 "${CELERY_LOG}" 2>/dev/null || echo "Could not read log file"
  echo "-----------------------------------"
  
  # For debugging, continue anyway
  echo "WARNING: Continuing without Celery for debugging purposes..."
  # Uncomment the next line to exit on Celery failure in production
  # exit 1
fi

### ---------- Start Gunicorn ----------
echo "Starting Gunicorn server..."
echo "Binding to 0.0.0.0:5000"
touch "${GUNICORN_LOG}"
chmod 666 "${GUNICORN_LOG}"

gunicorn app.app:app \
  -b 0.0.0.0:5000 \
  --workers 1 \
  --threads 4 \
  --timeout 600 \
  --graceful-timeout 60 \
  --capture-output \
  --access-logfile "${GUNICORN_LOG}" \
  --error-logfile "${GUNICORN_LOG}" \
  --pid /tmp/gunicorn.pid \
  --daemon

# Wait for Gunicorn to start
sleep 3

GUNICORN_PID=$(cat /tmp/gunicorn.pid 2>/dev/null || echo "")
if [ -z "${GUNICORN_PID}" ] || ! kill -0 "${GUNICORN_PID}" 2>/dev/null; then
  echo "ERROR: Gunicorn failed to start"
  echo "Gunicorn log output (last 50 lines):"
  echo "-----------------------------------"
  tail -n 50 "${GUNICORN_LOG}" 2>/dev/null || echo "Could not read log file"
  echo "-----------------------------------"
  exit 1
fi

echo "Gunicorn started successfully (PID: ${GUNICORN_PID})"

### ---------- Verify Services ----------
echo
echo "Current running processes:"
ps aux | grep -E "(gunicorn|celery|redis)" | grep -v grep || true
echo

# Verify port is listening
if command -v ss >/dev/null 2>&1; then
  ss -lntp | grep -q ":5000" || echo "WARNING: Port 5000 doesn't appear to be listening"
elif command -v netstat >/dev/null 2>&1; then
  netstat -tlnp 2>/dev/null | grep -q ":5000" || echo "WARNING: Port 5000 doesn't appear to be listening"
fi

echo "============================================"
if [ "${CELERY_STARTED}" = "1" ]; then
  echo "✅ All services started successfully"
else
  echo "⚠️  Application started (Celery failed)"
fi
echo "📍 Application: http://0.0.0.0:5000"
echo "📋 Logs:"
echo "    - Live logs: RunPod UI → Logs tab"
echo "    - Celery file: ${CELERY_LOG}"
echo "    - Gunicorn: ${GUNICORN_LOG}"
echo "============================================"
echo

# Keep container alive
if [ "${CELERY_STARTED}" = "1" ]; then
  echo "📊 Celery worker running (logs visible in RunPod UI)"
  echo "    Also streaming to: ${CELERY_LOG}"
  echo
  # Wait for Celery background process
  wait "${CELERY_BG_PID}"
else
  echo "📊 Keeping container alive for debugging..."
  echo "You can now connect to the container to investigate the issue."
  echo
  
  # Keep container running for debugging
  while true; do
    sleep 60
    echo "Container still running for debugging... ($(date))"
  done
fi
