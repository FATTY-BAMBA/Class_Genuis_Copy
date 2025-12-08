# app/app.py - Updated with new API format support

import os
import uuid
import time
import json
import redis
import hashlib
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from app.celery_app import celery as celery_app
from app.runpod_controller import start_pod, get_pod_status  # noqa: F401

# ==================== Setup Base Directory & Environment ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '..', '.env'))

# ==================== Persistent Storage ====================
PERSIST_BASE = os.getenv("PERSIST_BASE", "/workspace")
try:
    os.makedirs(PERSIST_BASE, exist_ok=True)
    _t = os.path.join(PERSIST_BASE, ".write_test")
    with open(_t, "w") as _f:
        _f.write("ok")
    os.remove(_t)
    print(f"✓ Using persistent storage at: {PERSIST_BASE}")
except Exception as e:
    print(f"✗ Cannot write to {PERSIST_BASE}: {e}")
    PERSIST_BASE = "/app"
    print(f"  Falling back to: {PERSIST_BASE}")

UPLOAD_FOLDER = os.path.join(PERSIST_BASE, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print(f"✓ Upload folder ready: {UPLOAD_FOLDER}")

# Templates and static files
TEMPLATES_DIR = "/app/templates"
STATIC_DIR = "/app/static"

# ==================== Flask App ====================
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB

# ==================== Redis ====================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.StrictRedis.from_url(REDIS_URL)

# ==================== Helper Functions ====================
def generate_unique_filename(original_filename, file_obj=None):
    """Generate a unique filename while preserving the original extension"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Get file extension
    base_name, extension = os.path.splitext(original_filename)
    safe_base = secure_filename(base_name)
    
    # Create hash from file content if available
    if file_obj:
        file_obj.seek(0)
        file_hash = hashlib.md5(file_obj.read(1024 * 1024)).hexdigest()[:8]  # First 1MB
        file_obj.seek(0)  # Reset for saving
    else:
        file_hash = hashlib.md5(original_filename.encode()).hexdigest()[:8]
    
    # Format: timestamp_hash_safename.ext
    unique_filename = f"{timestamp}_{file_hash}_{safe_base}{extension}"
    return unique_filename

def log_request_info(request_type, **kwargs):
    """Log request information for debugging"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "request_type": request_type,
        **kwargs
    }
    print(f"[REQUEST] {json.dumps(log_entry, indent=2, ensure_ascii=False)}")

def validate_units(units):
    """
    Validate Units array structure.
    
    Returns: (is_valid: bool, error_message: str or None)
    """
    if not units:
        return True, None
    
    if not isinstance(units, list):
        return False, "Units must be an array"
    
    for idx, unit in enumerate(units):
        if not isinstance(unit, dict):
            return False, f"Unit at index {idx} must be an object"
        
        if "UnitNo" not in unit:
            return False, f"Unit at index {idx} is missing 'UnitNo'"
        
        if "Title" not in unit:
            return False, f"Unit at index {idx} is missing 'Title'"
        
        if not isinstance(unit["UnitNo"], int):
            return False, f"Unit at index {idx}: 'UnitNo' must be an integer (got {type(unit['UnitNo']).__name__})"
        
        if not isinstance(unit["Title"], str):
            return False, f"Unit at index {idx}: 'Title' must be a string (got {type(unit['Title']).__name__})"
        
        if unit["UnitNo"] < 1:
            return False, f"Unit at index {idx}: 'UnitNo' must be positive (got {unit['UnitNo']})"
        
        if not unit["Title"].strip():
            return False, f"Unit at index {idx}: 'Title' cannot be empty"
    
    return True, None
    
# ==================== Health Endpoint ====================
@app.route("/healthz")
def healthz():
    status = {"status": "ok", "redis": False, "celery": False, "uploads_writable": False}

    # Check Redis
    try:
        redis_client.ping()
        status["redis"] = True
    except Exception as e:
        status["status"] = "degraded"
        print(f"Redis health check failed: {e}")

    # Check Celery
    try:
        inspector = celery_app.control.inspect()
        ping = inspector.ping(timeout=1) if inspector else None
        status["celery"] = bool(ping)
        if not status["celery"]:
            status["status"] = "degraded"
    except Exception as e:
        status["status"] = "degraded"
        print(f"Celery health check failed: {e}")

    # Check upload folder is writable
    try:
        test_file = os.path.join(UPLOAD_FOLDER, ".health_check")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        status["uploads_writable"] = True
    except Exception as e:
        status["status"] = "degraded"
        print(f"Upload folder not writable: {e}")

    return jsonify(status), 200 if status["status"] == "ok" else 503

# ==================== Main Routes ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle both file uploads and URL submissions from web interface"""
    from tasks.tasks import process_video_task
    
    file = request.files.get('file')
    video_url = request.form.get('video_url')
    num_pages = int(request.form.get('num_pages', 3))
    num_questions = int(request.form.get('num_questions', 10))
    
    # Generate unique video ID for tracking
    video_id = str(uuid.uuid4())
    
    try:
        if file and file.filename:
            # FILE UPLOAD PATH
            original_filename = file.filename
            unique_filename = generate_unique_filename(original_filename, file)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            log_request_info(
                "file_upload",
                video_id=video_id,
                original_filename=original_filename,
                saved_filename=unique_filename,
                save_path=video_path
            )
            
            # Save the file
            file.save(video_path)
            
            # Verify save
            if not os.path.exists(video_path):
                raise Exception(f"File save verification failed: {video_path}")
            
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            print(f"✓ File saved successfully: {unique_filename} ({file_size_mb:.2f} MB)")
            
            video_url = f"file://{video_path}"
            source_type = "upload"
            
        elif video_url:
            # URL SUBMISSION PATH
            log_request_info(
                "url_submission",
                video_id=video_id,
                video_url=video_url
            )
            
            original_filename = None
            unique_filename = None
            video_path = None
            source_type = "url"
            
        else:
            return jsonify({"error": "No file or video URL provided!"}), 400

        # Create video info object (web UI doesn't have Units structure)
        video_info = {
            "Id": video_id,
            "TeamId": request.form.get('team_id', 'test'),
            "SectionNo": request.form.get('section_no', '1'),
            "CreatedAt": datetime.now().isoformat(),
            "SourceType": source_type,
            "OriginalFilename": original_filename,
            "SavedFilename": unique_filename,
            "SavedPath": video_path,
            "VideoUrl": video_url,
            "SectionTitle": None,  # Not provided in web UI
            "Units": [],            # Not provided in web UI
            "NumQuestions": num_questions,
            "NumPages": num_pages
        }
        
    except Exception as e:
        print(f"❌ Upload processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    # Queue the task with retries
    task = None
    last_err = None
    for attempt in range(3):
        try:
            task = process_video_task.delay(video_url, video_info, num_questions, num_pages)
            print(f"✓ Task queued successfully: {task.id}")
            break
        except Exception as e:
            last_err = e
            print(f"Queue attempt {attempt + 1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))

    if not task:
        print(f"❌ Failed to queue task after 3 attempts: {last_err}")
        return jsonify({"error": "❌ Redis queue unavailable, please try again later."}), 503

    return jsonify({
        "queued": True,
        "task_id": task.id,
        "video_id": video_id,
        "source_type": source_type,
        "message": "✅ Upload received. Processing has started. The results will be sent to the 教學平台."
    }), 200

@app.route('/upload_url', methods=['POST'])
def upload_url():
    """
    API endpoint for external URL requests with enhanced educational metadata.
    
    Accepts new fields (backward compatible):
    - SectionTitle: Overall course section title (optional)
    - Units: Array of unit objects with UnitNo and Title (optional)
    
    Required fields: Id, TeamId, SectionNo, PlayUrl
    Optional fields: NumQuestions (default: 10), NumPages (default: 3), CreatedAt, SectionTitle, Units
    """
    from tasks.tasks import process_video_task

    try:
        data = request.get_json(force=True)
        
        # ==================== VALIDATION ====================
        # Required fields (unchanged from before)
        required_keys = {"Id", "TeamId", "SectionNo", "PlayUrl"}
        
        if not data or not required_keys.issubset(data.keys()):
            missing = required_keys - set(data.keys()) if data else required_keys
            return jsonify({"error": f"Missing required keys: {list(missing)}"}), 400

        play_url = data["PlayUrl"]
        
        # Validate PlayUrl format
        if not isinstance(play_url, str) or not play_url.strip():
            return jsonify({"error": "PlayUrl must be a non-empty string"}), 400
        
        if not play_url.startswith(("http://", "https://", "file://")):
            return jsonify({"error": f"Invalid PlayUrl format. Must start with http://, https://, or file:// (got: {play_url[:50]}...)"}), 400
        
        # ==================== FILE PATH HANDLING ====================
        # Handle file:// URLs that reference existing files
        if play_url.startswith("file://"):
            file_path = play_url.replace("file://", "")
            
            # Check multiple possible locations
            possible_paths = [
                file_path,
                os.path.join(UPLOAD_FOLDER, os.path.basename(file_path)),
                os.path.join("/app/uploads", os.path.basename(file_path)),
                os.path.join("/workspace/uploads", os.path.basename(file_path))
            ]
            
            found_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    found_path = path
                    play_url = f"file://{path}"
                    print(f"✓ Found file at: {path}")
                    break
            
            if not found_path:
                print(f"❌ File not found in any location: {file_path}")
                print(f"   Searched: {possible_paths}")
                return jsonify({"error": f"File not found: {file_path}"}), 404

        # ==================== EXTRACT NEW FIELDS (with defaults) ====================
        # NEW: Educational metadata (optional - backward compatible)
        section_title = data.get("SectionTitle", None)
        units = data.get("Units", [])
        
        # Validate SectionTitle if provided
        if section_title is not None and not isinstance(section_title, str):
            return jsonify({"error": f"SectionTitle must be a string (got {type(section_title).__name__})"}), 400
        
        # Validate Units structure if provided
        if units:
            is_valid, error_msg = validate_units(units)
            if not is_valid:
                return jsonify({"error": error_msg}), 400
        
        # Configuration with defaults (backward compatible)
        try:
            num_questions = int(data.get("NumQuestions", 10))  # Default: 10 questions
            num_pages = int(data.get("NumPages", 3))            # Default: 3 pages
            
            if num_questions < 1 or num_questions > 100:
                return jsonify({"error": f"NumQuestions must be between 1 and 100 (got {num_questions})"}), 400
            
            if num_pages < 1 or num_pages > 20:
                return jsonify({"error": f"NumPages must be between 1 and 20 (got {num_pages})"}), 400
                
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"NumQuestions and NumPages must be integers: {str(e)}"}), 400
        
        # ==================== BUILD VIDEO INFO ====================
        video_info = {
            # Required fields
            "Id": data["Id"],
            "TeamId": data["TeamId"],
            "SectionNo": data["SectionNo"],
            "CreatedAt": data.get("CreatedAt", datetime.now().isoformat()),
            "SourceType": "api_url",
            "VideoUrl": play_url,
            
            # NEW: Educational metadata (optional)
            "SectionTitle": section_title,
            "Units": units,
            
            # Processing configuration
            "NumQuestions": num_questions,
            "NumPages": num_pages,
        }
        
        # ==================== LOGGING ====================
        log_data = {
            "api_request": True,
            "video_id": data["Id"],
            "team_id": data["TeamId"],
            "section_no": data["SectionNo"],
            "play_url": play_url[:100] + "..." if len(play_url) > 100 else play_url,
            "num_questions": num_questions,
            "num_pages": num_pages,
        }
        
        # Add educational metadata to logs if provided
        if section_title:
            log_data["section_title"] = section_title
        if units:
            log_data["units_count"] = len(units)
        
        log_request_info("api_request", **log_data)
        
        # Log unit details if provided (pretty format)
        if section_title or units:
            print("=" * 60)
            print("📚 EDUCATIONAL METADATA RECEIVED")
            print("=" * 60)
            
            if section_title:
                print(f"📖 Section Title: {section_title}")
            
            if units:
                print(f"📑 Units ({len(units)}):")
                for unit in units:
                    print(f"   {unit['UnitNo']}. {unit['Title']}")
            
            print("=" * 60)
        else:
            print("ℹ️  No educational metadata (SectionTitle/Units) provided - processing as single video")
        
    except ValueError as e:
        print(f"❌ API request validation error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"❌ API request error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    # ==================== QUEUE TASK ====================
    task = None
    last_err = None
    for attempt in range(3):
        try:
            task = process_video_task.delay(play_url, video_info, num_questions, num_pages)
            print(f"✓ API task queued successfully: {task.id}")
            break
        except Exception as e:
            last_err = e
            print(f"API queue attempt {attempt + 1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))

    if not task:
        print(f"❌ Failed to queue API task after 3 attempts: {last_err}")
        return jsonify({"error": "❌ Redis queue unavailable, please try again later."}), 503

    # ==================== RESPONSE ====================
    response = {
        "status": "queued",
        "task_id": task.id,
        "video_id": data["Id"],
        "video_url": play_url[:100] + "..." if len(play_url) > 100 else play_url,
        "num_questions": num_questions,
        "num_pages": num_pages,
    }
    
    # Include educational metadata in response if provided
    if section_title:
        response["section_title"] = section_title
    if units:
        response["units_count"] = len(units)
        response["units"] = units
    
    return jsonify(response), 200

# ==================== Debug Routes (Optional) ====================
@app.route('/debug/uploads', methods=['GET'])
def debug_uploads():
    """Debug endpoint to check upload directory"""
    try:
        files = os.listdir(UPLOAD_FOLDER)
        file_info = []
        for f in files:
            path = os.path.join(UPLOAD_FOLDER, f)
            if os.path.isfile(path):
                size = os.path.getsize(path) / (1024 * 1024)  # MB
                mtime = datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
                file_info.append({
                    "name": f,
                    "size_mb": round(size, 2),
                    "modified": mtime
                })
        
        return jsonify({
            "upload_folder": UPLOAD_FOLDER,
            "file_count": len(file_info),
            "files": file_info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/test', methods=['POST'])
def test_api():
    """
    Test endpoint to validate API request format without processing.
    Useful for testing the new SectionTitle and Units fields.
    """
    try:
        data = request.get_json(force=True)
        
        # Check required fields
        required_keys = {"Id", "TeamId", "SectionNo", "PlayUrl"}
        missing = required_keys - set(data.keys()) if data else required_keys
        
        if missing:
            return jsonify({
                "valid": False,
                "error": f"Missing required keys: {list(missing)}",
                "received_keys": list(data.keys()) if data else []
            }), 400
        
        # Validate Units if provided
        units = data.get("Units", [])
        if units:
            is_valid, error_msg = validate_units(units)
            if not is_valid:
                return jsonify({
                    "valid": False,
                    "error": error_msg
                }), 400
        
        # Return validation success
        return jsonify({
            "valid": True,
            "message": "API request format is valid",
            "received": {
                "id": data["Id"],
                "team_id": data["TeamId"],
                "section_no": data["SectionNo"],
                "play_url": data["PlayUrl"][:50] + "...",
                "section_title": data.get("SectionTitle"),
                "units_count": len(units) if units else 0,
                "num_questions": data.get("NumQuestions", 10),
                "num_pages": data.get("NumPages", 3)
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "valid": False,
            "error": str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
