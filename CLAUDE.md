# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bird identification system for Nouvelle-Aquitaine region birds. The system consists of:
- **FastAPI inference API** (`api/`) - Serves a PyTorch image classification model
- **RTSP detector** (`detector/`) - Real-time bird detection using YOLOv8 and RTSP camera feed

## Development Commands

### Running the API (Docker)
```bash
docker-compose up --build
```
API available at `http://localhost:8000`

### Running the API (Local)
```bash
poetry install
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Running the Bird Detector
```bash
poetry install
python detector/bird_detector.py
```
Requires `.env` file with `RTSP_URL` configured.

### Testing RTSP Connection
```bash
python test_rtsp.py
```
Tests if the RTSP camera stream is accessible and can read frames.

### Running Tests
```bash
poetry install --with dev
pytest
```

## Architecture

### API Service (`api/`)
Three-layer architecture for bird species classification:

1. **`main.py`** - FastAPI application
   - Endpoints: `/identify`, `/identify/detailed`, `/species`, `/health`
   - Model loaded once at startup via `@app.on_event("startup")`
   - Expects model at `/model/model.pth` (mounted volume in Docker)

2. **`inference.py`** - `BirdPredictor` class
   - Loads PyTorch checkpoint containing model weights + class list
   - Handles image preprocessing (resize 256 → center crop 224 → normalize)
   - Returns top-k predictions with softmax probabilities
   - Accepts both file paths and bytes (for FastAPI uploads)

3. **`model.py`** - `BirdClassifier` neural network
   - Base: EfficientNetV2-S from TIMM library (`tf_efficientnetv2_s.in21k_ft_in1k`)
   - Custom classifier head: Dropout(0.3) → Linear(512) → ReLU → Dropout(0.2) → Linear(num_classes)
   - Number of classes determined from checkpoint at runtime

**Model Checkpoint Format**: The `.pth` file must contain:
```python
{
    'model_state_dict': {...},  # Model weights
    'classes': [...]            # List of class names (species)
}
```

**Docker Structure**: Files are organized in the container as:
- `/api/` - FastAPI application code (main.py, inference.py, model.py)
- `/model/` - Model checkpoint file (model.pth)
- `/detector/` - Detector script (bird_detector.py)
- `/captures/` - Shared directory for bird captures

### Detector Service (`detector/bird_detector.py`)
Standalone Python script that monitors RTSP camera feed for birds:

1. **YOLO Detection** - Uses YOLOv11n to detect birds (class 14 in COCO)
2. **Frame Processing** - Processes every Nth frame (configurable via `PROCESS_EVERY_N_FRAMES`)
3. **Cooldown Management** - Prevents duplicate detections within cooldown period
4. **API Integration** - Sends detected bird crops to FastAPI `/identify` endpoint
5. **Auto-reconnect** - Handles RTSP connection failures with automatic reconnection

Environment variables control detection behavior (see `.env` section).

## Key Configuration Files

### `pyproject.toml`
Poetry configuration with two dependency groups:
- **Main**: FastAPI, PyTorch, TIMM, Ultralytics (YOLO), OpenCV
- **Dev**: pytest, httpx

### `docker-compose.yaml`
Single service setup:
- Mounts `./api:/api` for hot reload
- Exposes port 8000
- Loads `.env` from `./api/.env`

### `.env` Variables
```bash
API_URL=http://api:8000                # API endpoint for detector (use 'api' for Docker networking)
RTSP_URL=rtsp://...                    # Camera stream URL
YOLO_CONFIDENCE=0.5                    # Detection confidence threshold
DETECTION_COOLDOWN=5                   # Seconds between detections
PROCESS_EVERY_N_FRAMES=10              # Process every Nth frame
CAPTURES_DIR=/captures                 # Directory to save detected bird images
HF_HOME=/tmp/huggingface               # HuggingFace cache location
TORCH_HOME=/tmp/torch                  # PyTorch cache location
YOLO_CONFIG_DIR=/tmp/ultralytics       # Ultralytics/YOLO cache location
```

## Troubleshooting

### RTSP Stream Timeout Issues

If you see `Stream timeout triggered after 30003.561662 ms`:

1. **Test the connection**: Run `python test_rtsp.py` to verify RTSP access
2. **Check network connectivity**:
   ```bash
   ping 192.168.1.74
   ```
3. **Test with other tools**:
   ```bash
   # Using ffplay
   ffplay rtsp://username:password@192.168.1.74:554/stream1

   # Using VLC
   vlc rtsp://username:password@192.168.1.74:554/stream1
   ```
4. **Verify RTSP URL format**: Common formats are:
   - `rtsp://username:password@ip:port/stream1`
   - `rtsp://username:password@ip:port/Streaming/Channels/101`
   - Check your camera's documentation for the correct path

5. **Network mode**: The detector container must be able to reach the camera. If the camera is on your local network, you may need to use `network_mode: "host"` in docker-compose.yaml

## Important Implementation Details

- **Model Loading**: The model expects a checkpoint file at `/model/model.pth` containing both `model_state_dict` and `classes` keys
- **Image Preprocessing**: Must match training preprocessing exactly (ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Bird Class ID**: YOLO uses class 14 for birds in the COCO dataset
- **Crop Margins**: Detector adds 10% margin around YOLO bounding boxes before sending to classifier
- **Docker User**: Runs as www-data (UID 1000, GID 1000) for proper permissions
- **Volume Mounts**:
  - `./api:/api` - Hot reload for API code changes
  - `./model:/model` - Model file accessible to container
  - `./detector:/detector` - Detector code
  - `./captures:/captures` - Shared directory for saving detected bird images
- **YOLO Model**: The YOLOv8n model is pre-downloaded during Docker build (as root) to avoid permission issues. The model is cached in `/tmp/ultralytics` which has write permissions for www-data user.
