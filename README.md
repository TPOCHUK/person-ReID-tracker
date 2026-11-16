# Person-ReID-Tracker

Local computer-vision service that tracks one person in a camera feed or video and attempts to recover the target after occlusion or a tracker ID change. It provides an annotated browser stream and a REST API without sending video to external services.

## Contents

- [Problem and solution](#problem-and-solution)
- [Results](#results)
- [Technologies](#technologies)
- [Usage](#usage)
- [Development](#development)
- [Testing](#testing)
- [Deploy and CI/CD](#deploy-and-cicd)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [Limitations](#limitations)

## Problem and solution

Person detectors work frame by frame and do not preserve identity. Detect Human combines **YOLOv8**, **ByteTrack**, and an **ONNX ReID model** to maintain one target over time without face recognition.

```text
Camera / video
      ↓
YOLOv8 + ByteTrack → target selection → ONNX ReID recovery
      ↓
FastAPI → REST state + MJPEG browser stream
```

Key capabilities:

- webcam, video file, and stream URL input;
- target recovery after short occlusions or ByteTrack ID changes;
- live MJPEG preview and target state API;
- concurrency-safe START/STOP and resource cleanup;
- performance, stability, resource, and HTTP benchmarks.

## Results

Measured on Windows 10, Python 3.13, 4 logical CPU cores and 8 GB RAM using CPU inference (`imgsz=256`, 1280×720 video):

| Check | Result |
|---|---:|
| Full pipeline: YOLO + tracking + ReID | **27.37 FPS** |
| Full-pipeline frame latency | **22.25 ms p50**, 76.48 ms p95 |
| YOLO + ByteTrack | **31.8 FPS** |
| 5-minute stability run | **4,695 frames**, no exception |
| API load check: 100 requests, 8 clients | **100% successful**, 19.60 ms p50, 23.54 ms p95 |
| Automated tests | **36 passed** |

Full-pipeline latency by stage:

| Stage | Mean | p50 | p95 |
|---|---:|---:|---:|
| Frame capture | 4.03 ms | 3.44 ms | 7.40 ms |
| YOLO + ByteTrack inference | 31.11 ms | 21.90 ms | 74.42 ms |
| Target tracking + conditional ReID | 0.81 ms | 0.01 ms | 0.05 ms |
| Complete CV pipeline | 32.12 ms | 22.25 ms | 76.48 ms |

The 120-second full-pipeline run processed **3,285 frames**. YOLO inference is the main bottleneck; ReID is conditional and does not run on every frame. During a separate five-minute run under high background system load, the pipeline processed **4,695 frames at 15.65 FPS** with no exception; total latency was 37.13 ms p50 and 152.00 ms p95.

Raw results: [YOLO inference](docs/benchmarks/inference-60s.json), [full pipeline](docs/benchmarks/full-120s.json), [stability](docs/benchmarks/stability-300s.json), and [HTTP server check](docs/benchmarks/server-check.json).

Results depend on hardware and workload. Accuracy metrics such as mAP and IDF1 are not claimed because the test video has no ground-truth annotations.

## Technologies

Python 3.13, FastAPI, OpenCV, PyTorch, ByteTrack, ONNX Runtime, Pytest, Ruff

## Usage

After starting the server, open [http://127.0.0.1:8000](http://127.0.0.1:8000) and click **START**. Interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

The first webcam is used by default. To process a video, change `.env`:

```env
CAMERA_SOURCE=videos/test.mp4
```

Main API endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/system/start` | Start video processing |
| `POST` | `/system/stop` | Stop and release resources |
| `GET` | `/status` | Get service state |
| `GET` | `/target` | Get target ID, bounding box, FPS, and latency |
| `GET` | `/video/stream` | Read the annotated MJPEG stream |

## Development

### Requirements

- Python **3.13**
- `yolov8n.pt` — person detector weights
- `yolo26n-reid.onnx` — appearance embedding model

Place both model files in the project root. Weights and test videos are intentionally excluded from Git.

### Installation

The following setup was verified on Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On Linux or macOS, activate the environment with `source .venv/bin/activate` and copy the configuration with `cp .env.example .env`.

### Development server

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Only one Uvicorn worker should be used because the camera, models, and tracking state are process-local.

## Testing

Install the development dependencies and run the checks:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check app tests tools bench.py
```

Unit and API tests use fakes, so they do not require model weights or a camera. To test the real pipeline:

```powershell
python bench.py --test full_pipeline --source videos/test.mp4 --duration 120 --warmup 10
python tools/check_server.py --source videos/test.mp4 --requests 100 --clients 8
```

Current validation summary:

- **36/36** unit and API tests passed;
- **100/100** HTTP requests succeeded with 8 concurrent clients;
- HTTP latency: **19.60 ms p50**, **23.54 ms p95**, **27.50 ms max**;
- MJPEG delivery, STOP cleanup, and repeated START/STOP passed;
- the five-minute stability benchmark completed without an exception.

## Deploy and CI/CD

GitHub Actions runs Ruff and Pytest on every push and pull request. The application itself must run on a machine with the model files and access to the camera or video source; GitHub Pages cannot host the Python/OpenCV backend.

For public deployment, add authentication, HTTPS, and explicit access controls. The default configuration is intended for local use.

## Contributing

Bug reports and focused pull requests are welcome. Before submitting a change:

1. Create a branch for the change.
2. Add or update tests.
3. Run Pytest and Ruff.
4. Describe the problem, solution, and verification steps in the pull request.

## Roadmap

- [x] YOLO person detection and ByteTrack integration
- [x] ReID-based target recovery
- [x] Browser preview and REST API
- [x] Automated tests and benchmark tooling
- [ ] Evaluate mAP, IDF1, and ReID errors on an annotated dataset
- [ ] Compare CPU and CUDA performance under identical conditions
- [ ] Add authentication for non-local deployment

## Limitations

- ReID compares visual appearance; similar clothing can cause a false match.
- Poor lighting, heavy occlusion, crowded scenes, and limited CPU resources reduce performance.
- This is not face recognition and does not verify a person's real identity.
- See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for implementation details and design decisions.
