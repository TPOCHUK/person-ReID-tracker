# Architecture

## Purpose

Person ReID Tracker processes one video source, selects one person, tracks that target across frames, and attempts appearance-based recovery when the tracker ID is lost. The service exposes the latest state as JSON and the annotated frames as MJPEG.

## Data flow

```text
OpenCV capture
      ↓
YOLOv8 detection + ByteTrack IDs
      ↓
TargetTracker ──→ ONNX ReID when recovery is required
      ↓
Annotated frame + immutable state snapshot
      ↓
FastAPI: /target and /video/stream
```

One frame is processed at a time. The web service and benchmark runner use the same `CVPipeline`, so benchmark timings represent the production processing path without HTTP or JPEG overhead.

## Main components

| Component | Responsibility |
|---|---|
| `app/main.py` | FastAPI lifecycle and application wiring |
| `app/service/system.py` | Camera lifecycle and processing loop |
| `app/camera/camera.py` | OpenCV video source adapter |
| `app/cv/pipeline.py` | Detection, tracking, overlay, timings, and snapshot |
| `app/cv/tracker.py` | Target selection and tracking state machine |
| `app/cv/reid.py` | Embedding extraction, update, and similarity search |
| `app/cv/factory.py` | Model and pipeline construction from settings |
| `app/api/router.py` | Control, status, target, and MJPEG endpoints |
| `app/benchmark/` | Performance and resource measurement |

## Target lifecycle

The first valid person nearest the frame center becomes the target. ByteTrack maintains its short-term `track_id`. If that ID disappears, the tracker periodically compares ReID embeddings for nearby candidates.

| State | Meaning |
|---|---|
| `idle` | No target is selected |
| `tracking` | The current target is visible |
| `lost` | The target ID is temporarily missing |
| `recovering` | ReID search has started but has not matched a candidate |

After `MAX_LOST_FRAMES`, the target is cleared. ReID embeddings describe visual appearance, not verified identity; the similarity threshold is not a probability.

## Concurrency and lifecycle

Camera reads, model inference, and JPEG encoding are blocking operations and run through `asyncio.to_thread()` so FastAPI can continue serving requests. An `asyncio.Lock` serializes START and STOP operations. Cleanup runs in `finally`, releasing the camera and resetting pipeline state after STOP, EOF, or an error.

The API reads a completed-frame snapshot instead of mutable tracker fields. This prevents partially updated state from being returned while the next frame is processed.

Run a single Uvicorn worker. Models, camera access, and tracking state are process-local; additional workers would create independent pipelines and compete for the same source.

## Configuration and extension points

`Settings` loads and validates `.env`. `build_pipeline()` connects the detector, ReID adapter, target tracker, and pipeline. HTTP code does not construct CV objects directly, and `CVPipeline` does not depend on FastAPI.

For changes:

1. Test target behavior in `tests/test_tracker.py`.
2. Test frame processing in `tests/test_pipeline.py`.
3. Test lifecycle and concurrency in `tests/test_system.py`.
4. Run the full-pipeline benchmark before comparing performance.

The current design supports one source, one selected target, and in-memory state. Multi-camera processing or persistent state should use separate pipeline instances and an explicit coordination layer.
