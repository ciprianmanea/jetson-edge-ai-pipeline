# Edge AI Perception Pipeline on Jetson Orin

Real-time object detection pipeline running **YOLOv8** on NVIDIA Jetson Orin Nano Super, using **DeepStream SDK** and **TensorRT** for optimized edge inference.

![Pipeline Demo](media/demo.gif)

## Performance Results

### Unconstrained Throughput (video file, no camera bottleneck)

| Precision | Avg FPS | Engine Size | Speedup vs FP32 |
|-----------|---------|-------------|-----------------|
| FP32      | 46      | 47.5 MB     | 1.0×            |
| FP16      | 94      | 24.6 MB     | 2.0×            |
| INT8      | 135     | 14.0 MB     | 2.9×            |

Pure inference throughput (trtexec, no pipeline overhead): INT8 achieves **208 FPS** (4.75ms GPU latency). The gap between 208 FPS (pure inference) and 135 FPS (full DeepStream pipeline) reveals that at high frame rates, video decode and object tracking become the bottleneck — not the GPU inference itself.

### Live Camera (USB webcam at 640×480 @ 30fps)

All three precisions handle the camera's 30fps output in real-time with significant GPU headroom, confirming the Orin Nano Super can support multi-stream or higher-resolution workloads.

*Measured on Jetson Orin Nano Super (8GB), JetPack 6.2.1, DeepStream 7.0, TensorRT 8.6.*

## Architecture
```
USB Camera (640×480 @ 30fps)
    │
    ▼
┌──────────────────────────────────────────────────┐
│               DeepStream Pipeline                │
│                                                  │
│  v4l2src → videoconvert → nvvideoconvert         │
│                                │                 │
│                          nvstreammux             │
│                                │                 │
│                            nvinfer               │
│                     (YOLOv8s + TensorRT)         │
│                                │                 │
│                          nvtracker               │
│                          (NvDCF)                 │
│                                │                 │
│                           nvdsosd                │
│                            │    │                │
│                       Display  Metadata → JSON   │
└──────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- NVIDIA Jetson Orin (Nano/NX/AGX) with JetPack 6.x
- Docker installed (included with JetPack)
- USB webcam

### Setup
```bash
# 1. Clone this repo
git clone https://github.com/ciprianmanea/jetson-edge-ai-pipeline.git
cd jetson-edge-ai-pipeline

# 2. Pull the DeepStream container
docker pull nvcr.io/nvidia/deepstream-l4t:7.0-samples-multiarch

# 3. Export YOLOv8s to ONNX (on a machine with Python + ultralytics)
pip install ultralytics
python3 -c "from ultralytics import YOLO; YOLO('yolov8s.pt').export(format='onnx', imgsz=640, opset=12, simplify=True)"
# Copy yolov8s.onnx to models/

# 4. Build TensorRT engines (on the Jetson, inside Docker)
./run.sh --shell
python3 scripts/convert_to_trt.py --onnx models/yolov8s.onnx --all
# For INT8 (requires calibration images in models/val2017/):
trtexec --onnx=models/yolov8s.onnx --saveEngine=models/yolov8s_int8.engine --int8 --fp16
exit

# 5. Run the pipeline
./run.sh                    # With display
./run.sh --headless         # Without display
./run.sh --benchmark        # Benchmark all precisions
```

## Project Structure
```
jetson-edge-ai-pipeline/
├── configs/
│   ├── config_infer_yolov8.txt       # nvinfer configuration
│   ├── deepstream_yolov8_usb.txt     # USB camera pipeline config
│   ├── deepstream_yolov8_file.txt    # Video file pipeline config
│   ├── coco_labels.txt               # COCO 80-class labels
│   └── nvdsinfer_yolov8_parser.cpp   # Custom C++ output parser for YOLOv8
├── scripts/
│   ├── convert_to_trt.py             # ONNX → TensorRT conversion (FP32/FP16/INT8)
│   ├── int8_calibrator.py            # INT8 calibration with COCO images
│   └── detection_logger.py           # Standalone TensorRT inference + JSON logging
├── models/                           # Model files (not tracked — see Setup)
├── benchmarks/                       # Performance results
├── output/                           # Detection logs (generated at runtime)
├── media/                            # Demo GIFs and screenshots
├── docs/
│   └── applicability.md              # How these patterns transfer beyond Jetson
├── run.sh                            # Docker-based launcher
├── LICENSE
└── README.md
```

## Technical Details

### Why ONNX → TensorRT?

You *can* run the ONNX model directly using ONNX Runtime, but it would be 3–5× slower. TensorRT performs hardware-specific optimizations during conversion: layer fusion (Conv+BN+ReLU becomes one kernel), kernel auto-tuning for the exact GPU, and memory layout optimization. The conversion is a one-time "compilation cost" that pays for itself on every inference.

### Model Size vs Runtime Memory

The 43MB ONNX file is not what gets loaded into memory. During inference, YOLOv8s at FP16 uses roughly 100–200MB of GPU memory (weights + activations + workspace). On the Orin Nano Super's 8GB unified memory, this is very manageable.

### Precision Tradeoffs

- **FP32 → FP16**: Direct mathematical conversion from 32-bit to 16-bit floats. Nearly lossless, 2× speedup, no calibration needed.
- **FP16 → INT8**: Requires calibration data because mapping continuous floats to 256 discrete integers needs knowledge of actual value ranges at each layer. We used COCO val2017 images for calibration via `trtexec`. The 2.9× total speedup comes with minimal accuracy loss for detection tasks.

### TensorRT Engines in Production

TensorRT engines are tied to the exact GPU architecture — an engine built on Orin cannot run on an RTX 4070 Ti. In production fleet deployments, the engine is built once on a representative device, signed, and distributed via OTA. Receiving devices load the pre-built engine in seconds. Storage integrity during OTA transfer is critical — a corrupted engine means a device that cannot run inference until the next successful update.

### Custom YOLOv8 Parser

DeepStream's `nvinfer` plugin requires a custom C++ shared library to parse YOLOv8's output tensor `[1, 84, 8400]` (4 box coordinates + 80 class scores × 8400 proposals). The parser performs confidence thresholding and non-maximum suppression before passing detections to the tracker.

### Object Tracking

NvDCF (NVIDIA Discriminative Correlation Filter) tracker provides persistent object IDs across frames, occlusion handling, and the ability to skip inference frames while maintaining tracking — useful for reducing GPU load in multi-stream scenarios.

## Applicability Beyond Jetson

The architecture patterns in this project transfer to any edge AI platform:

- **Model optimization pipeline** (ONNX → hardware-specific engine) applies equally to TensorFlow Lite on Google Coral, ONNX Runtime on Qualcomm, Core ML on Apple Silicon, and OpenVINO on Intel
- **Streaming inference architecture** (source → preprocess → infer → postprocess → output) is universal
- **Precision/speed tradeoff analysis** (FP32 vs FP16 vs INT8) is relevant on every platform
- **The real deployment challenges** — storage reliability during OTA model updates, I/O contention during simultaneous recording, fleet management — are hardware-agnostic

The framework names change; the systems engineering doesn't.

## Hardware

- **Board:** NVIDIA Jetson Orin Nano Super (8GB)
- **Storage:** 256GB NVMe SSD
- **Camera:** USB webcam (640×480 @ 30fps)
- **JetPack:** 6.2.1 (L4T R36.4.7)
- **DeepStream:** 7.0
- **TensorRT:** 8.6
- **CUDA:** 12.6

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
