#!/usr/bin/env python3
"""
Standalone detection logger for YOLOv8 on Jetson.
Runs TensorRT inference on USB camera and logs detections to JSONL format.

This script demonstrates direct TensorRT inference without DeepStream,
useful for lightweight deployments or custom pipelines.

Usage:
  python3 detection_logger.py --engine ../models/yolov8s_fp16.engine
  python3 detection_logger.py --engine ../models/yolov8s_int8.engine --duration 60
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

import numpy as np
import cv2
import tensorrt as trt

# COCO class labels
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

CONF_THRESH = 0.25
NMS_THRESH = 0.45
INPUT_SIZE = 640


class TRTInferencer:
    """TensorRT inference engine wrapper."""

    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        print(f"Loading TensorRT engine: {engine_path}")

        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Get binding info
        self.input_name = self.engine.get_binding_name(0)
        self.output_name = self.engine.get_binding_name(1)
        self.input_shape = self.engine.get_binding_shape(0)
        self.output_shape = self.engine.get_binding_shape(1)

        print(f"  Input:  {self.input_name} {self.input_shape}")
        print(f"  Output: {self.output_name} {self.output_shape}")

        # Allocate host buffers
        self.input_host = np.zeros(trt.volume(self.input_shape), dtype=np.float32)
        self.output_host = np.zeros(trt.volume(self.output_shape), dtype=np.float32)

        # Allocate device buffers using TRT's built-in allocation
        import ctypes
        self.cudart = ctypes.CDLL("libcudart.so")

        self.d_input = ctypes.c_void_p()
        self.d_output = ctypes.c_void_p()
        self.cudart.cudaMalloc(ctypes.byref(self.d_input),
                               ctypes.c_size_t(self.input_host.nbytes))
        self.cudart.cudaMalloc(ctypes.byref(self.d_output),
                               ctypes.c_size_t(self.output_host.nbytes))

        # Create CUDA stream
        self.stream = ctypes.c_void_p()
        self.cudart.cudaStreamCreate(ctypes.byref(self.stream))

        print(f"  Engine loaded successfully")

    def preprocess(self, frame):
        """Preprocess frame for YOLOv8: resize, normalize, CHW, batch."""
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return np.ascontiguousarray(img.flatten())

    def infer(self, preprocessed):
        """Run TensorRT inference."""
        import ctypes

        np.copyto(self.input_host, preprocessed)

        # H2D
        self.cudart.cudaMemcpyAsync(
            self.d_input, self.input_host.ctypes.data,
            ctypes.c_size_t(self.input_host.nbytes),
            ctypes.c_int(1), self.stream)

        # Execute
        self.context.execute_async_v2(
            bindings=[int(self.d_input.value), int(self.d_output.value)],
            stream_handle=int(self.stream.value))

        # D2H
        self.cudart.cudaMemcpyAsync(
            self.output_host.ctypes.data, self.d_output,
            ctypes.c_size_t(self.output_host.nbytes),
            ctypes.c_int(2), self.stream)

        # Synchronize
        self.cudart.cudaStreamSynchronize(self.stream)

        return self.output_host.reshape(self.output_shape)

    def __del__(self):
        try:
            self.cudart.cudaFree(self.d_input)
            self.cudart.cudaFree(self.d_output)
            self.cudart.cudaStreamDestroy(self.stream)
        except:
            pass


def postprocess(output, img_w, img_h):
    """
    Parse YOLOv8 output and apply NMS.
    Output shape: [1, 84, 8400]
    Returns list of detections: {class_id, label, confidence, bbox}
    """
    predictions = output[0]       # [84, 8400]
    predictions = predictions.T   # [8400, 84]

    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    mask = confidences > CONF_THRESH
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(confidences) == 0:
        return []

    # Convert cx,cy,w,h to x1,y1,x2,y2 and scale to image size
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2) / INPUT_SIZE * img_w
    boxes_xyxy[:, 1] = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2) / INPUT_SIZE * img_h
    boxes_xyxy[:, 2] = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2) / INPUT_SIZE * img_w
    boxes_xyxy[:, 3] = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2) / INPUT_SIZE * img_h

    # NMS using OpenCV
    boxes_for_nms = []
    for b in boxes_xyxy:
        boxes_for_nms.append([float(b[0]), float(b[1]),
                              float(b[2] - b[0]), float(b[3] - b[1])])

    indices = cv2.dnn.NMSBoxes(boxes_for_nms, confidences.tolist(),
                                CONF_THRESH, NMS_THRESH)

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            detections.append({
                "class_id": int(class_ids[i]),
                "label": COCO_LABELS[class_ids[i]] if class_ids[i] < len(COCO_LABELS) else f"class_{class_ids[i]}",
                "confidence": round(float(confidences[i]), 3),
                "bbox": {
                    "x1": round(float(boxes_xyxy[i, 0]), 1),
                    "y1": round(float(boxes_xyxy[i, 1]), 1),
                    "x2": round(float(boxes_xyxy[i, 2]), 1),
                    "y2": round(float(boxes_xyxy[i, 3]), 1)
                }
            })

    return detections


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Detection Logger")
    parser.add_argument("--engine", required=True, help="Path to TensorRT engine")
    parser.add_argument("--device", default="/dev/video0", help="Camera device")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--output-dir", default="/workspace/output", help="Output directory")
    parser.add_argument("--duration", type=int, default=30, help="Recording duration in seconds")
    parser.add_argument("--video", default=None, help="Video file instead of camera")
    args = parser.parse_args()

    # Initialize inference engine
    inferencer = TRTInferencer(args.engine)

    # Open video source
    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"Video source: {args.video}")
    else:
        cap = cv2.VideoCapture(args.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"Camera source: {args.device} @ {args.width}x{args.height}")

    if not cap.isOpened():
        sys.exit(f"ERROR: Could not open video source")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Actual resolution: {actual_w}x{actual_h}")

    # Setup output
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.output_dir, f"detections_{timestamp}.jsonl")
    summary_path = os.path.join(args.output_dir, f"summary_{timestamp}.json")

    print(f"\n{'='*60}")
    print(f"YOLOv8 Detection Logger")
    print(f"  Engine:   {args.engine}")
    print(f"  Source:   {'video' if args.video else 'camera'}")
    print(f"  Duration: {args.duration}s")
    print(f"  Output:   {log_path}")
    print(f"{'='*60}")
    print(f"Running... Press Ctrl+C to stop early.\n")

    # Main loop
    frame_count = 0
    total_detections = 0
    fps_history = []
    start_time = time.time()
    log_file = open(log_path, 'w')

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= args.duration:
                print(f"\nDuration reached ({args.duration}s)")
                break

            ret, frame = cap.read()
            if not ret:
                if args.video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            frame_start = time.time()

            # Preprocess
            preprocessed = inferencer.preprocess(frame)

            # Infer
            output = inferencer.infer(preprocessed)

            # Postprocess
            detections = postprocess(output, actual_w, actual_h)

            frame_time = time.time() - frame_start
            fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_history.append(fps)

            frame_count += 1
            total_detections += len(detections)

            # Log to JSONL
            record = {
                "frame": frame_count,
                "timestamp": datetime.now().isoformat(),
                "elapsed_s": round(elapsed, 3),
                "inference_ms": round(frame_time * 1000, 2),
                "fps": round(fps, 1),
                "num_detections": len(detections),
                "detections": detections
            }
            log_file.write(json.dumps(record) + "\n")

            # Print periodic status
            if frame_count % 30 == 0:
                avg_fps = np.mean(fps_history[-30:])
                det_per_frame = total_detections / frame_count
                print(f"  Frame {frame_count:5d} | FPS: {avg_fps:.1f} | "
                      f"Detections this frame: {len(detections)} | "
                      f"Avg detections/frame: {det_per_frame:.1f}")

    except KeyboardInterrupt:
        print(f"\nInterrupted by user")

    finally:
        total_time = time.time() - start_time
        log_file.close()
        cap.release()

        # Generate summary
        avg_fps = frame_count / total_time if total_time > 0 else 0
        summary = {
            "engine": args.engine,
            "source": args.video or args.device,
            "resolution": f"{actual_w}x{actual_h}",
            "duration_s": round(total_time, 2),
            "total_frames": frame_count,
            "total_detections": total_detections,
            "avg_detections_per_frame": round(total_detections / max(frame_count, 1), 2),
            "avg_fps": round(avg_fps, 1),
            "min_fps": round(min(fps_history) if fps_history else 0, 1),
            "max_fps": round(max(fps_history) if fps_history else 0, 1),
            "p95_inference_ms": round(np.percentile([1000/f for f in fps_history if f > 0], 95), 2) if fps_history else 0,
            "hardware": "Jetson Orin Nano Super 8GB",
            "timestamp": datetime.now().isoformat()
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Detection Log Summary")
        print(f"  Total frames:     {frame_count}")
        print(f"  Total detections: {total_detections}")
        print(f"  Avg FPS:          {avg_fps:.1f}")
        print(f"  Avg det/frame:    {total_detections / max(frame_count, 1):.1f}")
        print(f"  Duration:         {total_time:.1f}s")
        print(f"  Log file:         {log_path}")
        print(f"  Summary:          {summary_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
