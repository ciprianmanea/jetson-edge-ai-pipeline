#!/bin/bash
# Run the Edge AI Perception Pipeline in Docker
# Usage:
#   ./run.sh                           # Display mode (requires DISPLAY)
#   ./run.sh --headless                # No display
#   ./run.sh --file video.mp4          # Run on video file
#   ./run.sh --benchmark               # Benchmark all precisions

set -e

DISPLAY_VAR="${DISPLAY:-:1}"
CONTAINER="nvcr.io/nvidia/deepstream-l4t:7.0-samples-multiarch"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

case "${1}" in
  --headless)
    echo "Running in headless mode..."
    docker run --rm -it \
      --runtime nvidia --gpus all \
      --device /dev/video0:/dev/video0 \
      -v ${PROJECT_DIR}:/workspace \
      -w /workspace \
      ${CONTAINER} \
      deepstream-app -c configs/deepstream_yolov8_usb.txt
    ;;
  --file)
    VIDEO="${2:?Please specify a video file}"
    echo "Running on video file: ${VIDEO}..."
    docker run --rm -it \
      --runtime nvidia --gpus all \
      -v ${PROJECT_DIR}:/workspace \
      -w /workspace \
      ${CONTAINER} \
      deepstream-app -c configs/deepstream_yolov8_file.txt
    ;;
  --benchmark)
    echo "Running benchmarks across all precisions..."
    for PREC in fp32 fp16 int8; do
      ENGINE="models/yolov8s_${PREC}.engine"
      if [ ! -f "${ENGINE}" ]; then
        echo "SKIP: ${ENGINE} not found"
        continue
      fi
      echo ""
      echo "=== Benchmarking ${PREC} ==="
      docker run --rm -it \
        --runtime nvidia --gpus all \
        -v ${PROJECT_DIR}:/workspace \
        -w /workspace \
        ${CONTAINER} \
        trtexec --loadEngine=/workspace/${ENGINE} --warmUp=5000 --duration=30
    done
    ;;
  --shell)
    echo "Opening container shell..."
    docker run --rm -it \
      --runtime nvidia --gpus all \
      --device /dev/video0:/dev/video0 \
      -e DISPLAY=${DISPLAY_VAR} \
      -v /tmp/.X11-unix/:/tmp/.X11-unix \
      -v ${PROJECT_DIR}:/workspace \
      -w /workspace \
      ${CONTAINER} \
      bash
    ;;
  *)
    echo "Running with display (DISPLAY=${DISPLAY_VAR})..."
    xhost +local:docker 2>/dev/null || true
    docker run --rm -it \
      --runtime nvidia --gpus all \
      --device /dev/video0:/dev/video0 \
      -e DISPLAY=${DISPLAY_VAR} \
      -v /tmp/.X11-unix/:/tmp/.X11-unix \
      -v ${PROJECT_DIR}:/workspace \
      -w /workspace \
      ${CONTAINER} \
      deepstream-app -c configs/deepstream_yolov8_usb.txt
    ;;
esac
