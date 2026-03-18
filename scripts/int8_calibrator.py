#!/usr/bin/env python3
"""
INT8 Calibration for YOLOv8 TensorRT engine.
Uses cuda module from tensorrt for memory management.
"""

import os
import glob
import argparse
import time
import numpy as np
import cv2
import tensorrt as trt
from cuda import cudart

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class YoloV8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, images_dir, cache_file, batch_size=1, input_size=640, max_images=200):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.input_size = input_size

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
        self.image_paths = sorted(self.image_paths)[:max_images]
        print(f"Calibration: found {len(self.image_paths)} images (using up to {max_images})")

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No calibration images found in {images_dir}")

        self.current_index = 0
        self.nbytes = batch_size * 3 * input_size * input_size * np.dtype(np.float32).itemsize
        self.batch_data = np.zeros((batch_size, 3, input_size, input_size), dtype=np.float32)

        # Allocate GPU memory using cudart
        err, self.d_input = cudart.cudaMalloc(self.nbytes)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc failed: {err}")
        print(f"Allocated {self.nbytes / 1024 / 1024:.1f} MB GPU memory for calibration")

    def __del__(self):
        try:
            cudart.cudaFree(self.d_input)
        except:
            pass

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        return img

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.image_paths):
            return None

        batch_end = min(self.current_index + self.batch_size, len(self.image_paths))
        batch_count = 0

        for i in range(self.current_index, batch_end):
            img = self.preprocess_image(self.image_paths[i])
            if img is not None:
                self.batch_data[batch_count] = img
                batch_count += 1

        if batch_count == 0:
            self.current_index = batch_end
            return None

        self.current_index = batch_end

        if self.current_index % 50 == 0 or self.current_index >= len(self.image_paths):
            print(f"  Calibrating: {self.current_index}/{len(self.image_paths)} images processed")

        contiguous = np.ascontiguousarray(self.batch_data)
        err, = cudart.cudaMemcpy(self.d_input, contiguous.ctypes.data,
                                  self.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        if err != cudart.cudaError_t.cudaSuccess:
            print(f"WARNING: cudaMemcpy failed: {err}")
            return None

        return [int(self.d_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"Reading calibration cache from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print(f"Writing calibration cache to {self.cache_file}")
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


def build_int8_engine(onnx_path, images_dir, output_path, cache_file, workspace_gb=2, max_images=200):
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)

    calibrator = YoloV8Calibrator(
        images_dir=images_dir,
        cache_file=cache_file,
        batch_size=1,
        input_size=640,
        max_images=max_images
    )
    config.int8_calibrator = calibrator

    print(f"Building INT8 engine with {max_images} calibration images...")
    start = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    build_time = time.time() - start

    if serialized_engine is None:
        raise RuntimeError("INT8 engine build failed")

    with open(output_path, 'wb') as f:
        f.write(serialized_engine)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nINT8 Engine saved: {output_path}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Build time: {build_time:.1f} seconds")
    print(f"  Calibration images used: {max_images}")

    return build_time, size_mb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--images-dir", required=True, help="Directory with calibration images")
    parser.add_argument("--output", default=None, help="Output engine path")
    parser.add_argument("--cache", default="calibration.cache", help="Calibration cache file")
    parser.add_argument("--max-images", type=int, default=200, help="Max calibration images")
    parser.add_argument("--workspace", type=int, default=2, help="Workspace size in GB")
    args = parser.parse_args()

    output = args.output or args.onnx.replace(".onnx", "_int8.engine")

    build_int8_engine(
        onnx_path=args.onnx,
        images_dir=args.images_dir,
        output_path=output,
        cache_file=args.cache,
        workspace_gb=args.workspace,
        max_images=args.max_images
    )
