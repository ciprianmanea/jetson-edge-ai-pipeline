#!/usr/bin/env python3
"""
Convert ONNX model to TensorRT engine with FP32, FP16, and INT8 precision.
Usage: python3 convert_to_trt.py --onnx ../models/yolov8s.onnx --precision fp16
"""

import argparse
import os
import time
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_path, precision, output_path, workspace_gb=2):
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed")

    print(f"Network inputs: {network.num_inputs}, outputs: {network.num_outputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  Input {i}: {inp.name} | shape: {inp.shape} | dtype: {inp.dtype}")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 enabled")
        else:
            print("WARNING: FP16 not supported on this platform, falling back to FP32")
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.FP16)  # INT8 with FP16 fallback
            print("INT8 enabled (with FP16 fallback)")
            print("NOTE: For production INT8, you should provide a calibration dataset")
            # For a proper INT8 calibration, you would implement an INT8 calibrator
            # For now, TensorRT will use default quantization ranges
        else:
            print("WARNING: INT8 not supported, falling back to FP16")
            config.set_flag(trt.BuilderFlag.FP16)

    # Build engine
    print(f"Building TensorRT engine ({precision})... This may take 5-15 minutes on Orin Nano")
    start = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    build_time = time.time() - start

    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    # Save engine
    with open(output_path, 'wb') as f:
        f.write(serialized_engine)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Engine saved: {output_path}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Build time: {build_time:.1f} seconds")
    print(f"  Precision: {precision}")

    return build_time, size_mb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--output", default=None, help="Output engine path")
    parser.add_argument("--workspace", type=int, default=2, help="Workspace size in GB")
    parser.add_argument("--all", action="store_true", help="Build all precisions")
    args = parser.parse_args()

    if args.all:
        results = {}
        for prec in ["fp32", "fp16", "int8"]:
            out = args.onnx.replace(".onnx", f"_{prec}.engine")
            build_time, size_mb = build_engine(args.onnx, prec, out, args.workspace)
            results[prec] = {"time": build_time, "size": size_mb, "path": out}
        print("\n=== Build Summary ===")
        for prec, info in results.items():
            print(f"  {prec}: {info['size']:.1f} MB, built in {info['time']:.1f}s")
    else:
        output = args.output or args.onnx.replace(".onnx", f"_{args.precision}.engine")
        build_engine(args.onnx, args.precision, output, args.workspace)
