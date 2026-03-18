# Applicability Beyond Jetson

This document explains how the architecture patterns in this project
transfer to other edge AI platforms and why these skills matter regardless
of which hardware vendor you work with.

## Model Optimization Pipeline

The workflow of exporting a trained model to an intermediate representation
and converting it to a hardware-optimized engine is universal:

| Platform          | Intermediate    | Optimized Runtime       |
|-------------------|-----------------|-------------------------|
| NVIDIA Jetson     | ONNX            | TensorRT engine         |
| Google Coral      | TF SavedModel   | TFLite + Edge TPU       |
| Qualcomm RB5      | ONNX / DLC      | SNPE / QNN              |
| Apple Silicon     | Core ML Model   | Core ML (ANE)           |
| Intel (OpenVINO)  | ONNX            | OpenVINO IR             |
| AMD (Vitis AI)    | ONNX / TF       | Vitis AI DPU            |

The quantization decisions (FP32 → FP16 → INT8) and the resulting
accuracy/speed/size tradeoffs measured in this project's benchmarks
apply to every platform.

## Streaming Inference Architecture

The pipeline pattern — source → preprocess → infer → postprocess → output —
is identical across all edge AI frameworks. DeepStream, GStreamer, FFmpeg,
and platform-specific SDKs all implement variations of this dataflow.

Understanding how to profile and optimize each stage (and discovering that
the bottleneck often isn't inference but video decode, preprocessing, or
I/O) is a transferable skill.

## The Production Gap

The hardest part of edge AI deployment is rarely the inference itself.
It's everything around it:

- **Storage reliability**: What happens to recorded data and model files
  during unexpected power loss?
- **OTA model updates**: How do you safely update models on thousands of
  devices without bricking them?
- **I/O contention**: How does simultaneous recording affect inference
  throughput?
- **Fleet monitoring**: How do you track device health, inference accuracy,
  and performance degradation across a deployed fleet?

These challenges are hardware-agnostic. The solutions apply whether you're
deploying on Jetson, Coral, Qualcomm, or custom silicon.

## Key Takeaway

Edge AI is not just an inference problem. The storage, networking, and
deployment infrastructure layers are what separate a demo from a production
deployment. Understanding the full stack — from GPU kernels to file system
integrity — is what makes a solutions architect effective, regardless of
the platform.
