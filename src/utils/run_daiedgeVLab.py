from daiedge_vlab import dAIEdgeVLabAPI

TARGET = "jetsonorinnano" # "rpi4b" for Raspberry Pi 4B, "jetsonorinnano" for NVIDIA Jetson Orin Nano
RUNTIME = 'trt' # 'trt' for TensorRT, 'ort' for onnixruntime 
# MODEL = f"runs/detect/train5_full_datasetv2_labels_checked_yolo12s/weights/YOLOv12s_weights.onnx"
# MODEL = f"runs/detect/train6_full_datasetv2_labels_checked_yolo11s/weights/YOLOv11s_weights.onnx"
# MODEL = f"runs/detect/train7_full_datasetv2_labels_checked_yolo8s/weights/YOLOv8s_weights.onnx"
MODEL = f"runs/detect/train8_full_datasetv2_labels_checked_rt_detr/weights/RT_DETR_weights_quantized.onnx"
# MODEL = f"runs/detect/train6/weights/best.onnx"

api = dAIEdgeVLabAPI("setup.yaml")

# Start a benchmark for a given target, runtime and model
benchmark_id = api.startBenchmark(
    target = TARGET, 
    runtime = RUNTIME, 
    model_path = MODEL
    )

# Blocking method - wait for the results
result = api.waitBenchmarkResult(benchmark_id)

# Use the result
print("Report:\n", result["report"])
print("User log:\n", result["user_log"])
print("Error log:\n", result["error_log"])
print("Mean us : ", result["report"]["inference_latency"]["mean"])