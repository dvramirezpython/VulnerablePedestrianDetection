from ultralytics import RTDETR
from onnxruntime.quantization import quantize_dynamic, QuantType, preprocess


# Load model
dataset_path = 'dataset/dataset_all_weather_pedestrian_vulnerable_v2/'
model = RTDETR(model="runs/detect/train8_full_datasetv2_labels_checked_rt_detr/weights/RT_DETR_weights.pt")
quantized_model = model.export(format='tflite', int8=True, data=f'{dataset_path}data.yaml')

# Dummy input
# x = torch.rand(1, 3, 640, 640)

# Export to ONNX
# rtdetr_onnx_path = quantized_model.export(format="onnx",
#                                   batch=1,          # force batch = 1
#                                   imgsz=(640, 640),  # Specify input image size (height, width)
#                                   opset=16,          # Recommended opset version for compatibility
#                                   dynamic=False,      # Set to True for dynamic batch size and image size (if supported by your runtime)
#                                   half=False,        # Set to True for float16 conversion (requires onnxconverter-common)
#                                   simplify=True,    # Use onnxsim for model simplification if needed (pip install onnxsim)
#                                   device='0')       # Use 'cpu' for export, or '0' for CUDA device 0 if available


