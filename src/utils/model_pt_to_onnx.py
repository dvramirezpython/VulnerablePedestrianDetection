# Script to to translate a PyTorch model to ONNX format
import torch
import onnx
from ultralytics import YOLO
from ultralytics import RTDETR
# Load the PyTorch model
# pytorch_model_path = 'runs/detect/train5_full_datasetv2_labels_checked_yolo12s/weights/best.pt'
# pytorch_model_path = 'runs/detect/train6_full_datasetv2_labels_checked_yolo11s/weights/best.pt'
# pytorch_model_path = 'runs/detect/train7_full_datasetv2_labels_checked_yolo8s/weights/best.pt'
pytorch_model_path = 'runs/detect/train8_full_datasetv2_labels_checked_rt_detr/weights/best.pt'
model = RTDETR(pytorch_model_path)    
# Define a dummy input tensor with the appropriate shape (batch_size, channels, height, width)
dummy_input = torch.randn(1, 3, 640, 640)
# Export the model to ONNX format
onnx_model_path = 'best.onnx'
# model.export(format='onnx', imgsz=(640, 640), dynamic=True, simplify=True)
model.export(
    format='onnx',
    imgsz=640,
    batch=1,          # force batch = 1
    dynamic=False,    # disable dynamic axes
    simplify=True,
    opset=16          # safe opset for ARM
)
# Verify the ONNX model
onnx_model = onnx.load(f'runs/detect/train8_full_datasetv2_labels_checked_rt_detr/weights/{onnx_model_path}')
onnx.checker.check_model(onnx_model)
print(f"ONNX model has been successfully exported to {onnx_model_path}")
