import ncnn
import torch
import torch.nn as nn
from ultralytics import RTDETR
import pnnx
from onnxruntime.quantization import quantize_dynamic, QuantType, preprocess

class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model  # actual nn.Module

    def forward(self, x):
        outputs = self.model(x)
        # Ensure we only return tensors
        if isinstance(outputs, (list, tuple)):
            return tuple(outputs)  # torch.jit.trace likes tuple of tensors
        return outputs


# Load wrapper
wrapper = RTDETR(model="runs/detect/train8_full_datasetv2_labels_checked_rt_detr/weights/RT_DETR_weights.pt")
# model = ExportWrapper(wrapper)

# Dummy input
# x = torch.rand(1, 3, 640, 640)

# Export to ONNX
rtdetr_onnx_path = wrapper.export(format="onnx",
                                  batch=1,          # force batch = 1
                                  imgsz=(640, 640),  # Specify input image size (height, width)
                                  opset=16,          # Recommended opset version for compatibility
                                  dynamic=False,      # Set to True for dynamic batch size and image size (if supported by your runtime)
                                  half=False,        # Set to True for float16 conversion (requires onnxconverter-common)
                                  simplify=True,    # Use onnxsim for model simplification if needed (pip install onnxsim)
                                  device='0')       # Use 'cpu' for export, or '0' for CUDA device 0 if available


# Preprocess the ONNX before quantization (optional but recommended)
model_input="runs/detect/train8_full_datasetv2_labels_checked_rt_detr/weights/RT_DETR_weights.onnx"
model_preprocessed='runs/detect/train8_full_datasetv2_labels_checked_rt_detr/weights/RT_DETR_weights_preprocessed.onnx'
model_output='runs/detect/train8_full_datasetv2_labels_checked_rt_detr/weights/RT_DETR_weights_quantized.onnx'
preprocess.quant_pre_process(model_input, model_preprocessed)


# Perform dynamic quantization on the weights
# This should reduce the size by ~75%
quantize_dynamic(
    model_input=model_preprocessed,
    model_output=model_output,
    op_types_to_quantize=['MatMul', 'Conv'], # Common ops to target
    weight_type=QuantType.QUInt8
)

# Export
# opt_model = pnnx.export(
#     wrapper,
#     "runs/detect/train8_full_datasetv2_labels_checked_rt_detr/weights/best_ncnn.pt",
#     x,
#     check_trace=False
# )




# for image in os.listdir(dataset_path):
# results = model(source=dataset_path, 
#             verbose=True,
#             device=0,
#             save=True,
#             show=False)
#             # stream=True)



#__________________Testing________________---


# dataset_path = 'dataset/SeeingThroughFog/testing_dataset.v1i.yolov12/'
# # Run evaluation (mAP@50 and mAP@50-95 are computed automatically)
# yaml_filepath = f'{dataset_path}/data.yaml'
# results = opt_model.val(data=yaml_filepath, split="test")

# # Extract metrics
# print("\n📊 Evaluation Results:")
# print(f"mAP@50:     {results.results_dict['metrics/mAP50(B)']:.4f}")
# print(f"mAP@50-95:  {results.results_dict['metrics/mAP50-95(B)']:.4f}")
