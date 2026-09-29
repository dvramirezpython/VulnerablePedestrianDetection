# from rfdetr.visualize.training import plot_loss_metrics, plot_map_metrics
# from rfdetr.datasets._keypoint_schema import infer_coco_keypoint_schema

dataset = f"dataset/dataset_all_weather_pedestrian_vulnerable_v2"

from rfdetr import RFDETRMedium
import torch
import torch.nn.functional as F

model = RFDETRMedium()

RESOLUTION = 576
EPOCHS = 10
BATCH_SIZE = 4
# BATCH_SIZE = 'auto'
GRAD_ACCUM_STEPS = 2
LR = 1e-4


trained_model = model.train(
    dataset_dir=dataset,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    grad_accum_steps=GRAD_ACCUM_STEPS,
    lr=LR,
    output_dir='runs/detect/rf_detr'
)

_original_interpolate = F.interpolate
def patched_interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
    if mode == 'bicubic' and antialias:
        antialias = False 
    return _original_interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)
F.interpolate = patched_interpolate

onnx_path = 'runs/detect/rf_detr/rf_detr_medium.onnx'

model = RFDETRMedium(
    resolution=588,
    num_classes=1,
    patch_size=14,
    pretrain_weights='runs/detect/rf_detr/checkpoint_best_total.pth'
)

model.export(onnx_path)

print(f"Modelo exportado exitosamente a {onnx_path}")
