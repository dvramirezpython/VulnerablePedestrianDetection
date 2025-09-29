from ultralytics import YOLO

# dataset_path = 'dataset/video2.webm'
# dataset_path = 'dataset/PIE_dataset/video_0001.mp4'

# dataset_path = 'dataset/vulnerable-people-detect.v1i.yolov11_Dissanayake2025/'
# dataset_path = 'dataset/BGVP_dataset/BGVP_test_binary_detection.v2i.yolov12'
# dataset_path = 'dataset/SeeingThroughFog/test_night_imags_lut/'
dataset_path = 'dataset/SeeingThroughFog/testing_dataset.v1i.yolov12/'
# model = YOLO('runs/detect/train7_full_datasetv2_labels_checked_yolo8s/weights/best.pt')
# model = YOLO('runs/detect/train6_full_datasetv2_labels_checked_yolo11s/weights/best.pt')
model = YOLO('runs/detect/train5_full_datasetv2_labels_checked_yolo12s/weights/best.pt')

# for image in os.listdir(dataset_path):
# results = model(source=dataset_path, 
#             verbose=True,
#             device=0,
#             save=True,
#             show=False)
#             # stream=True)



#__________________Testing________________---



# Run evaluation (mAP@50 and mAP@50-95 are computed automatically)
yaml_filepath = f'{dataset_path}/data.yaml'
results = model.val(data=yaml_filepath, split="test")

# Extract metrics
print("Evaluation Results:")
print(f"mAP@50:     {results.results_dict['metrics/mAP50(B)']:.4f}")
print(f"mAP@50-95:  {results.results_dict['metrics/mAP50-95(B)']:.4f}")