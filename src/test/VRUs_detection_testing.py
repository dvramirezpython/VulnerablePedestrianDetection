from ultralytics import YOLO, RTDETR

# dataset_path = 'dataset/video2.webm'
# dataset_path = 'dataset/PIE_dataset/video_0001.mp4'

# dataset_path = 'dataset/vulnerable-people-detect.v1i.yolov11_Dissanayake2025/'
# dataset_path = 'dataset/BGVP_dataset/BGVP_test_binary_detection.v2i.yolov12'
# dataset_path = 'dataset/SeeingThroughFog/test_night_imags_lut/'
# dataset_path = 'dataset/SeeingThroughFog/testing_dataset.v1i.yolov12/'
dataset_path = 'dataset/PIE_dataset/video_0004.mp4'

# Tuned models
# model = YOLO('runs/detect/train7_full_datasetv2_labels_checked_yolo8s/weights/best.pt')
# model = YOLO('runs/detect/train6_full_datasetv2_labels_checked_yolo11s/weights/best.pt')
model = YOLO('runs/detect/train5_full_datasetv2_labels_checked_yolo12s/weights/best.pt')
# model = RTDETR('runs/detect/train8_full_datasetv2_labels_checked_rt_detr/weights/best.pt')

# Basic models
# model = YOLO('yolov8s.pt')
# model = YOLO('yolo11s.pt')
# model = YOLO('yolo12s.pt')
# model = RTDETR(model='rtdetr-l.pt')

#__________________Testing with a dataset________________---
def test_with_dataset(model, dataset_path):
    yaml_filepath = f'{dataset_path}/data.yaml'
    results = model.val(data=yaml_filepath, split="test")
    return results

#__________________Testing________________---
def test_image(model, image_path):
    results = model(source=image_path,
                    verbose=True,
                    device=0,
                    save=True)#,
                    # classes = [0])
    return results

#__________________Testing with multiple images________________---
def test_images(model, images_list):
    import os
    for image in os.listdir(dataset_path):
        results = model(source=dataset_path, 
                    verbose=True,
                    device=0,
                    save=True,
                    show=False) 
                    
                    # stream=True)
    return None


# Run evaluation (mAP@50 and mAP@50-95 are computed automatically)
# results = test_with_dataset(model, dataset_path)

# Extract metrics
# print("Evaluation Results:")
# print(f"mAP@50:     {results.results_dict['metrics/mAP50(B)']:.4f}")
# print(f"mAP@50-95:  {results.results_dict['metrics/mAP50-95(B)']:.4f}")

# Test one image
# image_path = 'dataset/SeeingThroughFog/testing_dataset.v1i.yolov12/challenging_test/night_snow.jpg'
results = test_image(model, dataset_path)   