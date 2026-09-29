from ultralytics import RTDETR

# Load model
model = RTDETR(model="runs/detect/train8_full_datasetv2_labels_checked_rt_detr/weights/RT_DETR_weights.pt")

source_path = 'dataset/PIE_dataset/video_0001.mp4'

#__________________Testing on video________________#
results = model(source=source_path, 
                device=0,
                conf=0.5,
                save=False,
                show=True,
                stream=False)



#__________________Testing________________---


# dataset_path = 'dataset/SeeingThroughFog/testing_dataset.v1i.yolov12/'
# # Run evaluation (mAP@50 and mAP@50-95 are computed automatically)
# yaml_filepath = f'{dataset_path}/data.yaml'
# results = opt_model.val(data=yaml_filepath, split="test")

# # Extract metrics
# print("\n Evaluation Results:")
# print(f"mAP@50:     {results.results_dict['metrics/mAP50(B)']:.4f}")
# print(f"mAP@50-95:  {results.results_dict['metrics/mAP50-95(B)']:.4f}")


#__________________Testing on video________________---

