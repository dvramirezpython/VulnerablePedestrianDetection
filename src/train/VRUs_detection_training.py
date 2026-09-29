from ultralytics import YOLO

dataset_path = 'dataset/dataset_all_weather_pedestrian_vulnerable_v2'
model = YOLO('yolov8s.pt')

results = model.train(data=f'{dataset_path}/data.yaml', 
                      epochs=250,
                      verbose=True,
                      device=0)
