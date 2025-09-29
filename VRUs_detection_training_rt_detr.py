'''
Fine tune of the model RT-DETR to detect pedestrian and vulnerable pedestrian
'''
from ultralytics import RTDETR

dataset_path = 'dataset/dataset_all_weather_pedestrian_vulnerable_v2/'

model = RTDETR(model='rtdetr-l.pt')

model.info()

result = model.train(data=f'{dataset_path}data.yaml', epochs = 250, 
                     imgsz = 640, device = 0, amp=False, batch=4)