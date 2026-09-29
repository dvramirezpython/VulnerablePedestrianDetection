from ultralytics import YOLO

model = YOLO("yolo26s.pt")

# Default tracker (TrackTrack)
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)
results = model.track(source="dataset/video.webm", 
                      tracker="bytetrack.yaml",
                      classes=0, 
                      stream=True, 
                      show=True, 
                      save=True, 
                      stream_buffer=False 
                      )
for r in results:
    pass
