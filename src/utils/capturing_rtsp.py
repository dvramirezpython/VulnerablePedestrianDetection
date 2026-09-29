import cv2
from ultralytics import YOLO
from ultralytics.utils import plotting 

# Open the RTSP stream
cap = cv2.VideoCapture("rtsp://192.72.1.1:554/liveRTSP/av4")
model = YOLO('yolo26l-pose.pt')  # Load the pose detection model
# model = YOLO('yolo26l.pt')  # Load the detection model
# model = YOLO('yolo26l-seg.pt')  # Load the segmentation model
# model = YOLO('yolo26l-sem.pt')  # Load the semantic segmentation model
# model = YOLO('yolo26l-depth.pt')  # Load the depth estimation model
# model = YOLO('yolo26l-obb.pt')  # Load the oriented bounding box model

# Check if the stream is opened successfully
if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

# Read frames from the stream
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Enhace frame brightness and contrast
    alpha = 1.0  # Contrast control (1.0-3.0)
    beta = 40    # Brightness control (0-100)
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # Perform object detection
    results = model(frame, device='cuda')

    # Display the frame
    # depth = results[0].depth.data.cpu().numpy()  # (H, W) float32, meters

    # Colorize with near = warm and save
        
    cv2.imshow("RTSP Stream", results[0].plot()) # For traditional detection, segmentation, and pose
    # cv2.imshow("RTSP Stream", 
    #            plotting.colorize_depth(depth, cmap="spectral")) # for depth estimation


    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()