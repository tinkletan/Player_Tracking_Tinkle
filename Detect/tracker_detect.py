import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
from ultralytics import YOLO

model = YOLO('D:/SIT220/detect/train/weights/best.pt')          ### weights from trained model

# Open the video file
video_path = r"D:\SIT220\Player_Tracking\test1.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, show=True, tracker="botsort.yaml")
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
