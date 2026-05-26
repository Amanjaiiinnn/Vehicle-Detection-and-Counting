from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("C:/Users/india/Desktop/Vehicle Detection/best (1).pt")

# Input video
video_path = "C:/Users/india/Desktop/Vehicle Detection/854671-hd_1920_1080_25fps.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

# Video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
out = cv2.VideoWriter(
    "C:/Users/india/Desktop/Vehicle Detection/output_video.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    # Save frame
    out.write(annotated_frame)

    # Optional display
    cv2.imshow("Detection", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing completed")