from ultralytics import YOLO

# Load trained model
model = YOLO("C:/Users/india/Desktop/Vehicle Detection/best (1).pt")

# Export to TFLite
model.export(
    format="tflite",
    imgsz=640
)

print("TFLite conversion completed")