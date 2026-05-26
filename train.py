from ultralytics import YOLO

def main():

    # Load pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    # Train model
    model.train(
        data="vehicles.yaml",   # dataset config
        epochs=100,             # total training cycles
        imgsz=640,              # image resize size
        batch=16,               # images per batch
        device=0,               # GPU device
        patience=20             # early stopping
    )

if __name__ == "__main__":
    main()