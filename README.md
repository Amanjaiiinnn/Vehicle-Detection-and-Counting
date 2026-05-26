# Vehicle Detection, Tracking, and Tripwire Counting Pipeline using YOLOv8 & TensorFlow Lite

This repository provides an end-to-end computer vision pipeline for building a localized vehicle detection and counting system. The workflow transforms a standard PyTorch-trained YOLOv8 model into highly optimized edge-deployment formats such as ONNX and TensorFlow Lite (TFLite), enabling efficient real-time inference on edge devices.

The project also integrates multi-object tracking and tripwire-based vehicle counting for accurate traffic analysis.

---

# 🏗️ System Architecture & Workflow

The pipeline is divided into three major phases:

1. **Model Training**
2. **Model Optimization & Quantization**
3. **Deployment & Real-Time Inference**

```text
[ Custom Dataset ]
        │
        ▼
[ train.py / Notebook ]
        │
        ▼
best.pt (PyTorch Model)
        │
 ┌──────┴───────────────────────────────┐
 ▼                                      ▼
[ export.py ]                    [ convert_tflite.py ]
 (ONNX Export)                     (TFLite Export)
        │                                      │
        ▼                                      ▼
best.onnx                         best_float16.tflite
 (Cloud / PC)                        (Edge / Mobile)
        │
 ┌──────┴───────────────────────────────┐
 ▼                                      ▼
[tflite_predict.py]              [vehicle_count.py]
(Native TF Runtime)               (Tracking & Counting)






🛠️ Code Deep-Dive & Detailed Usage
1. Model Training Phase
The training loop utilizes the yolov8n.pt (Nano) backbone, offering the optimal speed-to-accuracy trade-off for embedded applications. It features auto-evaluation and early stopping hooks.

Script: train.py (or step-by-step evaluation via vehicle_detection_model_train.ipynb)

Execution Parameters:

epochs=100: Maximum training cycles.

patience=20: Early stopping triggers if validation mAP50-95 stops rising for 20 successive epochs.

imgsz=640: Target input frame dimension scaling.

```

3. Real-Time Tracking & Tripwire Counting LogicThe script vehicle_count.py implements a persistent frame-by-frame tracker. Rather than counting every bounding box detected in a scene (which would over-count stalled or static vehicles), the system sets a virtual line threshold (LINE_Y).The Counting Algorithm:Object Tracking: Uses the Ultralytics built-in tracking algorithm to append a unique ID to every detected item across frames (persist=True).Centroid Derivation: The center coordinates $(C_x, C_y)$ of each vehicle bounding box are extracted:$$C_x = \frac{x_1 + x_2}{2}, \quad C_y = \frac{y_1 + y_2}{2}$$Tripwire Boundary Check: When a specific tracking ID’s $C_y$ spatial coordinate crosses the designated LINE_Y plane, the system evaluates if that ID has been counted. If unique, the absolute integer vehicle_count increments and the ID is written into a memory stack (counted_ids).


  # 🎯 Real-Time Vehicle Tracking & Tripwire Counting

The `vehicle_count.py` script implements a persistent multi-object tracking and counting system.

Instead of counting every detected bounding box in each frame (which would repeatedly count stationary vehicles), the system uses a **virtual tripwire line** (`LINE_Y`) to count vehicles only once when they cross a defined boundary.

---

## 📌 Counting Algorithm

1. **Object Tracking:** Uses the Ultralytics built-in tracking algorithm to append a unique ID to every detected item across frames (`persist=True`).
2. **Centroid Derivation:** The center coordinates $(C_x, C_y)$ of each vehicle bounding box are extracted:
   $$C_x = \frac{x_1 + x_2}{2}, \quad C_y = \frac{y_1 + y_2}{2}$$
3. **Tripwire Boundary Check:** When a specific tracking ID’s $C_y$ spatial coordinate crosses the designated `LINE_Y` plane, the system evaluates if that ID has been counted. If unique, the absolute integer `vehicle_count` increments and the ID is written into a memory stack (`counted_ids`).

## 1️⃣ Object Tracking

The system uses the built-in Ultralytics tracking functionality:

```python
persist=True

