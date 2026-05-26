Vehicle Detection, Tracking, and Tripwire Counting Pipeline using YOLOv8 & TensorFlow Lite

This repository provides an end-to-end computer vision pipeline for engineering a localized vehicle detection and counting system. The pipeline transitions a standard PyTorch-trained YOLOv8 architecture into highly optimized edge formats (ONNX & TFLite), culminating in a robust multi-object tracking system capable of edge-based tripwire counting.

---

## 🏗️ System Architecture & Workflow

The codebase is organized into three distinct phases: Training, Optimization/Quantization, and Deployment Inference.



[ Custom Dataset ] ──> [ train.py / Notebook ] ──> best.pt (PyTorch)
│
┌────────────────────────────────────┴────────────────────────────────────┐
▼                                                                         ▼
[ export.py (ONNX Export) ]                                           [ convert_tflite.py (TFLite Export) ]
│                                                                         │
▼                                                                         ▼
best.onnx (Cloud/PC)                                                    best_float16.tflite (Edge/Mobile)
│
┌────────────────────────────────────┴────────────────────────────────────┐
▼                                                                         ▼
[ tflite_predict.py (Native TF Runtime) ]                           [ vehicle_count.py (Object Tracking) ]



🛠️ Code Deep-Dive & Detailed Usage
1. Model Training Phase
The training loop utilizes the yolov8n.pt (Nano) backbone, offering the optimal speed-to-accuracy trade-off for embedded applications. It features auto-evaluation and early stopping hooks.

Script: train.py (or step-by-step evaluation via vehicle_detection_model_train.ipynb)

Execution Parameters:

epochs=100: Maximum training cycles.

patience=20: Early stopping triggers if validation mAP50-95 stops rising for 20 successive epochs.

imgsz=640: Target input frame dimension scaling.




3. Real-Time Tracking & Tripwire Counting LogicThe script vehicle_count.py implements a persistent frame-by-frame tracker. Rather than counting every bounding box detected in a scene (which would over-count stalled or static vehicles), the system sets a virtual line threshold (LINE_Y).The Counting Algorithm:Object Tracking: Uses the Ultralytics built-in tracking algorithm to append a unique ID to every detected item across frames (persist=True).Centroid Derivation: The center coordinates $(C_x, C_y)$ of each vehicle bounding box are extracted:$$C_x = \frac{x_1 + x_2}{2}, \quad C_y = \frac{y_1 + y_2}{2}$$Tripwire Boundary Check: When a specific tracking ID’s $C_y$ spatial coordinate crosses the designated LINE_Y plane, the system evaluates if that ID has been counted. If unique, the absolute integer vehicle_count increments and the ID is written into a memory stack (counted_ids).
