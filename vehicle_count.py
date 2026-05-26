from ultralytics import YOLO
import cv2

# Load model
model = YOLO(
    r"C:\Users\india\Desktop\Vehicle Detection\best (1)_saved_model\best (1)_float16.tflite",
    task="detect"
)

# Open video
cap = cv2.VideoCapture(
    r"C:\Users\india\Desktop\Vehicle Detection\media\1.mp4"
)

# Counting line position
LINE_Y = 700

# Store counted IDs
counted_ids = set()

# Total vehicle count
vehicle_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Tracking
    results = model.track(
        frame,
        persist=True,
        conf=0.50
    )

    boxes = results[0].boxes

    # Draw counting line
    cv2.line(
        frame,
        (0, LINE_Y),
        (frame.shape[1], LINE_Y),
        (0, 255, 255),
        3
    )

    if boxes.id is not None:

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()

        for box, track_id in zip(xyxy, ids):

            x1, y1, x2, y2 = box

            # Box center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            # Draw center point
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Count vehicle crossing line
            if cy > LINE_Y and track_id not in counted_ids:

                counted_ids.add(track_id)
                vehicle_count += 1

    # Show count
    cv2.putText(
        frame,
        f"Vehicle Count: {vehicle_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Vehicle Counting", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



# from ultralytics import YOLO
# import cv2

# # Load model
# model = YOLO(
#     r"C:\Users\india\Desktop\Vehicle Detection\best (1)_saved_model\best (1)_float16.tflite",
#     task="detect"
# )

# cap = cv2.VideoCapture(
#     r"C:\Users\india\Desktop\Vehicle Detection\traffic.mp4"
# )

# LINE_Y = 400

# counted_ids = set()
# vehicle_count = 0

# while True:

#     ret, frame = cap.read()

#     if not ret:
#         break

#     results = model.track(
#         frame,
#         persist=True,
#         conf=0.50
#     )

#     boxes = results[0].boxes

#     cv2.line(
#         frame,
#         (0, LINE_Y),
#         (frame.shape[1], LINE_Y),
#         (0, 255, 255),
#         3
#     )

#     if boxes.id is not None:

#         ids = boxes.id.cpu().numpy().astype(int)
#         xyxy = boxes.xyxy.cpu().numpy()

#         for box, track_id in zip(xyxy, ids):

#             x1, y1, x2, y2 = box

#             cx = int((x1 + x2) / 2)
#             cy = int((y1 + y2) / 2)

#             cv2.rectangle(
#                 frame,
#                 (int(x1), int(y1)),
#                 (int(x2), int(y2)),
#                 (0, 255, 0),
#                 2
#             )

#             if cy > LINE_Y and track_id not in counted_ids:

#                 counted_ids.add(track_id)
#                 vehicle_count += 1

#     cv2.putText(
#         frame,
#         f"Vehicle Count: {vehicle_count}",
#         (30, 50),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0, 255, 0),
#         2
#     )

#     cv2.imshow("Vehicle Counting", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()