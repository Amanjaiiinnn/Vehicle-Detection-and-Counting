# from ultralytics import YOLO

# # Load TFLite model
# model = YOLO("C:\\Users\\india\\Desktop\\Vehicle Detection\\best (1)_saved_model\\best (1)_float16.tflite")

# # Run prediction
# results = model("C:\\Users\\india\\Desktop\\Vehicle Detection\\free-video-854671.jpg")

# # Save output
# results[0].save("output.jpg")

# print("Done")



# from ultralytics import YOLO

# model = YOLO("best_float32.tflite")

# results = model.predict(
#     source="traffic.mp4",
#     save=True
# )





# from ultralytics import YOLO
# import cv2

# # Load TFLite model
# model = YOLO("C:\\Users\\india\\Desktop\\Vehicle Detection\\best (1)_saved_model\\best (1)_float16.tflite")

# # Read image
# img = cv2.imread("C:\\Users\\india\\Desktop\\Vehicle Detection\\free-video-854671.jpg")

# # Run prediction
# results = model(img, conf=0.50)

# # Draw boxes on image
# annotated = results[0].plot()

# # Save output image
# cv2.imwrite("output.jpg", annotated)

# # Show image
# cv2.imshow("Detection", annotated)

# cv2.waitKey(0)
# cv2.destroyAllWindows()



# from ultralytics import YOLO

# # Load TFLite model
# model = YOLO(
#     r"C:\Users\india\Desktop\Vehicle Detection\best (1)_saved_model\best (1)_float16.tflite"
# )

# # Run prediction
# results = model(
#     r"C:\Users\india\Desktop\Vehicle Detection\free-video-854671.jpg",
#     conf=0.50,
#     save=True
# )

# print("Done")




from ultralytics import YOLO

# Load TFLite model
model = YOLO(
    r"C:\Users\india\Desktop\Vehicle Detection\best (1)_saved_model\best (1)_float16.tflite"
)

# Run prediction on video
results = model.predict(
    source=r"C:\Users\india\Desktop\Vehicle Detection\3.mp4",
    conf=0.50,
    save=True
)

print("Video prediction completed")