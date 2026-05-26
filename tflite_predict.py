import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(
    model_path="C:\\Users\\india\\Desktop\\Vehicle Detection\\best (1)_saved_model\\best (1)_float16.tflite"
)

interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read image
img = cv2.imread("C:\\Users\\india\\Desktop\\Vehicle Detection\\free-video-854671.jpg")

# Resize image
input_size = 640

img_resized = cv2.resize(img, (input_size, input_size))

# Convert BGR to RGB
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

# Normalize
input_data = img_rgb.astype(np.float32) / 255.0

# Add batch dimension
input_data = np.expand_dims(input_data, axis=0)

# Set model input
interpreter.set_tensor(
    input_details[0]['index'],
    input_data
)

# Run inference
interpreter.invoke()

# Get output
output_data = interpreter.get_tensor(
    output_details[0]['index']
)

print("Output shape:", output_data.shape)

print(output_data)