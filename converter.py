import tensorflow as tf
import tf2onnx

# Path to your Keras .h5 model
keras_model_path = "plant_disease_model.h5"
onnx_model_path = "plant_disease_model.onnx"

# Load the Keras model
model = tf.keras.models.load_model(keras_model_path)

# Define input signature
input_signature = (tf.TensorSpec([None, 128, 128, 3], tf.float32, name="input"),)

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

# Save the ONNX model
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"✅ Conversion complete! ONNX model saved to: {onnx_model_path}")
