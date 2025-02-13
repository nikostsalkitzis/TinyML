import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("model.h5")

# Convert to TensorFlow Lite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset function (needed for full integer quantization)
def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 32, 32, 3).astype(np.float32)  # Sample images
        yield [data]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Ensure input and output are also int8
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Convert and save the quantized model
quantized_model = converter.convert()
with open("model_quantized.tflite", "wb") as f:
    f.write(quantized_model)

print("Quantized model saved as model_quantized.tflite")

