import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Select a few samples (e.g., 5 images) from the dataset
num_samples = 5
samples = x_test[:num_samples]
labels = y_test[:num_samples].flatten()  # Corresponding labels for reference

# Normalize the samples if required by the model
samples = samples.astype(np.float32) / 255.0

# Flatten the samples into 1D arrays (if required by your model)
samples_flattened = samples.reshape(num_samples, -1)

# Convert each sample to a C array
c_arrays = []
for idx, sample in enumerate(samples_flattened):
    c_array = ", ".join(map(lambda x: f"{x:.6f}", sample))
    c_arrays.append(f"float sample_{idx}[] = {{ {c_array} }};")

# Generate C header file content
header_content = """
#ifndef CIFAR10_SAMPLES_H
#define CIFAR10_SAMPLES_H

// CIFAR-10 sample data in C array format
""" + "\n\n".join(c_arrays) + f"""

// Total number of samples
#define NUM_SAMPLES {num_samples}

// Labels for the samples (for reference)
int sample_labels[NUM_SAMPLES] = {{ {', '.join(map(str, labels))} }};

#endif // CIFAR10_SAMPLES_H
"""

# Write the C arrays to a header file
with open("cifar10_samples.h", "w") as f:
    f.write(header_content)

print("C arrays for CIFAR-10 samples have been saved to cifar10_samples.h")

