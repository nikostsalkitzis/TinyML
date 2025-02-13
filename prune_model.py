import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers, models

# Load the pre-trained model
model = tf.keras.models.load_model("model.h5")

# Define pruning parameters
pruning_params = {
    "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.2, final_sparsity=0.8, begin_step=2000, end_step=10000
    )
}

# Function to apply pruning selectively
def apply_pruning_to_layers(model):
    pruned_layers = []
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense): 
            pruned_layers.append(tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params))
        else:
            pruned_layers.append(layer)  # Keep other layers unchanged

    return models.Sequential(pruned_layers)

# Apply pruning to specific layers
pruned_model = apply_pruning_to_layers(model)

# Recompile the model
pruned_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Fine-tune the pruned model
pruned_model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
)

# Strip pruning wrappers to reduce the final model size
pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

# Save the pruned model
pruned_model.save("model_pruned.h5")

print("Pruned model saved as model_pruned.h5")

