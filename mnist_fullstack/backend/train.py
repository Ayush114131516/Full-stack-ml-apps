import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# ── 1. Load & preprocess MNIST ──────────────────────────────────────────────
# Keras downloads it automatically (~11 MB) on first run.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale pixel values from [0, 255]  →  [0.0, 1.0]
# Neural networks train much better with small, normalised inputs.
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# Add a channel dimension: (60000, 28, 28) → (60000, 28, 28, 1)
# Conv2D layers expect  (batch, height, width, channels).
# Grayscale = 1 channel; RGB would be 3.
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test,  -1)

print(f"Training samples : {x_train.shape}")   # (60000, 28, 28, 1)
print(f"Test     samples : {x_test.shape}")    # (10000, 28, 28, 1)

# ── 2. Build the CNN ─────────────────────────────────────────────────────────
# Architecture overview:
#
#   Input (28×28×1)
#     └─ Conv2D(32)  → detects low-level features (edges, curves)
#     └─ MaxPool     → halves spatial size, keeps strongest activations
#     └─ Conv2D(64)  → detects higher-level patterns (loops, corners)
#     └─ MaxPool     → halves again
#     └─ Flatten     → turns 2D feature map into a 1D vector
#     └─ Dropout     → randomly zeros 50 % of neurons during training
#                      (prevents the model from just memorising examples)
#     └─ Dense(10)   → one output neuron per digit class (0–9)
#     └─ Softmax     → turns raw scores into probabilities that sum to 1

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dropout(0.5),

        layers.Dense(10, activation="softmax"),   # 10 classes → digits 0-9
    ],
    name="mnist_cnn",
)

model.summary()   # prints layer shapes & parameter counts

# ── 3. Compile ───────────────────────────────────────────────────────────────
# loss          : sparse_categorical_crossentropy
#                 → labels are plain integers (0-9), not one-hot vectors.
# optimizer     : adam — adaptive learning rate, works well out of the box.
# metrics       : accuracy — human-readable, not used to update weights.
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

# ── 4. Train ─────────────────────────────────────────────────────────────────
# validation_split=0.1 holds out 10 % of training data to monitor overfitting.
history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=15,
    validation_split=0.1,
    verbose=1,
)

# ── 5. Evaluate on the unseen test set ───────────────────────────────────────
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅  Test accuracy : {test_acc * 100:.2f}%")   # expect ~99 %

# ── 6. Save ───────────────────────────────────────────────────────────────────
# .h5  is the older HDF5 format — simple single-file, easy to share.
# FastAPI will load this at startup with:  tf.keras.models.load_model("mnist_model.h5")
model.save("mnist_model.h5")
print("-"*10+"Model saved → mnist_model.h5"+"-"*10)