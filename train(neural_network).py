import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from helper import Helper
import matplotlib.pyplot as plt
import tensorflow.keras as keras

# Load data
raw_data_normalized = Helper.load_raster("raw_RGB_image.tif")
true_data_normalized = Helper.load_raster("true_color_RGB_image.tif")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    raw_data_normalized, true_data_normalized, test_size=0.2, random_state=42
)

# Build a neural network model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            64, activation="relu", input_shape=(3,), name="input_layer"
        ),
        tf.keras.layers.Dense(32, activation="relu", name="hidden_layer2"),
        tf.keras.layers.Dense(32, activation="relu", name="hidden_layer3"),
        tf.keras.layers.Dense(
            3, activation="linear", name="output_layer"
        ),  # 3 output nodes for RGB values
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
#history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=32,
    epochs=20,
    callbacks=[early_stopping],
)
import pandas as pd

history_frame = pd.DataFrame(history.history)
history_frame.loc[:,['loss', 'val_loss']].plot()
plt.show()
# Save the model to disk
#filename = "model(neural_network).pickle"
#pickle.dump(model, open(filename, "wb"))
