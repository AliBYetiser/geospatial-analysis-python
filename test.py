import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from helper import Helper


# Get current working directory
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Load the model from disk
model_filenames = [
    "model(linear_regression_with_regularization).pickle",
    "model(neural_network).pickle",
]
loaded_models = [
    pickle.load(open(os.path.join(__location__, model), "rb"))
    for model in model_filenames
]

# Load test file
test_raw_filename = "raw_RGB_image.tif" # to be changed with test data other than the images we used for training
test_true_filename = "true_color_RGB_image.tif"
test_raw_normalized = Helper.load_raster(test_raw_filename)  # X_test
test_true_normalized = Helper.load_raster(test_true_filename)  # y_test

# Calculate Mean Squared Error as a measure of model performance
y_predictions = [model.predict(test_raw_normalized) for model in loaded_models]
prediction_rgb_images = [y_pred.reshape((693, 804, 3)) for y_pred in y_predictions]
mse = [mean_squared_error(test_true_normalized, y_pred) for y_pred in y_predictions]

# Let's plot MSE errors in a bar chart
mse_dict = {
    "Linear Regression with Regularization": mse[0],
    "Neural Network": mse[1],
}
models = list(mse_dict.keys())
mse = list(mse_dict.values())

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(3, 4)


# Plot the first figure
ax1 = fig.add_subplot(gs[0:2, 0:2])
ax1.imshow(prediction_rgb_images[0], cmap="viridis")
ax1.set_title("Linear regression with Regularization")
ax1.axis("off")

# Plot the second figure
ax2 = fig.add_subplot(gs[0:2, 2:4])
ax2.imshow(prediction_rgb_images[1], cmap="viridis")
ax2.set_title("Neural Network")
ax2.axis("off")

# Plot the MSE bar chart
ax3 = fig.add_subplot(gs[2, 1:-1])
ax3.bar(models, mse, color="maroon", width=0.4)
ax3.set_title("Regression models MSE errors")


# Adjust spacing between subplots
#plt.tight_layout()

# Show the entire plot
plt.show()
