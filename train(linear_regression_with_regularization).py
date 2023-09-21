import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from helper import Helper


# Load data
raw_data_normalized = Helper.load_raster("raw_RGB_image.tif")
true_data_normalized = Helper.load_raster("true_color_RGB_image.tif")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    raw_data_normalized, true_data_normalized, test_size=0.2, random_state=42
)

# Regularization reduces the variance of the model but introduces bias as it shrinks variables towards zero or a constant.

# Create and train the Ridge regression model with regularization parameter alpha

# Adjust the alpha value to control the strength of regularization
alpha = 0.01  # You can adjust this value
model = Ridge(alpha=alpha)
model.fit(X_train, y_train)

# Save the model to disk
filename = "model(linear_regression_with_regularization).pickle"
pickle.dump(model, open(filename, "wb"))
