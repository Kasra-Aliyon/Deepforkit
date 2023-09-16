# Deepforkit
 Day-ahead European Electricity Price Forecasting KIT

 Instructions for users: 

 1- Ensure you have all the required dependencies installed on your local device.
 2- Copy and paste the whole repository on your local device.
 3- Start using the data_download_kit (.ipynb) to download the data for your specific zone. Remember to use your own API key that could be requested from the Entso-e transparency platform.
 4- For large-scale predictions of historical prices, use prediction_kit (.ipynb).

Instructions for developers: 

1- Deepforkit is developed based on functional programming (instead of object-oriented programming) to allow for easier adjustments.
2- The data.py file is devoted to data-related modules like data downloading.
3- The models.py file is devoted to the model architecture suitable for hyperparameter optimization.
4- The hyperparameters.py file is devoted to hyperparameter optimization and feature selection tasks. 
5- The prediction.py file is devoted to prediction modules. 

Note: All functions have doc strings, and by following the .py files based on the order above, you can understand the underlying logic of Deepforkit.
