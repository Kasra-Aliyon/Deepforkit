import numpy as np
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from data import random_seed


def create_dnn_model(X_train_hist, X_train_fut, y_train, n_hidden, n_units,
                     dropout_rates, activations, batch_normalization, l1s):
    """
    Creates a deep neural network (DNN) model for time series forecasting.

    Args:
        X_train_hist (numpy.ndarray): The historical input features for training.
        X_train_fut (numpy.ndarray): The future input features for training.
        y_train (numpy.ndarray): The target values for training.
        n_hidden (int): The number of hidden layers in the model.
        n_units (list of int): A list containing the number of units for each hidden layer.
        dropout_rates (list of float): A list containing the dropout rate for each hidden layer.
        activations (list of str): A list containing the activation function for each hidden layer.
        batch_normalization (bool): A boolean indicating whether to include batch normalization in the model.
        l1s (list of float): A list containing the L1 regularization coefficient for each hidden layer.

    Returns:
        tensorflow.keras.models.Model: The compiled DNN model.
    """

    random_seed()

    # Create history input
    h_inputs = Input(shape=(X_train_hist.shape[1], X_train_hist.shape[2]), name='inputs_hist')
    h = Flatten(name='flatten_layer_hist')(h_inputs)

    # Create future input
    f_inputs = Input(shape=(X_train_fut.shape[1], X_train_fut.shape[2]), name='inputs_fut')
    f = Flatten(name='flatten_layer_fut')(f_inputs)

    # Concatenate history and future inputs
    x = concatenate([h, f], axis=-1, name='concatenated')

    # Add hidden layers
    for i in range(n_hidden):
        x = Dense(n_units[i], activation=activations[i],
                  kernel_regularizer=regularizers.L1(l1=l1s[i]),
                  kernel_initializer=initializers.GlorotNormal(seed=211))(x)

        if batch_normalization:
            x = BatchNormalization()(x)

        x = Dropout(dropout_rates[i])(x)

    # Create output layer
    output = Dense(units=y_train.shape[1], name='output')(x)

    # Build and return the model
    model = Model(inputs=[h_inputs, f_inputs], outputs=output)
    return model
