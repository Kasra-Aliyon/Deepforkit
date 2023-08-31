import pandas as pd
import numpy as np
import tensorflow as tf
import pytz
from sklearn.metrics import mean_absolute_error

from models import create_dnn_model
from data import create_ml_dataset, sMAPE, random_seed


def extract_hp_params(hp_params):
    """
    Extract hyperparameters from the dictionary.

    Args:
        hp_params (dict): A dictionary of hyperparameters.

    Returns:
        dict: A dictionary containing extracted hyperparameters.
    """
    extracted_params = {}
    extracted_params['n_hidden'] = hp_params['n_hidden']
    extracted_params['batch_size'] = hp_params['batch_size']
    extracted_params['learning_rate'] = hp_params['learning_rate']
    extracted_params['batch_normalization'] = hp_params['batch_normalization']
    extracted_params['dropout_rates'] = [hp_params[i] for i in [x for x in hp_params.keys() if 'dropout_rate' in str(x)]]
    extracted_params['activations'] = [hp_params[i] for i in [x for x in hp_params.keys() if 'activation' in str(x)]]
    extracted_params['l1s'] = [hp_params[i] for i in [x for x in hp_params.keys() if 'l1' in str(x)]]
    extracted_params['n_units'] = [hp_params[i] for i in [x for x in hp_params.keys() if 'n_unit' in str(x)]]
    extracted_params['days'] = get_days(hp_params)



    return extracted_params

def get_days(hp_params):
    """
    Get the list of days based on the specified hyperparameters.

    Args:
        hp_params (dict): A dictionary of hyperparameters.

    Returns:
        List: The list of days that is used by the model.
    """
    
    days = [1,7]
    if hp_params['use_hist_2']:
        days.append(2)
    if hp_params['use_hist_3']:
        days.append(3)
    days.sort() 

    return days

def get_columns_from_hp_params(hp_params):
    """
    Get columns based on the specified hyperparameters.

    Args:
        hp_params (dict): A dictionary of hyperparameters.

    Returns:
        tuple: Two lists containing historical features and future features.
    """
    hist_features = ['Price_DA']
    if 'use_Load_AC' in hp_params.keys() and hp_params['use_Load_AC']:
        hist_features.append('Load_AC')
    if 'use_Gen_SC' in hp_params.keys() and hp_params['use_Gen_SC']:
        hist_features.append('Gen_SC')

    if 'use_Sol_DA' in hp_params.keys() and hp_params['use_Sol_DA']:
        hist_features.append('Sol_DA')
    if 'use_Won_DA' in hp_params.keys() and hp_params['use_Won_DA']:
        hist_features.append('Won_DA')
    if 'use_Woff_DA' in hp_params.keys() and hp_params['use_Woff_DA']:
        hist_features.append('Woff_DA')

    fut_features = ['Load_DA']
    if 'use_hour' in hp_params.keys() and hp_params['use_hour']:
        fut_features.extend(['hour_sin', 'hour_cos'])
    if 'use_dow' in hp_params.keys() and hp_params['use_dow']:
        fut_features.extend(['dow_sin', 'dow_cos'])
    if 'use_woy' in hp_params.keys() and hp_params['use_woy']:
        fut_features.extend(['woy_sin', 'woy_cos'])
    if 'use_weekend' in hp_params.keys() and hp_params['use_isweekend']:
        fut_features.append('isweekend')

    return hist_features, fut_features


def prepare_train_test_datasets(df, date, calibration_years, hist_features, fut_features, extracted_hp_params, n_in=24, n_out=24, stride=24, t_column='Price_DA', scale='Standard'):
    """
    Prepare training and testing datasets.

    Args:
        df (DataFrame): The main dataframe.
        date (str): The date for which to make the prediction.
        calibration_years (int): The number of years for calibration.
        hist_features (list): A list of historical feature names.
        fut_features (list): A list of future feature names.
        days (list): A list of days to be considered for creating the dataset.
        n_in (int, optional): The number of input time steps. Defaults to 24.
        n_out (int, optional): The number of output time steps. Defaults to 24.
        stride (int, optional): The stride for creating samples. Defaults to 24.
        t_column (str, optional): The target column name. Defaults to 'Price_DA'.
        scale (str, optional): The scaling method. Defaults to 'Standard'.  

    Returns:
        tuple: Prepared training and testing datasets.
    """
    days = extracted_hp_params['days']

    next_day = pd.to_datetime(date)
    df_train = df[next_day - pd.Timedelta(hours=1) - pd.Timedelta(weeks=52 * calibration_years): next_day - pd.Timedelta(hours=1)]
    df_test = df.loc[next_day - pd.Timedelta(days=days[-1]): next_day + pd.Timedelta(hours=23)]

    df_valid = pd.DataFrame()
    X_train_hist, X_train_fut, y_train, _, _, _, X_test_hist, X_test_fut, y_test = create_ml_dataset(df_train, df_valid, df_test, hist_features, days, n_in, n_out, stride, t_column, fut_features, scale)

    return X_train_hist, X_train_fut, y_train, X_test_hist, X_test_fut, y_test

def train_and_predict_next_day(X_train_hist, X_train_fut, y_train, X_test_hist, X_test_fut, extracted_hp_params):
    """
    Train the model and make predictions for the next day.
    Args:
    X_train_hist (ndarray): Training data for historical features.
    X_train_fut (ndarray): Training data for future features.
    y_train (ndarray): Training data labels.
    X_test_hist (ndarray): Test data for historical features.
    X_test_fut (ndarray): Test data for future features.
    extracted_hp_params (dict): A dictionary of extracted hyperparameters.

    Returns:
        ndarray: Predicted prices for the next day.

        n_hidden, n_units,
                     dropout_rates, activations, batch_normalization, l1s
    """
    random_seed()

    n_hidden = extracted_hp_params['n_hidden']
    n_units = extracted_hp_params['n_units']
    dropout_rates = extracted_hp_params['dropout_rates']
    activations = extracted_hp_params['activations']
    batch_normalization = extracted_hp_params['batch_normalization']
    l1s = extracted_hp_params['l1s']
    learning_rate = extracted_hp_params['learning_rate']
    batch_size = extracted_hp_params['batch_size']

    model = create_dnn_model(X_train_hist, X_train_fut, y_train, n_hidden, n_units,
                    dropout_rates, activations, batch_normalization, l1s)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mae', metrics=[tf.keras.metrics.MAE])
    model.fit([X_train_hist, X_train_fut], y_train, epochs=1000, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)], verbose=0, batch_size=batch_size, shuffle=True)

    return model.predict([X_test_hist, X_test_fut], verbose=0)

def recalibrate_predict_over_horizon(df, horizon, hp_params, calibration_years=10):
    """
    Recalibrates the deep learning model daily over the specified horizon, 
    making predictions for each day using the given hyperparameters and 
    calibration years. Then, does the same for the next day, and so on.

    Args:
        df (pd.DataFrame): The input dataframe containing historical data.
        horizon (pd.DataFrame): A time series indicating the prediction horizon.
        hp_params (dict): A dictionary containing the best hyperparameters from optimization.
        calibration_years (int, optional): The number of years to use for model calibration. Defaults to 10.

    Returns:
        tuple: A tuple containing two elements:
            1. pd.DataFrame: The predicted prices over the horizon.
            2. pd.DataFrame: A DataFrame containing performance metrics (MAE, sMAPE, rMAE) for each day (24 Hours).
    """

    random_seed()

    predictions = []
    day_metrics = dict()
    extracted_hp_params = extract_hp_params(hp_params)
    hist_features, fut_features = get_columns_from_hp_params(hp_params)

    for day in horizon.resample(rule='D').last().index:
        X_train_hist, X_train_fut, y_train, X_test_hist, X_test_fut, y_test = prepare_train_test_datasets(df, day, calibration_years, hist_features, fut_features, extracted_hp_params)
        next_day_predicted_prices = np.squeeze(train_and_predict_next_day(X_train_hist, X_train_fut, y_train, X_test_hist, X_test_fut, extracted_hp_params).reshape(-1, 1))
        predictions.append(next_day_predicted_prices)

        next_day_real_prices = np.array(df.loc[day: day + pd.Timedelta(hours=23)]['Price_DA'])
        MAE = mean_absolute_error(next_day_real_prices, next_day_predicted_prices)
        sMAPE_val = sMAPE(next_day_real_prices, next_day_predicted_prices)
        _, naive_MAE, _ = naive_predict_over_horizon(df, horizon.loc[day: day + pd.Timedelta(hours=23)])
        rMAE = MAE / naive_MAE

        day_metrics[day.date()] = {'MAE': MAE, 'sMAPE': sMAPE_val, 'rMAE': rMAE}
        day_metrics_DataFrame = pd.DataFrame.from_dict(day_metrics, orient='index', columns=['MAE', 'sMAPE', 'rMAE'])
        overall_MAE = day_metrics_DataFrame['MAE'].mean()
        overall_sMAPE = day_metrics_DataFrame['sMAPE'].mean()
        overall_rMAE = day_metrics_DataFrame['rMAE'].mean()

        print('for {}, MAE is:{:0.2f} & sMAPE is:{:0.2f}% & rMAE is:{:0.2f} ||| daily mean of MAE & sMAPE & rMAE till now are :{:0.2f} & {:0.2f}% & {:0.2f}'.format(day.date(), MAE, sMAPE_val, rMAE, overall_MAE, overall_sMAPE, overall_rMAE))

    predicted_prices = np.vstack(predictions)
    predicted_prices_dataframe = pd.DataFrame(predicted_prices.flatten(), index=horizon.index, columns=['Price_DA'])

    day_metrics_DataFrame.set_index(pd.to_datetime(day_metrics_DataFrame.index), inplace=True)


    return predicted_prices_dataframe, day_metrics_DataFrame


def predict_next_day(df, hp_params, date, calibration_years=10):
    """
    Predicts the next day's prices using a deep learning model trained with 
    the given hyperparameters and calibration years.

    Args:
        df (pd.DataFrame): The input dataframe containing historical data.
        hp_params (dict): A dictionary containing the best hyperparameters from optimization.
        date (str): A string representing the date for which to predict the next day's prices.
        calibration_years (int, optional): The number of years to use for model calibration. Defaults to 10.

    Returns:
        np.ndarray: The predicted prices for the next day as a numpy array.
    """
    random_seed()

    extracted_hp_params = extract_hp_params(hp_params)
    hist_features, fut_features = get_columns_from_hp_params(hp_params)
    X_train_hist, X_train_fut, y_train, X_test_hist, X_test_fut, y_test = prepare_train_test_datasets(df, pd.to_datetime(date), calibration_years, hist_features, fut_features, extracted_hp_params)

    next_day_predicted_prices = np.squeeze(train_and_predict_next_day(X_train_hist, X_train_fut, y_train, X_test_hist, X_test_fut, extracted_hp_params).reshape(-1, 1))
    next_day_predicted_prices_df = pd.DataFrame(next_day_predicted_prices, index=pd.date_range(start=date, periods=24, freq='H'), columns=['Price_DA'])

    return next_day_predicted_prices_df


def predict_over_horizon(df, horizon, hp_params, calibration_years=10):
    """
    Predicts prices over a given horizon using a deep learning model trained with the given hyperparameters and calibration years.

    Args:
        df (pd.DataFrame): The input dataframe containing historical data.
        horizon (pd.DatetimeIndex): The datetime index representing the target prediction horizon.
        hp_params (dict): A dictionary containing the best hyperparameters from optimization.
        calibration_years (int, optional): The number of years to use for model calibration. Defaults to 10.

    Returns:
        pd.DataFrame: The predicted prices over the given horizon.
        pd.DataFrame: The evaluation metrics (MAE, sMAPE, rMAE) for the predictions on a daily basis.
    """
    random_seed()

    extracted_hp_params = extract_hp_params(hp_params)
    hist_features, fut_features = get_columns_from_hp_params(hp_params)
    days = extracted_hp_params['days']

    df_test = df.loc[horizon.index[0] - pd.Timedelta(days=days[-1]):horizon.index[-1]]
    df_train = df.loc[df_test.index[0] + pd.Timedelta(days=days[-1]) - pd.Timedelta(weeks=52*calibration_years) :df_test.index[0] - pd.Timedelta(hours=1) + pd.Timedelta(days=days[-1]) ]
    df_valid = pd.DataFrame()
    X_train_hist, X_train_fut, y_train, X_valid_hist, X_valid_fut, y_valid, X_test_hist, X_test_fut, y_test = create_ml_dataset(
        df_train, df_valid, df_test, hf_columns=hist_features, days=days,n_in=24, n_out=24, stride=24, t_column='Price_DA', fut_columns=fut_features, scale='Standard')

    n_hidden = extracted_hp_params['n_hidden']
    n_units = extracted_hp_params['n_units']
    dropout_rates = extracted_hp_params['dropout_rates']
    activations = extracted_hp_params['activations']
    batch_normalization = extracted_hp_params['batch_normalization']
    l1s = extracted_hp_params['l1s']
    learning_rate = extracted_hp_params['learning_rate']
    batch_size = extracted_hp_params['batch_size']

    model = create_dnn_model(X_train_hist, X_train_fut, y_train,n_hidden, n_units,
                     dropout_rates, activations, batch_normalization, l1s)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mae', metrics=[tf.keras.metrics.MAE], optimizer=optimizer)

    model.fit(x=[X_train_hist, X_train_fut], y=y_train, epochs=1000, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)], verbose=0, batch_size=batch_size, shuffle=True)
    
    test_predictions = model.predict([X_test_hist, X_test_fut], verbose=0)
    test_predictions_df = pd.DataFrame(test_predictions.flatten(), index=horizon.index, columns=['Price_DA'])

    day_metrics = compute_day_metrics(horizon, test_predictions_df, df)

    return test_predictions_df, day_metrics


def predict_over_horizon_hyperparameters(df_train, df_valid, df_test, hp_params):
    """
    Predicts prices over a given horizon using a deep learning model trained with the given hyperparameters. Here, the training, validation and test sets are the same as hyperparameter optimization.

    Args:
        df (pd.DataFrame): The input dataframe containing historical data.
        horizon (pd.DatetimeIndex): The datetime index representing the target prediction horizon.
        hp_params (dict): A dictionary containing the best hyperparameters from optimization.
        calibration_years (int, optional): The number of years to use for model calibration. Defaults to 10.

    Returns:
        pd.DataFrame: The predicted prices over the given horizon.
        pd.DataFrame: The evaluation metrics (MAE, sMAPE, rMAE) for the predictions on a daily basis.
    """
    random_seed()

    df = pd.concat([df_train, df_valid, df_test])

    extracted_hp_params = extract_hp_params(hp_params)
    hist_features, fut_features = get_columns_from_hp_params(hp_params)
    days = extracted_hp_params['days']

    test_data = df.loc[df_test.index[0] - pd.Timedelta(days=days[-1]):df_test.index[-1]]
    valid_data = df.loc[df_valid.index[0] : test_data.index[0] - pd.Timedelta(hours=1) + pd.Timedelta(days=days[-1])]

    X_train_hist, X_train_fut, y_train, X_valid_hist, X_valid_fut, y_valid, X_test_hist, X_test_fut, y_test = create_ml_dataset(
        df_train, valid_data, test_data, hf_columns=hist_features, days=days,n_in=24, n_out=24, stride=24, t_column='Price_DA', fut_columns=fut_features, scale='Standard')

    n_hidden = extracted_hp_params['n_hidden']
    n_units = extracted_hp_params['n_units']
    dropout_rates = extracted_hp_params['dropout_rates']
    activations = extracted_hp_params['activations']
    batch_normalization = extracted_hp_params['batch_normalization']
    l1s = extracted_hp_params['l1s']
    learning_rate = extracted_hp_params['learning_rate']
    batch_size = extracted_hp_params['batch_size']

    model = create_dnn_model(X_train_hist, X_train_fut, y_train,n_hidden, n_units,
                     dropout_rates, activations, batch_normalization, l1s)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mae', metrics=[tf.keras.metrics.MAE], optimizer=optimizer)

    model.fit(x=[X_train_hist, X_train_fut], y=y_train, epochs=1000, validation_data=([X_valid_hist, X_valid_fut],y_valid), callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)], verbose=0, batch_size=batch_size, shuffle=True)
    
    test_predictions = model.predict([X_test_hist, X_test_fut], verbose=0)
    test_predictions_df = pd.DataFrame(test_predictions.flatten(), index=df_test.index, columns=['Price_DA'])

    day_metrics = compute_day_metrics(df_test, test_predictions_df, df)

    return test_predictions_df, day_metrics






def compute_day_metrics(horizon, test_predictions_df, df):
    """
    Computes MAE, sMAPE, and rMAE for each day in the given horizon.

    Args:
    horizon(pd.DataFrame) : A DataFrame containing the actual prices for the given horizon.
    test_predictions (pd.DataFrame) : A DataFrame containing the predicted prices for the given horizon.
    df (pd.DataFrame) :  The DataFrame containing historical data used for the naive predictions.
    Returns:
    day_metrics_df (pd.DataFrame) : A DataFrame containing the calculated MAE, sMAPE, and rMAE for each day in the horizon.
    """
    

    day_metrics = dict()

    for day in horizon.resample(rule='D').last().index:
        MAE = mean_absolute_error(horizon.loc[day:day + pd.Timedelta(hours=23)]['Price_DA'], test_predictions_df.loc[day:day + pd.Timedelta(hours=23)]['Price_DA'])
        sMAPE_value = sMAPE(horizon.loc[day:day + pd.Timedelta(hours=23)]['Price_DA'].values, test_predictions_df.loc[day:day + pd.Timedelta(hours=23)].values)
        _, naive_MAE, _ = naive_predict_over_horizon(df, horizon.loc[day:day + pd.Timedelta(hours=23)])
        rMAE = MAE / naive_MAE

        day_metrics[day.date()] = {'MAE': MAE, 'sMAPE': sMAPE_value, 'rMAE': rMAE}
    
    day_metrics_df = pd.DataFrame.from_dict(day_metrics, orient='index', columns=['MAE', 'sMAPE', 'rMAE'])
    day_metrics_df.index = pd.to_datetime(day_metrics_df.index)

    return day_metrics_df


def naive_predict_over_horizon(df, horizon):
    """
    Generate naive predictions for the given horizon using historical data. It is just the price of the same hour of the previous week. It then compares the predictions with the actual prices.
    and compute the Mean Absolute Error (MAE) and symmetric Mean Absolute Percentage Error (sMAPE).

    Args:
    df(pd.DataFrame) : The DataFrame containing historical data.
    horizon(pd.DataFrame) : A DataFrame containing the actual prices for the given horizon.

    Returns:
    test_predictions(pd.Series) : A Series containing the naive predictions for the given horizon.
    test_MAE(float) : The Mean Absolute Error between the naive predictions and the actual prices.
    test_sMAPE(float) : The symmetric Mean Absolute Percentage Error between the naive predictions and the actual prices.
    """
    test_space_start = horizon.index[0] - pd.Timedelta(weeks=1)
    test_space_end = horizon.index[-1]
    test_space = df.loc[test_space_start:test_space_end]

    test_predictions = test_space['Price_DA'].shift(24 * 7)
    test_predictions = test_predictions.dropna()

    test_MAE = mean_absolute_error(horizon['Price_DA'], test_predictions)
    test_sMAPE = sMAPE(horizon['Price_DA'], test_predictions)

    return test_predictions, test_MAE, test_sMAPE





