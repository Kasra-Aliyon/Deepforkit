import optuna
import pandas as pd
import numpy as np
import os
import glob
import json

from hyperparameters import optimize_hyperparameters
from data import wrangle_data, split_dataframe_by_years, sMAPE

from prediction import predict_over_horizon, predict_over_horizon_hyperparameters, recalibrate_predict_over_horizon, predict_next_day, naive_predict_over_horizon

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def large_scale_predictor(zone, first_year, last_year):
    """
    Runs a large-scale prediction for a given zone, from a specified first year to a specified last year.

    Parameters:
    zone (str): The zone for which to run the prediction.
    first_year (int): The first year to consider for the prediction.
    last_year (int): The last year to consider for the prediction.

    Creates the required directory structure if it does not exist, loads and wrangles the data, 
    optimizes hyperparameters, and runs the prediction. 
    The results (predictions and metrics) are saved to pickle files in the appropriate directory.
    Finally, all prediction and metric files are read and concatenated into two comprehensive DataFrames,
    which are then saved to csv files.
    """



    directory = zone
    if not os.path.exists(directory):
        os.makedirs(directory)

    df = wrangle_data(source_folder='data', return_dfs=True, store = False, destination_folder=None)[zone]

    for test_year in range(first_year,last_year+1):

        with open('config.json', 'r') as f:
            config = json.load(f)
        
        num_validation_years = config['num_validation_years']
        num_train_years = config['num_train_years']
        n_trials = config['n_trials']
        n_jobs = config['n_jobs']
        calibration_years = config['calibration_years']


        
        df_train, df_valid, df_test = split_dataframe_by_years(dataframe=df, test_year=test_year, num_validation_years=num_validation_years, num_train_years=num_train_years )
        study_name = zone + '_' + str(test_year)
        optimize_hyperparameters(df_train, df_valid, df_test, study_name=study_name, n_trials=n_trials, n_jobs=n_jobs)
        hp_study = optuna.load_study(study_name=study_name, storage='sqlite:///hyperparameter_optimization_trials/' + study_name + '.db')
        hp_params = hp_study.best_params


        preds, metrics = recalibrate_predict_over_horizon(df, df_test, hp_params, calibration_years=calibration_years)
        new_metric = sMAPE(preds.values.flatten(), df_test['Price_DA'].values.flatten())


        preds.to_pickle(directory + '/' + str(test_year)  +'_preds.pkl')
        metrics.to_pickle(directory + '/' + str(test_year)  +'_metrics.pkl')

    files = glob.glob(zone +'/'+"*preds.pkl")
    data_frames = []
    for file in files:
        with open(file, 'rb') as f:
            data_frames.append(pd.read_pickle(f))
    # concatenate all DataFrames in the list into one DataFrame
    Predictions = pd.concat(data_frames)

    files = glob.glob(zone +'/'+"*metrics.pkl")
    data_frames = []
    for file in files:
        with open(file, 'rb') as f:
            data_frames.append(pd.read_pickle(f))
    # concatenate all DataFrames in the list into one DataFrame
    Metrics = pd.concat(data_frames)

    Predictions.to_csv(directory + '/' + 'Predictions.csv')
    Metrics.to_csv(directory + '/' + 'Metrics.csv')



def date_index_setter(df):
    """
    Sets the DataFrame index to the date.

    Parameters:
    df (pandas.DataFrame): The DataFrame for which to set the index.

    Returns:
    df (pandas.DataFrame): The DataFrame with the date as its index.
    """

    df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    index = pd.to_datetime(df['Date'])
    df.set_index(index, inplace=True)
    df.drop(columns=['Date'], inplace=True)
    return df


def large_scale_reader(zone):
    """
    Reads the results of a large-scale prediction for a given zone.

    Parameters:
    zone (str): The zone for which to read the prediction results.

    Reads the prediction and metric files from the appropriate directory, 
    sets the DataFrame index to the date using the date_index_setter function, 
    and returns the predictions and metrics as separate DataFrames.

    Returns:
    Predictions (pandas.DataFrame): The DataFrame containing the predictions.
    Metrics (pandas.DataFrame): The DataFrame containing the metrics.
    """


    directory = zone

    Predictions = pd.read_csv(directory + '/' + 'Predictions.csv', parse_dates=True )
    Predictions = date_index_setter(Predictions)
    Predictions.rename(columns={'Price_DA': 'Preds'}, inplace=True)  

    Metrics = pd.read_csv(directory + '/' + 'Metrics.csv', parse_dates=True )
    Metrics = date_index_setter(Metrics)  

    return Predictions, Metrics