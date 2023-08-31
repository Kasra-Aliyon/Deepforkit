from models import create_dnn_model
from data import create_ml_dataset, sMAPE, random_seed
from prediction import naive_predict_over_horizon

import tensorflow as tf
import numpy as np
import pandas as pd
import optuna
from optuna.integration import TFKerasPruningCallback
from sklearn.metrics import mean_absolute_error
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import os

def create_trial_params(df_train, trial):
    """
    Creates a dictionary of hyperparameters that exists within a trial.

    Args:
        trial (optuna.trial.Trial): An optuna Trial object.

    Returns:
        dict: A dictionary of hyperparameters for the trial.
    """
    params = {
        'n_hidden': trial.suggest_int('n_hidden', 3, 4),
        'learning_rate': trial.suggest_float('learning_rate', 5e-4, 1e-1, log=True),
        'batch_size': trial.suggest_int('batch_size', 7, 84, 7),
        'batch_normalization': trial.suggest_categorical('batch_normalization', [True, False]),
        'use_hist_2': trial.suggest_categorical('use_hist_2', [True, False]),
        'use_hist_3': trial.suggest_categorical('use_hist_3', [True, False]),
    }

    if 'Load_AC' in df_train.columns:
        params['use_Load_AC'] = trial.suggest_categorical('use_Load_AC', [True, False])
    
    if 'Gen_SC' in df_train.columns:
        params['use_Gen_SC'] = trial.suggest_categorical('use_Gen_SC', [True, False])

    if 'Sol_DA' in df_train.columns:
        if not (df_train['Sol_DA'] == 0).all():
            params['use_Sol_DA'] = trial.suggest_categorical('use_Sol_DA', [True, False])

    if 'Won_DA' in df_train.columns:
        if not (df_train['Won_DA'] == 0).all():
            params['use_Won_DA'] = trial.suggest_categorical('use_Won_DA', [True, False])

    if 'Woff_DA' in df_train.columns:
        if not (df_train['Woff_DA'] == 0).all():
            params['use_Woff_DA'] = trial.suggest_categorical('use_Woff_DA', [True, False])

    if ('hour_sin' in df_train.columns) and ('hour_cos' in df_train.columns):
        params['use_hour'] = trial.suggest_categorical('use_hour', [True, False])

    if ('dow_sin' in df_train.columns) and ('dow_cos' in df_train.columns):
        params['use_dow'] = trial.suggest_categorical('use_dow', [True, False])

    if ('woy_sin' in df_train.columns) and ('woy_cos' in df_train.columns):
        params['use_woy'] = trial.suggest_categorical('use_woy', [True, False])
    
    if 'isweekend' in df_train.columns:
        params['use_isweekend'] = trial.suggest_categorical('use_isweekend', [True, False])

    return params


def create_trial_layer_params(trial, n_hidden):

    """
    Creates a dictionary of layer hyperparameters that exists within a trial.

    Args:
        trial (optuna.trial.Trial): An optuna Trial object.
        n_hidden (int): The number of hidden layers in the model.

    Returns:
        dict: A dictionary of layer hyperparameters for the trial.
    """
    layer_params = {
        'dropout_rates': [trial.suggest_float('dropout_rate_Layer_' + str(i + 1), 0.0, 0.4) for i in range(n_hidden)],
        'activations': [trial.suggest_categorical('activation_Layer_' + str(i + 1), ['relu', 'tanh', 'sigmoid', 'linear']) for i in range(n_hidden)],
        'l1s': [trial.suggest_float('l1_Layer_' + str(i + 1), 1e-5, 1e-1, log=True) for i in range(n_hidden)],
        'n_units': [trial.suggest_int('n_units_Layer_' + str(i + 1), 50, 300, 5) for i in range(n_hidden)],
    }

    return layer_params

def get_hf_columns(df_train, trial_params):
    """
    Returns a list of history columns to use in the model based on the trial_params extracted from trial.

    Args:
        df_train (pandas.DataFrame): The training dataset.
        trial_params (dict): A dictionary of hyperparameters for the trial.

    Returns:
        list: A list of column names to use as history and future features in the model.
    """

    hf_columns = ['Price_DA']

    if 'Load_AC' in df_train.columns:
        if trial_params['use_Load_AC']:
            hf_columns.append('Load_AC')
    if 'Gen_SC' in df_train.columns:
        if trial_params['use_Gen_SC']:
            hf_columns.append('Gen_SC')

    if 'Sol_DA' in df_train.columns:
        if not (df_train['Sol_DA'] == 0).all():
            if trial_params.get('use_Sol_DA'):
                hf_columns.append('Sol_DA')

    if 'Won_DA' in df_train.columns:            
        if not (df_train['Won_DA'] == 0).all():
            if trial_params.get('use_Won_DA'):
                hf_columns.append('Won_DA')

    if 'Woff_DA' in df_train.columns:
        if not (df_train['Woff_DA'] == 0).all():
            if trial_params.get('use_Woff_DA'):
                hf_columns.append('Woff_DA')

    return hf_columns


def get_fut_columns(df_train, trial_params):
    """
    Returns a list of future columns to use in the model based on the trial_params extracted from trial.

    Args:
        trial_params (dict): A dictionary of hyperparameters for the trial.

    Returns:
        list: A list of column names to use as future features in the model.
    """
    fut_columns = ['Load_DA']


    if ('hour_sin' in df_train.columns) and ('hour_cos' in df_train.columns):
        if trial_params['use_hour']:
            fut_columns.extend(['hour_sin', 'hour_cos'])

    if ('dow_sin' in df_train.columns) and ('dow_cos' in df_train.columns):       
        if trial_params['use_dow']:
            fut_columns.extend(['dow_sin', 'dow_cos'])

    if ('woy_sin' in df_train.columns) and ('woy_cos' in df_train.columns):
        if trial_params['use_woy']:
            fut_columns.extend(['woy_sin', 'woy_cos'])

    if 'isweekend' in df_train.columns:
        if trial_params['use_isweekend']:
            fut_columns.append('isweekend')

    return fut_columns


def evaluate_performance(y_true, y_pred, df, time_frame):
    """
    Calculates evaluation metrics for a model.

    Args:
        y_true (numpy.ndarray): The true values of the target variable.
        y_pred (numpy.ndarray): The predicted values of the target variable.
        df (pandas.DataFrame): The dataset used for evaluation.
        time_frame (pandas.DataFrame): The time frame for which to evaluate the model.

    Returns:
        tuple: A tuple of evaluation metrics: mean absolute error (MAE), symmetric mean absolute percentage error (sMAPE), and relative MAE (rMAE).
    """

    mae = mean_absolute_error(y_true, y_pred)
    smape = sMAPE(y_true, y_pred)

    _, naive_mae, _ = naive_predict_over_horizon(df, time_frame)
    rmae = mae / naive_mae

    return mae, smape, rmae

def train_and_evaluate_model(trial, df_train, df_valid, df_test, hf_columns, fut_columns):
    """
    Trains and evaluates a deep neural network model.

    Args:
        trial (optuna.trial.Trial): An optuna Trial object.
        df_train (pandas.DataFrame): The training dataset.
        df_valid (pandas.DataFrame): The validation dataset.
        df_test (pandas.DataFrame): The test dataset.
        hf_columns (list): A list of history and future columns to use in the model.
        fut_columns (list): A list of future columns to use in the model.

    Returns:
        float: The mean absolute error (MAE) of the model on the validation set.
    """
    random_seed()

    df=pd.concat([df_train, df_valid, df_test])
    trial_params = create_trial_params(df_train, trial)
    layer_params = create_trial_layer_params(trial, trial_params['n_hidden'])

    days = [1, 7]
    if trial_params['use_hist_2']:
        days.append(2)
    if trial_params['use_hist_3']:
        days.append(3)
    days.sort()

    X_train_hist, X_train_fut, y_train, X_valid_hist, X_valid_fut, y_valid, X_test_hist, X_test_fut, y_test = create_ml_dataset(
        df_train, df_valid, df_test, hf_columns, days, n_in=24, n_out=24, stride=24, t_column='Price_DA',
        fut_columns=fut_columns, scale='Standard')

    model = create_dnn_model(X_train_hist, X_train_fut, y_train, trial_params['n_hidden'], layer_params['n_units'],
                             layer_params['dropout_rates'], layer_params['activations'],
                             trial_params['batch_normalization'], layer_params['l1s'])

    optimizer = tf.keras.optimizers.Adam(learning_rate=trial_params['learning_rate'])
    model.compile(loss='mae', metrics=tf.keras.metrics.MAE, optimizer=optimizer)

    callbacks = [
        TFKerasPruningCallback(trial, 'val_mean_absolute_error'),
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    ]

    model.fit(
        x=[X_train_hist, X_train_fut], y=y_train,
        validation_data=([X_valid_hist, X_valid_fut], y_valid),
        epochs=1000, verbose=0, callbacks=callbacks,
        batch_size=trial_params['batch_size'], shuffle=True)

    valid_pred = model.predict([X_valid_hist, X_valid_fut], verbose=0)
    valid_mae, valid_smape, valid_rmae = evaluate_performance(y_valid, valid_pred, df, df_valid)

    test_pred = model.predict([X_test_hist, X_test_fut], verbose=0)
    test_mae, test_smape, test_rmae = evaluate_performance(y_test, test_pred, df, df_test)

    trial.set_user_attr('val_MAE', valid_mae)
    trial.set_user_attr('val_sMAPE', valid_smape)
    trial.set_user_attr('val_rMAE', valid_rmae)

    trial.set_user_attr('test_MAE', test_mae)
    trial.set_user_attr('test_sMAPE', test_smape)
    trial.set_user_attr('test_rMAE', test_rmae)

    print(f"MAE for Validation Set is: {valid_mae:.2f} | sMAPE for Validation Set is: {valid_smape:.2f}% | rMAE for Validation Set is: {valid_rmae:.2f}")
    print(f"MAE for Test Set is: {test_mae:.2f} | sMAPE for Test Set is: {test_smape:.2f}% | rMAE for Test Set is: {test_rmae:.2f}")

    return valid_mae


def optimize_hyperparameters(df_train, df_valid, df_test, study_name, n_trials, n_jobs):
    """
    Runs hyperparameter optimization for a deep neural network model, and then saves the results to a file with the study_name in the hyperparameter_optimization_trials folder in the current directory.

    Args:
        df_train (pandas.DataFrame): The training dataset.
        df_valid (pandas.DataFrame): The validation dataset.
        df_test (pandas.DataFrame): The test dataset.
        study_name (str): The name of the study for the hyperparameter optimization.
        n_trials (int): The number of trials to run for the optimization.
        n_jobs (int): The number of parallel jobs to use for the optimization.

    Returns:
        optuna.study.Study: The optuna Study object containing the results of the hyperparameter optimization.
    """
    random_seed()

    sampler = TPESampler(seed=211)
    pruner = HyperbandPruner()

    folder_name = 'hyperparameter_optimization_trials'
    current_directory = os.path.dirname(__file__)
    directory_name = os.path.join(current_directory, folder_name)

    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"Directory '{directory_name}' created successfully to store the results of the hyperparameter optimization!")
    else:
        print(f"Directory '{directory_name}' already exists, using it to store the results of the hyperparameter optimization!")

    database_path = os.path.join(directory_name, f'{study_name}.db')
    study = optuna.create_study(sampler=sampler, pruner=pruner, study_name=study_name,
                                 storage=f'sqlite:///{database_path}', load_if_exists=True)

    def objective(trial):
        trial_params = create_trial_params(df_train, trial)
        hf_columns = get_hf_columns(df_train, trial_params)
        fut_columns = get_fut_columns(df_train, trial_params)

        return train_and_evaluate_model(trial, df_train, df_valid, df_test, hf_columns, fut_columns)

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)









