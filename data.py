import pandas as pd
import numpy as np
from entsoe import EntsoePandasClient
import os
import glob
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
import keras.backend as K
import random

# Downloading functions

def create_or_use_directory(folder_name):
    """
    Create a new directory with the given folder name or use an existing one.

    Args:
    folder_name (str): The name of the folder to create or use.

    Returns:
    str: The full path of the created or used directory.
    """
    current_directory = os.path.dirname(__file__)
    directory_name = os.path.join(current_directory, folder_name)

    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"Directory '{directory_name}' created successfully!")
    else:
        print(f"Directory '{directory_name}' already exists, using it now!")

    return directory_name


def download_entsoe_data(client, country, start, end):
    """
    Download ENTSO-E data for a specific country and time span.

    Args:
    client (EntsoePandasClient): An instance of the EntsoePandasClient class to use for querying the data.
    country (str): The country code for the desired data.
    start (pd.Timestamp): The start time of the desired data span.
    end (pd.Timestamp): The end time of the desired data span.

    Returns:
    pd.DataFrame: A Pandas DataFrame containing the downloaded data.
    """
    span = slice(start + pd.Timedelta(days=1), end - pd.Timedelta(days=1, hours=1))
    df = pd.DataFrame()
    df['Price_DA'] = client.query_day_ahead_prices(country, start=start, end=end).loc[span]
    df['Load_DA'] = client.query_load_forecast(country, start=start, end=end).loc[span]
    df['Load_AC'] = client.query_load(country, start=start, end=end).loc[span]
    # ...
    if country in ['NO_1', 'NO_2', 'NO_3', 'NO_4', 'NO_5']:
        query = client.query_generation_forecast(country, start=start,end=end).loc[span]
        df['Gen_SC'] = pd.concat([query[query[0].isna()]['Actual Aggregated'],query[query[0].notna()][0]], axis=0)
    elif country in ['ES', 'PT', 'DE_AT_LU']:
        df['Gen_SC'] = client.query_generation_forecast(country, start=start,end=end).loc[span]['Actual Aggregated'].resample('H').mean()
    else:
        df['Gen_SC'] = client.query_generation_forecast(country, start=start,end=end).loc[span]

    df['Sol_DA'] = client.query_wind_and_solar_forecast(country, start=start,end=end, psr_type='B16').loc[span]
    df['Won_DA'] = client.query_wind_and_solar_forecast(country, start=start,end=end, psr_type='B19').loc[span]

    if country in ['DK_1', 'DK_2', 'NL']:
        df['Woff_DA'] = client.query_wind_and_solar_forecast(country, start=start,end=end, psr_type='B18').loc[span]
    # ...
    df.index = df.index.tz_convert('CET')
    df.fillna(0, inplace=True)

    ### 25 hour and 23 hour days ###
    df.index = df.index.tz_localize(None)
    df = df.resample('H').mean()
    df = df.interpolate(method='ffill')
    ###---------------------------###
    return df


def download_data_with_retry(client, country, start, end):
    """
    Download ENTSO-E data for a specific country and time span with retry logic.

    Args:
    client (EntsoePandasClient): An instance of the EntsoePandasClient class to use for querying the data.
    country (str): The country code for the desired data.
    start (pd.Timestamp): The start time of the desired data span.
    end (pd.Timestamp): The end time of the desired data span.

    Returns:
    pd.DataFrame or None: A Pandas DataFrame containing the downloaded data if successful, or None if unsuccessful.
    """
    success = False
    for attempt in range(1, 6):
        try:
            df = download_entsoe_data(client, country, start, end)
            print(f"Data for {country} downloaded successfully!")
            success = True
            break
        except Exception as e:
            print(f"Error in attempt {attempt} for {country}: {e}")
            if attempt < 5:
                time.sleep(5)
            else:
                print(f"Failed to download data for {country} after 5 attempts.")
    return df if success else None


def save_data_to_csv(df, country, directory_name):
    """
    Save a Pandas DataFrame to a CSV file with the given name in the given directory.

    Args:
    df (pd.DataFrame): The DataFrame to save.
    country (str): The country code to use for the CSV filename.
    directory_name (str): The full path of the directory
    """
    local_filename = os.path.join(directory_name, country + '.csv')
    df.to_csv(local_filename)


def download_data(first_year=2015, last_year=2022, zones=['BE', 'CH', 'DK_1', 'DK_2', 'EE', 'ES', 'FI', 'FR', 'NL', 'NO_1', 'NO_2', 'NO_3', 'NO_4', 'NO_5', 'PL', 'PT', 'SE_1', 'SE_2', 'SE_3', 'SE_4'],
                  folder_name='data', api_key='986f0997-2b9b-43d0-93ed-8514d256f86d'):
    """
    Download ENTSO-E data for a specified time span and list of zones.

    Args:
    first_year (int): The first year to download data for. Default is 2015.
    last_year (int): The last year to download data for. Default is 2022.
    zones (list): A list of country codes for the zones to download data for. Default is.
    folder_name (str): The name of the folder to save the downloaded data to. Default is 'data'.
    api_key (str): The API key to use for the ENTSO-E data queries. Default is '986f0997-2b9b-43d0-93ed-8514d256f86d'.

    Returns:
    None: This function does not return anything, but saves the downloaded data as CSV files in the specified folder.
    """
    client = EntsoePandasClient(api_key=api_key)
    start = pd.Timestamp(pd.to_datetime(str(first_year)) - pd.Timedelta(days=1), tz='CET')
    end = pd.Timestamp(pd.to_datetime(str(last_year + 1)) + pd.Timedelta(days=1), tz='CET')

    directory_name = create_or_use_directory(folder_name)

    for country in zones:
        local_filename = os.path.join(directory_name, country + '.csv')

        if os.path.exists(local_filename):
            print(f"{local_filename} already exists. Skipping download.")
            continue

        df = download_data_with_retry(client, country, start, end)

        if df is not None:
            save_data_to_csv(df, country, directory_name)
        else:
            print(f"Skipping {country} due to failed download attempts.")

# Wrangling functions

def create_destination_directory(current_directory, destination_folder):
    directory_path = os.path.join(current_directory, destination_folder)
    """
    Create a new directory at the specified location if it does not exist.

    Args:
        current_directory (str): The current working directory where the new directory should be created.
        destination_folder (str): The name of the directory to be created.

    Returns:
        str: The absolute path to the newly created directory.
    """

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully ")
    else:
        print(f"Directory '{directory_path}' already exists, using it to store wrangled data!")
    return directory_path

def get_zone_name(csv_directory):
    """
    Extract the zone name from a CSV file path.

    Args:
        csv_directory (str): The path to the CSV file.

    Returns:
        str: The zone name extracted from the filename.

    """        
    return csv_directory.split('\\')[-1].split('.')[0]

def read_csv_files(csv_directories):
    """
    Read in a list of CSV files as Pandas DataFrames.

    Args:
        csv_directories (list): A list of paths to the CSV files.

    Returns:
        list: A list of Pandas DataFrames, one for each input CSV file.

    """
    return [pd.read_csv(directory) for directory in csv_directories]

def add_calendar_features(df):
    """
    Add calendar-based features to a Pandas DataFrame.

    This function calculates the sine and cosine of the hour of day, day of week, and week of year, and adds them
    as new columns to the input DataFrame. It also adds a binary 'isweekend' column indicating whether each row
    falls on a weekend (Saturday or Sunday).

    Args:
        df (pandas.DataFrame): The input DataFrame to which calendar features will be added.

    Returns:
        None.
    """
    max_hour = df.index.hour.values.max() + 1
    max_dayofweek = df.index.day_of_week.values.max() + 1
    max_weekofyear = df.index.isocalendar().week.values.max()
    
    df['hour_sin'] = np.sin(df.index.hour.values * (2 * np.pi / max_hour))
    df['hour_cos'] = np.cos(df.index.hour.values * (2 * np.pi / max_hour))
    df['dow_sin'] = np.sin(df.index.day_of_week.values * (2 * np.pi / max_dayofweek))
    df['dow_cos'] = np.cos(df.index.day_of_week.values * (2 * np.pi / max_dayofweek))
    df['woy_sin'] = np.sin(df.index.isocalendar().week.values * (2 * np.pi / max_weekofyear))
    df['woy_cos'] = np.cos(df.index.isocalendar().week.values * (2 * np.pi / max_weekofyear))
    df['isweekend'] = (df.index.day_of_week.values > 4).astype(int)

def wrangle_single_dataframe(df):
    """
    Modify a single Pandas DataFrame to prepare it for analysis.

    This function renames the first column of the input DataFrame to 'Date', converts the 'Date' column to a
    DatetimeIndex with hourly frequency, and removes the original 'Date' column. It also calls the
    'add_calendar_features()' function to add additional calendar-based features to the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame to be modified.

    Returns:
        None.
    """
    
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df.set_index(df.columns[0], inplace=True)

    add_calendar_features(df)

def store_dataframes(files, zones, directory_path):
    """
    Store a list of Pandas DataFrames to CSV files in a specified directory.

    This function takes a list of Pandas DataFrames, a list of corresponding zone names, and a directory path, and
    saves each DataFrame as a CSV file with a name that matches its corresponding zone name. The files are saved
    in the specified directory.

    Args:
        files (List[pandas.DataFrame]): A list of Pandas DataFrames to be saved.
        zones (List[str]): A list of zone names corresponding to the input DataFrames.
        directory_path (str): The path to the directory where the files will be saved.

    Returns:
        None.
    """

    for i, df in enumerate(files):
        df.to_csv(directory_path + "\\" + zones[i] + '.csv')

def wrangle_data(source_folder, destination_folder, store, return_dfs):
    """
    Wrangles time-series data stored in CSV files from a source directory, adding calendar features
    and storing the resulting data in CSV files in a destination directory, if specified. The function
    returns a dictionary mapping zone names to the resulting pandas DataFrames, if `return_dfs` is `True`.

    Args:
        source_folder: A string specifying the path to the source directory containing the CSV files.
        destination_folder: A string specifying the path to the destination directory where the wrangled CSV
                            files will be stored, if `store` is `True`.
        store: A boolean indicating whether to store the wrangled CSV files in the destination directory or not.
        return_dfs: A boolean indicating whether to return the resulting pandas DataFrames as a dictionary mapping
                    zone names to DataFrames or not.

    Returns:
        If `return_dfs` is `True`, returns a dictionary mapping zone names to the resulting pandas DataFrames. If
        `store` is `True` and `return_dfs` is `False`, returns `None`.
    """
    current_directory = os.path.dirname(__file__)

    if store:
        directory_path = create_destination_directory(current_directory, destination_folder)

    csv_directories = glob.glob(os.path.join(current_directory, source_folder, '*.csv'))
    zones = [get_zone_name(csv_directory) for csv_directory in csv_directories]
    dataframes = read_csv_files(csv_directories)

    for df in dataframes:
        wrangle_single_dataframe(df)

    if store:
        store_dataframes(dataframes, zones, directory_path)

    if return_dfs:
        return dict(zip(zones, dataframes))



# Split the main dataframe into train, validation, and test sets based on the specified years.

def split_dataframe_by_years(dataframe, test_year, num_validation_years, num_train_years):
    """
    
    Split the input dataframe into train, validation, and test sets based on the specified years.
    
    Args:
    - dataframe: a pandas dataframe with a datetime index.
    - test_year: an integer representing the year to be used as the test set.
    - num_validation_years: an integer representing the number of years to be used for the validation set.
    - num_train_years: an integer representing the number of years to be used for the train set.
    
    Returns:
    - A tuple containing three pandas dataframes: train_set, validation_set, and test_set.

    """

    test_set = dataframe[dataframe.index.year == test_year]
    
    if num_validation_years == 0:
        validation_set = pd.DataFrame()
    else:
        validation_set = dataframe[
            (dataframe.index.year >= test_year - num_validation_years) & 
            (dataframe.index.year < test_year)
                                    ]

    train_set = dataframe[
        (dataframe.index.year < test_year - num_validation_years) & 
        (dataframe.index.year >= test_year - num_validation_years - num_train_years)
                            ]

    return train_set, validation_set, test_set


#Make the data ready for use for the ML models format:(samples, timesteps, features)

def split_flexible_history(series, days=[1, 2, 3, 7], n_in=24, n_out=24, stride=24):
    """
    
    Splits a time series into input-output pairs that can be used for training a neural network with flexible history
    window sizes.

    Args:
        series (pandas.Series): The time series to split into input-output pairs.
        days (list): A list of integers representing the days in the past to include in the input sequences, in decreasing
            order of importance. Default is [1, 2, 3, 7].
        n_in (int): The number of input time steps in each sequence. Default is 24.
        n_out (int): The number of output time steps in each sequence. Default is 24.
        stride (int): The number of time steps to move the window forward for each sequence. Default is 24.

    Returns:
        Tuple of numpy.ndarrays representing the input sequences and output sequences respectively.
        The input array has shape (num_samples, num_input_features), where num_samples is the number of input-output pairs
        generated and num_input_features is the total number of time steps included in each input sequence.
        The output array has shape (num_samples, n_out), where num_samples is the number of input-output pairs
        generated and n_out is the number of time steps in each output sequence.
    
    """
    days.sort()
    X, y = [], []

    for i in range(0, len(series), stride):
        start, end = [], []
        for day in days[::-1]:
            start.append((days[::-1][0] - day) * 24 + i)
            end.append((days[::-1][0] - day) * 24 + i + n_in)

        out_start = days[::-1][0] * 24 + i
        out_end = out_start + n_out

        if out_end > len(series):
            break

        x_list = [series[start[idx]:end[idx]].values for idx in range(len(start))]
        seq_x = np.concatenate(x_list)
        seq_y = series[out_start:out_end].values

        X.append(seq_x)
        y.append(seq_y)

    return np.vstack(X), np.vstack(y)

def extract_history_features(df, hf_columns=['Price_DA', 'Load_AC'], days=[1, 2, 3, 7], n_in=24, n_out=24, stride=24, t_column='Price_DA'):
    """
    Extracts history features from a Pandas DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame containing the time series data.
    hf_columns (List[str]): A list of names of the columns in df that should be used as history features.
    days (list): A list of integers indicating the number of days in the past to use for history features.
    n_in (int): The number of time steps to use for input sequence.
    n_out (int): The number of time steps to use for output sequence.
    stride (int): The number of time steps to skip between subsequences.
    t_column (str): The name of the column to use as target.

    Returns:
    tuple: A tuple containing the input and output sequences as NumPy ndarrays.

    """
    hf_column_indices = {name: i for i, name in enumerate(hf_columns)}
    X, y = [], []

    for i, col in enumerate(hf_columns):
        X.append(split_flexible_history(df[col], days=days, n_in=n_in, n_out=n_out, stride=stride)[0])
        y.append(split_flexible_history(df[col], days=days, n_in=n_in, n_out=n_out, stride=stride)[1])

    target_index = hf_column_indices[t_column]
    X = np.dstack(X)
    y = y[target_index]

    return X, y

def extract_future_features(df, days=[1, 2, 3, 7], n_in=24, n_out=24, stride=24, fut_columns=['Load_DA', 'isweekend'], t_column='Price_DA'):
    """
    Extracts future features from a given dataframe and returns the feature and target data for training.

    Args:
    - df (pd.DataFrame): A pandas DataFrame containing the data to be processed.
    - days (list): A list of integers representing the number of days to include in the feature sequences.
    - n_in (int): An integer representing the number of past time steps to include in each input sequence.
    - n_out (int): An integer representing the number of future time steps to predict.
    - stride (int): An integer representing the step size between two consecutive subsequences.
    - fut_columns (List[str]): A list of column names representing the future features to be included in the feature sequences.
    - t_column (str): A string representing the name of the target column.

    Returns:
    - X: A numpy array of shape (n_samples, n_input_steps, n_features) representing the input data.
    - y: A numpy array of shape (n_samples, n_output_steps) representing the target data.
    """
    days.sort()
    X, y = [], []

    for i in range(0, len(df), stride):
        out_start = days[::-1][0] * 24 + i
        out_end = out_start + n_out

        if out_end > len(df):
            break

        x_list = [df[col][out_start:out_end] for col in fut_columns]
        seq_x = np.dstack(x_list)
        seq_y = df[t_column][out_start:out_end]

        X.append(seq_x)
        y.append(seq_y)

    return np.vstack(X), np.vstack(y)

def create_ml_dataset(df_train, df_valid, df_test, hf_columns, days, n_in, n_out, stride, t_column, fut_columns, scale):

    """
    Creates a machine learning dataset from the provided train, valid, and test dataframes.
    if df_valid is None, then the validation data is returned as None.
    if scale is not None, then the data is scaled using the specified method. 'MinMax' and 'Standard' scaling are supported.

    Args:
    - df_train (pd.DataFrame): The training data.
    - df_valid (pd.DataFrame): The validation data.
    - df_test (pd.DataFrame): The test data.
    - hf_columns (list[str]): List of column names to use as historical features.
    - days (list): List of days to use for historical feature extraction.
    - n_in (int): The number of hours of historical data to use as input.
    - n_out (int): The number of hours of future data to predict.
    - stride (int): The number of hours to move the sliding window when extracting historical features.
    - t_column (str): The column to use as the target variable.
    - fut_columns (listlist[str]): List of column names to use as future features.
    - scale (str or None): The scaling method to use, either 'MinMax' or 'Standard', or None for no scaling.

    Returns:
    - X_train_hist (ndarray): The historical features for the training dataset.
    - X_train_fut (ndarray): The future features for the training dataset.
    - y_train (ndarray): The target variable for the training dataset.
    - X_valid_hist (ndarray): The historical features for the validation dataset.
    - X_valid_fut (ndarray): The future features for the validation dataset.
    - y_valid (ndarray): The target variable for the validation dataset.
    - X_test_hist (ndarray): The historical features for the test dataset.
    - X_test_fut (ndarray): The future features for the test dataset.
    - y_test (ndarray or None): The target variable for the test dataset, or None if it is not provided.
    """
    # Initialize dataset variables
    X_train_hist, X_train_fut, y_train = None, None, None
    X_valid_hist, X_valid_fut, y_valid = None, None, None

    # Extract history and future features for the train, valid, and test datasets
    if not df_train.empty:
        X_train_hist, y_train_hist = extract_history_features(df_train, hf_columns, days, n_in, n_out, stride, t_column)
        X_train_fut, y_train_fut = extract_future_features(df_train, days, n_in, n_out, stride, fut_columns, t_column)

    if not df_valid.empty:
        X_valid_hist, y_valid_hist = extract_history_features(df_valid, hf_columns, days, n_in, n_out, stride, t_column)
        X_valid_fut, y_valid_fut = extract_future_features(df_valid, days, n_in, n_out, stride, fut_columns, t_column)

    X_test_hist, y_test_hist = extract_history_features(df_test, hf_columns, days, n_in, n_out, stride, t_column)
    X_test_fut, y_test_fut = extract_future_features(df_test, days, n_in, n_out, stride, fut_columns, t_column)

    # Ensure the history and future target values are the same
    if not df_train.empty and np.all(y_train_hist == y_train_fut):
        y_train = y_train_hist

    if not df_valid.empty and np.all(y_valid_hist == y_valid_fut):
        y_valid = y_valid_hist

    y_test = y_test_hist if np.all(y_test_hist == y_test_fut) else None

    # Scale the input features if a scaler is specified
    if scale is not None:
        if scale == 'MinMax':
            scaler_hist = MinMaxScaler()
            scaler_fut = MinMaxScaler()
        elif scale == 'Standard':
            scaler_hist = StandardScaler()
            scaler_fut = StandardScaler()

        scaler_hist.fit(X_train_hist.reshape(-1, len(hf_columns)))
        scaler_fut.fit(X_train_fut.reshape(-1, len(fut_columns)))

        def scale_and_reshape(X, scaler, feature_len):
            X_scaled = scaler.transform(X.reshape(-1, feature_len))
            return X_scaled.reshape(-1, X.shape[1], feature_len)

        if not df_train.empty:
            X_train_hist = scale_and_reshape(X_train_hist, scaler_hist, len(hf_columns))
            X_train_fut = scale_and_reshape(X_train_fut, scaler_fut, len(fut_columns))

        if not df_valid.empty:
            X_valid_hist = scale_and_reshape(X_valid_hist, scaler_hist, len(hf_columns))
            X_valid_fut = scale_and_reshape(X_valid_fut, scaler_fut, len(fut_columns))

        X_test_hist = scale_and_reshape(X_test_hist, scaler_hist, len(hf_columns))
        X_test_fut = scale_and_reshape(X_test_fut, scaler_fut, len(fut_columns))

    return X_train_hist, X_train_fut, y_train, X_valid_hist, X_valid_fut, y_valid, X_test_hist, X_test_fut, y_test


def sMAPE(actual, predicted):
    """
    Calculate the net promoter score mean absolute percentage error (sMAPE).

    Args:
        actual (np.ndarray): Actual values.
        predicted (np.ndarray): Predicted values.

    Returns:
        float: The sMAPE value.
    """
    numerator = np.abs(actual - predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    smape = np.mean(numerator / denominator) * 100

    return smape


def tf_sMAPE(actual, predicted):
    """
    Calculate the symmetric mean absolute percentage error (SMAPE).

    Args:
        actual (tf.Tensor): Actual values.
        predicted (tf.Tensor): Predicted values.

    Returns:
        float: The sMAPE value.
    """

    actual = tf.py_function(lambda actual: actual.numpy(), [actual], Tout=tf.float32)
    predicted = tf.py_function(lambda predicted: predicted.numpy(), [predicted], Tout=tf.float32)

    absolute_percentage_error = K.abs((actual - predicted) / (K.abs(actual) + K.abs(predicted)))
    smape = K.mean(absolute_percentage_error) * 200

    return smape


def random_seed(seed=211):
    """
    Set the random seed for numpy, tensorflow, and random.

    Args:
        seed (int): The random seed to set.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    



