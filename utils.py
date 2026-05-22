import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def get_indicators(directory):
    """
    Reads multiple CSV files from a directory, processes them into time series data, 
    and combines them into a single DataFrame with interpolated missing values.

    Parameters:
    directory (str): Path to the directory containing CSV files.

    Returns:
    pd.DataFrame: A DataFrame where each column represents an indicator from a CSV file, 
                  indexed by month.
    """
    df_list = []
    
    # Loop through files in directory (skip system files)
    for file in os.listdir(directory):
        try:
            df = pd.read_csv(os.path.join(directory, file))
            df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors='coerce')  # Convert first column to datetime
            df.set_index(df.columns[0], inplace=True, drop=True)  # Set first column as index
            df[df.columns[0]] = pd.to_numeric(df[df.columns[0]], errors='coerce')  # Convert data to numeric
            df.index = df.index.to_period('M')  # Convert index to monthly period
            df = pd.DataFrame(df.groupby(by=df.index)[df.columns[0]].mean())  # Group by month and take mean
            df.columns = [file[:-4]]  # Use filename (without extension) as column name
            df_list.append(df)
        except:
            pass
    
    indicator_df = pd.concat(df_list, axis=1)
    indicator_df.index.rename('Month', inplace=True)  # Rename index to 'Month'
    indicator_df.index = indicator_df.index.to_timestamp()  # Convert index back to timestamp
    indicator_df.sort_index(inplace=True)  # Sort by date
    indicator_df = indicator_df.interpolate(method='linear', limit_area='inside')  # Interpolate missing values

    return indicator_df


def get_metadata(manheim_df, indicator_df):
    """
    Computes correlation statistics between indicators and manheim.

    Parameters:
    manheim_df (pd.DataFrame): DataFrame containing manheim values, with a datetime index.
    indicator_df (pd.DataFrame): DataFrame containing various indicator time series.

    Returns:
    pd.DataFrame: A metadata DataFrame containing:
        - 'Indicator': Name of the indicator.
        - 'Month Start': First available date of the indicator series.
        - 'Corr': Correlation between the indicator and manheim.
        - 'Detrended Corr': Correlation after removing trends from both series.
        - 'Abs Detrended Corr': Absolute value of the detrended correlation.
    """

    rows = []
    for indicator in indicator_df.columns:
        series = indicator_df[indicator].loc[manheim_df.index]
        corr = np.corrcoef(series, manheim_df.manheim)[0, 1]
        detrended_corr = np.corrcoef(signal.detrend(series), signal.detrend(manheim_df.manheim))[0, 1]
        abs_detrended_corr = abs(detrended_corr)
        month_start = indicator_df[indicator].dropna().index[0]
        row = [indicator, month_start, corr, detrended_corr, abs_detrended_corr]
        rows.append(row)
    meta_df = pd.DataFrame(rows, columns=['Indicator', 'Month Start', 'Corr', 'Detrended Corr', 'Abs Detrended Corr'])
    meta_df.set_index('Indicator', inplace=True)
    meta_df.sort_values(by='Month Start', ascending=True, inplace=True)
    return meta_df

def exponential_decay_weights(size, decay_rate=0.1):
    """
    Generates a set of normalized weights that follow an exponential decay pattern.

    Parameters:
    size (int): The number of weights to generate.
    decay_rate (float, optional): The rate at which the weights decay. Default is 0.2.

    Returns:
    np.ndarray: A normalized array of weights summing to 1.
    """

    indices = np.arange(size)
    weights = np.exp(-decay_rate * (size - 1 - indices))
    return weights / weights.sum()

def ensemble(forecasted, observed, horizon, periods=3, gamma=1e2):
    """
    Computes weighted ensemble predictions based on historical forecast errors.

    Parameters:
    forecasted (list or np.array): A list or array containing forecasted values.
    observed (list or np.array): A list or array containing actual observed values.
    horizon (int): Number of months to forecast.
    periods (int): Number of past periods to consider for computing weights. Default is 3.
    gamma (float): Scaling factor for weight calculation. Default is 1e2.

    Returns:
    tuple: 
        - weighted_preds (list): List of weighted ensemble predictions.
        - min_preds (list): List of minimum forecasted values for each period.
        - max_preds (list): List of maximum forecasted values for each period.
    """
    weighted_preds = []
    min_preds = []
    max_preds = []
    weight_list = []
    for i in range(periods+horizon,len(forecasted)):
        mapes = []
        for j in range(len(forecasted[i])):
            mapes.append(mape(forecasted[i-periods-horizon:i-horizon,j].flatten(), observed[i-periods-horizon:i-horizon].flatten()))
        weights = np.exp(-gamma * np.array(mapes))
        weights /= weights.sum()
        weight_list.append(weights)
        weighted_preds.append(weights.T @ np.array(forecasted[i]))
        min_preds.append(forecasted[i].min())
        max_preds.append(forecasted[i].max())
    return weighted_preds, min_preds, max_preds

def mape(pred, obs):
    return np.mean(np.abs((np.array(pred).flatten() - np.array(obs).flatten()) / np.array(obs).flatten()))

def mse(pred, obs):
    return np.mean((np.array(pred).flatten() - np.array(obs).flatten()) ** 2)