
import pandas as pd
import os
import functools
import numpy as np

def clean_data(path):
    """
    extract only columns needed for cost-to-go optim
    """
    # load data 
    df = pd.read_csv(path)
    # clean data
    col_set = [
        "FlightDate",
        "HourRange",
        "TotalPassengerCount",
        "AverageWait",
        "BoothsUsed",
        "FlightCount",]
    df = df[col_set]
    return df

def collect_data(list_of_files):
    month_files = list_of_files
    month_data = list(map(lambda x: clean_data(path="data/{}".format(x)), month_files))
    collected = functools.reduce(lambda x,y: pd.concat([x,y]), month_data).reset_index(drop=True)
    return collected

def average_by_hour(df):
    features = ["TotalPassengerCount", "AverageWait", "BoothsUsed"]
    df2 = []
    for f in features:
        temp = df.groupby(["HourRange"])[f].mean().reset_index()
        df2.append(temp)
    df2 = functools.reduce(lambda x,y: pd.merge(x, y, on = "HourRange"), df2)
    return df2

def arrival_rate(df, month, hour, day):

    # Filter data for the specified month and hour
    filtered_data = df[(df['Month'] == month) & (df['Hour'] == hour) & (df['WeekDay'] == day)]
 
    arrival_rate = filtered_data['TotalPassengerCount'].mean()

    # if nan then return 0
    if np.isnan(arrival_rate):
        return 0
    else:
        return arrival_rate.astype(int)
    
def service_rate(df):
    # drop rows with 0 AverageWait
    df = df[df['AverageWait'] != 0]
    df['ServiceRate'] = df['TotalPassengerCount'] / df['BoothsUsed'] 
    return df['ServiceRate'].mean()

