
import pandas as pd
import os
import functools
import numpy as np

def map_time_to_period(hour):
    """
    Maps a given time range to 'morning' or 'afternoon'.
    
    Parameters:
        hour_range (str): A time range in the format "HHMM - HHMM".
    
    Returns:
        str: "morning" or "afternoon" based on the range.
    """
    # Extract the start hour from the range
    start_hour = hour
    # Map time to period
    if 6 <= start_hour < 12:
        return "morning"
    elif 12 <= start_hour < 18:
        return "afternoon"
    elif 18 <= start_hour < 24:
        return "evening"
    else:
        return "night"
    
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

    # transform FlightDate to datetime
    df['FlightDate'] = pd.to_datetime(df['FlightDate'])

    # convert HourRange to Hour
    df['Hour'] = df['HourRange'].str.split('-').str[0].str.strip()
    df['Hour'] = (df['Hour'].astype(int)/100).astype(int)

    # WeekDay column
    df['WeekDay'] = df['FlightDate'].dt.day_name()

    # Month column
    df['Month'] = df['FlightDate'].dt.month_name()

    # time_of_day column
    df["time_of_day"] = list(map(lambda x: map_time_to_period(x), df['Hour']))

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

def hourly_arrival_rate(df, hour, day = None, month = None):

    if month is not None and day is not None:
        filtered_data = df[(df['Month'] == month) & (df['Hour'] == hour) & (df['WeekDay'] == day)]
    elif month is not None:
        filtered_data = df[(df['Month'] == month) & (df['Hour'] == hour)]
    elif day is not None:
        filtered_data = df[(df['Hour'] == hour) & (df['WeekDay'] == day)]
    else:
        filtered_data = df[(df['Hour'] == hour)]
 
    arrival_rate = filtered_data['TotalPassengerCount'].mean()

    # if nan then return 0
    if np.isnan(arrival_rate):
        return 0
    else:
        return arrival_rate.astype(int)
    
def shift_arrival_rate(df, time_of_day, day = None, month = None):
    # Filter data for the specified month and hour
    if month is not None and day is not None:
        filtered_data = df[(df['Month'] == month) & (df['WeekDay'] == day) & (df['time_of_day'] == time_of_day)]
    elif day is not None:
        filtered_data = df[(df['WeekDay'] == day) & (df['time_of_day'] == time_of_day)]
    elif month is not None:
        filtered_data = df[(df['Month'] == month) & (df['time_of_day'] == time_of_day)]
    else:
        filtered_data = df[(df['time_of_day'] == time_of_day)]
 
    arrival_rate = filtered_data['TotalPassengerCount'].mean()

    # if nan then return 0
    if np.isnan(arrival_rate):
        return 0
    else:
        return arrival_rate.astype(int)


def service_rate(df):
    # drop rows with 0 AverageWait
    df = df[df['AverageWait'] != 0]

    # calculate the service rate as the weight average of the TotalPassengerCount and BoothsUsed
    df['ServiceRate'] = df['TotalPassengerCount'] / df['BoothsUsed']
    return sum(df['ServiceRate']*df['TotalPassengerCount']) / sum(df['TotalPassengerCount'])


