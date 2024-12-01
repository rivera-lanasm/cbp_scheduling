
import pandas as pd
import os
import functools

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
        "BoothsUsed"]
    df = df[col_set]
    return df

def collect_data():
    month_files = ["jfk_2022.csv", "jfk_2023.csv"]
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

