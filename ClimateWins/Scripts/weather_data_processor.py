#Allows the user to choose which stations to drop from the data frame to be analyzed as well as which variable columns to drop for all stations

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def weatherData_select(df):
    """
    Modify the dataframe based on user input by dropping selected stations and variables.

    Parameters:
    df (pd.DataFrame): The original dataframe containing weather data.

    Returns:
    pd.DataFrame: The modified dataframe with selected stations and variables dropped.
    """
    # Create a copy of the original dataframe
    dfw = df.copy()

    # Prompt the user to select stations to drop
    stations_to_drop = input("Enter stations to drop (separated by comma): ")
    if stations_to_drop:
        stations_to_drop_list = [station.strip().lower() for station in stations_to_drop.split(',')]
        for station in stations_to_drop_list:
            dfw = dfw.loc[:, ~dfw.columns.str.lower().str.startswith(station+'_')]

    # Prompt the user to select variables to drop
    variables_to_drop = input("Enter variables to drop (separated by comma): ")
    if variables_to_drop:
        variables_to_drop_list = [variable.strip().lower() for variable in variables_to_drop.split(',')]
        columns_to_drop = [col for col in dfw.columns if any(var in col.lower() for var in variables_to_drop_list)]
        dfw = dfw.drop(columns=columns_to_drop, errors='ignore')

    return dfw

