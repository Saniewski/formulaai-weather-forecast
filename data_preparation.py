from pickle import dump, load
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf


'''
This function does initial data preparation based on original file provided in the challenge.
Function returns a pandas DataFrame which is then used by preprocess function.
It also writes the DataFrame to a csv file.
'''
def prepare_data(path='weather.csv', out_path='weather_processed.csv'):
    df = pd.read_csv(path)

    # prepare columns and offsets to be exctracted from the original dataset
    columns = [
        "M_SESSION_UID",
        "M_SESSION_TIME",
        "TIMESTAMP",
        "M_TRACK_ID",
        "M_TRACK_TEMPERATURE",
        "M_GAME_PAUSED",
        "M_FORECAST_ACCURACY",
        "M_AIR_TEMPERATURE",
        "M_NUM_WEATHER_FORECAST_SAMPLES",
        "M_SESSION_TYPE",
        "M_WEATHER_FORECAST_SAMPLES_M_SESSION_TYPE",
        "M_TIME_OFFSET",
        "M_WEATHER_FORECAST_SAMPLES_M_WEATHER",
        "M_WEATHER_FORECAST_SAMPLES_M_TRACK_TEMPERATURE",
        "M_TRACK_TEMPERATURE_CHANGE",
        "M_WEATHER_FORECAST_SAMPLES_M_AIR_TEMPERATURE",
        "M_AIR_TEMPERATURE_CHANGE",
        "M_RAIN_PERCENTAGE",
        "M_WEATHER",
    ]

    forecast_columns = [
        "M_WEATHER_FORECAST_SAMPLES_M_WEATHER",
        "M_WEATHER_FORECAST_SAMPLES_M_TRACK_TEMPERATURE",
        "M_TRACK_TEMPERATURE_CHANGE",
        "M_WEATHER_FORECAST_SAMPLES_M_AIR_TEMPERATURE",
        "M_AIR_TEMPERATURE_CHANGE",
        "M_RAIN_PERCENTAGE",
    ]

    time_offsets = (5, 10, 15, 30, 45, 60)

    # drop nulls and clear data from wrong values and outliers
    df = df[columns].dropna()
    df = df[(df['M_GAME_PAUSED'] == 0) & (df['M_SESSION_TYPE'] != 0)]
    df = df.drop(columns='M_WEATHER_FORECAST_SAMPLES_M_SESSION_TYPE').drop_duplicates()
    df = df[(df['M_TRACK_TEMPERATURE'] > 15) &
            (df['M_AIR_TEMPERATURE'] > 15) &
            (df['M_WEATHER_FORECAST_SAMPLES_M_TRACK_TEMPERATURE'] > 15) &
            (df['M_WEATHER_FORECAST_SAMPLES_M_AIR_TEMPERATURE'] > 15)]
    df = df[df['M_TIME_OFFSET'].isin(time_offsets)]

    # sort data
    df.sort_values(by=['M_SESSION_UID', 'M_SESSION_TIME'], inplace=True)

    # create the target data structure
    df_tmp = df[["M_SESSION_UID",
                "M_SESSION_TIME",
                "TIMESTAMP",
                "M_TRACK_ID",
                "M_TRACK_TEMPERATURE",
                "M_FORECAST_ACCURACY",
                "M_AIR_TEMPERATURE",
                "M_WEATHER"]].drop_duplicates()

    # squeeze forecast values to additional columns
    df_forecasts = df[["M_SESSION_UID", "M_SESSION_TIME"]]
    for column in forecast_columns:
        for offset in time_offsets:
            df_forecasts[f'{column}_{offset}'] = df[df["M_TIME_OFFSET"] == offset][column]
    df_forecasts = df_forecasts.groupby(by=["M_SESSION_UID", "M_SESSION_TIME"], dropna=False).median().round(0)

    # join data with forecasts
    df_final = df_tmp.join(df_forecasts, on=["M_SESSION_UID", "M_SESSION_TIME"], how='left')

    df_classes = pd.DataFrame()
    for idx, (_, sess) in enumerate(df_final.groupby(by='M_SESSION_UID')):
        sess = pd.DataFrame(sess)
        sess['M_SESSION_UID'] = idx
        df_classes = df_classes.append(sess)
    df_final = df_classes

    # replace 6 to 4 in weather variables
    df_final['M_WEATHER'].replace(6.0, 4.0)
    for offset in time_offsets:
        df_final[f'M_WEATHER_FORECAST_SAMPLES_M_WEATHER_{offset}'].replace(6.0, 4.0)

    df_final.to_csv(path_or_buf=out_path)

    return df_final

'''
This function takes as input data from prepare_data function and returns inputs for the neural network model.
Arguments can be either path to a csv file or a pandas DataFrame.
'''
def preprocess(path='weather_processed.csv', data=None, train_model=False):

    # load data
    if data is None:
        df = pd.read_csv(path)
    else:
        df = data

    # clean data
    df = df.dropna()
    try:
        df.drop(columns=['Unnamed: 0'])
    except:
        pass

    df['M_SESSION_TIME'] = df['M_SESSION_TIME'].astype(int)

    if train_model:
        series = []
        for idx, (_, sess) in enumerate(df.groupby(by='M_SESSION_UID')):
            if len(sess) < 50:
                continue
            sess = pd.DataFrame(sess)
            sess['M_SESSION_UID'] = idx
            series.append(sess)
        train, valid = train_test_split(series, test_size = 0.2, random_state=1)
        train, valid = pd.concat(train), pd.concat(valid)
        df = train

        # fit new scaler
        scaler = StandardScaler()
        scaler.fit(df)

        # scale data
        scaled_train = pd.DataFrame(scaler.transform(df))
        scaled_train.columns = df.columns
        scaled_valid = pd.DataFrame(scaler.transform(valid))
        scaled_valid.columns = df.columns

        # dump scaler
        dump(scaler, open('scaler.pkl', 'wb'))

    else:
        # load scaler which was used while fitting the model
        scaler = load(open('scaler.pkl', 'rb'))

        # scale data
        scaled_valid = pd.DataFrame(scaler.transform(df))
        scaled_valid.columns = df.columns

    # define columns
    y_columns = [
        'M_RAIN_PERCENTAGE_5',
        'M_RAIN_PERCENTAGE_10',
        'M_RAIN_PERCENTAGE_15',
        'M_RAIN_PERCENTAGE_30',
        'M_RAIN_PERCENTAGE_45',
        'M_RAIN_PERCENTAGE_60',
        'M_WEATHER_FORECAST_SAMPLES_M_WEATHER_5',
        'M_WEATHER_FORECAST_SAMPLES_M_WEATHER_10',
        'M_WEATHER_FORECAST_SAMPLES_M_WEATHER_15',
        'M_WEATHER_FORECAST_SAMPLES_M_WEATHER_30',
        'M_WEATHER_FORECAST_SAMPLES_M_WEATHER_45',
        'M_WEATHER_FORECAST_SAMPLES_M_WEATHER_60',
        ]

    x_columns = [
        'M_SESSION_UID',
        'M_SESSION_TIME',
        'TIMESTAMP',
        'M_TRACK_ID',
        'M_TRACK_TEMPERATURE',
        'M_AIR_TEMPERATURE',
        'M_WEATHER',
        'M_WEATHER_FORECAST_SAMPLES_M_TRACK_TEMPERATURE_5',
        'M_TRACK_TEMPERATURE_CHANGE_5',
        'M_WEATHER_FORECAST_SAMPLES_M_AIR_TEMPERATURE_5',
        'M_AIR_TEMPERATURE_CHANGE_5',
        'M_WEATHER_FORECAST_SAMPLES_M_TRACK_TEMPERATURE_10',
        'M_TRACK_TEMPERATURE_CHANGE_10',
        'M_AIR_TEMPERATURE_CHANGE_10',
        'M_WEATHER_FORECAST_SAMPLES_M_TRACK_TEMPERATURE_15',
        'M_TRACK_TEMPERATURE_CHANGE_15',
        'M_WEATHER_FORECAST_SAMPLES_M_AIR_TEMPERATURE_10',
        'M_WEATHER_FORECAST_SAMPLES_M_AIR_TEMPERATURE_15',
        'M_AIR_TEMPERATURE_CHANGE_15',
        'M_WEATHER_FORECAST_SAMPLES_M_TRACK_TEMPERATURE_30',
        'M_TRACK_TEMPERATURE_CHANGE_30',
        'M_WEATHER_FORECAST_SAMPLES_M_AIR_TEMPERATURE_30',
        'M_AIR_TEMPERATURE_CHANGE_30',
        'M_WEATHER_FORECAST_SAMPLES_M_WEATHER_45',
        'M_WEATHER_FORECAST_SAMPLES_M_TRACK_TEMPERATURE_45',
        'M_TRACK_TEMPERATURE_CHANGE_45',
        'M_WEATHER_FORECAST_SAMPLES_M_AIR_TEMPERATURE_45',
        'M_AIR_TEMPERATURE_CHANGE_45',
        'M_WEATHER_FORECAST_SAMPLES_M_TRACK_TEMPERATURE_60',
        'M_TRACK_TEMPERATURE_CHANGE_60',
        'M_WEATHER_FORECAST_SAMPLES_M_AIR_TEMPERATURE_60',
        'M_AIR_TEMPERATURE_CHANGE_60'
        ]

    if train_model:
        x_train, y_train = scaled_train[x_columns], scaled_train[y_columns]
    x_valid, y_valid = scaled_valid[x_columns], scaled_valid[y_columns]

    n_input = 50
    n_features = len(x_columns)

    if train_model:
        train_ts = tf.keras.preprocessing.sequence.TimeseriesGenerator(x_train.to_numpy(), y_train.to_numpy(), length = n_input, batch_size=10000)
    valid_ts = tf.keras.preprocessing.sequence.TimeseriesGenerator(x_valid.to_numpy(), y_valid.to_numpy(), length = n_input, batch_size=10000)

    if train_model:
        return train_ts, valid_ts, n_input, n_features

    return valid_ts, n_input, n_features
