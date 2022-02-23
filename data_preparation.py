import pandas as pd

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
