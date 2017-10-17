# coding: utf-8

"""
Bikes availability prediction (i.e. probability) using xgboost.
"""


import logging
import daiquiri

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import xgboost as xgb


SEED = 1337
np.random.seed(SEED)

daiquiri.setup(logging.INFO)
logger = daiquiri.getLogger("prediction")


def datareader(fpath):
    """Read a CSV file ane return a DataFrame
    """
    logger.info("read the file '%s'", fpath)
    coldate = 'last_update'
    return pd.read_csv(fpath, parse_dates=[coldate])


def complete_data(df):
    """Add some columns

    - day of the week
    - hour of the day
    - minute (10 by 10)
    """
    logger.info("complete some data")
    def group_minute(value):
        if value <= 10:
            return 0
        if value <= 20:
            return 10
        if value <= 30:
            return 20
        if value <= 40:
            return 30
        if value <= 50:
            return 40
        return 50
    df = df.copy()
    df['day'] = df['ts'].apply(lambda x: x.weekday())
    df['hour'] = df['ts'].apply(lambda x: x.hour)
    minute = df['ts'].apply(lambda x: x.minute)
    df['minute'] = minute.apply(group_minute)
    return df


def cleanup(df):
    """Clean up

    - keep OPEN station
    - drop duplicates
    - rename some columns
    - drop some columns
    - drop lines when stands == bikes == 0
    """
    logger.info("cleanup processing")
    columns_to_drop = ['availability', 'status',
                       'bike_stands', 'availabilitycode']
    df = (df.copy()
          .query("status == 'OPEN'")
          .drop(columns_to_drop, axis=1)
          .drop_duplicates()
          .rename_axis({"available_bike_stands": "stands",
                        "available_bikes": "bikes",
                        "last_update": "ts",
                        "number": "station"}, axis=1)
          .query("stands > 0 and bikes > 0")) # or 'Gris' availability value...
    return df


def availability(df, threshold):
    """Set an 'availability' column according to a threshold

    if the number of bikes is less than `threshold`, the availability (of bikes,
    not stands) is low.
    """
    logger.info("set the availability level")
    df = df.copy()
    key = 'availability'
    df[key] = 'medium'
    low_mask = df['bikes'] < threshold
    high_mask = np.logical_and(np.logical_not(low_mask),
                               df['stands'] < threshold)
    df.loc[low_mask, key] = 'low'
    df.loc[high_mask, key] = 'high'
    return df


def bikes_probability(df):
    logger.info("bikes probability")
    df['probability'] = df['bikes'] / (df['bikes'] + df['stands'])
    return df


def extract_bonus_by_station(df):
    """Return a series with station id and bonus oui/non

    turn the french yes/no into 1/0
    """
    logger.info("extract the bonus for each station")
    result =  (df.groupby(["station", "bonus"])["bikes"]
               .count()
               .reset_index())
    result['bonus'] = result['bonus'].apply(lambda x: 1 if x == 'Oui' else 0)
    return result[["station", "bonus"]].set_index("station")


def time_resampling(df, freq="10T"):
    """Normalize the timeseries
    """
    logger.info("Time resampling for each station by '%s'", freq)
    df = (df.groupby("station")
          .resample(freq, on="ts")[["ts", "bikes", "stands"]]
          .mean()
          .bfill())
    return df.reset_index()


def prepare_data_for_training(df, date, freq='1H', start=None, periods=1,
                              observation='availability'):
    """Prepare data for training

    date: datetime / Timestamp
        date for the prediction
    freq: str
        the delay between the latest available data and the prediction. e.g. one hour
    start: Timestamp
        start of the history data (for training)
    periods: int
        number of predictions

    Returns 4 DataFrames: two for training, two for testing
    """
    logger.info("prepare data for training")
    logger.info("sort values (station, ts)")
    data = df.sort_values(['station', 'ts']).set_index(["ts", "station"])
    logger.info("compute the future availability at '%s'", freq)
    label = data[observation].copy()
    label.name = "future"
    label = (label.reset_index(level=1)
             .shift(-1, freq=freq)
             .reset_index()
             .set_index(["ts", "station"]))
    logger.info("merge data with the future availability")
    result = data.merge(label, left_index=True, right_index=True)
    logger.info("availability label as values")
    if observation == 'availability':
        result[observation] = result[observation].replace({"low": 0, "medium": 1, "high": 2})
        result['future'] = result['future'].replace({"low": 0, "medium": 1, "high": 2})
    result.reset_index(level=1, inplace=True)
    freq = freq.replace("T", "m")
    if start is not None:
        result = result[result.index >= start]
    cut = date - pd.Timedelta(freq)
    stop = date + periods * pd.Timedelta(freq)
    logger.info("cut date %s", cut)
    logger.info("stop date %s", stop)
    logger.info("split train and test according to a prediction date")
    train = result[result.index <= cut].copy()
    train_X = train.drop([observation, "future"], axis=1)
    train_Y = train['future'].copy()
    # time window
    mask = np.logical_and(result.index > date, result.index <= stop)
    test = result[mask].copy()
    test_X = test.drop([observation, "future"], axis=1)
    test_Y = test['future'].copy()
    return train_X, train_Y, test_X, test_Y


def fit(train_X, train_Y, test_X, test_Y):
    """Train the xgboost model

    Return the booster trained model
    """
    logger.info("fit")
    # param = {'objective': 'multi:softmax'}
    # param = {'objective': 'reg:linear'}
    param = {'objective': 'reg:logistic'}
    param['eta'] = 0.2
    param['max_depth'] = 6
    param['silent'] = 1
    param['nthread'] = 2
    param['num_class'] = train_Y.nunique()
    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 10
    bst = xgb.train(param, xg_train, num_round, watchlist)
    return bst


def prediction(bst, test_X, test_Y):
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    pred = bst.predict(xg_test)
    return pred


def error_rate(bst, test_X, test_Y):
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    pred = bst.predict(xg_test)
    error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
    logger.info('Test error using softmax = %s', error_rate)
    return error_rate


if __name__ == '__main__':
    DATAFILE = "./data/lyon.csv"
    THRESHOLD = 3

    raw = datareader(DATAFILE)
    df_clean = cleanup(raw)
    bonus = extract_bonus_by_station(df_clean)
    df_clean = df_clean.drop("bonus", axis=1)
    # df = (df_clean.pipe(time_resampling)
    #       .pipe(complete_data)
    #       .pipe(lambda x: availability(x, THRESHOLD)))
    df = (df_clean.pipe(time_resampling)
          .pipe(complete_data)
          .pipe(bikes_probability))

    # Note: date range is are 2017-07-08 15:20:28  -  2017-09-26 14:58:45
    start = pd.Timestamp("2017-07-11") # Tuesday
    # predict_date = pd.Timestamp("2017-07-26T19:30:00") # wednesday
    predict_date = pd.Timestamp("2017-07-26T10:00:00") # wednesday
    train_X, train_Y, test_X, test_Y = prepare_data_for_training(df,
                                                                 predict_date,
                                                                 freq='30T',
                                                                 start=start,
                                                                 periods=2,
                                                                 observation='probability')
    # train_X, train_Y, test_X, test_Y = prepare_data_for_training(df, predict_date, freq='1H', start=start, periods=2)

    bst = fit(train_X, train_Y, test_X, test_Y)
    err = error_rate(bst, test_X, test_Y)
    print("Error rate: {}".format(err))
    pred = prediction(bst, test_X, test_Y)
