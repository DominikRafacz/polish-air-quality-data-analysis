import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pmd

from pmdarima.preprocessing import BoxCoxEndogTransformer
from pmdarima.pipeline import Pipeline

import utils


def group_and_reindex(data: pd.DataFrame, granularity: str = 'daily'):
    data = data.sort_values('timestamp') \
        .groupby('timestamp') \
        .agg({'measurement': np.nanmean})

    if granularity == 'daily':
        data.index.names = ['date']
        data = data.reset_index()
        data['day'] = data.date.dt.day
        data['month'] = data.date.dt.month
        data['year'] = data.date.dt.year
    elif granularity == 'weekly':
        data = data.groupby([data.index.year, data.index.isocalendar().week]) \
            .mean()
        data.index.names = ['year', 'week']
        data = data.reset_index()
    elif granularity == 'monthly':
        data = data.groupby([data.index.year, data.index.month]) \
            .mean()
        data.index.names = ['year', 'month']
        data = data.reset_index()

    return data


def train_test_split_on_year(data: pd.DataFrame, year: int = 2020):
    return data[data.year < year].copy(), data[data.year >= year].copy()


def transpose_confidences(confidences: np.ndarray):
    return [row[0] for row in confidences], [row[1] for row in confidences]


def auto_transform(data_train, data_test):
    transformer = BoxCoxEndogTransformer(lmbda2=1e-6)

    trans_train, _ = transformer.fit_transform(data_train.measurement)
    ret_train = data_train.copy()
    ret_train['measurement_transformed'] = trans_train

    trans_test, _ = transformer.transform(data_test.measurement)
    ret_test = data_test.copy()
    ret_test['measurement_transformed'] = trans_test

    return transformer, ret_train, ret_test


def auto_model(data: pd.DataFrame, granularity: str, transform: bool = False, ret_decomposition: bool = False):
    period_len = utils.get_period_length(granularity)
    data = group_and_reindex(data, granularity)
    data_train, data_test = train_test_split_on_year(data)

    if transform:
        transformer, data_train_trans, data_test_trans = auto_transform(data_train, data_test)
        model = pmd.auto_arima(data_train_trans.measurement_transformed, m=period_len, trace=True, suppress_warnings=True)
        pred_trans, conf_int_trans = model.predict_in_sample(start=data_test.index[0], end=data_test.index[-1], return_conf_int=True)
        lower_trans, upper_trans = transpose_confidences(conf_int_trans)

        pred, lower, upper = (transformer.inverse_transform(values)[0] for values in [pred_trans, lower_trans, upper_trans])

        data_train.loc[:, 'measurement_transformed'] = data_train_trans.measurement_transformed
        data_test.loc[:, 'measurement_transformed'] = data_test_trans.measurement_transformed
        data_test.loc[:, 'prediction_transformed'] = pred_trans
        data_test.loc[:, 'lower_confidence_transformed'] = lower_trans
        data_test.loc[:, 'upper_confidence_transformed'] = upper_trans

        model = {'transformer': transformer, 'model': model}
    else:
        model = pmd.auto_arima(data_train.measurement, m=period_len, trace=True, suppress_warnings=True)

        pred, conf_int = model.predict_in_sample(start=data_test.index[0], end=data_test.index[-1], return_conf_int=True)
        lower, upper = transpose_confidences(conf_int)

    data_test.loc[:, 'prediction'] = pred
    data_test.loc[:, 'lower_confidence'] = lower
    data_test.loc[:, 'upper_confidence'] = upper

    if ret_decomposition:
        return model, data_train, data_test, seasonal_decompose(data.measurement, period=period_len)
    else:
        return model, data_train, data_test


def calc_diffs(data, transformed=None, years=(2020, 2021)):
    if transformed or (transformed is None and 'measurement_transformed' in data.columns):
        var_mes, var_pre = 'measurement_transformed', 'prediction_transformed'
    else:
        var_mes, var_pre = 'measurement', 'prediction'
    return data[data.year.isin(years)][var_mes] - data[data.year.isin(years)][var_pre]


def calc_rmse(diff, split_pos_neg=False):
    if split_pos_neg:
        return np.sqrt(np.square(diff[diff > 0]).mean()), np.sqrt(np.square(diff[diff < 0]).mean())
    else:
        return np.sqrt(np.square(diff).mean())
