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


def transform_confidences(confidences: np.ndarray):
    return [row[0] for row in confidences], [row[1] for row in confidences]


def auto_model(data: pd.DataFrame, granularity: str, transform: bool = False):
    period_len = utils.get_period_length(granularity)
    data = group_and_reindex(data, granularity)

    decomposition = seasonal_decompose(data.measurement, period=period_len)

    data_train, data_test = train_test_split_on_year(data)

    if transform:
        model = Pipeline([
            ('boxcox', BoxCoxEndogTransformer(lmbda2=1e-6)),
            ('arima', pmd.AutoARIMA(trace=True,
                                    suppress_warnings=True,
                                    m=period_len))
        ])
        model.fit(data_train.measurement)
    else:
        model = pmd.auto_arima(data_train.measurement, m=period_len, trace=True, suppress_warnings=True)

    pred, conf_int = model.predict_in_sample(start=data_test.index[0], end=data_test.index[-1], return_conf_int=True)
    lower, upper = transform_confidences(conf_int)

    data_test_ret = data_test.copy()
    data_test.loc[:, 'prediction'] = pred
    data_test.loc[:, 'lower_confidence'] = lower
    data_test.loc[:, 'upper_confidence'] = upper
    return decomposition, model, data_train, data_test
