import matplotlib.pyplot as plt
from functools import wraps

# source: https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle
import pandas as pd

from utils import generate_ticks

DEFAULT_THEME = 'pitayasmoothie-dark.mplstyle'
DEFAULT_RC_PARAMS = {
    'figure.dpi': 300,
    'figure.figsize': (12, 9)
}


def with_theme_and_params(plotting_func):
    """
    Decorator for wrapping function with plotting context

    :param plotting_func: Plotting function to be wrapped with default rc_params and default theme
    :return: Wrapped function.
    """
    @wraps(plotting_func)
    def wrapper(*args, **kwargs):
        theme = kwargs['theme'] if 'theme' in kwargs else DEFAULT_THEME
        rc_params = kwargs['rc_params'] if 'rc_params' in kwargs else DEFAULT_RC_PARAMS
        show = kwargs['show'] if 'show' in kwargs else True
        with plt.style.context(theme), plt.rc_context(rc_params):
            ret = plotting_func(*args, **kwargs)
            if show:
                plt.show()
            else:
                return ret
    return wrapper


@with_theme_and_params
def plot_map_with_pollutions(geo_data_with_measurements, title, **kwargs):
    """
    Function for plotting map with pollutions

    :param geo_data_with_measurements: GeoDataFrame with columns 'geometry' and 'measurement'
    :param title: Title of the plot
    :param kwargs: used by wrappers
    :return: tuple (figure object, axes object)
    """
    fig, ax = plt.subplots()

    geo_data_with_measurements.plot(column='measurement', legend=True, ax=ax)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


@with_theme_and_params
def plot_timeline_of_pollution(data, pollution, granularity, **kwargs):
    fig, ax = plt.subplots()

    ax.plot(data.measurement)

    ax.set_title(f'{pollution} pollution in Poland averaged {granularity}')
    ax.set_xlabel('Time')
    ax.set_ylabel(f'Average concentration of {pollution} in the air [μg/m³]')

    ax.set_xticks(*generate_ticks(data))

    return fig, ax


@with_theme_and_params
def plot_decomposition_of_pollution(decomposition, data, pollution, granularity, **kwargs):
    fig, axs = plt.subplots(2, 2, sharey=True)

    ticks, tick_labels = generate_ticks(data)

    ax_top = plt.subplot(2, 1, 1)
    ax_top.plot(decomposition.observed)
    ax_top.plot(decomposition.trend)
    ax_top.set_xticks(ticks, tick_labels)
    ax_top.set_title(f'Trend of the measurements of {pollution} in Poland, averaged {granularity}')
    ax_top.set_ylabel(f'{pollution} in the air [μg/m³]')
    ax_top.legend(['Measurements', 'Overall trend'])

    axs[1][0].plot(decomposition.seasonal)

    axs[1][1].scatter(decomposition.resid.index, decomposition.resid)

    for index, title in zip([0, 1], ['Seasonal trend of measurements', 'Residuals']):
        axs[1][index].axhline(color="white")
        axs[1][index].set_xticks(ticks, tick_labels)
        axs[1][index].set_title(title)

    axs[1][0].set_ylabel('Difference [μg/m³]')

    return fig, axs


@with_theme_and_params
def plot_historical_and_predictions(model_results, pollution, granularity, **kwargs):
    fig, ax = plt.subplots()

    ax.plot(pd.concat([model_results['df_train'].measurement, model_results['df_test'].measurement]))
    ax.plot(model_results['df_test'].index, model_results['df_test'].prediction, linestyle='dashed')
    ax.fill_between(model_results['df_test'].index, model_results['df_test'].lower_confidence,
                    model_results['df_test'].upper_confidence, color='gray', alpha=0.2)
    ax.legend(['Historical data', '2020 and 2021 predictions', 'Confidence interval'])
    ax.set_title(f'Comparison of historical {granularity} data to trend prediction for {pollution} in Poland')
    ax.set_ylabel(f'Average concentration of {pollution} in the air [μg/m³]')
    ax.set_xlabel(f'Time')
    ax.set_xticks(*generate_ticks(pd.concat([model_results['df_train'][['year']], model_results['df_test'][['year']]])))

    return fig, ax

