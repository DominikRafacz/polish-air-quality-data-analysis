import matplotlib.pyplot as plt
import pandas as pd
import pmdarima as pmd

from utils import generate_ticks, get_period_label, get_period_length
from functools import wraps
from matplotlib import gridspec


# source: https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle
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


@with_theme_and_params
def plot_acf_and_frequency(data, pollution, granularity, transformed=False, **kwargs):
    fig = pmd.utils.tsdisplay(data.measurement_transformed if transformed else data.measurement,
                              show=False, lag_max=get_period_length(granularity))
    axs = fig.axes
    axs[0].set_title(f'Average {pollution} pollution in Poland averaged {granularity} {", transformed with Box-Cox" if transformed else ""}')
    axs[0].set_xticks(*generate_ticks(data))
    axs[0].set_ylabel(f'{"Transformed m" if transformed else "M"}easurement value')
    axs[1].set_title('Autocorrelation function')
    axs[1].set_xlabel(f'{get_period_label(granularity, uppercase=True)} lag')
    axs[1].set_ylabel('Autocorrelation value')
    axs[2].set_title('Frequency of specific values')
    axs[2].set_ylabel('Number of occurences')
    axs[2].set_xlabel(f'Value of {"transformed " if transformed else " "}measurement')

    return fig, axs


@with_theme_and_params
def plot_full_pre_model_analysis(data_train, data_test, decomposition, pollution, granularity, transformed: bool, **kwargs):
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 2)
    ax_top = fig.add_subplot(gs[0, 0:])
    ax_sea = fig.add_subplot(gs[1, 0])
    ax_res = fig.add_subplot(gs[1, 1], sharey=ax_sea)
    ax_acf = fig.add_subplot(gs[2, 0])
    ax_his = fig.add_subplot(gs[2, 1])

    var_name = 'measurement_transformed' if transformed else 'measurement'

    all_data = pd.concat([data_train[['year', var_name]],
                          data_test[['year', var_name]]])

    ax_top.plot(all_data[var_name])
    ax_top.plot(decomposition.trend)
    ax_top.set_xticks(*generate_ticks(all_data))
    ax_top.axvline(len(data_train), zorder=-1)
    ax_top.set_title(f'Trend of the measurements of {pollution} in Poland, averaged {granularity}')
    ax_top.set_ylabel('Transformed value')
    ax_top.legend([f'Measurements{" transformed " if transformed else ""}', 'Overall trend'])

    ax_sea.plot(decomposition.seasonal)

    ax_res.scatter(decomposition.resid.index, decomposition.resid)

    for ax, title in zip([ax_sea, ax_res], ['Seasonal trend of values', 'Residuals']):
        ax.axhline(zorder=-1)
        ax.set_xticks(*generate_ticks(data_train))
        ax.set_title(title)
        ax.set_ylabel('Value')

    pmd.utils.plot_acf(data_train[var_name], ax=ax_acf, show=False,
                       title='Autocorrelation function', lags=get_period_length(granularity))
    ax_acf.set_xlabel(f'{get_period_label(granularity, uppercase=True)} lag')
    ax_acf.set_ylabel('Autocorrelation value')

    ax_his.hist(data_train[var_name], bins=25)
    ax_his.set_title('Frequency of specific values')
    ax_his.set_ylabel('Number of occurences')
    ax_his.set_xlabel(f'Value of {"transformed " if transformed else ""}measurement')

    fig.set_tight_layout(True)
