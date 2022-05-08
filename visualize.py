import matplotlib.pyplot as plt
import pandas as pd
import pmdarima as pmd

from utils import generate_ticks, get_period_label, get_period_length
from functools import wraps
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy import corrcoef

# source: https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle
DEFAULT_THEME = 'pitayasmoothie-dark.mplstyle'
DEFAULT_RC_PARAMS = {
    'figure.dpi': 300,
    'figure.figsize': (12, 9)
}


def get_theme_colors(theme=DEFAULT_THEME):
    with plt.style.context(theme):
        return plt.rcParams['axes.prop_cycle'].by_key()['color']


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
def plot_map_with_values(geo_data_with_values, values_column: str, title: str, **kwargs):
    """
    Function for plotting map with continuous values

    :param geo_data_with_values: GeoDataFrame with columns 'geometry' and column of name the same as 'values_column'
        parameter
    :param values_column: Name of the column containing values to plot
    :param title: Title of the plot
    :param kwargs: used by wrappers
    :return: tuple (figure object, axes object)
    """
    fig, ax = plt.subplots()

    geo_data_with_values.plot(column=values_column, legend=True, ax=ax)

    ax.set_title(title)

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
def plot_timeline_of_pollution_for_regions(regional_data, regions, focus_region_colors, pollution, **kwargs):
    fig, ax = plt.subplots()

    for region in regions:
        data = regional_data[region]
        data = data[data.year < 2020]
        ax.plot(data.measurement, color=focus_region_colors[region])

    ax.set_title(f'{pollution} pollution in Poland split by regions')
    ax.set_xlabel('Time')
    ax.set_xticks(*generate_ticks(data))
    ax.set_ylabel(f'Average concentration of {pollution} in the air [μg/m³]')

    plt.show()


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
    ax.set_ylabel(f'Average {pollution} in the air [μg/m³]')
    ax.set_xlabel(f'Time')
    ax.set_xticks(*generate_ticks(pd.concat([model_results['df_train'][['year']], model_results['df_test'][['year']]])))

    return fig, ax


@with_theme_and_params
def plot_historical_and_predictions_and_mobility(model_results, mobility_data, pollution, granularity, **kwargs):
    fig, axs = plt.subplots(3, 1, sharex=True)

    df_test_joined = model_results['df_test'].merge(mobility_data, on=['year', 'month'], how='left')

    colors = get_theme_colors()

    corr = {
        var: corrcoef(
        model_results['df_test'].measurement_transformed - model_results['df_test'].prediction_transformed,
        mobility_data[mobility_data.year.isin([2020, 2021])][var]
        )[0, 1] for var in ['mobility_driving', 'mobility_walking']
    }

    axs[0].plot(df_test_joined.measurement)
    axs[0].plot(df_test_joined.index, df_test_joined.prediction, linestyle='dashed')
    axs[0].fill_between(df_test_joined.index, df_test_joined.lower_confidence,
                        df_test_joined.upper_confidence, color='gray', alpha=0.2)
    axs[0].legend(['Historical data', '2020 and 2021 predictions', 'Confidence interval'])
    axs[0].set_title(f'Comparison of historical {granularity} data to trend prediction for {pollution} in Poland')
    axs[0].set_ylabel(f'{pollution} in the air [μg/m³]')

    axs[1].plot(df_test_joined.measurement_transformed - df_test_joined.prediction_transformed, color=colors[2])
    axs[1].axhline(zorder=-1)
    axs[1].set_title('Difference between measurement and predicted value')
    axs[1].legend(['Difference value'])

    axs[2].plot(df_test_joined.mobility_driving)
    axs[2].plot(df_test_joined.mobility_walking)
    axs[2].axhline(zorder=-1)
    axs[2].set_title('Mobility of people with Apple devices in Poland')
    axs[2].set_ylabel('Mobility relative to previous time periods [%]')
    axs[2].legend(['Driving', 'Walking'])

    axs[2].set_xlabel(f'Time')
    axs[2].set_xticks(range(0, 24, 3), ['2020 Jan', 'Apr', 'Jul', 'Oct', '2021 Jan', 'Apr', 'Jul', 'Oct'])

    axs[2].text(16, -35, f'Correlation between diff and driving {corr["mobility_driving"]:.2f}', color=colors[0])
    axs[2].text(16, -50, f'Correlation between diff and walking {corr["mobility_walking"]:.2f}', color=colors[1])

    return fig, axs


@with_theme_and_params
def plot_acf_and_frequency(data, pollution, granularity, transformed=False, **kwargs):
    fig = pmd.utils.tsdisplay(data.measurement_transformed if transformed else data.measurement,
                              show=False, lag_max=get_period_length(granularity))
    axs = fig.axes
    axs[0].set_title(
        f'Average {pollution} pollution in Poland averaged {granularity} {", transformed with Box-Cox" if transformed else ""}')
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
def plot_full_pre_model_analysis(data_train, data_test, decomposition, pollution, granularity, transformed: bool,
                                 **kwargs):
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
    return fig, gs


@with_theme_and_params
def plot_mobility_data(data, **kwargs):
    dates = ['2020-01-13', '2020-04-01', '2020-07-01', '2020-10-01',
             '2021-01-01', '2021-04-01', '2021-07-01', '2021-10-01',
             '2022-01-01', '2022-04-10']
    indices = [None] * 10

    for i, date in enumerate(dates):
        temp_df = data[data.date == date]
        indices[i] = temp_df.index[0]

    fig, ax = plt.subplots()
    ax.plot(data.mobility_driving)
    ax.plot(data.mobility_walking)

    ax.set_title('Relative mobility of Apple devices users in Poland')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mobility relative to previous time periods [%]')
    ax.set_xticks(indices)
    ax.set_xticklabels(['Jan 2020', 'Apr 2020', 'Jul 2020', 'Oct 2020', 'Jan 2021',
                        'Apr 2021', 'Jul 2021', 'Oct 2021', 'Jan 2022',
                        'Apr 2022'])
    ax.legend(['Driving', 'Walking'])
    return fig, ax


@with_theme_and_params
def plot_comparison_of_regions(model_results_dict, regions, region_labels, pollution, region_colors=None,
                               one_panel=False, data_start=2019, **kwargs):
    data_historical = dict()
    for region in regions:
        data = pd.concat([model_results_dict[region]['df_train'][['measurement', 'year']],
                          model_results_dict[region]['df_test'][['measurement', 'year']]])
        data_historical[region] = data[data.year >= data_start]

    data_test = {region: model_results_dict[region]['df_test'] for region in regions}

    if one_panel:
        fig, ax = plt.subplots(1, 1)

        for region in regions:
            ax.plot(data_historical[region].measurement, color=region_colors[region])
            ax.plot(data_test[region].prediction, linestyle='dashed', color=region_colors[region], alpha=0.5)

            ax.set_title(f'Comparison of {pollution} pollution historical data and predictions in different regions')
            ax.set_xticks(*generate_ticks(data_historical[region]))
            ax.legend(handles=[Line2D([], [], color='gray', marker='', label='Historical data'),
                               Line2D([], [], color='gray', marker='', label='Prediction', alpha=0.5,
                                      linestyle='dashed')])

        ax.set_xticks(*generate_ticks(data_historical[regions[0]]))

        return fig, ax
    else:
        fig, axs = plt.subplots(len(regions), 1, sharex=True)

        for ax, region in zip(axs, regions):
            if region_colors is None:
                ax.plot(data_historical[region].measurement)
                ax.plot(data_test[region].prediction, linestyle='dashed')
            else:
                ax.plot(data_historical[region].measurement, color=region_colors[region])
                ax.plot(data_test[region].prediction, linestyle='dashed', color=region_colors[region], alpha=0.5)
            ax.fill_between(data_test[region].index, data_test[region].lower_confidence,
                            data_test[region].upper_confidence, color='gray', alpha=0.2)
            ax.set_title(region_labels[region])
        axs[0].set_xticks(*generate_ticks(data_historical[regions[0]]))

        if region_colors is None:
            axs[0].legend(['Historical data', 'Prediction', 'Confidence intervals'])
        else:
            axs[0].legend(handles=[Line2D([], [], color='gray', marker='', label='Historical data'),
                                   Line2D([], [], color='gray', marker='', label='Prediction', alpha=0.5,
                                          linestyle='dashed'),
                                   Patch(edgecolor='gray', facecolor='gray', alpha=0.2, label='Confidence intervals')])

        fig.suptitle(f'Comparison of {pollution} pollution historical data and predictions in different regions')

        return fig, axs
