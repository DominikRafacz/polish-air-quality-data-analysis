import matplotlib.pyplot as plt
from functools import wraps

# source: https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle
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
        with plt.style.context(theme), plt.rc_context(rc_params):
            return plotting_func(*args, **kwargs)
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
def plot_timeline_of_pollution(data, pollution, granularity):
    fig, ax = plt.subplots()

    ax.plot(data.measurement)

    ax.set_title(f'{pollution} pollution in Poland averaged {granularity}')
    ax.set_xlabel('Time')
    ax.set_ylabel(f'Average concentration of {pollution} in the air [μg/m³]')

    ticks, tick_labels = generate_ticks(data)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)

    return fig, ax