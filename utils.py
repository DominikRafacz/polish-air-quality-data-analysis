import pandas as pd


def remove_redundant(x: list):
    return list(set(x))


def construct_file_name(year: int, pollutant: str, exposition: int):
    if year == 2021:
        return 'data/2021.xlsx'
    else:
        return f'data/{year}/{year}_{pollutant}_{exposition}g.xlsx'


def get_period_length(granularity: str) -> int:
    match = {'daily': 365, 'weekly': 52, 'monthly': 12}
    return match[granularity]


def get_period_label(granularity: str, uppercase: bool = False) -> str:
    match = {'daily': 'Day', 'weekly': 'Week', 'monthly': 'Month'} if uppercase \
        else {'daily': 'day', 'weekly': 'week', 'monthly': 'month'}
    return match[granularity]


def generate_ticks(data: pd.DataFrame) -> ([int], [int]):
    start_year = data.year.min()
    end_year = data.year.max()
    ticks = [*data[~data['year'].duplicated()].index, len(data)]
    tick_labels = list(range(start_year, end_year + 2))
    return ticks, tick_labels


def unpack_to_dict(names, values):
    return {name: value for name, value in zip(names, values)}
