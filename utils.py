import pandas as pd


def remove_redundant(x: list):
    return list(set(x))


def construct_file_name(year: int, pollutant: str, exposition: int):
    if year == 2021:
        return 'data/2021.xlsx'
    else:
        return f'data/{year}/{year}_{pollutant}_{exposition}g.xlsx'


def get_period_length(granularity: str) -> int:
    if granularity == 'daily':
        return 365
    elif granularity == 'weekly':
        return 52
    elif granularity == 'monthly':
        return 12


def generate_ticks(data: pd.DataFrame, granularity: str) -> ([int], [int]):
    start_year = data.year.min()
    end_year = data.year.max()
    period_len = get_period_length(granularity)
    ticks = [period_len * i for i in range(0, end_year - start_year + 2)]
    tick_labels = list(range(start_year, end_year + 2))
    return ticks, tick_labels
