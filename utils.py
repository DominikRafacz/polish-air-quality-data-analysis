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
