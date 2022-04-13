def remove_redundant(x: list):
    return list(set(x))


def construct_file_name(year: int, pollutant: str, exposition: int):
    return f'data/{year}/{year}_{pollutant}_{exposition}g.xlsx'

