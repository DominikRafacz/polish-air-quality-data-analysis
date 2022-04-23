from os import path

from utils import remove_redundant

POLLUTANTS = {
    'NO2', 'NOx', 'SO2', 'CO', 'O3', 'PM10', 'PM25'
}

REGIONS = [
    'DOLNOSLASKIE', 'KUJAWSKO-POMORSKIE', 'LUBELSKIE', 'LODZKIE',
    'LUBUSKIE', 'MALOPOLSKIE', 'MAZOWIECKIE', 'OPOLSKIE', 'PODLASKIE',
    'PODKARPACKIE', 'POMORSKIE', 'SWIETOKRZYSKIE', 'SLASKIE',
    'WARMINSKO-MAZURSKIE', 'WIELKOPOLSKIE', 'ZACHODNIOPOMORSKIE'
]


def validate_year(year: int):
    if year < 2000 or year > 2021:
        raise ValueError('"year" should be within range [2000, 2021]')


def validate_pollutant(pollutant: str):
    if pollutant not in POLLUTANTS:
        raise ValueError(f'"pollutant" should be one of {POLLUTANTS}')


def validate_exposition(exposition: int):
    if exposition not in [1, 24]:
        raise ValueError(f'"exposition" should be either 1 or 24')


def validate_region(region: str):
    if region not in REGIONS:
        raise ValueError(f'"region" should be one of {REGIONS}')


def validate_and_wrap_multiple(parameters, validation_function, parameter_name, expected_type, allow_none=False):
    if type(parameters) is list:
        for param in parameters:
            validation_function(param)
        return remove_redundant(parameters)
    elif type(parameters) is expected_type:
        validation_function(parameters)
        return [parameters]
    elif allow_none and parameters is None:
        return None
    raise TypeError(f'"{parameter_name}" has to be a list of values or a single value')


def handle_file_existence(file_name, on_nonexistent):
    if not path.exists(file_name):
        if on_nonexistent == 'error':
            raise ValueError(f'File "{file_name}" does not exists!')
        elif on_nonexistent == 'message':
            print(f'File "{file_name}" does not exists, skipping.')
            return 'skip'
    return None
