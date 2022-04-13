import pandas as pd

from validation import *
from utils import construct_file_name


def _read_normalized_data(year: int, pollutant: str, exposition: int):
    file_name = construct_file_name(year, pollutant, exposition)

    data = pd.read_excel(io=file_name, sheet_name=0, skiprows=[0, 2, 3, 4, 5]) \
        .rename(columns={'Kod stacji': 'timestamp'}) \
        .melt(id_vars=['timestamp'], var_name='station_code', value_name='measurement')

    data['pollutant'] = pollutant
    data['exposition'] = exposition

    return data


def read_normalized_data(year: int, pollutant: str, exposition: int):
    validate_year(year)
    validate_pollutant(pollutant)
    validate_exposition(exposition)
    return _read_normalized_data(year, pollutant, exposition)


def read_stations_metadata():
    data = pd.read_excel(io='data/Metadane - stacje i stanowiska pomiarowe.xlsx', sheet_name=0,
                         index_col=0, usecols='A,B,D:G,K,N,O')
    data.columns = ['station_code', 'station_name', 'station_code_legacy',
                    'active_since', 'active_until', 'region', 'latitude', 'longitude']
    data['region'] = data['region'].replace(
        to_replace=['DOLNOŚLĄSKIE', 'KUJAWSKO-POMORSKIE', 'LUBELSKIE', 'ŁÓDZKIE','LUBUSKIE', 'MAŁOPOLSKIE',
                    'MAZOWIECKIE', 'OPOLSKIE', 'PODLASKIE', 'PODKARPACKIE', 'POMORSKIE', 'ŚWIĘTOKRZYSKIE', 'ŚLĄSKIE',
                    'WARMIŃSKO-MAZURSKIE', 'WIELKOPOLSKIE', 'ZACHODNIOPOMORSKIE'],
        value=REGIONS
    )
    return data


def match_station_region(measurements_data: pd.DataFrame, stations_metadata: pd.DataFrame):
    stations_metadata = stations_metadata[['station_code', 'region']]
    return measurements_data.merge(stations_metadata, on='station_code')


def _read_and_filter_multiple_datafiles(years, pollutants, expositions, regions, stations_metadata):
    ret = []
    for year in years:
        for pollutant in pollutants:
            for exposition in expositions:
                data = read_normalized_data(year, pollutant, exposition)
                data = match_station_region(data, stations_metadata)
                if regions is not None:
                    data = data[data['region'].isin(regions)]
                ret.append(data)
    return ret


def query_data_range(years: int | list[int], pollutants: str | list[str],
                     expositions: int | list[int], regions: str | list[str] | None = None):
    years = validate_and_wrap_multiple(years, validate_year, 'year', int)
    pollutants = validate_and_wrap_multiple(pollutants, validate_pollutant, 'pollutant', str)
    expositions = validate_and_wrap_multiple(expositions, validate_exposition, 'exposition', int)
    regions = validate_and_wrap_multiple(regions, validate_region, 'region', str, allow_none=True)

    stations_metadata = read_stations_metadata()
    full_data = _read_and_filter_multiple_datafiles(years, pollutants, expositions, regions, stations_metadata)

    return pd.concat(full_data)
