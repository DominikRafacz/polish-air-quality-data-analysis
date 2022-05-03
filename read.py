import pandas as pd
import geopandas as gpd

from validation import *
from utils import construct_file_name


def _construct_2021_sheet_name(pollutant, exposition):
    if pollutant == 'SO2' and exposition == 24:   #typo case
        return '2021_SO2_24_H'
    else:
        return f'2021_{pollutant}_{exposition}H'


def _match_read_function(year: int, pollutant: str, exposition: int):
    skip_rows = [1, 2] if year <= 2015 else [0, 2, 3, 4, 5]
    decimal = ',' if 2016 <= year <= 2018 else '.'
    sheet_name = 0 if year < 2021 else _construct_2021_sheet_name(pollutant, exposition)
    return lambda io: pd.read_excel(io=io, sheet_name=sheet_name, skiprows=skip_rows, decimal=decimal)


def _read_normalized_data(year: int, pollutant: str, exposition: int, on_nonexistent: str):
    file_name = construct_file_name(year, pollutant, exposition)
    if handle_file_existence(file_name, on_nonexistent) is not None:
        return None
    read_excel_function = _match_read_function(year, pollutant, exposition)

    data = read_excel_function(io=file_name)
    data.columns.values[0] = 'timestamp'
    data = data.melt(id_vars=['timestamp'], var_name='station_code', value_name='measurement')

    data['pollutant'] = pollutant
    data['exposition'] = exposition

    return data


def read_normalized_data(year: int, pollutant: str, exposition: int, on_nonexistent: str = 'error'):
    validate_year(year)
    validate_pollutant(pollutant)
    validate_exposition(exposition)
    return _read_normalized_data(year, pollutant, exposition, on_nonexistent)


def read_stations_metadata():
    data = pd.read_excel(io='data/metadata.xlsx', sheet_name=0,
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


def _match_station_region(measurements_data: pd.DataFrame, stations_metadata: pd.DataFrame):
    stations_metadata = stations_metadata[['station_code', 'region']]
    return measurements_data.merge(stations_metadata, on='station_code')


def _match_station_region_legacy(measurements_data: pd.DataFrame, stations_metadata: pd.DataFrame):
    stations_metadata = stations_metadata[['station_code_legacy', 'region']]
    stations_metadata = stations_metadata.rename(columns={'station_code_legacy': 'station_code'})
    return measurements_data.merge(stations_metadata, on='station_code')


def match_station_region(measurements_data: pd.DataFrame, stations_metadata: pd.DataFrame):
    matched = _match_station_region(measurements_data, stations_metadata)
    matched_legacy = _match_station_region_legacy(measurements_data, stations_metadata)
    ret = pd.concat([matched, matched_legacy])
    return ret[~ret.duplicated()]


def _read_and_filter_multiple_datafiles(years, pollutants, expositions, regions, stations_metadata, on_nonexistent):
    ret = []
    for year in years:
        for pollutant in pollutants:
            for exposition in expositions:
                data = read_normalized_data(year, pollutant, exposition, on_nonexistent)
                if data is None:
                    continue
                data = match_station_region(data, stations_metadata)
                if regions is not None:
                    data = data[data['region'].isin(regions)]
                ret.append(data)
    return ret


def query_data_range(years: int | list[int], pollutants: str | list[str],
                     expositions: int | list[int], regions: str | list[str] | None = None,
                     on_nonexistent: str = 'message'):
    years = validate_and_wrap_multiple(years, validate_year, 'year', int)
    pollutants = validate_and_wrap_multiple(pollutants, validate_pollutant, 'pollutant', str)
    expositions = validate_and_wrap_multiple(expositions, validate_exposition, 'exposition', int)
    regions = validate_and_wrap_multiple(regions, validate_region, 'region', str, allow_none=True)

    stations_metadata = read_stations_metadata()
    full_data = _read_and_filter_multiple_datafiles(years, pollutants, expositions,
                                                    regions, stations_metadata, on_nonexistent)
    return pd.concat(full_data).reset_index(drop=True)


def get_geo_data():
    geo_data = gpd.read_file('data/Wojew¢dztwa.shp')
    geo_data['region'] = ['SLASKIE', 'OPOLSKIE', 'SWIETOKRZYSKIE', 'POMORSKIE', 'PODLASKIE', 'ZACHODNIOPOMORSKIE',
                          'DOLNOSLASKIE', 'WIELKOPOLSKIE', 'PODKARPACKIE', 'MALOPOLSKIE', 'WARMINSKO-MAZURSKIE',
                          'LODZKIE', 'MAZOWIECKIE', 'KUJAWSKO-POMORSKIE', 'LUBELSKIE', 'LUBUSKIE']
    geo_data = geo_data[['geometry', 'region']]
    return geo_data


def get_mobility_data():
    df = pd.read_csv("data/mobility_trend_2022_04_10.csv")
    df = df[df.region == "Poland"].T.iloc[6:] - 100  # set the 100 as the baseline, and convert data relative to it
    df = df.reset_index()  # reset index
    df.columns = ["date", "mobility_driving", "mobility_walking"]  # rename columns
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df.date.dt.day
    df['month'] = df.date.dt.month
    df['year'] = df.date.dt.year
    return df
