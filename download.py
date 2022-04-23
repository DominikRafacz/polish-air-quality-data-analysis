import requests
import zipfile
import os

files_ids = {
    '2000.zip': 223,
    '2001.zip': 224,
    '2002.zip': 225,
    '2003.zip': 226,
    '2004.zip': 202,
    '2005.zip': 203,
    '2006.zip': 227,
    '2007.zip': 228,
    '2008.zip': 229,
    '2009.zip': 230,
    '2010.zip': 231,
    '2011.zip': 232,
    '2012.zip': 233,
    '2013.zip': 234,
    '2014.zip': 302,
    '2015.zip': 236,
    '2016.zip': 242,
    '2017.zip': 262,
    '2018.zip': 303,
    '2019.zip': 322,
    '2020.zip': 424,
    '2021.xlsx': 442,
    'metadata.xlsx': 405
}

basic_url = 'https://powietrze.gios.gov.pl/pjp/archives/downloadFile/'


def unzip(filename):
    with zipfile.ZipFile(filename) as zipped:
        zipped.extractall(filename.split('.')[0])


def download_all():
    for file_name, file_id in files_ids.items():
        print(f'Downloading: {file_name}')
        req = requests.get(basic_url + str(file_id))
        with open(f'data/{file_name}', 'wb') as file:
            file.write(req.content)

        if file_id not in (405, 442):
            unzip(f'data/{file_name}')
            os.remove(f'data/{file_name}')


if __name__ == '__main__':
    download_all()
