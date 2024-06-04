import pandas as pd
import numpy as np


def Read(filename: str,
         skiprows: int = 7,
         header: int = 1,
         sep: str = ' ',
         shift_header_entries: int = 2,
         extract_columns=None):
    r"""
    Reads data in the default format of LIGGGHTS using pandas.read_csv

    Parameters
    ----------
    filename: str
        path of the file to read
    skiprows: int
        number of rows to skip before reading the header (same as pandas argument)
    header: int
        number of header rows to determine column names (same as pandas argument)
    sep: str
        delimiter of the data
    purge_header_entries: int
        number of header entries that need to be removed to align column name with data
    extract_columns
        allows the direct extraction of data into a tuple of numpy arrays, by default the whole
        data frame is returned

    Returns
    -------
    If the extract_column argument is set to None, the whole pandas dataframe is returned, otherwise
    a single numpy array or a tuple of numpy arrays is returned
    """
    data = pd.read_csv(filename, skiprows=skiprows, header=header, sep=sep)
    dummy_names = [f'ToDrop{n}' for n in range(shift_header_entries)]
    data.columns = data.columns[shift_header_entries:].append(pd.Index(dummy_names))
    data = data.drop(columns=dummy_names)

    if extract_columns is None:
        return data
    else:
        if isinstance(extract_columns, list):
            return (np.asarray(data[e]) for e in extract_columns)
        else:
            return np.asarray(data[extract_columns])

filename = '/home/drieder/Data/cylinder1000000.liggghts'
data = Read(filename=filename)

radius = np.asarray(data.radius)

r, x, y, z = Read(filename=filename, extract_columns=['radius', 'x', 'y', 'z'])
