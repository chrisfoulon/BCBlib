import os
from pathlib import Path
from typing import Collection, Union, List

import pandas as pd


def str_to_column_id(column, df):
    if isinstance(column, str):
        if column not in df.columns:
            try:
                column = int(column)
            except ValueError:
                raise ValueError(f'{column} must either be column names or integers')
        if column not in df.columns:
            raise ValueError(f'{column} is not an existing column')
    return column


def nan_filtered_column(column, df):
    filtered_df = df[column].dropna()


def import_spreadsheet(spreadsheet: Union[pd.DataFrame, str, bytes, os.PathLike],
                       header: Union[int, List[int], None] = 0):
    """

    Parameters
    ----------
    spreadsheet :
        input dataframe or path to a xls(x) or csv file
    header :
        Only used when spreadsheet is a pathlike
        index or list of indices of the column names. None, in case the spreadsheet has no header.

    Returns
    -------

    """
    if isinstance(spreadsheet, pd.DataFrame):
        df = spreadsheet
    else:
        if not Path(spreadsheet).is_file():
            raise ValueError(f'{spreadsheet} is no an existing file path')
        if spreadsheet.endswith('.csv'):
            df = pd.read_csv(spreadsheet, header=header)
        else:
            df = pd.read_excel(spreadsheet, header=header)
    return df
