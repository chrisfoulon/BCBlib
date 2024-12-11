from tqdm import tqdm
import pandas as pd
import numpy as np


def filter_invalid_columns(df):
    """
    Filter invalid columns from the dataframe:
    1. Remove constant columns.
    2. Convert boolean columns to integers.
    3. Remove object columns that cannot be converted to datetime or time-only.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.

    Returns
    -------
    pandas.DataFrame
        The filtered dataframe with valid columns only.
    """
    # Remove constant columns
    df = df.loc[:, df.nunique() > 1]

    # Convert boolean columns to integers
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)

    # Handle string/object columns
    columns_to_remove = []
    for col in df.select_dtypes(include=['object']).columns:
        try:
            # Attempt to parse as time-only
            if df[col].str.match(r'^\d{1,2}:\d{2}(:\d{2})?$').all():
                df[col] = pd.to_datetime(df[col], format='%H:%M:%S', errors='raise').dt.time
            else:
                # Attempt to parse as full datetime
                df[col] = pd.to_datetime(df[col], errors='raise')
        except (ValueError, TypeError):
            # If conversion fails, mark the column for removal
            columns_to_remove.append(col)

    # Remove columns that couldn't be converted
    df.drop(columns=columns_to_remove, inplace=True)

    return df


def select_columns_to_keep(df, min_threshold=90, min_rows=1000):
    """
    Select columns with 90%+ data while keeping at least 1,000 rows.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.
    min_threshold : int, optional
        The minimum percentage of non-null values to consider columns (default is 90).
    min_rows : int, optional
        The minimum number of rows to retain in the filtered dataset (default is 1,000).

    Returns
    -------
    pandas.DataFrame
        The filtered dataframe with selected columns.
    """
    # Filter columns based on non-null percentage
    columns_to_consider = df.columns[
        (df.notnull().mean() * 100 >= min_threshold) & (df.notnull().mean() * 100 < 100)
    ]

    # Extract columns with 100% coverage to add to the final dataset later
    always_include_columns = df.columns[df.notnull().mean() * 100 == 100]

    # Start with all columns in the consideration set
    current_columns = list(columns_to_consider)

    # Iteratively remove columns to maximize rows
    pbar = tqdm(total=len(current_columns), desc="Optimizing columns")
    while len(current_columns) > 1:
        best_removed_column = None
        max_rows_after_removal = 0

        for column in current_columns:
            # Test removing the current column
            temp_columns = [col for col in current_columns if col != column]
            temp_data = df[temp_columns].dropna()
            num_rows = len(temp_data)

            # Check if this removal improves the row count
            if num_rows > max_rows_after_removal:
                max_rows_after_removal = num_rows
                best_removed_column = column

        # Remove the column that results in the most rows and check stopping condition
        if best_removed_column:
            current_columns.remove(best_removed_column)
            if max_rows_after_removal >= min_rows:
                break  # Stop when the number of rows after removal meets the threshold
        else:
            break  # No further improvement possible

        pbar.update(1)

    pbar.close()

    # Combine the optimized columns with always included columns
    final_columns = current_columns + list(always_include_columns)
    return df[final_columns].dropna()


def normalize_dataframe(df, method='min-max'):
    """
    Normalize a dataframe with numerical columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe to normalize.
    method : str, optional
        The normalization method to use. Options are:
        - 'zscore': Apply z-score normalization (mean=0, std=1).
        - 'min-max': Scale such that min=0 and max=1.
        - 'zero-max': Scale such that 0=0 and max=1.

    Returns
    -------
    pandas.DataFrame
        A dataframe with normalized numerical columns.

    Raises
    ------
    ValueError
        If an invalid method is provided.

    Examples
    --------
    Normalize with z-score:

    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    >>> normalize_dataframe(df, method='zscore')

    Normalize with min-max:

    >>> normalize_dataframe(df, method='min-max')

    Normalize with zero-max:

    >>> normalize_dataframe(df, method='zero-max')
    """
    method = method.lower()
    if method == 'none':
        # Do nothing, just return the original DataFrame
        return df
    # Validate method
    if method not in ['zscore', 'min-max', 'zero-max']:
        raise ValueError("Invalid method. Choose 'zscore', 'min-max', or 'zero-max'.")

    df_normalized = df.copy()

    for col in df_normalized.select_dtypes(include=[np.number]).columns:
        if method == 'zscore':
            # Z-score normalization
            mean = df_normalized[col].mean()
            std = df_normalized[col].std()
            df_normalized[col] = (df_normalized[col] - mean) / std
        elif method == 'min-max':
            # Min-max normalization
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        elif method == 'zero-max':
            # Zero-max normalization
            max_val = df_normalized[col].max()
            df_normalized[col] = df_normalized[col] / max_val

    return df_normalized