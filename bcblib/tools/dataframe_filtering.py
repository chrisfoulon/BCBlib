from sklearn.preprocessing import RobustScaler
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


def normalize_dataframe(df, method='min-max', scaling_factors=None):
    """
    Normalize a dataframe with numerical, datetime, and timedelta columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe to normalize.
    method : str, optional
        The normalization method to use. Options are:
          - 'zscore': Apply z-score normalization (mean=0, std=1).
          - 'min-max': Scale such that min=0 and max=1.
          - 'zero-max': Scale such that 0=0 and max=1.
          - 'robust': Scale using robust methods (median and quantiles).
    scaling_factors : dict or None, optional
        A dictionary of precomputed scaling factors per column. For example:
          - For 'zscore': {col: (mean, std)},
          - For 'min-max': {col: (min, max)},
          - For 'zero-max': {col: max},
          - For 'robust': {col: scaler}.
        If provided, these are used instead of computing new factors.
    
    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        A tuple containing:
          - A dataframe with normalized columns.
          - A dictionary of scaling factors computed or provided, keyed by column name.
    
    Raises
    ------
    ValueError
        If an invalid method is provided.
    
    Examples
    --------
    >>> norm_df, scales = normalize_dataframe(df, method='zscore')
    >>> norm_df, scales = normalize_dataframe(df, method='min-max')
    >>> norm_df, scales = normalize_dataframe(df, method='zero-max')
    >>> norm_df, scales = normalize_dataframe(df, method='robust')
    """
    method = method.lower()
    if method == 'none':
        return df, {}

    if method not in ['zscore', 'min-max', 'zero-max', 'robust']:
        raise ValueError("Invalid method. Choose 'zscore', 'min-max', 'zero-max', or 'robust'.")

    df_normalized = df.copy()
    computed_scaling = {} if scaling_factors is None else scaling_factors.copy()

    for col in df_normalized.columns:
        col_dtype = df_normalized[col].dtype

        # Process numerical columns.
        if np.issubdtype(col_dtype, np.number):
            values = df_normalized[col]
        # Process datetime columns by converting to seconds.
        elif np.issubdtype(col_dtype, np.datetime64):
            values = df_normalized[col].astype('int64') / 1e9
        # Process timedelta columns by converting to total seconds.
        elif np.issubdtype(col_dtype, np.timedelta64):
            values = df_normalized[col].dt.total_seconds()
        else:
            continue

        if col not in computed_scaling:
            if method == 'zscore':
                mean_val = values.mean()
                std_val = values.std()
                computed_scaling[col] = (mean_val, std_val)
            elif method == 'min-max':
                min_val = values.min()
                max_val = values.max()
                computed_scaling[col] = (min_val, max_val)
            elif method == 'zero-max':
                max_val = values.max()
                computed_scaling[col] = max_val
            elif method == 'robust':
                scaler = RobustScaler()
                # Reshape for 1D data to fit sklearn's expected format
                if values.ndim == 1:
                    values_reshaped = values.values.reshape(-1, 1)
                    scaler.fit(values_reshaped)
                else:
                    scaler.fit(values.values.reshape(-1, 1))
                computed_scaling[col] = scaler

        if method == 'zscore':
            mean_val, std_val = computed_scaling[col]
            if std_val == 0:
                df_normalized[col] = 0
            else:
                df_normalized[col] = (values - mean_val) / std_val
        elif method == 'min-max':
            min_val, max_val = computed_scaling[col]
            denom = max_val - min_val
            if denom == 0:
                df_normalized[col] = 0
            else:
                df_normalized[col] = (values - min_val) / denom
        elif method == 'zero-max':
            max_val = computed_scaling[col]
            if max_val == 0:
                df_normalized[col] = 0
            else:
                df_normalized[col] = values / max_val
        elif method == 'robust':
            scaler = computed_scaling[col]
            if values.ndim == 1:
                values_reshaped = values.values.reshape(-1, 1)
                scaled_values = scaler.transform(values_reshaped).flatten()
            else:
                scaled_values = scaler.transform(values.values.reshape(-1, 1)).flatten()
            df_normalized[col] = scaled_values

    return df_normalized, computed_scaling


def inverse_normalize_dataframe(df_normalized, scaling_factors, method='min-max'):
    """
    Reverse the normalization applied to a dataframe.
    
    Parameters
    ----------
    df_normalized : pandas.DataFrame
        The normalized dataframe to restore to original scale.
    scaling_factors : dict
        The scaling factors dictionary returned by normalize_dataframe.
    method : str, optional
        The normalization method that was used. Options are:
          - 'zscore': Z-score normalization.
          - 'min-max': Min-max scaling.
          - 'zero-max': Zero-max scaling.
          - 'robust': Robust scaling.
    
    Returns
    -------
    pandas.DataFrame
        A dataframe with the normalization reversed.
    
    Raises
    ------
    ValueError
        If an invalid method is provided.
    
    Examples
    --------
    >>> original_df = inverse_normalize_dataframe(norm_df, scales, method='zscore')
    >>> original_df = inverse_normalize_dataframe(norm_df, scales, method='min-max')
    >>> original_df = inverse_normalize_dataframe(norm_df, scales, method='zero-max')
    >>> original_df = inverse_normalize_dataframe(norm_df, scales, method='robust')
    """
    method = method.lower()
    if method == 'none':
        return df_normalized

    if method not in ['zscore', 'min-max', 'zero-max', 'robust']:
        raise ValueError("Invalid method. Choose 'zscore', 'min-max', 'zero-max', or 'robust'.")
    
    df_original = df_normalized.copy()
    
    for col in df_normalized.columns:
        if col not in scaling_factors:
            continue

        values = df_normalized[col]
        
        if method == 'zscore':
            mean_val, std_val = scaling_factors[col]
            df_original[col] = values * std_val + mean_val
        elif method == 'min-max':
            min_val, max_val = scaling_factors[col]
            df_original[col] = values * (max_val - min_val) + min_val
        elif method == 'zero-max':
            max_val = scaling_factors[col]
            df_original[col] = values * max_val
        elif method == 'robust':
            scaler = scaling_factors[col]
            if values.ndim == 1:
                values_reshaped = values.values.reshape(-1, 1)
                original_values = scaler.inverse_transform(values_reshaped).flatten()
            else:
                original_values = scaler.inverse_transform(values.values.reshape(-1, 1)).flatten()
            df_original[col] = original_values
            
    return df_original

