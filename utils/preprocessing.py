import logging

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import matplotlib.pyplot as plt
from scipy import interpolate
from typing import Callable, Union


from .frequency import parse_freq, convert_freq_to_seconds, calc_freq_num


logger = logging.getLogger(__name__)

class InterpolationError(Exception):
    '''
    Should be raised when during interpolation numbers of not NaN entries less than 2.
    '''
    def __init__(self, message):
        self.message = message


def resample_df(
        df: pd.DataFrame, 
        freq=None, 
        method: Callable = None, 
        origin: str = 'epoch', 
        label='left',
        offset: Union[str, bool] = 'auto', 
        scale=None, 
        return_freq=True, 
        return_offset=False
) -> pd.DataFrame:
    '''
    Returns resampled pd.DataFrame with specified frequency or with calculated initial data frequency.

    Parameters
    ----------
    df : pd.DataFrame
        fields: 'timestamp', 'target'
        initial data can be regular or irregular time series

    freq : str, default None
        frequency to resample if given

    method : Callable, default None
        should return aggregated data. If None will be applied function that retern mean value.

    origin : str, default 'epoch'
        The origin of the time grid. Used if (offset == 'auto' and label != 'right').
        One of list ['epoch', 'start_day']. If 'epoch' the origin of the time grid timestamp equal '01.01.1970 00:00:00',
        if 'start_day' equal first day in train dataset and time '00:00:00'.

    label : str, default 'left'
        On which border of the window ['left', 'right'] to form grid timestamps. If 'right' offset must be False.

    offset : Union[str, bool], default 'auto'
        If True offset equal to rounded half of the frequency (freq or calculated frequency) value will be added
        into timestamp. Only works when label is "left". If 'auto' will be automatically selected
        offset will be added or not

    scale : Union[int, float], default None, optional
        multiply freq value or calculated frequency value by scale.

    return_freq : bool, default True
        if True freq or calculated freq will be return.

    return_offset : bool, default False
        if True offset will be return.

    Returns
    -------
    resampled_ts : pd.DataFrame
        fields: 'timestamp', 'target'

    freq : str, optional
        freq with specified unit or calculated frequency in seconds. If return_freq is True.

    offset : bool, optional
        resampled data timestamp offset. If return_offset is True.

    '''
    origin_ts = df.copy()
    
    if freq is not None:
        freq = convert_freq_to_seconds(freq)
        freq_num, _ = parse_freq(freq)
    else:
        freq_num = calc_freq_num(origin_ts)
        freq = f'{freq_num}s'

    if scale is not None:
        freq_num *= scale
        freq = f'{freq_num}s'

    if (not isinstance(offset, bool)) and (offset != 'auto'):
        raise ValueError(f'The offset should be boolean or equal "auto", but not {offset}')

    if (offset == 'auto') and (label != 'right'):
        df = origin_ts.copy()
        try:
            df['timestamp'] = df['timestamp'].dt.tz_convert(None)
        except:
            pass

        if origin == 'epoch':
            first_date = pd.to_datetime(f"1970-01-01 00:00:00")
        elif origin == 'start_day':
            first_date = df['timestamp'].iloc[0]
        else:  # TODO for other origins (example datetime, timestamp origin)
            raise ValueError('origin must be "epoch" or "start_day" if offset == "auto"')

        df['timedelta'] = df['timestamp'] - pd.to_datetime(
            f"{first_date.year}-{first_date.month}-{first_date.day} 00:00:00")

        if (df.timedelta % to_offset(freq)).dt.total_seconds().mean() > freq_num / 2:
            offset = True
        else:
            offset = False

    if method is None:
        resampled_ts = origin_ts.groupby(
            pd.Grouper(key='timestamp', freq=freq, origin=origin, label=label)).mean()
    else:
        resampled_ts = origin_ts.groupby(
            pd.Grouper(key='timestamp', freq=freq, origin=origin, label=label)).apply(method)

    if label == 'right':
        offset = False
    if offset:
        loffset_num = round(freq_num / 2)
        loffset = f'{loffset_num}s'
        resampled_ts.index = resampled_ts.index + to_offset(loffset)

    resampled_ts = resampled_ts.reset_index()
    
    #print('df.timedelta % to_offset(freq)).dt.total_seconds().mean() = ', (df.timedelta % to_offset(freq)).dt.total_seconds().mean())
    #print('freq_num / 2', freq_num / 2)
    
    #print('df 0 = ')
    #display(df)
    #print('resampled_ts 0 = ')
    #display(resampled_ts)
    
    #print('to_offset(loffset) = ', to_offset(loffset))

    if return_offset:
        return resampled_ts, freq, offset
    if return_freq:
        return resampled_ts, freq

    return resampled_ts


def interpolate_scipy(df: pd.DataFrame, fill_value=None) -> pd.DataFrame:
    '''
    Returns pd.DataFrame after inplace fill NaN values with scipy.interpolate.interp1d method

    Note: changes the dataframe inplace

    Parameters
    ----------
    df : pd.DataFrame
        fields: 'timestamp', 'target'

        dataframe to be filled with interpolation and extrapolation

    Returns
    -------
    df : pd.DataFrame
        fields: 'timestamp', 'target'

    Raises
    ------
    InterpolationError:
        If numbers of not NaN entries less than 2
    '''
    timezone = df['timestamp'].dt.tz
    df_copy = df[['timestamp', 'target']].dropna()

    if df_copy.shape[0] < 2:
        raise InterpolationError('Numbers of not NaN entries should be greater than 2')
    if timezone is not None:
        df_copy['timestamp'] = df_copy['timestamp'].dt.tz_convert(None)

    df_copy['total_seconds'] = (df_copy['timestamp'] - pd.to_datetime('1970-01-01')).dt.total_seconds()

    if fill_value == 'extrapolate':
        timestamp_to_fill = df[df['target'].isna()].timestamp
    else:
        first_notna_time = df[df['target'].notna()]['timestamp'].min()
        last_notna_time = df[df['target'].notna()]['timestamp'].max()
        fill_mask = ((df['timestamp'] > first_notna_time) & (df['timestamp'] < last_notna_time))
        timestamp_to_fill = df[fill_mask & df['target'].isna()].timestamp

    interp_func = interpolate.interp1d(x=df_copy['total_seconds'], y=df_copy['target'],
                                       fill_value=fill_value)
    if timezone is not None:
        seconds_to_fill = (timestamp_to_fill.dt.tz_convert(None) - pd.to_datetime('1970-01-01')).dt.total_seconds()
    else:
        seconds_to_fill = (timestamp_to_fill - pd.to_datetime('1970-01-01')).dt.total_seconds()

    values_to_fill = interp_func(seconds_to_fill)
    df.loc[df['timestamp'].isin(timestamp_to_fill), 'target'] = values_to_fill

    return df


def r2_modified(y_true: pd.Series, y_pred: pd.Series, y_base: pd.Series, eps: float = 1e-10, plot_chart=False):
    '''
    Modified coefficient of determination metric.

    Parameters
    ----------
    y_true:
        pd.Series of shape (n_samples,).
        Correct target values.

    y_pred:
        pd.Series of shape (n_samples,).
        Estimated target values.

    y_base:
        pd.Series of shape (n_samples,).
        "Baseline" estimated target values.

    eps:
        The value added to the divisor to avoid division by zero.

    plot_chart:
        if True then charts are drawn

    Returns
    -------
    r2_modified : float
    '''
    if y_true.shape[0] != y_base.shape[0]:
        raise ValueError('y_true and y_base should be same size')
    if not (y_true.index == y_base.index).all():
        raise ValueError('y_true and upsampled_df.target_init should be same timestamps')

    r2_modified = np.sum((y_true.values - y_pred.values) ** 2) / (np.sum((y_true.values - y_base.values) ** 2) + eps)

    if plot_chart:
        plt.figure(figsize=(10, 7))

        plt.plot(y_base, label='base')
        plt.plot(y_true, label='origin')
        plt.plot(y_pred, label=f'prediction r2 = {1 - r2:0.3f}')

        plt.legend()
        plt.plot()

    return r2_modified


def r2_base_model(resampled_df_init: pd.DataFrame, init_freq: str):
    '''
    "Baseline" estimated target values calculate.

    Parameters
    ----------
    resampled_df_init : pd.DataFrame
        resampled dataframe with initial frequency init_freq.

    init_freq : str
        initial frequency of data.

    Returns
    -------
    y_base : pd.Series
        "Baseline" estimated target values
    '''
    df_true = resampled_df_init.dropna()
    history_depth = (df_true['timestamp'].max() - df_true['timestamp'].min()).total_seconds()

    if history_depth <= 86400:
        freq_base_num = round(history_depth / 2)
    else:
        freq_base_num = 86400
       
    upsampled_df = downsample_and_upsample_ts(freq_base_num, df_true, df_true, init_freq,
                                              offset=True, extrapolate=True)
    y_base = upsampled_df.target
    return y_base


def downsample_and_upsample_ts(
    freq_num: int, df_init: pd.DataFrame,
    resampled_df_init: pd.DataFrame, 
    freq_init: str,
    offset: Union[bool, str] = True, 
    extrapolate=False
) -> pd.DataFrame:

    '''
    Apply downsample and upsample operations successively to timeseries than merge with identity resampled timeseries.

    Parameters
    ----------
    freq_num : int
        frequency to data downsampling.

    df_init : pd.DataFrame
        initial data for downsampling.

    resampled_df_init : pd.DataFrame
        identity resampled data with freq_init.

    freq_init : str
        initial frequency of data.

    offset : Union[bool, str]
        offset for downsample and upsample operations. Default True.
    extrapolate:
        If True then points outside the data range will be extrapolated.

    Returns
    -------
    upsampled_df : pd.DataFrame
        fields: 'target', 'target_init'
    '''
    freq_num = int(np.ceil(freq_num))
    freq = f'{freq_num}s'
    downsampled_df = resample_df(df_init, freq, return_freq=False, offset=offset)
    upsampled_df = resample_df(downsampled_df, freq_init, return_freq=False, offset=offset)
    upsampled_df = pd.concat([
        upsampled_df.set_index('timestamp'),
        resampled_df_init.rename(columns={'target': 'target_init'}).set_index('timestamp')
    ], axis=1).reset_index()
    
    # вариант с pd.merge() дает такой же результат,
    # на дистанции может оказаться надежнее, но точно медленнее, чем pd.concat
    # upsampled_df = pd.merge(upsampled_df, resampled_df_init, on='timestamp', how='outer', suffixes=('', '_init'))

    upsampled_df = upsampled_df.sort_values('timestamp')
    if upsampled_df['target'].isna().sum() > 0:
        if extrapolate:
            fill_value = 'extrapolate'
        else:
            fill_value = None
        upsampled_df = interpolate_scipy(upsampled_df, fill_value=fill_value)

    upsampled_df = upsampled_df.set_index('timestamp').dropna()
    return upsampled_df


def binary_search_r2(
    grid, 
    threshold, 
    metric_df, 
    resampled_df_init, 
    loss_func, 
    offset_init, 
    iters=100,
    return_calcs=False, 
    verbose=True
):
    '''
    Binary search algorithm for determining the best downsampling frequency that satisfies
    a quality (result of loss function) no worse than the threshold.

    Parameters
    ----------
    grid : array-like, List
        grid of the frequencies to binary search.

    threshold : float
        loss function threshold for binary search in range [0, threshold].

    metric_df : pd.DataFrame
        initial data for apply downsampling and upsampling.

    resampled_df_init : pd.DataFrame
        (можно реализовать внутри функции с контролированием offset-а)
        identity resampled data with initial frequency of data.

    loss_func : Callable
        a function that returns the error between the identity resampled data with initial frequency
        and the downsample-upsampled data.

    offset_init : bool
        offset of identity resampled data with initial frequency.

    iters : int
        maximum binary search iterations number.

    return_calcs : bool, default False
        returns calculation results.

    verbose : bool, default True
        prints iterations logs.

    Returns
    -------
    freq : int
        binary search result

    offset : bool
        offset of downsampled data with freq

    calc_result : List, optional
        a calculation results
    '''
    offset = True
    calc_result = []
    freq_init_num = grid[0]
    freq_init = f'{freq_init_num}s'
    grid = np.array(grid)
    grid = grid[
        grid <= (resampled_df_init['timestamp'].max() - resampled_df_init['timestamp'].min()).total_seconds() / 2]

    if grid.shape[0] == 0:
        return

    first = 0
    last = len(grid) - 1
    loss = 0
    base_model_preds = r2_base_model(resampled_df_init, init_freq=freq_init)

    i = 0
    while (first <= last) and (i < iters):
        ind = first + (last - first) // 2
        if ind == 0:
            offset = offset_init

        freq_iter = grid[ind]
        upsampled_df = downsample_and_upsample_ts(freq_iter, metric_df, resampled_df_init, freq_init,
                                                  offset=offset, extrapolate=True)
        loss_iter = loss_func(upsampled_df.target_init, upsampled_df.target, base_model_preds)

        if verbose:
            logger.info('iter: %d, ind: %d, first: %d, last: %d, freq_iter: %d, loss_iter: %d',
                        i, ind, first, last, freq_iter, loss_iter)
        calc_result.append([freq_iter, loss_iter])
        if (loss_iter >= loss) and (round(loss_iter, 5) <= threshold):
            loss = loss_iter
            freq = freq_iter
            first = ind + 1
        else:
            last = ind - 1
        i += 1

    if return_calcs:
        return freq, offset, calc_result

    return freq, offset


def freq_num_grid_geomspace(min_freq_num: int, max_freq_num: int, n_nodes: int = 100) -> np.ndarray:
    '''
    Returns a grid of frequencies with n_nodes according to a geometric progression

    Parameters
    ----------
    min_freq_num : int
        Minimum value of the grid

    max_freq_num : int
        Maximum value of the grid

    n_nodes : int, default 100
        Number of nodes in the grid

        Note: The parameter n_nodes represents the size of the output grid and should be greater than 1
              since geometric progression needs more than one value to compute the grid.

    Returns
    -------
    grid : np.ndarray
    '''
    grid = np.geomspace(min_freq_num, max_freq_num, num=n_nodes, endpoint=True)
    grid = list(map(lambda x: np.ceil(x / 10) * 10 if x > 60 else x, grid))
    grid = np.unique(np.array([min_freq_num, ] + grid).astype(int))
    return grid


def freq_num_grid(min_freq_num: int, max_freq_num: int, n_nodes: int = 100) -> np.ndarray:
    '''
    Returns a regular grid of frequencies with n_node

    Parameters
    ----------
    min_freq_num : int
        Minimum value of the grid

    max_freq_num : int
        Maximum value of the grid

    n_nodes : int, default 100
        Number of nodes in the grid

        Note: The parameter n_nodes represents the size of the output grid and should be greater than 1
              since geometric progression needs more than one value to compute the grid.

    Returns
    -------
    grid : np.array
    '''
    grid = np.linspace(min_freq_num, max_freq_num, n_nodes)
    grid = list(map(lambda x: np.ceil(x / 10) * 10 if x > 10 else x, grid))
    grid = np.unique(np.array([min_freq_num, ] + grid).astype(int))
    return grid


def grid_geom_space(min_val: int, max_val: int, size: int) -> np.ndarray:
    '''
    Returns a grid of frequencies with n_nodes <= size according to a geometric progression with ceil delta parameter

    Parameters
    ----------
    min_val : int
        Minimum value of the grid

    max_val : int
        Maximum value of the grid

    size : int
        Maximum size of the grid

        Note: The parameter size represents the maximum size of the output grid and should be greater than 1
              since geometric progression needs more than one value to compute the grid.

    Returns
    -------
    grid : np.ndarray

    Raises
    ------
    ValueError
        If size < 2
            and
        If max_val < min_val
    '''
    if size < 2:
        raise ValueError('The parameter size should be greater than 2 or equal')
    if max_val < min_val:
        raise ValueError('The max_val should be greater than min_val')
    d = (max_val / min_val) ** (1 / (size - 1))
    d = int(np.ceil(d))
    grid = np.unique([min_val * (d ** power) for power in range(size)])
    return grid[grid <= max_val]


def optimize_freq(
    metric_df: pd.DataFrame,
    min_history_size: int = 2 * 86400,
    loss_func: Callable = r2_modified,
    threshold: float = 0.001,
    max_freq_num: int = 86400,
    grid_func: Callable = freq_num_grid_geomspace,
    n_grid: int = 10000,
    verbose: bool = False
) -> str:
    '''
    Parameters
    ----------
    metric_df:
        Input time series dataframe
    min_history_size:
        A minimum history size.
    loss_func:
        Quality function for comparing two time series: original and resampled.
    threshold:
        The threshold value of the quality function below which the solution satisfies us,
        while the solution tends to the threshold value.
    max_freq_num:
        A expected maximum frequency.
    grid_func:
        Function which return frequency grid.
    n_grid:
        Size of frequency grid for search.
    verbose:
        Print logs if True.

    Returns
    -------
    freq_optim_min, str
        A minimum optimum frequency.
    '''
    history_size = round((metric_df['timestamp'].max() - metric_df['timestamp'].min()).total_seconds())

    if history_size < min_history_size:
        logger.warning('The history_size %d in freq optimizer is less than min_history_size %d',
                       history_size, min_history_size)
    history_size_actual = round((metric_df['timestamp'].max() - metric_df['timestamp'].min()).total_seconds())

    if history_size_actual < round(0.9 * history_size):
        logger.warn(f'''The actual history_size is less than expected history_size '
                      on {round(100 * history_size_actual / history_size)}%, 
                      actual history_size: {history_size_actual}, history_size: {history_size}''')

    init_resampled_df, init_freq, init_offset = resample_df(metric_df, return_offset=True)
    init_freq_num = parse_freq(init_freq)[0]

    # подбирает оптимальный freq только если freq < 86400 секунд
    if init_freq_num >= max_freq_num:
        return f'{init_freq_num}s'

    grid = grid_func(init_freq_num, max_freq_num, n_grid)

    optim_freq_num, optim_offset = binary_search_r2(
        grid,
        threshold,
        metric_df,
        init_resampled_df,
        loss_func,
        init_offset,
        return_calcs=False,
        verbose=verbose)
    
    freq_optim_min = f'{optim_freq_num}s'
    logger.info('init_freq_num: %s, optim_freq_min: %s', init_freq_num, freq_optim_min)

    return freq_optim_min


def get_offset_for_regular_ts(df, freq=None, return_freq=False):
    resampled, freq, offset = resample_df(df, freq, return_offset=True)
    if not (resampled.timestamp == df.timestamp).all():
        offset = bool(1 - int(offset))

    resampled, freq, offset = resample_df(df, freq, return_offset=True, offset=offset)
    if not (resampled.timestamp == df.timestamp).all():
        raise ValueError('Offset detected not correct')
    if return_freq:
        return offset, freq
    return offset


if __name__ == '__main__':
    from db import load_metric_df

    metric_id = 'krm_office_temp'
    retention_policy = 'INFINITE'
    metric_df = load_metric_df(metric_id, retention_policy, history_size=86400*7)
    min_optim_freq = optimize_freq(metric_df)
    resampled_df = resample_df(metric_df, freq=min_optim_freq, return_freq=False)

    plt.plot(metric_df.timestamp, metric_df.target, label='origin')
    plt.plot(resampled_df.timestamp, resampled_df.target, label=f'resampled freq = {min_optim_freq}')
    plt.legend()
    plt.show()
