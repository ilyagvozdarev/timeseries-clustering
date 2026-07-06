import re
import string
import pandas as pd


def parse_freq(freq: str):
    '''
    Returns frequency value and frequency unit after freq parsing

    Parameters
    ----------
    freq : str
        data frequency

    Returns
    -------
    freq_num : int
        frequency value
    freq_str : str
        frequency unit
    '''
    freq_num_str = re.findall(r'\d+', freq)
    if len(freq_num_str) > 0:
        freq_num = int(freq_num_str[0])
        freq_str = freq.replace(freq_num_str[0], '')
    else:
        freq_num = 1
        freq_str = freq
    return freq_num, freq_str


def convert_freq_to_seconds(freq: str):
    '''
    Converts frequency unit to seconds as string

    Parameters
    ----------
    freq : str
        data frequency
    '''
    if freq is None:
        return
    freq_num, freq_str = parse_freq(freq)

    if freq_str == 'T':
        return f'{freq_num * 60}s'
    elif freq_str == 's':
        return f'{freq_num}s'
    elif freq_str == 'H':
        return f'{freq_num * 3600}s'
    elif freq_str == 'D':
        return f'{freq_num * 24 * 3600}s'
    elif freq_str == 'W':
        return f'{freq_num * 7 * 24 * 3600}'
    elif freq_str == 'M':
        return f'{int(freq_num * 365.25 * 2 * 3600)}s'
    elif freq_str == 'Y':
        return f'{int(freq_num * 365.25 * 24 * 3600)}s'
    raise ValueError(f'''Unknown data frequency: {freq}''')


def calc_freq_num(raw_data: pd.DataFrame) -> int:
    '''
    Returns the raw data frequency value in seconds, calculated as the median of the first time differences

    Parameters
    ----------
    raw_data : pd.DataFrame
        raw initial data with required fields: 'timestamp'

    Returns
    -------
    freq_num : int
        calculated raw data frequency value
    '''
    if not isinstance(raw_data, pd.DataFrame):
        raise TypeError('raw_data data must be a pd.DataFrame')
    if raw_data.shape[0] < 3:
        raise ValueError('DataFrame have at least 3 rows')
    
    freq_num = int(raw_data['timestamp'].dt.round('s').diff(1).median().total_seconds())

    if freq_num == 0:
        raise ValueError('freq_num must be not equal 0 seconds')
    
    return freq_num


def freq_format(freq: str) -> str:
    if not isinstance(freq, str):
        raise ValueError(f'freq should be str type given: {type(freq)}')
    if len(re.findall(rf'[{string.ascii_uppercase}]', freq.upper())) == 0:
        raise ValueError(f'''Unknown data frequency: {freq}''')

    scopes = ['year', 'month', 'week', 'day', 'hour', 'min', 'sec']
    freq_sec = {
        'sec': 1,
        'min': 60,
        'hour': 3600,
        'day': 86400,
        'week': 7 * 86400,
        'month': int(30.4375 * 86400),
        'year': int(365.25 * 86400)
    }
    freq_aliases = {
        'sec': 's',
        'min': 'min',
        'hour': 'H',
        'day': 'D',
        'week': 'W',
        'month': 'M',
        'year': 'Y'
    }
    freq_num, freq_str = parse_freq(convert_freq_to_seconds(freq))

    for scope in scopes:
        if freq_num % freq_sec[scope] == 0:
            value = freq_num // freq_sec[scope]
            if value == 1:
                return f'{freq_aliases[scope]}'
            else:
                return f'{value}{freq_aliases[scope]}'

    raise ValueError(f'''Unknown data frequency: {freq}''')

