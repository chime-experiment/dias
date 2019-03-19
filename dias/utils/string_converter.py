# Helper functions to convert to and from strings.
# ----------------------------------------------------------------------------

import os
import re
from datetime import timedelta, datetime


TIMEDELTA_REGEX = re.compile(
        r'((?P<hours>\d+?)h)?((?P<minutes>\d+?)m)?((?P<seconds>\d+?)s)?')
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
DATA_UNITS = {"B": 1, "kB": 10**3, "MB": 10**6, "GB": 10**9}
DATA_UNITS_SORTED = ["B", "kB", "MB", "GB"]


def str2timedelta(time_str):
    """
    Convert a string to a timedelta.

    Parameters
    ----------
    time_str : String
        A string representing a timedelta in the form `<int>h`, `<int>m`,
        `<int>s` or a combination of the three.

    Returns
    -------
    An instance of datetime.timedelta.

    """
    # Check for simple numeric seconds
    try:
        seconds = int(time_str)
        return timedelta(seconds=seconds)
    except ValueError:
        pass

    # Otherwise parse time
    parts = TIMEDELTA_REGEX.match(time_str)
    if not parts:
        return
    parts = parts.groupdict()
    time_params = {}
    for (name, param) in parts.items():
        if param:
            time_params[name] = int(param)
    return timedelta(**time_params)


def str2total_seconds(time_str):
    return str2timedelta(time_str).total_seconds()


def str2datetime(time_str):
    return datetime.strptime(time_str, DATETIME_FORMAT)


def str2timestamp(time_str):
    return str2datetime(time_str).timestamp()


def datetime2str(dt):
    return dt.strftime(DATETIME_FORMAT)


def timestamp2str(ts):
    return datetime.utcfromtimestamp(ts).strftime(DATETIME_FORMAT)


def str2bytes(size):
    try:
        number, unit = [string.strip() for string in size.split()]
    except ValueError:
        raise ValueError("Didn't understand data size: '{}' (use SI prefixes "
                         "and a whitespace between number and unit)."
                         .format(size))
    except AttributeError:
        raise ValueError("Data size ('{}') should be of type string (is of"
                         " type {})".format(size, type(size)))

    return int(float(number) * DATA_UNITS[unit])


def bytes2str(num):
    for unit in DATA_UNITS_SORTED:
        if abs(num) < 1000.0:
            return "%3.1f %s" % (num, unit)
        num /= 1000.0
    return "%.1f %s" % (num, 'TB')


# This performs shell-expansion on a string
def str2path(s):
    return os.path.expandvars(os.path.expanduser(s))
