"""Helper functions to convert to and from strings."""
from dias.exception import DiasException

import os
import re
from datetime import timedelta, datetime


TIMEDELTA_REGEX = re.compile(
    r"((?P<hours>\d+?)h)?((?P<minutes>\d+?)m)?((?P<seconds>\d+?)s)?"
)
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATA_UNITS = {"B": 1, "kB": 10**3, "MB": 10**6, "GB": 10**9}
DATA_UNITS_SORTED = ["B", "kB", "MB", "GB"]


def str2timedelta(time_str):
    """
    Convert a string to a timedelta.

    Parameters
    ----------
    time_str : str
        A string representing a timedelta in the form `<int>h`, `<int>m`,
        `<int>s` or a combination of the three.

    Returns
    -------
    :class:`datetime.timedelta`
        The converted timedelta.
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
    if not time_params:
        raise DiasException("Unable to parse time string: '{}'".format(time_str))
    return timedelta(**time_params)


def str2total_seconds(time_str):
    """
    Convert that describes a timedelta directly to seconds.

    Parameters
    ----------
    time_str : str
        A string representing a timedelta in the form `<int>h`, `<int>m`,
        `<int>s` or a combination of the three.

    Returns
    -------
    float
        Timedelta in seconds.
    """
    return str2timedelta(time_str).total_seconds()


def str2datetime(time_str):
    """
    Convert a string to :class:`datetime.datetime`.

    Parameters
    ----------
    time_str : str
        A string representing a datetime in the format `%Y-%m-%d %H:%M:%S`,
        where `%Y` is the year, `%m` is the month, `%d` is the day, `%H` is the
        hour, `%M` is the minute and `%S` is the second.

    Returns
    -------
    :class:`datetime:datetime`
        The converted datetime object.
    """
    return datetime.strptime(time_str, DATETIME_FORMAT)


def str2timestamp(time_str):
    """
    Convert a string describing a datetime directly to a timestamp.

    Parameters
    ----------
    time_str : str
        A string representing a datetime in the format `%Y-%m-%d %H:%M:%S`,
        where `%Y` is the year, `%m` is the month, `%d` is the day, `%H` is the
        hour, `%M` is the minute and `%S` is the second.

    Returns
    -------
    float
        POSIX timestamp.
    """
    return str2datetime(time_str).timestamp()


def datetime2str(dt):
    """
    Convert a datetime to a string.

    Parameters
    ----------
    dt : :class:`datetime.datetime`
        A datetime object.

    Returns
    -------
    str
        A string representing the datetime in the format `%Y-%m-%d %H:%M:%S`,
        where `%Y` is the year, `%m` is the month, `%d` is the day, `%H` is the
        hour, `%M` is the minute and `%S` is the second.
    """
    return dt.strftime(DATETIME_FORMAT)


def timestamp2str(ts):
    """
    Convert a timestamp to a string.

    Parameters
    ----------
    int
        POSIX timestamp.

    Returns
    -------
    str
        A string representing the timestamp in the format `%Y-%m-%d %H:%M:%S`,
        where `%Y` is the year, `%m` is the month, `%d` is the day, `%H` is the
        hour, `%M` is the minute and `%S` is the second.
    """
    return datetime.utcfromtimestamp(ts).strftime(DATETIME_FORMAT)


def str2bytes(size):
    """
    Convert a data size string to number of bytes.

    Note: I don't know if this works for floats.
    TODO: test and change the description.

    Parameters
    ----------
    size : str
        A string describing a data size in the format `<int> B`,
        `<int> kB`, `<int> MB` or `<int> GB`, where `B` is Bytes,
        `k` is kilo (`1e3`), `M` is Mega (`1e6`) and `G` is Giga (`1e9`).

    Returns
    -------
    int
        Number of bytes.
    """
    try:
        number, unit = [string.strip() for string in size.split()]
    except ValueError:
        raise ValueError(
            "Didn't understand data size: '{}' (use SI prefixes "
            "and a whitespace between number and unit).".format(size)
        )
    except AttributeError:
        raise ValueError(
            "Data size ('{}') should be of type string (is of"
            " type {})".format(size, type(size))
        )

    return int(float(number) * DATA_UNITS[unit])


def bytes2str(num):
    """
    Convert number of bytes to a data size string.

    Parameters
    ----------
    num : float
        Number of bytes

    Returns
    -------
    str
        A string describing a data size in the format `<float> B`,
        `<float> kB`, `<float> MB` or `<float> GB`, where `B` is Bytes,
        `k` is kilo (`1e3`), `M` is Mega (`1e6`) and `G` is Giga (`1e9`).
    """
    for unit in DATA_UNITS_SORTED:
        if abs(num) < 1000.0:
            return "%3.1f %s" % (num, unit)
        num /= 1000.0
    return "%.1f %s" % (num, "TB")


def str2path(s):
    """
    Perform shell-expansion on a string.

    Parameters
    ----------
    s : str
        A path.

    Returns
    -------
    str
        The same path, but `~` or `~user` are replaced by the user's home
        directory and substrings of the form `$name` or `${name}` are replaced
        by the value of environment variable name. Malformed variable names and
        references to non-existing variables are left unchanged.
    """
    return os.path.expandvars(os.path.expanduser(s))


def str2xpath(s):
    """
    Perform shell-expansion on a string, and check that the resultant path exists.

    Raises ValueError if the path doesn't exist.

    Parameters
    ----------
    s : str
        A path.

    Returns
    -------
    str
        The same path, but `~` or `~user` are replaced by the user's home
        directory and substrings of the form `$name` or `${name}` are replaced
        by the value of environment variable name. Malformed variable names and
        references to non-existing variables are left unchanged.
    """
    path = str2path(s)

    if not os.path.exists(path):
        raise ValueError("Path {} does not exist.".format(path))
    return path
