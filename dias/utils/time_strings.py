# Helper functions to convert between string and datetime as well as timedelta
# ----------------------------------------------------------------------------

import re
from datetime import timedelta,datetime


TIMEDELTA_REGEX = re.compile(r'((?P<hours>\d+?)h)?((?P<minutes>\d+?)m)?((?P<seconds>\d+?)s)?')
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def str2timedelta(time_str):
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
