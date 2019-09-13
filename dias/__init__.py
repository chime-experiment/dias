"""dias, a data integrity analysis system."""
from .exception import DiasException, DiasUsageError, DiasConfigError
from .exception import DiasConcurrencyError
from .analyzer import Analyzer
from .chime_analyzer import CHIMEAnalyzer
from .task import Task
from .task_queue import TaskQueue
from .config_loader import ConfigLoader
from .job import Job
from .scheduler import Scheduler, stop_scheduler

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
