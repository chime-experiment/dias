# Analyzer Base Class
# -------------------

import logging
from caput import config
from dias.utils import str2timedelta, str2datetime
from prometheus_client import Gauge


class Analyzer(config.Reader):
    """Base class for all dias analyzers.
    All dias analyzers should inherit from this class, with functionality added
    by over-riding `setup`, `run` and/or `shutdown`.
    In addition, input parameters may be specified by adding class attributes
    which are instances of `config.Property`. These will then be read from the
    task config file when it is initialized.  The class attributes
    will be overridden with instance attributes with the same name but with the
    values specified in the config file.
    Attributes
    ----------
    Methods
    -------
    __init__
    setup
    run
    finish
    """

    # Config values
    start_time = config.Property(proptype=str2datetime)
    log_level = config.Property(proptype=logging.getLevelName)
    period = config.Property(proptype=str2timedelta)

    def __init__(self, name, write_dir, state_dir):
        """Constructor of analyzer base class.
        """
        self.name = name
        self.write_dir = write_dir
        self.state_dir = state_dir

    def init_logger(self, log_level_override=None):
        """Set up the logger. Call this after reading the config."""
        self.logger = logging.getLogger('dias[{0}]'.format(self.name))
        if log_level_override:
            self.log_level = log_level_override

        self.logger.setLevel(self.log_level)

    def add_task_metric(self, metric_name, description, labelnames=[], unit=''):
        """Add a gauge metric. It will be exported with the full name
        `dias_task_<task name>_<metric_name>_<unit>`.
        Pass the metric name without the prefix and unit according to
        prometheus naming conventions
        (https://prometheus.io/docs/practices/naming/#metric-names).
        Use a base unit as described here
        (https://prometheus.io/docs/practices/naming/#base-units)."""
        name = 'dias_task_{}_{}'.format(self.name, metric_name)
        return Gauge(name, description, labelnames=labelnames, unit=unit)

    def add_data_metric(self, name, description, labelnames=[], unit=''):
        """Add a gauge metric. It will be exported with the full name
        `dias_data_<task name>_<metric_name>_<unit>`.
        Pass the metric name without the prefix and unit according to
        prometheus naming conventions
        (https://prometheus.io/docs/practices/naming/#metric-names).
        Use a base unit as described here
        (https://prometheus.io/docs/practices/naming/#base-units)."""
        name = 'dias_data_{}_{}'.format(self.name, name)
        return Gauge(name, description, labelnames=labelnames, unit=unit)

    # Overridable Attributes
    # -----------------------

    def setup(self):
        """Initial setup stage of analyzer.
        """
        pass

    def finish(self):
        """Final clean-up stage of analyzer.
        """
        pass

    def run(self):
        """Main task stage of analyzer. Will be called by the dias framework
        according to the period set in the task config.
        """
        pass
