# Analyzer Base Class
# -------------------

import logging
from caput import config
from dias.utils.time_strings import str2timedelta, str2datetime
from dias.config_loader import DEFAULT_LOG_LEVEL

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
    period = config.Property(proptype=str2timedelta)
    log_level = config.Property(default=DEFAULT_LOG_LEVEL,
                                proptype=logging.getLevelName)

    def __init__(self, name, write_dir):
        """Constructor of analyzer base class.
        """
        self.name = name
        self.write_dir = write_dir

    def init_logger(self):
        """Set up the logger. Call this after reading the config."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.log_level)

    def task_metric(self, metric_name, value, documentation=None, labels=dict(),
                    unit=''):
        """Add a prometheus task metric. Use this to export metrics about tasks
        internals. The metric will be exported with the full name:
        `dias_task_<task name>_<metric_name>`."""
        labels['analyzer'] = __name__
        metric_name = 'dias_task_{}_{}'.format(self.name, metric_name)
        self._prometheus.add_metric(metric_name, value, documentation,
                                    timestamp=None, labels=labels, unit=unit)

    def data_metric(self, metric_name, value, documentation=None,
                    labels=dict(), unit=''):
        """Add a prometheus data metric. Use this to export
        metrics about the data you are analyzing.
        The metric will be exported with the full name:
        `dias_data_<task name>_<metric_name>`."""
        labels['task'] = self.name
        labels['analyzer'] = __name__
        metric_name = 'dias_data_{}_{}'.format(self.name, metric_name)
        self._prometheus.add_metric(metric_name, value, documentation,
                                     timestamp=None, labels=labels, unit=unit)


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
