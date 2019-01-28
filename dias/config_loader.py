import importlib
from caput import config
import logging
import yaml
import os
from dias.utils.time_strings import str2timedelta
import copy

# This is how a log line produced by dias will look:
LOG_FORMAT = '[%(asctime)s] %(name)s: %(message)s'
DEFAULT_LOG_LEVEL = 'INFO'
DEFAULT_ARCHIVE_DIR = ''

# Minimum value for config value trigger_interval dias allows (in minutes)
MIN_TRIGGER_INTERVAL_MINUTES = 10

class DiasUsageError(Exception):
    """Exception raised for errors in the usage of dias.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

class ConfigLoader(config.Reader):

    # Config variables
    task_config_dir = config.Property(
        proptype=str,
        key='task_config_dir')
    task_write_dir = config.Property(proptype=str)
    prometheus_client_port = config.Property(proptype=int)
    log_level = config.Property(default='INFO', proptype=logging.getLevelName)
    trigger_interval = config.Property(default='1h', proptype=str2timedelta)

    # For CHIMEAnalyzer
    archive_data_dir = config.Property(default=DEFAULT_ARCHIVE_DIR,
                                       proptype=str)

    def __init__(self, config_path, limit_task = None):
        self.config_path = config_path

        self.tasks = list()

        # Read and apply dias global config
        try:
            global_file = open(self.config_path, "r")
        except Exception as exc:
            raise DiasUsageError('Failed to open dias config file: {}'
                                 .format(exc))
        self.global_config = yaml.load(global_file)
        self.read_config(self.global_config)
        global_file.close()

        # Check config values
        if self.trigger_interval.seconds * 60 < MIN_TRIGGER_INTERVAL_MINUTES:
            msg = 'Config value `trigger_interval` is too small ({}). '\
                    'dias does not allow values smaller than {} minutes.'\
                .format(self.trigger_interval,
                        MIN_TRIGGER_INTERVAL_MINUTES)
            raise AttributeError(msg)

        # Set the default task config dir, which is the
        # subdirectory "task" in the directory containing dais.conf
        if self.task_config_dir is None:
            self.task_config_dir = os.path.join(
                    os.path.dirname(self.config_path), "tasks")

        # Load all the analyzers
        self.load_analyzers(limit_task)

        # Don't do anything if we have no tasks
        if len(self.tasks) < 1:
            raise IndexError("No tasks have been defined.")

    def load_analyzers(self, limit_task):
        """
        Locate and load all task config files
        """

        for config_file in os.listdir(self.task_config_dir):
            # Only accept files ending in .conf as task configs.
            # Task config files starting with an underscore (_) are disabled.
            if config_file.endswith(".conf") and not\
                    config_file.startswith("_"):

                # Remove .conf from the config file name to get the name of the
                # task
                task_name = os.path.splitext(config_file)[0]

                # If we've limited ourselves to a particular task,
                # skip everything else
                if limit_task is not None and task_name != limit_task:
                    continue

                # caput config reader class for task config
                task_file = open(os.path.join(self.task_config_dir,
                                              config_file),"r")

                # use any values configured on global level
                task_config = copy.deepcopy(self.global_config)

                # override with values from task config if specified
                task_config.update(yaml.load(task_file))

                # Load the analyzer for this task from the task config
                analyzer_class = \
                    self._import_analyzer_class(task_config['analyzer'])

                # This is where we tell the task to write its output
                write_dir = os.path.join(self.task_write_dir, task_name)

                task = analyzer_class(task_name, write_dir)
                task.read_config(task_config)

                self.tasks.append(task)

                task_file.close()


    def _import_analyzer_class(self, name):
        """
        Finds the Analyser class given by name.  If name includes a module,
        first imports the module.

        Returns the class specified, if found.  On error, raises ImportError.
        """

        # Split the name into a module and a classname
        (modulename, separator, classname) = name.rpartition('.')

        # Check if we successfully split
        if separator == "":
            # No module, look for name in globals list
            try:
                class_ = globals()[classname]
            except KeyError:
                raise ImportError("Analyzer class {0} not found".format(classname))
        else:
            # Try to load the module
            try:
                ext_module = importlib.import_module(modulename)
            except ImportError:
                raise ImportError("Analyzer module {0} not found".format(
                    modulename))

            # Now, find the class in the module
            try:
                class_ = getattr(ext_module, classname)
            except AttributeError:
                raise ImportError(
                        "Analyzer class {0} not found in module {1}".format(
                            classname, modulename))

        return class_
