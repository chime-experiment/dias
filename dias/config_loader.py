import importlib
import logging
import yaml
import os
from dias.utils.time_strings import str2timedelta
from dias.task import Task
import copy

# This is how a log line produced by dias will look:
LOG_FORMAT = '[%(asctime)s] %(name)s: %(message)s'
DEFAULT_LOG_LEVEL = 'INFO'

# Minimum value for config value trigger_interval dias allows (in minutes)
MIN_TRIGGER_INTERVAL_MINUTES = 10

class ConfigLoader:
    def __init__(self, config_path, limit_task=None):
        self.config_path = config_path

        # Read and apply dias global config
        global_file = open(self.config_path, "r")
        self.global_config = yaml.load(global_file)
        global_file.close()

        # Set log level, if necessary.  We do this in global_config
        # instead of setting self.log_level directly so that it will
        # be propagated into tasks.
        self.global_config.setdefault('log_level', DEFAULT_LOG_LEVEL)

        # Check config values
        if str2timedelta(self['trigger_interval']).seconds * 60 \
                < MIN_TRIGGER_INTERVAL_MINUTES:
            msg = 'Config value `trigger_interval` is too small ({}). '\
                    'dias does not allow values smaller than {} minutes.'\
                .format(self['trigger_interval'], MIN_TRIGGER_INTERVAL_MINUTES)
            raise AttributeError(msg)

        # Set the default task config dir, which is the
        # subdirectory "task" in the directory containing dais.conf
        if not 'task_config_dir' in self or self['task_config_dir'] is None:
            self['task_config_dir'] = os.path.join(
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

        self.tasks = list()

        for config_file in os.listdir(self['task_config_dir']):
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
                task_file = open(os.path.join(self['task_config_dir'],
                    config_file),"r")

                # use any values configured on global level
                task_config = copy.deepcopy(self.global_config)

                # override with values from task config if specified
                task_config.update(yaml.load(task_file))

                # This is where we tell the task to write its output
                write_dir = os.path.join(self['task_write_dir'], task_name)

                # create the task object
                task = Task(task_name, task_config, write_dir)

                # Load the analyzer for this task from the task config
                analyzer_class = \
                    self._import_analyzer_class(task_config['analyzer'])

                task.analyzer = analyzer_class(task_name, write_dir)
                task.analyzer.read_config(task_config)

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

    # Data model
    def __getitem__(self, key):
        return self.global_config[key]

    def __setitem__(self, key, value):
        self.global_config[key] = value

    def __contains__(self, key):
        return self.global_config.__contains__(key)
    
    def __iter__(self):
        return self.global_config.__iter__()
