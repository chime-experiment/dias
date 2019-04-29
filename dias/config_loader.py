"""dias config loader."""
import importlib
import logging
import yaml
import os
from dias.utils import str2timedelta, str2path, str2bytes
from dias import DiasConfigError, DiasUsageError, Task
import copy

# This is how a log line produced by dias will look like:
LOG_FORMAT = '[%(asctime)s] %(name)s: %(message)s'
DEFAULT_LOG_LEVEL = 'INFO'

# Minimum value for config value trigger_interval dias allows (in minutes)
MIN_TRIGGER_INTERVAL_MINUTES = 10


class ConfigLoader:
    """
    dias Config loader class.

    Imports analyzer modules, loads dias and task config and can check config
    parameters.

    Hint
    ----
    Config values can be accessed by name, as if this were a `dict`:
    ``config[name]``

    Assignment is supported: ``config[name] = value``

    as is iteration: ``iterator = iter(config)``.
    """

    def __init__(self, config_path, limit_task=None):
        """
        Construct the config loader.

        Parameters
        ----------
        config_path : str
            Full path to YAML config file.
        limit_task : :class:`Task` or `None`
            If this is not `None`, only the config of the given task is loaded.
            Default: `None`.
        """
        logging.basicConfig(format=LOG_FORMAT)

        self.config_path = config_path

        # Read and apply dias global config
        try:
            global_file = open(self.config_path, "r")
        except Exception as exc:
            raise DiasUsageError('Failed to open dias config file: {}'
                                 .format(exc))
        self.global_config = yaml.safe_load(global_file)

        global_file.close()

        # Validate the config variables

        # The default task config dir is the subdirectory "task" in
        # the directory containing dais.conf
        self._check_config_variable(
                'task_config_dir', proptype=str2path,
                default=os.path.join(
                    os.path.dirname(self.config_path), "tasks"))

        self._check_config_variable('task_write_dir', proptype=str2path)
        self._check_config_variable('task_state_dir', proptype=str2path)
        # now shell-expand these paths
        self['task_config_dir'] = str2path(self['task_config_dir'])
        self['task_write_dir'] = str2path(self['task_write_dir'])
        self['task_state_dir'] = str2path(self['task_state_dir'])

        self._check_config_variable('prometheus_client_port', proptype=int)
        self._check_config_variable(
                'log_level', default=DEFAULT_LOG_LEVEL,
                proptype=logging.getLevelName)
        self._check_config_variable(
                'trigger_interval', default='1h', proptype=str2timedelta)

        if str2timedelta(self['trigger_interval']).seconds * 60 \
                < MIN_TRIGGER_INTERVAL_MINUTES:
            msg = 'Config value `trigger_interval` is too small ({}). '\
                    'dias does not allow values smaller than {} minutes.'\
                    .format(self['trigger_interval'],
                            MIN_TRIGGER_INTERVAL_MINUTES)
            raise AttributeError(msg)

        # Load all the analyzers
        self._load_analyzers(limit_task)

        # Don't do anything if we have no tasks
        if len(self.tasks) < 1:
            if limit_task is None:
                raise IndexError("No tasks have been defined.")
            else:
                raise AttributeError("Task {} not found.".format(limit_task))

    def _check_config_variable(self, key, proptype, default=None, config=None):
        """
        Validate a config variable from the global config.

        Parameters
        ----------
        config : dict
            A dictionary in which to look for the given key.
        key : str
            The key (name) of the variable.
        proptype : function
            A function validating the loaded value.
        default : a type accepted by the proptype function
            The default value (default: None).

        Raises
        ------
        DiasConfigError
            If the key is not found in the config and no default is provided,
            if the proptype function does not accept the value or
            if proptype returns None when given the value.
        """
        if config is None:
            config = self.global_config
        try:
            value = config[key]
        except KeyError:
            if default is None:
                raise DiasConfigError("Could not find variable {} in config."
                                      .format(key))
            else:
                value = self[key] = default

        try:
            value = proptype(value)
        except Exception as exc:
            raise DiasConfigError("Value ({}) for config variable {} not "
                                  "accepted: {}".format(value, key, exc))
        finally:
            if value is None:
                raise DiasConfigError("Value ({}) for config variable {} not "
                                      "accepted: Not of type {}."
                                      .format(value, key, proptype))

    def _load_analyzers(self, limit_task):
        """
        Locate and load all task config files.

        Instantiate all :class:Task objects by locating and loading all task
        config files. The tasks are stored in a list as `self.tasks`.

        Parameters
        ----------
        limit_task : string or None
            If not None, only the task named `limit_task` will be loaded: all
            other tasks are ignored.
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
                task_file = open(
                        os.path.join(self['task_config_dir'], config_file),
                        "r")

                # use any values configured on global level
                task_config = copy.deepcopy(self.global_config)

                # override with values from task config if specified
                task_config.update(yaml.safe_load(task_file))

                # check task config vars
                # start_time is optional and default is None, so don't touch it
                self._check_config_variable('log_level', logging.getLevelName,
                                            'INFO', task_config)
                self._check_config_variable('period', str2timedelta,
                                            None, task_config)
                self._check_config_variable('data_size_max', str2bytes,
                                            None, task_config)
                self._check_config_variable('state_size_max', str2bytes,
                                            None, task_config)

                # This is where we tell the task to write its output
                write_dir = os.path.join(self['task_write_dir'], task_name)

                # This is where they can write a state until next time dias
                # starts up.
                state_dir = os.path.join(self['task_state_dir'], task_name)

                # create the task object
                task = Task(task_name, task_config, write_dir, state_dir)

                # Load the analyzer for this task from the task config
                analyzer_class = \
                    self._import_analyzer_class(task_config['analyzer'])

                task.analyzer = analyzer_class(task_name, write_dir, state_dir)
                task.analyzer.read_config(task_config)

                self.tasks.append(task)

                task_file.close()

    def _import_analyzer_class(self, name):
        """
        Find the Analyser class given by name.

        If name includes a module, first imports the module.

        Parameters
        ----------
        name : str
            Full name of a module to import.

        Returns
        -------
        The class specified, if found.

        Raises
        ------
        ImportError
            In case a class or module couldn't be found.
        """
        # Split the name into a module and a classname
        (modulename, separator, classname) = name.rpartition('.')

        # Check if we successfully split
        if separator == "":
            # No module, look for name in globals list
            try:
                class_ = globals()[classname]
            except KeyError:
                raise ImportError(
                        "Analyzer class {0} not found".format(classname))
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
        """
        Return `self[key]`.

        Note
        ----
        For those interested in the implementation: since key can be a slice,
        this function can return multiple tasks or none at all.
        """
        return self.global_config[key]

    def __setitem__(self, key, value):
        """Set `self[key] = value`."""
        self.global_config[key] = value

    def __contains__(self, key):
        """Test `key in self` and `key not in self`."""
        return self.global_config.__contains__(key)

    def __iter__(self):
        """Return `iter(self)`."""
        return self.global_config.__iter__()
