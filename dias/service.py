# dias Service
# ------------


import importlib
import logging
from caput import config
import yaml
import os
from dias import prometheus
from dias.utils.time_strings import str2timedelta
import copy

# This is how a log line produced by dias will look:
LOG_FORMAT = '[%(asctime)s] %(name)s: %(message)s'

# Minimum value for config value trigger_interval dias allows (in minutes)
MIN_TRIGGER_INTERVAL_MINUTES = 10

class service(config.Reader):

    # Config variables
    task_config_dir = config.Property(
        default=os.path.join(os.getcwd(), 'tasks'), proptype=str,
        key='task_config_dir')
    task_write_dir = config.Property(proptype=str)
    prometheus_client_port = config.Property(proptype=int)
    log_level = config.Property(default='INFO', proptype=logging.getLevelName)
    trigger_interval = config.Property(default='1h', proptype=str2timedelta)

    # For CHIMEAnalyzer
    archive_data_dir = config.Property(default='', proptype=str)

    def __init__(self, config_path):
        self.config_path = config_path

        self.tasks = list()

        # Read and apply dias global config
        global_file = open(self.config_path, "r")
        self.global_config = yaml.load(global_file)
        self.read_config(self.global_config)
        global_file.close()

        # Set the module logger.
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)
        logging.basicConfig(format=LOG_FORMAT)

        # Check config values
        if self.trigger_interval.seconds * 60 < MIN_TRIGGER_INTERVAL_MINUTES:
            msg = 'Config value `trigger_interval` is too small (' +\
                  self.trigger_interval + '). dias does not allow' +\
                  'values smaller than ' + MIN_TRIGGER_INTERVAL_MINUTES +\
                  ' minutes.'
            raise AttributeError(msg)

        # Start prometheus client
        self.prometheus = prometheus.Prometheus(self.prometheus_client_port)

        # Setup tasks
        self.load_analyzers()
        self.setup_tasks()

    def setup_tasks(self):
        for task in self.tasks:
            task.setup()

    def run_tasks(self):
        pass

    def load_analyzers(self):
        """
        Locate and load all task config files
        """

        for config_file in os.listdir(self.task_config_dir):
            # Only accept files ending in .conf as task configs.
            # Task config files starting with an underscore (_) are disabled.
            if config_file.endswith(".conf") and not\
                    config_file.startswith("_"):

                # caput config reader class for task config
                task_file = open(os.path.join(self.task_config_dir,
                                              config_file),"r")

                # use any values configured on global level
                task_config = copy.deepcopy(self.global_config)

                # override with values from task config if specified
                task_config.update(yaml.load(task_file))

                # Load the analyzer for this task from the task config
                analyzer_class = self.import_analyzer_class(task_config['analyzer'])

                # Remove .conf from the config file name to get the name of the
                # task
                task_name = config_file[:-5]

                # This is where we tell the task to write its output
                write_dir = os.path.join(self.task_write_dir, task_name)

                # Create the directory if it doesn't exist
                if not os.path.isdir(write_dir):
                    self.logger.info('Creating new write directory for task `'
                                      + task_name + '`: ' + write_dir)
                    os.makedirs(write_dir)
                else:
                    self.logger.info('Set write directory for task `'
                                     + task_name + '` to existing path: ' +
                                     write_dir)

                task = analyzer_class(task_name, write_dir, self.prometheus)
                task.read_config(task_config)
                task.init_logger()

                self.tasks.append(task)

                task_file.close()


    def import_analyzer_class(self, name):
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
                clas = globals()[classname]
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
                clas = getattr(ext_module, classname)
            except AttributeError:
                raise ImportError(
                        "Analyzer class {0} not found in module {1}".format(
                            classname, modulename))

        return clas
