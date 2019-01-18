# dias Service
# ------------


import importlib
import logging
from caput import config
import yaml
import os
from dias.analyzers.sample_analyzer import *

# Set the module logger.
logger = logging.getLogger(__name__)


class service(config.Reader):
    task_config_dir = config.Property(default=os.getcwd() + '/tasks', proptype=str,
                                      key='task_config_dir')

    def __init__(self, config_path):
        self.config_path = config_path
        self.tasks = list()

        self.load_analyzers()

        # Setup tasks
        # self.setup_tasks()

    def setup_tasks(self):
        for task in self.tasks:
            task.setup()

    def run_tasks(self):
        pass
    
    def load_analyzers(self):
        """
        Locate and load all task config files
        """

        # caput config reader class for global config
        global_file = open(self.config_path, "r")
        self.read_config(yaml.load(global_file))
        global_file.close()

        for config_file in os.listdir(self.task_config_dir):

            if config_file.endswith(".conf"):

                # caput config reader class for task config
                task_file = open(os.path.join(self.task_config_dir,
                                              config_file),"r")
                task_config = yaml.load(task_file)

                # Load the analyzer for this task from the task config
                task = self.import_analyzer_class(task_config['analyzer'])
                task.read_config(self, task_config)
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