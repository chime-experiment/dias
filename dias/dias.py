import importlib
import logging
from caput import config
import yaml
import os

# Set the module logger.
logger = logging.getLogger(__name__)

def run_tasks():
    pass

class config_global(config.Reader):
    task_config_dir = config.Property(default=os.getcwd(), proptype=str,
                                      key='task_config_dir')

class config_task(config.Reader):
    analyzer = config.Property(proptype=str, key='analyzer')
    
def load_analyzers():
    """
    Locate and load all task config files
    """
    
    # caput config reader class for global config
    global_config = config_global()
    global_file = open("dias.conf","r")
    global_config.read_config(yaml.load(global_file))
    global_file.close()
    
    config_dir = global_config.task_config_dir
    for config_file in os.listdir(config_dir):
        
        if config_file.endswith("ad.conf"):
                        
            # caput config reader class for task config
            task_config = config_task()
            task_file = open(os.path.join(config_dir, config_file),"r")
            task_config.read_config(yaml.load(task_file))
            task_file.close()
    

def import_analyzer_class(name):
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
