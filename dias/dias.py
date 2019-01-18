import importlib
import logging

# Set the module logger.
logger = logging.getLogger(__name__)

def run_tasks():
    pass

def load_analyzers():
    pass

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
