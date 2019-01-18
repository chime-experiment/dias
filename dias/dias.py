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
    Imports the Analyser class given by name.  If name includes a module,
    first imports the module.

    Returns the imported class.
    """

    # Split the name into a module and a classname
    (modulename, separator, classname) = name.rpartition('.')

    try:
        # Check if we successfully split
        if separator == "":
            # No module, look for name in the global list
            # Throws KeyError if not found
            clas = globals()[classname]
        else:
            # Load the module.  Throws ImportError on error
            ext_module = importlib.import_module(modulename)

            # Find the class in the module.  Throws AttributeError on error
            clas = getattr(ext_module, classname)
    except ImportError:
        raise ImportError("Analyzer module {0} not found".format(module_path))
    except KeyError:
        raise ImportError("Analyzer class {0} not found".format(classname))
    except AttributeError:
        raise ImportError("Analyzer class {0} not found in module {1}".format(
            modulename, classname))

    return clas
 
