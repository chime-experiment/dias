# Analyzer Base Class
# -------------------

import logging
from caput import config
from dias.utils.time_strings import str2timedelta, str2datetime


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

    # Set the module logger.
    logger = logging.getLogger(__name__)
    FORMAT = '[%(asctime)s] %(name)s: %(message)s'
    logging.basicConfig(format=FORMAT)
    logger.setLevel(logging.INFO)


    def __init__(self, name, write_dir):
        """Constructor of analyzer base class.
        """
        self.name = name
        self.write_dir = write_dir

    start_time = config.Property(proptype=str2datetime)
    period = config.Property(proptype=str2timedelta)

    #prometheus = dias.Prometheus()


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