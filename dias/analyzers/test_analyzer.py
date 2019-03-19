"""Analyzer to test the dias scheduler."""
from dias import Analyzer
from dias.utils import str2timedelta
from time import sleep
from caput import config


class TestAnalyzer(Analyzer):
    """
    Test dias analyzer.

    This is a trivial dias analyzer used to test the scheduler.

    Metrics
    -------
    None

    Output Data
    -----------
    None

    State Data
    ----------
    None

    Config Variables
    ----------------

    Attributes
    ----------
    wait_time : String
        The amount of time the task sleeps in its run method.
    """

    wait_time = config.Property(proptype=str2timedelta)

    def setup(self):
        """Set up the analyzer."""
        self.logger.info('Task awake.')

    def run(self):
        """
        Run the analyzer.

        Sleeps the configured wait time.
        """
        seconds = self.wait_time.total_seconds()

        self.logger.info('Task running for {0} seconds.'.format(seconds))
        sleep(seconds)
        self.logger.info('Task exiting.')
