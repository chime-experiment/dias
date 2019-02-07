

from dias.analyzer import Analyzer
from datetime import datetime
from time import sleep
from dias.utils.time_strings import str2timedelta, datetime2str
from caput import config

class TestAnalyzer(Analyzer):
    """\
Test dias analyzer.

This is a trivial dias analyzer used to test the scheduler.
"""

    wait_time = config.Property(proptype=str2timedelta)

    def setup(self):
        self.logger.info('Task awake.')

    def run(self):
        seconds = self.wait_time.total_seconds()

        self.logger.info('Task running for {0} seconds.'.format(seconds))
        sleep(seconds)
        self.logger.info('Task exiting.')
