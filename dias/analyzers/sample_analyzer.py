"""Example dias analyzer.

This is a basic example for how to write an analyzer for dias.
"""


from dias import analyzer
from datetime import datetime
from caput import config
from dias.utils.time_strings import str2timedelta, datetime2str


class SampleAnalyzer(analyzer.Analyzer):
    """Sample Analyzer for dias.
    This subclass of dias.analyzer.Analyzer describes the new analyzer.
    """

    # Config parameter for this anlyzer can be specified by assigning class
    # attributes a caput.config.Property
    offset = config.Property(proptype=str2timedelta, default='10s')

    def setup(self):
        """Setup stage: this is called when dias starts up."""
        self.logger.info('Starting up. My name is ' + self.name +
                            ' and I am of type ' + __name__ + '.')
        self.run_counter = 0

    def run(self):
        """Main task stage: analyze data from the last period.
        """

        # Calculate the start and end of the passed period, which in this
        # example is the time we want to analyze data of.
        end_time = datetime.now() - self.offset
        start_time = end_time - self.period

        self.logger.info('Analyzing data between ' + datetime2str(start_time) +
                         ' and ' + datetime2str(end_time) + '.')
        self.logger.info('If I had any data, I would probably throw stuff at ' +
                         self.write_dir)

        # Export a task metric that counts how often this task ran.
        self.run_counter += 1
        self.task_metric('runs_count', self.run_counter)

    def finish(self):
        """Final stage: this is called when dias shuts down."""
        self.logger.info('Shutting down.')
