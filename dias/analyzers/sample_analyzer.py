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
        self.logger.info('Starting up. My name is {} and I am of type {}.'
                         .format(self.name, __name__))

        # Add a task metric that counts how often this task ran.
        # It will be exported as dias_task_<task_name>_runs_total.
        self.run_counter = self.add_task_metric("runs",
                                                "Number of times the task ran.",
                                                unit="total")


    def run(self):
        """Main task stage: analyze data from the last period.
        """

        # Calculate the start and end of the passed period, which in this
        # example is the time we want to analyze data of.
        end_time = datetime.now() - self.offset
        start_time = end_time - self.period

        self.logger.info('Analyzing data between {} and {}.'
                         .format(datetime2str(start_time),
                                 datetime2str(end_time)))
        self.logger.info('If I had any data, I would probably throw stuff at '\
                '{}.'.format(self.write_dir))

        # Increment (+1).
        self.run_counter.inc()

    def finish(self):
        """Final stage: this is called when dias shuts down."""
        self.logger.info('Shutting down.')
