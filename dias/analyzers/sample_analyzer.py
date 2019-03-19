"""Example dias analyzer.

This is a basic example for how to write an analyzer for dias.
"""


from dias import CHIMEAnalyzer
from datetime import datetime
from caput import config
from dias.utils import str2timedelta, datetime2str


class SampleAnalyzer(CHIMEAnalyzer):
    """
    Sample Analyzer for dias.

    Example subclass of dias.analyzer.Analyzer. Shows how a dias analyzer can
    be implemented.
    """

    # Config parameter for this anlyzer can be specified by assigning class
    # attributes a caput.config.Property
    offset = config.Property(proptype=str2timedelta, default='10s')

    def setup(self):
        """
        Set up the analyzer.

        Setup stage. This is called by the framework when dias starts up.
        """
        self.logger.info('Starting up. My name is {} and I am of type {}.'
                         .format(self.name, __name__))
        self.logger.debug('I could load everything I saved the last time I '
                          'finished in {}.'.format(self.state_dir))

        # Add a task metric that counts how often this task ran.
        # It will be exported as dias_task_<task_name>_runs_total.
        self.run_counter = self.add_task_metric(
                "runs",
                "Number of times the task ran.",
                unit="total")

    def run(self):
        """
        Analyze data from the last period.

        Main task stage, called by the dias framework.
        """
        # Calculate the start and end of the passed period, which in this
        # example is the time we want to analyze data of.
        end_time = datetime.now() - self.offset
        start_time = end_time - self.period

        self.logger.info('Analyzing data between {} and {}.'
                         .format(datetime2str(start_time),
                                 datetime2str(end_time)))
        self.logger.info(
                'If I had any data, I would probably throw stuff at '
                '{}.'.format(self.write_dir))

        # Increment (+1).
        self.run_counter.inc()

    def finish(self):
        """
        Shut down the analyzer.

        Final stage: this is called by the framework when dias shuts down.
        """
        self.logger.info('Shutting down.')
        self.logger.debug('I could save some stuff I would like to keep until '
                          'next setup in {}.'.format(self.state_dir))

    def delete_callback(self, deleted_files):
        """
        Notify the analyzer about files that have been deleted after a run.

        Gets called by the dias framework.
        """
        self.logger.debug('Oh no, I still needed all of those: {}'
                          .format(deleted_files))
