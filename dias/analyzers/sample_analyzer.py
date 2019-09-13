"""One-line description of the module.

This is a basic example for how to write an analyzer for dias.
"""


from dias import CHIMEAnalyzer
from datetime import datetime
from caput import config
from dias.utils import str2timedelta, datetime2str


class SampleAnalyzer(CHIMEAnalyzer):
    """
    One-line description.

    Long description.

    Metrics
    -------
    dias_task_<task name>_my_metric_total
    .....................................
    Description.

    Output Data
    -----------
    None

    State Data
    ----------
    None

    Config
    ------

    Attributes
    ----------
    my_value : type
        Description.
    """

    # Config parameter for this anlyzer can be specified by assigning class
    # attributes a caput.config.Property
    my_value = config.Property(proptype=str2timedelta, default="10s")

    def setup(self):
        """
        Describe the method in one line.

        A longer description.
        Don't overwrite this method if your analyzer doesn't need it!
        """
        self.logger.info(
            "Starting up. My name is {} and I am of type {}.".format(
                self.name, __name__
            )
        )
        self.logger.debug(
            "I could load everything I saved the last time I "
            "finished in {}.".format(self.state_dir)
        )

        # Add a task metric that counts how often this task ran.
        # It will be exported as dias_task_<task_name>_runs_total.
        self.my_metric = self.add_task_metric(
            "runs", "Number of times the task ran.", unit="total"
        )

    def run(self):
        """
        Describe the method in one line.

        A longer description.
        """
        # Calculate the start and end of the passed period, which in this
        # example is the time we want to analyze data of.
        end_time = datetime.now() - self.my_value
        start_time = end_time - self.period

        self.logger.info(
            "Analyzing data between {} and {}.".format(
                datetime2str(start_time), datetime2str(end_time)
            )
        )
        self.logger.info(
            "If I had any data, I would probably throw stuff at "
            "{}.".format(self.write_dir)
        )

        # Increment (+1).
        self.run_counter.inc()

    def finish(self):
        """
        Describe the method in one line.

        A longer description.
        Don't overwrite this method if your analyzer doesn't need it!
        """
        self.logger.info("Shutting down.")
        self.logger.debug(
            "I could save some stuff I would like to keep until "
            "next setup in {}.".format(self.state_dir)
        )

    def delete_callback(self, deleted_files):
        """
        Describe the method in one line.

        A longer description.
        Don't overwrite this method if your analyzer doesn't need it!
        """
        self.logger.debug(
            "Oh no, I still needed all of those: {}".format(deleted_files)
        )
