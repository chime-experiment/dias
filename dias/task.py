import os
from dias.utils.time_strings import str2timestamp, str2total_seconds

class Task:
    """\
The Task class is used by the scheduler to hold a task's instantiated
analyzer along with associated bookkeeping data
"""
    def __init__(self, task_name, task_config, write_dir, state_dir):
        self.write_dir = write_dir
        self.state_dir = state_dir
        self.name = task_name

        self.runcount = 0

        # Extract important stuff from the task config
        self.period = task_config['period']
        if 'start_time' in task_config:
            self.start_time = task_config['start_time']
        else:
            self.start_time = None

    def prepare(self, reference_time, log_level_override=None,
            start_now=False):
        """Prepare a task for execution."""

        # initialse the analyzer's logger
        self.analyzer.init_logger(log_level_override)

        # Initialise start_time
        if self.start_time is None:
            if start_now:
                self.start_time = reference_time
            else:
                self.start_time = reference_time + random.random() * self.period
        else:
            # Convert to seconds
            self.start_time = str2timestamp(self.start_time)

        # Convert period to seconds
        self.period = str2total_seconds(self.period)

        # Advance start time into the non-past:
        while self.start_time <= reference_time:
            self.start_time += self.period

        # Create the task's output directory if it doesn't exist
        if not os.path.isdir(self.write_dir):
            self.analyzer.logger.info(
                    'Creating new output directory: {0}: {}.'.format(
                        self.write_dir))
            os.makedirs(self.write_dir)
        else:
            self.analyzer.logger.info('Write directory for task: {0}.'
                    .format(self.write_dir))

        # Create the task's state directory if it doesn't exist
        if not os.path.isdir(self.state_dir):
            self.analyzer.logger.debug('Creating new state directory ' \
                    'for task `{}`: {}.'.format(self.name, self.state_dir))
            os.makedirs(self.state_dir)
        else:
            self.analyzer.logger.debug('Set state directory for task `{}`: {}.'
                    .format(self.name, self.state_dir))

    def running(self):
        return self.runcount > 0

    def runner(self):
        """This serves as the entry point for a running task.  It executes in
        a worker thread."""

        # Run the task
        self.runcount += 1
        self.analyzer.logger.info("Running.")
        result = self.analyzer.run()
        self.analyzer.logger.info("Completed; result: {0}".format(repr(result)))
        self.runcount -= 1

        # TODO Clean up, check for disk space overage, etc.

        # Return the result
        return result

    def increment(self):
        """Increment start_time by period"""
        self.start_time += self.period
    
    # Rich comparison
    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return self.name != other.name

    def __ge__(self, other):
        if self.start_time == other.start_time:
            return self.name >= other.name
        return self.start_time >= other.start_time

    def __gt__(self, other):
        if self.start_time == other.start_time:
            return self.name > other.name
        return self.start_time > other.start_time

    def __le__(self, other):
        if self.start_time == other.start_time:
            return self.name <= other.name
        return self.start_time <= other.start_time

    def __lt__(self, other):
        if self.start_time == other.start_time:
            return self.name < other.name
        return self.start_time < other.start_time
