import logging
import os
from dias import prometheus

# This is how a log line produced by dias will look:
LOG_FORMAT = '[%(asctime)s] %(name)s: %(message)s'

# Minimum value for config value trigger_interval dias allows (in minutes)
MIN_TRIGGER_INTERVAL_MINUTES = 10

def stop_scheduler(pidfile):
    """\
Stop a running scheduler.  The PID of the scheduler is given in the file provided.

This function will block until the scheduler has terminated.
"""
    raise NotImplementedError

class Scheduler:

    def __init__(self, config, local_prometheus = False, log_stdout = False):
        self.config = config

        # Set the module logger.
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.log_level)
        logging.basicConfig(format=LOG_FORMAT)

        # Start prometheus client
        self.prometheus = prometheus.Prometheus(
                self.config.prometheus_client_port)

        # Prepare tasks
        self.prepare_tasks()

    def prepare_tasks(self):
        for task in self.config.tasks:
            # Associate prometheus client
            task._prometheus = self.prometheus

            # initialse the task's logger
            task.init_logger()

            # Create the task's output directory if it doesn't exist
            if not os.path.isdir(task.write_dir):
                self.logger.info('Creating new write directory for task ' \
                        '`{}`: {}.'.format(task.name, task.write_dir))
                os.makedirs(task.write_dir)
            else:
                self.logger.info('Set write directory for task `{}`: {}.'
                        .format(task.name, task.write_dir))


    def setup_tasks(self):
        for task in self.config.tasks:
            task.setup()

    def next_task(self):
        """Returns the next scheduled task"""
        return self.config.tasks[0]

    def run_tasks(self):
        raise NotImplementedError

    def finish_tasks(self):
        for task in self.config.tasks:
            task.finish()
