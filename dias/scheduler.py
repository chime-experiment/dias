import logging
import os
import threading
import time
from prometheus_client import make_wsgi_app
from wsgiref.simple_server import make_server

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

    def __init__(self, config, log_stdout=False):
        self.config = config

        # Set the module logger.
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.log_level)
        logging.basicConfig(format=LOG_FORMAT)

        # Set a server to export (expose to prometheus) the data (in a thread)
        app = make_wsgi_app()
        httpd = make_server('', config.prometheus_client_port, app)
        t = threading.Thread(target=httpd.serve_forever)
        t.daemon = True
        t.start()
        self.logger.info('Starting prometheus client on port {}.'
                         .format(httpd.server_port))

        # Prepare tasks
        self.prepare_tasks()

    def prepare_tasks(self):

        # This is the notional start time of the scheduler
        reference_time = time.time()

        for task in self.config.tasks:
            # initialse the task's logger
            task.init_logger(self.config.log_level_override)

            # Pick a start time if the task hasn't declared one
            if task.start_time is None:
                if self.config.start_now:
                    task.start_time = reference_time
                else:
                    task.start_time = reference_time + random.random() * task.period

            # Advance start time into the non-past:
            while task.start_time <= reference_time:
                task.start_time += task.period

            # Create the task's output directory if it doesn't exist
            if not os.path.isdir(task.write_dir):
                self.logger.info('Creating new write directory for task ' \
                        '`{}`: {}.'.format(task.name, task.write_dir))
                os.makedirs(task.write_dir)
            else:
                self.logger.info('Set write directory for task `{}`: {}.'
                        .format(task.name, task.write_dir))

        # Sort tasks by start_time
        self.config.tasks.sort(key=lambda task: task.start_time,
                reverse=True)

        self.logger.info("Initialised {0} tasks".format(len(self.config.tasks)))
        if self.config.log_level == 'DEBUG':
            for i in range(len(self.config.tasks)):
                self.logger.debug("  {0}: {1} @ {2}".format(i, self.config.tasks[i].name,
                    self.config.tasks[i].start_time))


    def setup_tasks(self):
        for task in self.config.tasks:
            task.setup()

    def next_task(self):
        """Returns the next scheduled task"""
        return self.config.tasks[0]

    def start(self):
        """This is the entry point for the scheduler main loop"""
        raise NotImplementedError

    def finish_tasks(self):
        for task in self.config.tasks:
            task.finish()
