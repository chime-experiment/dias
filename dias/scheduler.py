# dias Service
# ------------


import logging
import os
import threading
from prometheus_client import make_wsgi_app
from wsgiref.simple_server import make_server

# This is how a log line produced by dias will look:
LOG_FORMAT = '[%(asctime)s] %(name)s: %(message)s'

# Minimum value for config value trigger_interval dias allows (in minutes)
MIN_TRIGGER_INTERVAL_MINUTES = 10

class Scheduler:

    def __init__(self, config, log_stdout = False):
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
        for task in self.config.tasks:
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