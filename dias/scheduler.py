"""dias Scheduler."""
import concurrent.futures
import logging
import threading
import time
from dias import Job, TaskQueue, DiasConcurrencyError
from dias.utils import timestamp2str
from prometheus_client import make_wsgi_app
from wsgiref.simple_server import make_server

# Minimum value for config value trigger_interval dias allows (in minutes)
MIN_TRIGGER_INTERVAL_MINUTES = 10


def stop_scheduler(pidfile):
    """
    Stop a running scheduler.

    TODO: Not implemented yet.
    This function will block until the scheduler has terminated.

    Parameters
    ----------
    pidfile : None
        Pidfile providing the PID of the scheduler.

    Raises
    ------
    NotImplementedError
        Because someone has to implement this.
    """
    raise NotImplementedError


def _prometheus_client(barrier, logger, port):
    """
    Boilerplate for the prometheus client thread.

    This function is the entrypoint for the prometheus client thread.
    It is responsible for starting the WSGI app that implements the client.
    """
    # Create the WSGI HTTP app
    app = make_wsgi_app()
    httpd = make_server('', port, app)
    logger.info(
            'Starting prometheus client on port {}.'.format(httpd.server_port))

    # Signal we're ready
    barrier.wait()

    # Go
    httpd.serve_forever()


class Scheduler:
    """
    dias Scheduler.

    Puts tasks in a queue and runs them one by one when told. Also calls finish
    method of tasks.
    """

    def __init__(self, config):
        """
        Construct the dias scheduler.

        Parameters
        ----------
        config : dict
            dias configuration. Expected to contain the `log_level` and
            `prometheus_client_port`.
        """
        self.config = config
        self.jobs = list()

        # Set the module logger.
        self.logger = logging.getLogger('dias')
        self.logger.setLevel(config['log_level'])

        # Synchronization barrier
        barrier = threading.Barrier(2)

        # Create the prometheus client thread
        self.prom_client = threading.Thread(
            target=_prometheus_client,
            args=(barrier, self.logger, config['prometheus_client_port']))
        self.prom_client.daemon = True
        self.prom_client.start()

        # Wait for prometheus client start-up
        barrier.wait()

        # Prepare tasks
        self.__init_task_queue()

    def __init_task_queue(self):
        # This is the notional start time of the scheduler
        reference_time = time.time()

        # Get all the tasks ready
        for task in self.config.tasks:
            task.prepare(
                    reference_time,
                    log_level_override=self.config['log_level_override'],
                    start_now=self.config['start_now'])

        # Create the tasks queue
        self.queue = TaskQueue(self.config.tasks)

        # Don't need this anymore
        del self.config.tasks

        self.logger.info("Initialised {0} tasks".format(len(self.queue)))
        if self.config['log_level'] == 'DEBUG':
            for i in range(len(self.queue)):
                self.logger.debug("  {0}: {1} @ {2}".format(
                    i, self.queue[i].name, self.queue[i].start_time))

    def next_task(self):
        """
        Get the next scheduled task.

        Returns
        -------
        :class:`Task`
            The next task in the queue.
        """
        return self.queue.next_task()

    def __execute_task(self, task):
        """
        Submit a task to the executor.

        Parameters
        ----------
        :class:`Task` object
            the task to be executed.
        """
        # Create a new job. This will submit the task to the executor
        try:
            # Raises DiasConcurrencyError if the task is currently running
            job = Job(task, self.executor)

            # Remember the job
            self.jobs.append(job)
        except DiasConcurrencyError:
            self.logger.warning(
                    "Job running long.  "
                    "Skipping execution of task {0}".format(task.name))

        # Re-schedule the task for next time
        task.increment()
        self.queue.update(task)

    def start(self, pidfile=None):
        """
        Start the scheduler main loop.

        This is the entry point for the scheduler main loop.

        Parameters
        ----------
        pidfile : None
            TODO: Not implemented.
        """
        # This is the executor for workers
        self.executor = concurrent.futures.ThreadPoolExecutor()

        # Service loop
        while True:
            task_next = self.queue.next_task()
            self.logger.debug(
                    "Next task scheduled is: {0} at {1} UTC".format(
                        task_next.name, timestamp2str(task_next.start_time)))

            # If it's time to execute the next task, do so
            if time.time() >= task_next.start_time:
                # Create a new job for the task and remember it
                self.__execute_task(task_next)

                # short-circuit the loop to look for another task ready to be
                # scheduled
                continue

            # Look for jobs that have completed
            remaining = []
            for job in self.jobs:
                if job.done():
                    pass
                else:
                    remaining.append(job)
            self.jobs = remaining

            # Wait for next iteration
            time.sleep(10)

    def finish_tasks(self):
        """
        Finish all tasks.

        Calls the finish method of the analyzers of each task.
        """
        for task in self.queue:
            task.analyzer.finish()
