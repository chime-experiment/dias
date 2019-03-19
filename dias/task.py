"""dias Task."""
import os
import random
import traceback

from dias.utils import str2timestamp, str2total_seconds, bytes2str
from pathlib import Path
from prometheus_client import Gauge, Counter

# This is a cache of task metrics
task_metrics = {}


class Task:
    """
    dias Task.

    The Task class is used by the scheduler to hold a task's instantiated
    analyzer along with associated bookkeeping data
    """

    def __init__(self, task_name, task_config, write_dir, state_dir):
        self.write_dir = write_dir
        self.state_dir = state_dir
        self.data_space_used = 0
        self.state_space_used = 0
        self.name = task_name

        self.runcount = 0

        # The first time we run, create per-task prometheus metrics.
        # Labels are the task name and the directory type
        # ('write' or 'state').

        # We're only allowed to define these once, so we cache them
        if not task_metrics:
            task_metrics['data_written'] = Gauge(
                    'data_written',
                    'Total amount of data written, '
                    'including files deleted due to '
                    'disk space overage.',
                    labelnames=['task', 'directory'],
                    namespace='dias',
                    unit='bytes')
            task_metrics['disk_space'] = Gauge(
                    'disk_space',
                    'Total amount of data on disk.',
                    labelnames=['task', 'directory'],
                    namespace='dias',
                    unit='bytes')
            task_metrics['runs'] = Counter(
                    'runs', 'Total times task ran.',
                    labelnames=['task'],
                    namespace='dias', unit='total')

        self.data_written_metric = task_metrics['data_written']
        self.disk_space_metric = task_metrics['disk_space']
        self.metric_runs_total = task_metrics['runs']

        # Initialize counter with zero. prometheus_client does not export a
        # value until the counter is incremented.
        self.metric_runs_total.labels(task=task_name).inc(0)

        # Extract important stuff from the task config
        self.period = task_config['period']
        if 'start_time' in task_config:
            self.start_time = task_config['start_time']
        else:
            self.start_time = None

    def prepare(self, reference_time,
                log_level_override=None,
                start_now=False):
        """
        Prepare a task for execution.

        Parameters
        ----------
        reference_time : int
            POSIX timestamp indicating when task should be executed, depending
            on :param:`start_now`.
        log_level_override : str or None
            If this is not None, the analyzers log level will set to this
            instead of the global log level of dias.
        start_now : bool
            If `True`, the task will be executed at :param:`start_time`,
            otherwise a random time up to the task's period will be added to
            :param:`start_time`.
        """
        # initialse the analyzer's logger
        self.analyzer.init_logger(log_level_override)

        # Initialise start_time
        if self.start_time is None:
            if start_now:
                self.start_time = reference_time
            else:
                self.start_time = reference_time \
                        + random.random() * self.period
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
            self.analyzer.logger.debug(
                    'Creating new output directory: {0}'
                    .format(self.write_dir))
            os.makedirs(self.write_dir)
        else:
            self.analyzer.logger.debug(
                    'Write directory: {0}.'.format(self.write_dir))

        # Create the task's state directory if it doesn't exist
        if not os.path.isdir(self.state_dir):
            self.analyzer.logger.debug('Creating new state directory: {}.'
                                       .format(self.state_dir))
            os.makedirs(self.state_dir)
        else:
            self.analyzer.logger.debug(
                    'Set state directory: {}.'.format(self.state_dir))

        # Run the setup
        self.analyzer.setup()

    def running(self):
        """
        Tell if the task is running.

        Returns
        -------
        bool
            `True` if the task is running, `False` otherwise.
        """
        return self.runcount > 0

    def runner(self):
        """
        Execute run method of the task.

        This serves as the entry point for a running task.  It executes in
        a worker thread.

        Returns
        -------
        The result of the task.
        """
        # Run the task
        self.runcount += 1
        self.analyzer.logger.info("Start-up.")

        try:
            result = self.analyzer.run()
        except Exception as e:
            self.analyzer.logger.error("Task failed: {}".format(e))
            result = "Failed"
            self.analyzer.logger.error(traceback.format_exc())
        else:
            self.analyzer.logger.info(
                "Shut-down; result: {0}".format(repr(result)))

        self.runcount -= 1

        # Check for disk space overage, delete files, export metrics
        before_data_space_used = self.data_space_used
        (data_size, self.data_space_used) = self.cleanup(
            self.analyzer.write_dir, self.analyzer.data_size_max)
        data_written = data_size - before_data_space_used
        self.data_written_metric.labels(task=self.name, directory='write')\
            .set(data_written)
        self.disk_space_metric.labels(task=self.name, directory='write')\
            .set(self.data_space_used)

        # For now don't enforce state data size limit.
        before_state_space_used = self.state_space_used
        (state_size, self.state_space_used) = self.cleanup(
            self.analyzer.state_dir, self.analyzer.state_size_max, check=True)
        state_written = state_size - before_state_space_used
        self.data_written_metric.labels(task=self.name, directory='state')\
            .set(state_written)
        self.disk_space_metric.labels(task=self.name, directory='state')\
            .set(self.state_space_used)

        if not result == "Failed":
            self.metric_runs_total.labels(task=self.name).inc()

        # Return the result
        return result

    def cleanup(self, dir, max_size, check=False):
        """
        Delete old files in case of disk space overage and inform analyzer.

        Parameters
        ----------
        dir : str
            Directory to clean up.
        max_size : float
            Maximum disk space allowed in directory.
        check : bool
            If `True`, no files are deleted. Default: `False`.

        Returns
        -------
        tuple(int, int)
            Total data size before and after cleanup in bytes.
        """
        self.analyzer.logger.debug("Cleaning up {}: data size maximum: {}"
                                   .format(dir, bytes2str(max_size)))

        # gather all files their modification times and sizes (recursively)
        files = list()
        for r in list(Path(dir).rglob("*")):
            if r.is_file():
                files.append(r)

        # sort by modification time (new to old)
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # Delete old files if task disk space over limit
        total_data_size = 0
        disk_usage = 0
        deleted_files = list()
        for f in files:
            total_data_size = total_data_size + f.stat().st_size
            if total_data_size > max_size:
                if check:
                    self.analyzer.logger.warning(
                        "File '{}' of size {} exceeds total state size limit"
                        " ({}).".format(f.absolute(),
                                        bytes2str(f.stat().st_size),
                                        bytes2str(max_size)))
                else:
                    self.analyzer.logger.info("Deleting file: {}"
                                              .format(f.absolute()))
                    deleted_files.append(f)
                    try:
                        # Remove file or symbolic link
                        f.unlink()
                    except Exception as e:
                        self.analyzer.logger.warning(
                            "Unable to delete file '{}': {}"
                            .format(f.absolute(), e))
                        deleted_files.pop()
                        disk_usage = total_data_size
            else:
                disk_usage = total_data_size

        # Inform analyzer
        if len(deleted_files) > 0:
            self.analyzer.delete_callback(deleted_files)

        return (total_data_size, disk_usage)

    def increment(self):
        """Increment start_time by period."""
        self.start_time += self.period

    # Rich comparison
    def __eq__(self, other):
        """Tell if the name of this task equals the name of another task."""
        return self.name == other.name

    def __ne__(self, other):
        """Tell if the name of this task is unequal the name of another one."""
        return self.name != other.name

    def __ge__(self, other):
        """Compare the name of this task alphabetically with another one."""
        if self.start_time == other.start_time:
            return self.name >= other.name
        return self.start_time >= other.start_time

    def __gt__(self, other):
        """Compare the name of this task alphabetically with another one."""
        if self.start_time == other.start_time:
            return self.name > other.name
        return self.start_time > other.start_time

    def __le__(self, other):
        """Compare the name of this task alphabetically with another one."""
        if self.start_time == other.start_time:
            return self.name <= other.name
        return self.start_time <= other.start_time

    def __lt__(self, other):
        """Compare the name of this task alphabetically with another one."""
        if self.start_time == other.start_time:
            return self.name < other.name
        return self.start_time < other.start_time
