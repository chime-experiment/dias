"""dias Job."""
from dias import DiasException


class DiasConcurrencyError(DiasException):
    """
    dias concurrency exception.

    Raised when the scheduler tries to start a task that is already running.
    """


class Job:
    """The Job class is used to store metadata about a running task."""

    def __init__(self, task, executor):
        self.task = task

        if task.running():
            raise DiasConcurrencyError

        # execute
        self.future = executor.submit(task.runner)

    def done(self):
        """
        Tell if the job is finished.

        Returns
        -------
        bool
            `True` if the job has finished, `False` otherwise.
        """
        return self.future.done()
