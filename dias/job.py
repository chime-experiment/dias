from dias import DiasException


class DiasConcurrencyError(DiasException):
    """\
Raised when the scheduler tries to start a
task that is already running.
"""


class Job:
    """\
The Job class is used to store metadata about a running task.
"""

    def __init__(self, task, executor):
        self.task = task

        if task.running():
            raise DiasConcurrencyError

        # execute
        self.future = executor.submit(task.runner)

    def done(self):
        """Is the job finished?"""
        return self.future.done()
