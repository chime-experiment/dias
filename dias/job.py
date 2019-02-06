class Job:
    """\
The Job class is used to store metadata about a running task.
"""
    def __init__(self, task, executor):
        self.task = task

        # execute
        self.future = executor.submit(task.runner)

    def done(self):
        """Is the job finished?"""
        return self.future.done()
