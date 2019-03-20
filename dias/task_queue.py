"""dias task queue."""
import threading


class TaskQueue:
    """
    The dias task queue.

    The TaskQueue contains the list of all tasks to be executed by the
    scheduler. It uses threading.Lock to ensure thread-safety.

    Hint
    ----
    Tasks in the task queue can be accessed as if they were in a list:

    ``queue[key]`` or ``queue[slice]``

    The first task in the queue, (`queue[0]`), is the next to be
    scheduled.  Use `len` to find the number of tasks in the queue:

    ``len(queue)``

    Adding or removing elements from the queue is not supported.
    """

    def __init__(self, tasks):
        """
        Construct the task queue.

        Parameters
        ----------
        tasks : dict
            A dict of :class:`Task` instances to be taken care of by the
            scheduler. The keys are the task names (String).
        """
        self.tasks = tasks

        self.lock = threading.Lock()

        # Sort the tasks
        self.sort()

    def __len__(self):
        """
        Get the number of tasks in the queue.

        Note
        ----
        This implements `len(queue)`.

        Returns
        -------
        int
            Number of tasks.
        """
        return len(self.tasks)

    def __getitem__(self, key):
        """
        Return self[key].

        Note
        ----
        For those interested in the implementation: since key can be a slice,
        this function can return multiple tasks or none at all.

        Parameters
        ----------
        key : str
            A task name.

        Returns
        -------
        :class:`Task`
            The task instance.
        """
        return self.tasks[key]

    def next_task(self):
        """
        Get the next task to be scheduled.

        Note
        ----
        ``self.next_task()`` is equivalent to ``self[0]``.

        Returns
        -------
        :class:`Task`
            The next task in the queue.
        """
        return self.tasks[0]

    def next_time(self):
        """
        Get the start time of the next task to be scheduled.

        Returns
        -------
        :class:`datetime.datetime`
            The start time of the next task in the queue.
        """
        return self.next_task().start_time

    def sort(self):
        """
        Sort the task queue.

        After sorting, the next task to schedule will be first. Really, once
        sorted, the queue stays sorted, so this function needs only be called
        by the constructor when the queue is first made.
        """
        self.lock.acquire()
        self.tasks.sort()
        self.lock.release()

    def update(self, task):
        """
        Update the task queue for a new start time of task.

        Increment the start time of `task` by its period and then relocate
        it in the queue to keep the task queue sorted.

        Parameters
        ----------
        task : :class:`Task`
            The task to update in the queue.
        """
        self.lock.acquire()

        # Delete the old task by linear search
        for index, item in enumerate(self.tasks):
            if item == task:
                del self.tasks[index]
                break

        # Insert at the new location by binary search
        new, end = 0, len(self.tasks)
        while new < end:
            mid = (new + end) // 2
            if task < self.tasks[mid]:
                end = mid
            else:
                new = mid + 1
        self.tasks.insert(new, task)

        self.lock.release()
