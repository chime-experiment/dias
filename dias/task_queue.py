import threading


class TaskQueue:
    """\
The TaskQueue contains the list of all tasks to be executed by the
scheduler.  It uses threading.Lock to ensure thread-safety.
In some ways this works like a list, but be careful."""

    def __init__(self, tasks):
        self.tasks = tasks

        self.lock = threading.Lock()

        # Sort the tasks
        self.sort()

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, key):
        return self.tasks[key]

    def next_task(self):
        """Returns the next task to be scheduled"""
        return self.tasks[0]

    def next_time(self):
        """Returns the start time of the next task to be scheduled"""
        return self.next_task().start_time

    def sort(self):
        """Sorts the task queue by start time"""
        self.lock.acquire()
        self.tasks.sort()
        self.lock.release()

    def update(self, task):
        """Updates the task queue for a new start time of task"""
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
