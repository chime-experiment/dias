## Configuration


The path to the configuration file can be passed to `dias` by
`-c path/to/dias.conf`, otherwise `dias` will search for a configuration file
at `../conf/dias.conf`. This should be a YAML file with the following keys:

* **log_level:** String. Global log level: `CRITICAL`, `ERROR`, `WARNING`,
  `INFO`, `DEBUG` or `NOTSET` (default: `INFO`)
* **trigger_interval:** String. A time interval specifying hours, minutes
  and/or seconds (e.g. `2h15m30s` or `12h30m`). This tells `dias` how often to
  look for tasks that need to be triggered (minimum: `10m`, default: `1h`).
* **task_config_dir:** Path to the [task files](#tasks) (default: `tasks` in
  the same directory as the config file)
* **task_write_dir:** String. Path to write task output data to.
* **task_state_dir:** String. Path to write task state data to.
* **prometheus_client_port:** Int. Port to run the dias client on.

For an example, see [`conf/dias.conf`](https://github.com/chime-experiment/dias/blob/master/conf/dias.conf).

## Defining a task

The other piece is the configuration file which tells the `dias` scheduler
about your analysis task.  Create a YAML file in the `tasks` directory.
You can call it whatever you want, but the name must end in `.conf`.
Whatever you call it will end up being the task's _name_.

This file contains two types of configuration data:

* task configuration data
* data specifying how and when the `dias` scheduler should run the task

The first of these are any `caput` config properties that were defined in your
analyzer.  The second of these is the properties:

* **analyzer**: the import path to the analyzer class containing the code to execute
* **period**: indicating the schedule for this task
* **start_time**: a time indicating when you want the task to first run.
  This is mostly used to determine the _phase_ of your task. If this value is
  in the future, the scheduler won't run your task until that time arrives.
  This is optional.  If not given, the scheduler will start the task at an
  arbitrary time within one `period` of the start of the scheduler.
* **data_size_max**: the amount of data this task can write in it's data
  directory. *dias* deletes old files if this is exceeded. This should be a
  string containing of a number followed by a whitespace and the SI-unit (e.g.
  `1 kB`)
* **state_size_max**: the amount of data this task can write in it's state
  directory. *dias* deletes old files if this is exceeded. This should be a
  string containing of a number followed by a whitespace and the SI-unit (e.g.
  `10 MB`)

For this example, we might have a file called `trivial_task.conf` containing:

```YAML
# Assuming the filename we used for
# the TrivialAnalyzer class was
# trivial_analyzer.py
analyzer: "dias.analyzers.trivial_analyzer.TrivialAnalyzer" 
period: "1h"
this_thing: 33.3
that_thing: 12
the_other_thing: "a string"
start_time: '2018-01-03 17:13:13'
data_size_max: '1 GB'
state_size_max: '0 B'
```
