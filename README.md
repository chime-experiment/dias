# dias
`dias` is a data integrity analysis system designed to be used for automated, non-realtime data processing tasks.

A deployed dias system consists of a number of _analyzers_ which are automatically run by the _scheduler_ service as configured _tasks_.

Dependencies are:
* [caput](https://github.com/radiocosmology/caput)
* [ch_util](https://bitbucket.org/chime/ch_util/src/master/ch_util/) for the CHIME-specific analyzer

## Configuration
The path to the configuration file can be passed to `dias` by `-c path/to/dias.conf`, otherwise `dias` will search for a configuration file at `../conf/dias.conf`. This should be a YAML file with the following keys:
 * **log_level:** String. Global log level: `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG` or `NOTSET` (default: `INFO`)
 * **trigger_interval:** String. A time interval specifying hours, minutes and/or seconds (e.g. `2h15m30s` or `12h30m`). This tells `dias` how often to look for tasks that need to be triggered (minimum: `10m`, default: `1h`).
 * **task_config_dir:** Path to the [task files](#tasks) (default: `tasks` in the same directory as the config file)
 * **task_write_dir:** String. Path to write task output data to.
 * **task_state_dir:** String. Path to write task state data to.
 * **prometheus_client_port:** Int. Port to run the dias client on.
 
 For an example, see [`conf/dias.conf`](conf/dias.conf).

## Analyzers
An _analyzer_ is a code block implementing a particular data analysis task.  It should be created by subclassing either the base `dias.analyzer.Analyzer` class or else (more likely for CHIME data analysis) the CHIME-specific `dias.chime_analyzers.CHIMEAnalyzer`, which is itself a subclass of the base `Analyzer`.

The base analyzer inherits from the [caput](https://github.com/radiocosmology/caput) `config.Reader` class.

### How to create a simple dias analyzer

The easiest way to create an analyzer is to create a new file in the `dias/analyzers` subpackage directory.  Import the CHIME analyzer (or the base analyzer, if you don't need CHIME-specific stuff).  Dias analyzers use the `caput.config.Reader` for configuration data, so import that, too:
```python
from dias import CHIMEAnalyzer
from caput import config
```
Make a subclass of the base `CHIMEAnalyzer` class:
```python
class TrivialAnalyzer(CHIMEAnalyzer):
```
Use `caput.config` to define some configuration properties as attributes of your analyzer, which can be read from the configuration file later:
```python
class TrivialAnalyzer(CHIMEAnalyzer):
    this_thing      = config.Property(proptype=float, default='3.14')
    that_thing      = config.Property(proptype=int, default='0')
    the_other_thing = config.Property(proptype=str, default='pie')
```
Define a `run()` method that contains the analyzer's code.  This method will be called when it is time to run your analysis task:
```python
   def run(self):
       #do stuff here
```
The analyzer may also define:
 - A `setup()` method which will be run once, when the dias scheduler process is first started.
 ```python
    def setup(self):
        # Do stuff that should be done when dias starts up.
 ```
 - A `finish()` method which will be run once if the dias scheduler terminates.
```python
    def finish(self):
        # Do stuff that should be done when dias shuts down.
 ```
 
 ### Other useful utilities dias provides
 * **name:** A string containing the task's name (i.e. the name of the config file).
 ```python
    def run(self):
        print("I am " + self.name)
 ```
 * **logger:** A logger object which can be used to write to the dias log file (for text output, instead of using print statements). See the [python logging facility](https://docs.python.org/3/library/logging.html) on how to use it.
 ```python
    def run(self):
        self.logger.debug("Some message only needed for debugging my analyzer.")
        self.logger.warn("Something terrible happened!")
```
will result in
```
[2019-01-18 14:07:57,903] dias[trivial_task]: Some message only needed for debugging my analyzer.
[2019-01-18 14:07:57,905] dias[trivial_task]: Something terrible happened!
```
(where "trivial_task" is the name of the task we define in the `trivial_task.conf` file below).
* **add_task_metric(metric_name, description (optional), labelnames (optional), unit (optional)):**

A method the analyzer base class provides to create housekeeping metrics of the task. It returns a [`prometheus_client.Gauge`](https://github.com/prometheus/client_python#gauge) that can be used to update the value of the metric.
The example
```python
    def setup(self):
        self.some_metric = self.add_task_metric("something", unit='total')
        self.some_metric.set(1.2)

    def run(self):
        self.some_metric.inc()
```
will export a prometheus metric called `dias_task_<task_name>_something_total` with the initial value *1.2* that gets incremented on each run of the task.
* **add_data_metric(metric_name, description (optional), labelnames (optional), unit (optional)):**

 A method the analyzer base class provides to create metrics describing the analyzed data.
 It returns a [`prometheus_client.Gauge`](https://github.com/prometheus/client_python#gauge) that can be used to update the value of the metric.
The example
```python
    def setup(self):
        self.some_metric = self.add_data_metric("some_time", unit='seconds')

    def run(self):
        self.some_metric.set(1)

```
will export a prometheus metric called `dias_data_<task_name>_some_time_seconds`
 which will be set to *1* on every run of the task.

 * **Metric labels:** If a list of strings is passed as the parameter `labelnames`
 when calling `add_task_metric` or `add_data_metric`, the value of the metric
 can be set depending on label values, as described [here](https://github.com/prometheus/client_python#labels).
 Example with labels being `frequency` and `input`:
 ```python
 m = self.add_data_metric("my_metric", labelnames=['frequency, 'input'])
 m.labels(frequency=7.5, input='some_value').inc()
 ```

* **period:** A `datetime.timedelta` object, defining the time between task runs. The value is set in the tasks config file.
* **start_time:** A `datetime.datetime` object, indicating the time at which the running task was scheduled for execution.
* **write_dir:** The directory name for output data as a string. The analyzer code should write all its output data into this directory. The value is set in the tasks config file.
* **state_dir:** A directory where the task is allowed to store a small amount of data to save its state on disk. The value is set in the tasks config file.

#### CHIMEAnalyzer
The CHIME-specific Analyzer also provides
* **finder:** An appropriately configured instance of `ch_util.data_index.Finder`
```python
TODO: How to use.
    def run(self):
        f = self.Finder(...)
        f.only_corr()
```
It works exactly the same as calling `ch_util.data_index.Finder(...)`, except this one sets the `node_spoof` parameter for you (and will ignore any `node_spoof` you specify).

A configuration using a CHIMEAnalyzer should include `archive_data_dir` with the `node_spoof` parameter passed to `Finder`.
 
## Defining a task
The other piece is the configuration file which tells the `dias` scheduler about your analysis task.  Create a YAML file in the `tasks` directory.  You can call it whatever you want, but the name must end in `.conf`.  Whatever you call it will end up being the task's _name_.

This file contains two types of configuration data:
* task configuration data
* data specifying how and when the `dias` scheduler should run the task

The first of these are any `caput` config properties that were defined in your analyzer.  The second of these is the properties:
* **analyzer**: the import path to the analyzer class containing the code to execute
* **period**: indicating the schedule for this task
* **start_time**: a time indicating when you want the task to first run.  This is mostly used to determine the _phase_ of your task.  If this value is in the future, the
scheduler won't run your task until that time arrives.  This is optional.  If not given, the scheduler will start the task at an arbitrary time within one `period` of the
start of the scheduler.

For this example, we might have a file called `trivial_task.conf` containing:
```YAML
analyzer: "dias.analyzers.trivial_analyzer.TrivialAnalyzer" # Assuming the filename we used for
                                                           # the TrivialAnalyzer class was
                                                           # trivial_analyzer.py
period: "1h"
this_thing: 33.3
that_thing: 12
the_other_thing: "a string"
```
## Testing analyzers

After defining your task by creating the `trivial_task.conf` file, it's time to test it.

We'll do this without installing dias first, and instead run things in-place.  First make sure python can find dias:
```
export PYTHONPATH=/path/to/dias
```
where `/path/to/dias` is the path containing dias's `setup.py`.  Then you can test things by running the script:
```
scripts/dias tryrun trivial_task
```

With the `tryrun` action, the `dias` script will:

* start a new dias instance
* load the global configuration as well as the configuration file `trivial_task.conf` (and ignore all other task config files)
* run your analyzer's `setup()`, if present
* run your analyzer's `run()` once, immediately
* run your analyzer's `finish()` once, if present
* finally, exit

Output that your task sends to the `logger` will be written to standard output (i.e. your terminal).  It will also instantiate a prometheus client running on a random port which you can inspect to view the test task's prometheus output.  When running in this mode, prometheus metrics aren't sent to the prometheus database (so they won't be available in grafana).

### Using an installed dias
If you _do_ decide to install dias using `setup.py`, you'll have to tell the `script/dias` program where to find the config files (which
aren't installed by `setup.py`.  Do this with the `-c` option to `script/dias`:
```
scripts/dias -c /path/to/dias/conf/dias.conf tryrun trivial_task
```
where `/path/to/dias/conf/dias.conf` is the `dias.conf` file located in your working copy of dias.
