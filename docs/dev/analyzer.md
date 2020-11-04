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
    this_thing = config.Property(proptype=float, default='3.14')
    that_thing = config.Property(proptype=int, default='0')
    the_other_thing = config.Property(proptype=str,
                                      default='pie')
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
 
 - A method that is called after `run()` to inform the analyzer in case files
  have been deleted from its `write_dir` due to data size overage.
```python
    def delete_callback(self, deleted_files):
        # The 'deleted_files' parameter is a list of
        # 'pathlib.Path' objects for each file in
        # 'self.write_dir' that has been deleted.
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
        self.logger.debug("Some message only needed for debugging"
                          " my analyzer.")
        self.logger.warn("Something terrible happened!")
```
will result in
```
[2019-01-18 14:07:57,903] dias[trivial_task]: Some message only
needed for debugging my analyzer.
[2019-01-18 14:07:57,905] dias[trivial_task]: Something terrible
happened!
```
(where "trivial_task" is the name of the task we define in the `trivial_task.conf` file below).
* **add_task_metric(metric_name, description (optional), labelnames (optional), unit (optional)):**

A method the analyzer base class provides to create housekeeping metrics of the task. It returns a [`prometheus_client.Gauge`](https://github.com/prometheus/client_python#gauge) that can be used to update the value of the metric.
The example
```python
    def setup(self):
        self.some_metric = self.add_task_metric("something",
                                                unit='total')
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
        self.some_metric = self.add_data_metric("some_time",
                                                unit='seconds')

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
     def setup(self):
         self.some_metric = self.add_data_metric(
             "my_metric", labelnames=['frequency', 'input'])
         
     def run(self):
         self.some_metric.labels(frequency=7.5,
                                 input='some_value').set(1)
         self.some_metric.labels(frequency=8.5,
                                 input='some_value').inc()
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
 
### Documenting analyzers

Follow the
[numpy docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html)
when documenting your analyzer. To document the class that implements your
analyzer, you need to add the sections `Metrics`, `Output Data`, `State Data`
and an empty `Config Variables` section followed by the section `Attributes`
(this is just so the config variables get displayed nicely
[here](/user/analyzers). Any of the sections that don't apply to your analyzer,
should contain a simple `None`. Otherwise fill them out following this example:

```python
"""
One-line description.

Long description about what this analyzer does.

Metrics
-------
dias_task_<task_name>_my_metric_seconds
.......................................
Description of my task metric.
Labels
    label name : Description of the label.


dias_data_<task_name>_my_other_metric_total
...........................................
Description of my data metric.
Labels
    label name : Description of the label.

Output Data
-----------

File naming
...........
`<YYYYMMDD>_<VAR>_my_file.h5`
    Descrition of the file name. In this example, YYYYMMDD is the date when
    data was analyzed and VAR probably stands for something, too.

Indexes
.......
foo
    Description of this index map.
bar
    Description of that index map.

Datasets
.........
my_dataset
    Description of this dataset.

State Data
----------
Free descrition of the state data that gets written to disc by this analyzer.

Config Variables
----------------

Attributes
----------
my_config_var : int
    Descrition of my config variable.
my_other_conf_var: str
    Description. Default : 'foo'.
"""
```

### Testing analyzers

After [defining your task](/user/config) by creating the `trivial_task.conf` file, it's time to test it.

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

#### Testing prometheus metrics

When you test an analyzer with `scripts/dias tryrun <task name>`, you will see
something like
```
[2019-02-19 21:00:38,008] dias: Starting prometheus client on
port 36261.
```
in first few lines of the output. Note the port number at the end of the line
that was randomly chosen for this *tryrun*. While your task is being run, you
can see the metrics it exports using
```
curl localhost:<port number>
```
or by accessing `localhost:<port number>` with your browser.
Since this will only work until the scripts exits, you can make it pause
indefinitely after running your task using
```
scripts/dias tryrun -p <task name>
```
Abort the paused dias script by hitting `CTRL-C`.

#### Location of output data

In tryrun mode, the default [`dias.conf`](https://github.com/chime-experiment/dias/blob/master/conf/dias.conf) will tell your analyzer to write
date to `~/dias_tmp/<task_name>/data`.  Task state files (if any) will be put
into `~/dias_tmp/<task_name>/state`.

#### Using an installed dias
If you _do_ decide to install dias using `setup.py`, you'll have to tell the `script/dias` program where to find the config files (which aren't installed by `setup.py`).  Do this with the `-c` option to `script/dias`:
```
scripts/dias -c /path/to/dias/conf/dias.conf tryrun trivial_task
```
where `/path/to/dias/conf/dias.conf` is the `dias.conf` file located in your working copy of dias.
