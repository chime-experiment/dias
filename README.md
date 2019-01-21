# dias
`dias` is a data integrity analysis system designed to be used for automated, non-realtime data processing tasks.

A deployed dias system consists of a number of _analyzers_ which are automatically run by the _scheduler_ service as configured _tasks_.

Dependencies are:
* [caput](https://github.com/radiocosmology/caput)
* [ch_util](https://bitbucket.org/chime/ch_util/src/master/ch_util/) for the CHIME-specific analyzer

## Analyzers
An _analyzer_ is a code block implementing a particular data analysis task.  It should be created by subclassing either the base `dias.analyzer.Analyzer` class or else (more likely for CHIME data analysis) the CHIME-specific `dias.chime_analyzer.CHIMEAnalyzer`, which is itself a subclass of the base `Analyzer`.

The base analyzer inherits from the [caput](https://github.com/radiocosmology/caput) `config.Reader` class.

### How to create a simple dias analyzer

The easiest way to create an analyzer is to create a new file in the `dias/analyzers` subpackage directory.  Import the CHIME analyzer (or the base analyzer, if you don't need CHIME-specific stuff).  Dias analyzers use the `caput.config.Reader` for configuration data, so import that, too:
```python
from dias.chime_analyzer import CHIMEAnalyzer
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
        self.name()
 ```
 * **logger:** A logger object which can be used to write to the dias log file (for text output, instead of using print statements). See the [python logging facility](https://docs.python.org/3/library/logging.html) on how to use it.
 ```python
    def run(self):
        self.logger.debug("Some message only needed for debugging my analyzer.")
        self.logger.warn("Something terrible happened!")
```
will result in
```
[2019-01-18 14:07:57,903] trivial_task: Some message only needed for debugging my analyzer.
[2019-01-18 14:07:57,905] trivial_task: Something terrible happened!
```
(where "trivial_task" is the name of the task we defined in the `trivial_task.conf` file below).
* **task_metric(metric_name, value, documentation=None, labels=dict(), unit=''):** A method the analyzer base class provides to export housekeeping metrics of the task.
```python
    def run(self):
        self.task_metric("something_total", 7)
```
will export a prometheus metric called `dias_task_<task_name>_something_total` with the value `7`.
* **data_metric(metric_name, value, documentation=None, labels=dict(), unit=''):** A method the analyzer base class provides to export metrics desribing the analyzed data.
```python
    def run(self):
        self.data_metric("my_metric_seconds", 23.555)
```
will export a prometheus metric called `dias_data_<task_name>_my_metric_seconds` with the value `23.555`.

* **period:** A `datetime.timedelta` object, defining the time between task runs. The value is set in the tasks config file.
* **start_time:** A `datetime.datetime` object, defining the phase of the task as the first time the task is run. The value is set in the tasks config file.
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
 
## Tasks
The other piece is the configuration file which tells the `dias` scheduler about your analysis task.  Create a YAML file in the `tasks` directory.  You can call it whatever you want, but the name must end in `.conf`.  Whatever you call it will end up being the task's _name_.

This file contains two types of configuration data:
* task configuration data
* data specifying how and when the `dias` scheduler should run the task

The first of these are any `caput` config properties that were defined in your analyzer.  The second of these is the properties:
* **analyzer**: the import path to the analyzer class containing the code to execute
* **period**: indicating the schedule for this task
* **start_time**: a time indicating the _phase_ of your task (optional)

For this example, we might have a file called `trivial_task.conf` containing:
```YAML
analyzer: "dias.analyzer.trivial_analyzer.TrivialAnalyzer" # Assuming the filename we used for
                                                           # the TrivialAnalyzer class was
                                                           # trivial_analyzer.py
period: "1h"
this_thing: 33.3
that_thing: 12
the_other_thing: "a string"
```
