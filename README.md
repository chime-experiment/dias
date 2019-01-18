# dias
`dias` is a data integrity analysis system designed to be used for automated, non-realtime data processing tasks.

A deployed dias system consists of a number of _analyzers_ which are automatically run by the _scheduler_ service as configured _tasks_.

Dependencies are:
* [caput](https://github.com/radiocosmology/caput)
* [ch_util](https://bitbucket.org/chime/ch_util/src/master/ch_util/) for the CHIME-specific analyzer

## Analyzers
An _analyzer_ is a code block implementing a particular data analysis task.  It should be created by subclassing either the base `dias.analyzer.Analyzer` class or else (more likely for CHIME data analysis) the CHIME-specific `dias.chime_analyzer.CHIMEAnalyzer`, which is itself a subclass of the base `Analyzer`.

The base analyzer inherits from the [caput](https://github.com/radiocosmology/caput) `config.Reader` class.

### How to create a simple dias analzyer

The easiest way to create an analyzer is to create a new file in the `dias/analyzers` subpackage directory.  Import the CHIME analyzer (or the base analyzer, if you don't need CHIME-specific stuff).  Dias analyzers use the `caput.config.Reader` for configuration data, so import that, too:
```python
from dias.chime_analyzer import CHIMEAnalyzer
from caput import config
```
Make a subclass of the base `CHIMEAnalyzer` class:
```python
class TrivialAnalyzer(CHIMEAnalyzer):
```
Use `caput.config` to define some configuration properties, which can be read from the configuration file later:
```python
    this_thing      = config.Property(proptype=float, default='3.14')
    that_thing      = config.Property(proptype=int, default='0')
    the_other_thing = config.Property(proptype=string, default='pie')
```
Define a `run()` method that contains the analyzer's code.  This method will be called when it is time to run your analysis task:
```python
   def run():
       #do stuff here
```
The other piece is the configuration file which tells the `dias` scheduler about your analysis task.  Create a YAML file in the `tasks` directory.  You can call it whatever you want, but the name must end in `.conf`.  Whatever you call it will end up being the task's _name_.

This file contains two types of configuration data:
* task configuration data
* data specifying how and when the `dias` scheduler should run the task

The first of these are any `caput` config properties that were defined in your analyzer.  The second of these is the properties:
* **analyzer**: the import path to the analyzer class containing the code to execute
* **period**: indicating the schedule for this task
* **start_time**: a time indicating the _phase_ of your task (optional)

For this example, we might have a file called `my_task.conf` containing:
```YAML
analyzer: "dias.analyzer.my_analyzer.MyAnalyzer" # Assuming the filename we used for the MyAnalyzer class was my_analyzer.py
period: "1h"
this_thing: 33.3
that_thing: 12
the_other_thing: "a string"
```
