"""Analyzer Base Class."""

import logging
import os
import re
import glob

from datetime import datetime
from datetime import timedelta

from caput import config
from dias.utils import str2timedelta, str2datetime, str2bytes
from dias import DiasUsageError
from prometheus_client import Gauge


class Analyzer(config.Reader):
    """
    Base class for all dias analyzers.

    All dias analyzers should inherit from this class, with functionality added
    by over-riding `setup`, `run` and/or `shutdown`.
    In addition, input parameters may be specified by adding class attributes
    which are instances of `config.Property`. These will then be read from the
    task config file when it is initialized.  The class attributes
    will be overridden with instance attributes with the same name but with the
    values specified in the config file.

    Attributes
    ----------
    period : String
        A time period (e.g. '1h'), indicating the schedule for this task
    start_time : String
        A time (e.g. '2018-01-03 17:13:13') indicating when you want the task
        to first run. This is mostly used to determine the phase of your task.
        If this value is in the future, the scheduler won't run your task until
        that time arrives. This is optional. If not given, the scheduler will
        start the task at an arbitrary time within one period of the start of
        the scheduler.
    data_size_max : String
        The amount of data this task can write in it's data directory. dias
        deletes old files if this is exceeded. This should be a string
        containing of a number followed by a whitespace and the SI-unit (e.g.
        '1 kB')
    state_size_max : String
        the amount of data this task can write in it's state directory. dias
        deletes old files if this is exceeded. This should be a string
        containing of a number followed by a whitespace and the SI-unit
        (e.g. '10 MB')

    Methods
    -------
    __init__
    setup
    run
    finish
    """

    # Config values
    start_time = config.Property(proptype=str2datetime)
    log_level = config.Property(proptype=logging.getLevelName)
    period = config.Property(proptype=str2timedelta)
    data_size_max = config.Property(proptype=str2bytes)
    state_size_max = config.Property(proptype=str2bytes)

    def __init__(self, name, write_dir, state_dir, tracker=None):
        """Construct the analyzer base class.

        Parameters
        ----------
        name : String
            Task name.
        write_dir : String
            Path to write output data to.
        state_dir : String
            Path to write state data to.
        tracker : :class: dias.utils.Tracker
            A file Tracker to associate with the analyzer.
            Helps the analyzer keep track of which files it has not processed, yet.
        """
        self.name = name
        self.write_dir = write_dir
        self.state_dir = state_dir
        self._tracker = tracker

    def init_logger(self, log_level_override="NOTSET"):
        """
        Set up the logger.

        Called by :class:`Task` after reading the config.

        Parameters
        ----------
        log_level_override : String
            If this is passed, it will override the global log level of dias.
        """
        self.logger = logging.getLogger("dias[{0}]".format(self.name))
        if log_level_override:
            self.log_level = log_level_override

        self.logger.setLevel(self.log_level)

    def add_task_metric(self, metric_name, description="", labelnames=[], unit=""):
        """
        Add a gauge metric.

        The metric will be exported with the full name
        `dias_task_<task name>_<metric_name>_<unit>`.
        Pass the metric name without the prefix and unit according to
        prometheus naming conventions
        (https://prometheus.io/docs/practices/naming/#metric-names).
        Use a base unit as described here
        (https://prometheus.io/docs/practices/naming/#base-units).

        Parameters
        ----------
        name : String
            Name of the metric.
        description : String
            Description of the metric (optional).
        labelnames : list of Strings
            Names of the labels for the metric (optional).
        unit : String
            The base unit of the metric (optional)

        Returns
        -------
        :class:`prometheus_client.Gauge`
            The metric object to be kept and updated by the analyzer.
        """
        name = "dias_task_{}_{}".format(self.name, metric_name)
        return Gauge(name, description, labelnames=labelnames, unit=unit)

    def add_data_metric(self, name, description="", labelnames=[], unit=""):
        """
        Add a gauge metric.

        The metric will be exported with the full name
        `dias_data_<task name>_<metric_name>_<unit>`.
        Pass the metric name without the prefix and unit according to
        prometheus naming conventions
        (https://prometheus.io/docs/practices/naming/#metric-names).
        Use a base unit as described here
        (https://prometheus.io/docs/practices/naming/#base-units).

        Parameters
        ----------
        name : String
            Name of the metric.
        description : String
            Description of the metric (optional).
        labelnames : list of Strings
            Names of the labels for the metric (optional).
        unit : String
            The base unit of the metric (optional)

        Returns
        -------
        :class:`prometheus_client.Gauge`
            The metric object to be kept and updated by the analyzer.
        """
        name = "dias_data_{}_{}".format(self.name, name)
        return Gauge(name, description, labelnames=labelnames, unit=unit)

    def find_all_archive(self, instrument, data_product="*"):
        """
        Return a list of unlocked files located in archive.

        Parameters
        ----------
        data_product : str
            Refers to the type of file being looked for. Should match file names for data types.
            Default: Returns files of all types.
            Data product types are located at the end of folder names.


        Returns : list of str
            Where each str represents a path to a file.
        """
        glob_str = os.path.join(
            self.archive_data_dir, "*_{0}_{1}".format(instrument, data_product), "*.h5"
        )
        return sorted(glob.glob(glob_str))

    def filter_files_by_time(
        self, files=[], start_time=datetime.min, stop_time=datetime.max
    ):
        """
        Filter a list of files by time range. Date of file is inferred from the filename.

        Parameters
        ----------
        start_time : :class:datetime.datetime
            Inclusive of start_time.

        end_time : :class:datetime.datetime
            Inclusive of end_time.
        """
        file_list = []
        for f in files:
            # obtain file's date from name
            # folders have naming structure YYYYmmddT*
            # files have naming structure *_%.h5
            # where * is the number of seconds after the folder_datetime that the data was acquired
            # % is the fraction of a second
            folder_datetime = re.search("(\d*T\d*)Z.*", f).groups()[0]
            folder_datetime = datetime.strptime(folder_datetime, "%Y%m%dT%H%M%S")
            # we are not going to care about the fractions of a second
            file_seconds = re.search("(\d*)_\d\d\d\d.h5", f).groups()[0]

            file_time = folder_datetime + timedelta(seconds=int(file_seconds))

            if (file_time >= start_time) and (file_time <= stop_time):
                file_list.append(f)

        return file_list

    # Overridable Attributes
    # -----------------------

    def setup(self):
        """
        Set up the analyzer.

        Initial setup stage of analyzer. Called by the dias framework when dias
        is set up.
        """
        pass

    def finish(self):
        """
        Shut down the analyzer.

        Final clean-up stage of analyzer. Called by the dias framework.
        """
        pass

    def run(self):
        """
        Run the analyzer.

        Main task stage of analyzer. Will be called by the dias framework
        according to the period set in the task config.
        """
        pass

    def delete_callback(self, deleted_files):
        """
        Tell the analyzer that files have been deleted after the task has run.

        Called after run() to inform the analyzer about files that have been
        deleted from its write_dir due to data size overage. This is called
        by the dias framework.

        Parameters
        ----------
        deleted_files : list of pathlib.Path objects
            Files that have been deleted.
        """
        pass

    def new_files(self, filetypes, start=None, end=None, only_unprocessed=True):
        """
        Return a list of files unprocessed by dias_task_name, of its filetypes of interest.

        If dates are provided, will return all files of type filestypes within the time range,
        regardless if processed.

        Parameters
        ----------
        filetypes : List of String or String
            list of filetypes of interest.
        start : float, datetime, or None
            Float is expected to be a Unix timestamp.
            If provided, will return a list of files that contain data after start and now. Files will be returned, even if previously processed..
        end : float, datetime, or None
            Float is expected to be a Unix timestamp.
            If provided, will return a list of files that contain data before end.
        only_unprocessed: boolean
            If True, will return only un-processed files. If false, will return all files fitting the alternate constraints.
        """
        if self._tracker is not None:
            return self._tracker.new_files(
                self.name, filetypes, start, end, only_unprocessed
            )
        else:
            raise DiasUsageError("Analyzer does not have a tracker configured")

    def register_done(self, list_of_files):
        """
        Register files in list_of_files as processed by analyzer.

        Parameters
        ----------
        list_of_files : List of Strings
            List of filepaths to be registered as processed. They need to already exist in the database.
        """
        if self._tracker is not None:
            self._tracker.register_done(self.name, list_of_files)
        else:
            raise DiasUsageError("Analyzer does not have a tracker configured.")

    def add_output_file(self, start, end, filepath):
        """
        Register filepath in Output Table, with the timespan it covers.

        Parameters
        ----------
        start : float or datetime.datetime
            Earliest timestamp in output file.
        end : float or datetime.datetime
            Latest timestamp in output file.
        filepath : String
            absolute path to filepath
        """
        if self._tracker is not None:
            self._tracker.add_output_file(self.name, start, end, filepath)
        else:
            raise DiasUsageError("Analyzer does not have a tracker configured.")

    def get_acquisitions(self, file_list):
        """
        Group files in file_list by acquisition.

        Parameters
        ----------
        file_list : List of strings
            list of filenames

        Returns
        -------
        dict : key acquisition directory : value list of filenames
            Filenames are grouped by acquisition
        """
        if self._tracker is not None:
            return self._tracker.get_acquisitions(file_list)
        else:
            raise DiasUsageError("Analyzer does not have a tracker configured.")
