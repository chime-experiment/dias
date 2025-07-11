"""
DIAS File Tracker.

Contains interface to SQLite database, which tracks which files have been processed by each analyzer in a configured directory.
Currently, only tracks the configured Filetypes for the configured Analyzers.

The database contains 4 tables:
    * File Type (id, name_of_filetype)
    * Analyzer (id, name_of_analyzer, filetype_it_works_with)
    * File Table (id, filepath, filetype_of_file_id, start_time, end_time, exists_in_staging)
    * Processed (id, file_id, analyzer_id)

Parameters
----------
path : String
    Path to location of files. (default: location of current staging directory.
conf : dict
    dictionary of configuration key:pair values for file_index.db location

Example
-------
client = dias.utils.Tracker('/mnt/gong/staging/', '/mnt/gong/dias/file_index.db', write=True)
my_todo = client.new_files('my_analyzer', filetypes="chimecal")
self.do_things(my_todo)
client.register_done('my_analyzer', my_todo)

or

client = dias.utils.Tracker('/mnt/gong/staging/', '/mnt/gong/dias/file_index.db')
my_todo = client.new_files('my_analyzer', filetypes=["chimecal", "chimestack"], start=1592304524.2424)
"""

import re
import os
import glob
import h5py
import yaml
import logging

from collections import namedtuple, defaultdict
from datetime import datetime
from pathlib import Path

from caput import time as ctime
from dias import DiasUsageError
from caput import config
from peewee import (
    Model,
    CharField,
    SqliteDatabase,
    ForeignKeyField,
    DateTimeField,
    BooleanField,
    OperationalError,
    JOIN,
    chunked,
)


# Initialize db in Tracker.__init__()
db = SqliteDatabase(None)

FILETYPES = ["chimestack_corr", "chimecal_corr", "chimetiming_corr", "chime_flaginput"]


class BaseModel(Model):
    """Base Model defining the database connection."""

    class Meta:
        """The definition of the database connection."""

        database = db


class FileType(BaseModel):
    """
    Model definition for table of filetypes that DIAS uses.

    Attributes
    ----------
    name : String
        name of filetype; matches how they would be written in directory names
    id : Integer
        unique id for filetype in database
    """

    name = CharField(unique=True)


class Analyzer(BaseModel):
    """
    Model definition for the table of analyzers.

    Attributes
    ----------
    name : String
        name of analyzer
    """

    name = CharField(unique=True)


class File(BaseModel):
    """
    Model definition for table of files that Dias analyzers interact with.

    Attributes
    ----------
    filepath : String
        Absolute path for file
    file_type_id : Integer
        reference to filetype of that file
    start_time : Float
        timestamp of earliest datapoint in file
    end_time : Float
        timestamp of latest datapoint in file
    exists : Boolean
        does the file still exist at filepath
    """

    filepath = CharField(unique=True)
    file_type_id = ForeignKeyField(FileType, backref="files")
    start_time = DateTimeField()
    end_time = DateTimeField()
    exists = BooleanField()


class Processed(BaseModel):
    """
    Model for table of analyzer/file pairs which tracks which files have already been processed by the analyzers.

    Attributes
    ----------
    file_id : Integer
        Refers to a file in a file table
    analyzer_id : Integer
        Refers to an analyzer in the analyzer table
    """

    file_id = ForeignKeyField(File, backref="processed")
    analyzer_id = ForeignKeyField(Analyzer, backref="processed")

    class Meta:
        """Meta for the Model."""

        # file/analyzer pairs must be unique
        indexes = ((("file_id", "analyzer_id"), True),)


class Tracker:
    """
    Utility that keeps track of processed files for DIAS analyzers.

    Interfaces with an SQLite database to assess which files have/haven't been already processed by various analyzers.
    Creates the database, if it does not exist, and keeps it up-to-date with the current state of the staging directory.

    Parameters
    ----------
    path : String
        the location of the staging directory that the user is interested in
    file_tracking_db : String
        the location of the file index database that the tracker is referencing
    write : Boolean
        flips between read_and_write and read_only mode. False is read_only mode, and is the default

    Methods
    -------
    __init__
    create_tables
    _get_filetypes
    insert_files
    remove_files
    update_file_table
    new_files
    register_done
    add_output_file
    _ensure_time_unix
    """

    def __init__(
        self,
        path="/mnt/gong/staging",
        file_tracking_db="/mnt/gong/dias/file_index.db",
        write=False,
    ):
        """Construct the Tracker client, and create SQLite tables, if they do not exist."""
        self.base_path = path
        self.write = write

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel("INFO")

        self.file_tracking_db = file_tracking_db
        # create connection to sqlite database
        db.init(os.path.expanduser(self.file_tracking_db))
        if not self.write:
            self.logger.info(
                "Tracker is in read-only mode. You will not be able to update the database."
            )
            return

        try:
            self.create_tables()
        except OperationalError:
            raise DiasUsageError(
                "Unable to access file index db at location {0} for Tracker.".format(
                    self.file_tracking_db
                )
            )

    def create_tables(self):
        """Create tables in database if they do not exist."""
        file_types = {}

        if not self.write:
            self.logger.info(
                "Tracker is in read-only mode. You will not be able to update the database."
            )
            return

        with db:

            db.create_tables([FileType, Analyzer, Processed, File], safe=True)

            for ft in FILETYPES:
                if not FileType.select().where(FileType.name == ft).exists():
                    FileType.create(name=ft)

            self.update_file_table()

    def _get_filetypes(self):
        """
        Get a dictionary filled with peewee rows referencing all of the filetypes in SQL db.

        Returns
        -------
        dict
            Key: String - name of filetype
            Value: Model instance referencing row for filetype
        """
        return {ft: FileType.get(FileType.name == ft) for ft in FILETYPES}

    def insert_files(self, files):
        """
        Insert files into File Table of SQL db. For them sets exists = True.

        Files that already exist in db are ignored.

        Parameters
        ----------
        files : List of String
            list of filepaths
        """
        if not self.write:
            self.logger.info(
                "Tracker is in read-only mode. You will not be able to update the database."
            )
            return

        # pattern used to extract filetype of file from its name
        pattern = re.compile(".*Z_(.*)/.*")

        file_type_rows = self._get_filetypes()

        self.logger.debug("Adding {0} DIAS input files to database".format(len(files)))

        # remove already logged files from list
        files_in_table = [f.filepath for f in File.select()]

        files = [f for f in files if f not in files_in_table]

        for f in files:
            lock_file = "." + Path(f).name + ".lock"
            path_lock = Path(f).parent / lock_file
            if path_lock.is_file():
                # file is being written to, do not include in table
                continue
            with h5py.File(f, "r") as source:
                file_type = pattern.search(f).group(1)
                if not file_type == "chime_flaginput":
                    File.insert(
                        {
                            "filepath": f,
                            "file_type_id": file_type_rows[file_type],
                            "start_time": source["index_map/time"][0]["ctime"],
                            "end_time": source["index_map/time"][-1]["ctime"],
                            "exists": True,
                        }
                    ).on_conflict_replace().execute()
                else:
                    File.insert(
                        {
                            "filepath": f,
                            "file_type_id": file_type_rows[file_type],
                            "start_time": source["index_map/update_time"][0],
                            "end_time": source["index_map/update_time"][-1],
                            "exists": True,
                        }
                    ).on_conflict_replace().execute()

    def remove_files(self, files):
        """
        Set exists = False for files in File.

        Parameters
        ----------
        files : List of String
            list of filepaths
        """
        if not self.write:
            self.logger.info(
                "Tracker is in read-only mode. You will not be able to update the database."
            )
            return

        self.logger.debug(
            "Removing {0} files, which are not present on disk anymore, from db".format(
                len(files)
            )
        )

        with db:
            for f in files:
                file_row = File.get(File.filepath == f)
                file_row.exists = False
                file_row.save()

    def update_file_table(self):
        """Insert novel files that are in `self.path` into db, and set exists=False for files that are not at `self.path`, anymore."""
        if not self.write:
            self.logger.info(
                "Tracker is in read-only mode. You will not be able to update the database."
            )
            return

        files_on_disk = []
        for ft in FILETYPES:
            for file_ in glob.glob(
                os.path.join(self.base_path, "*_{0}".format(ft), "*.h5")
            ):
                # skip locked files
                path = Path(file_)
                if not Path(file_).with_name(f".{path.name}.lock").exists():
                    files_on_disk.append(file_)

        files_in_db = [
            query.filepath for query in (File.select().where(File.exists == True))
        ]

        self.insert_files([f for f in files_on_disk if f not in files_in_db])
        self.remove_files([f for f in files_in_db if not os.path.isfile(f)])

    def new_files(
        self,
        dias_task_name,
        filetypes,
        start=None,
        end=datetime.utcnow().timestamp(),
        only_unprocessed=True,
        update=True,
    ):
        """
        Return a list of files unprocessed by dias_task_name, of its filetype of interest.

        Optionally updates the File Table with the current state of `self.path`.

        Parameters
        ----------
        dias_task_name : String
            Name of task whose files you wish to track.
        filetypes: List of String or String
            list of filetypes of interest
        update : Boolean
            If true, updates the File Table with the current state of `self.path`.
        start : float, datetime, or None
            Float is expected to be a Unix timestamp.
            If provided, will return a list of files that contain data after start.
        end : float, datetime, or None
            Float is expected to be a Unix timestamp. Default: now
            If start is not provided, only files before end will be returned.
        only_unprocessed: boolean
            If true, will return only un-processed files. If false, will return all files fitting the alternate constraints.
        """
        if update and self.write:
            self.update_file_table()

        if isinstance(filetypes, str):
            filetypes = [FileType.get(FileType.name == filetypes)]
        elif isinstance(filetypes, (list, tuple)):
            filetypes = list(FileType.select().where(FileType.name << filetypes))

        dias_task_analyzer, _ = Analyzer.get_or_create(name=dias_task_name)

        if not end:
            end = (datetime.utcnow().timestamp(),)

        end = self._ensure_time_unix(end)

        if only_unprocessed:
            ## Select files processed by analyzer
            files_processed = (
                File.select()
                .join(Processed)
                .join(Analyzer)
                .where(Processed.analyzer_id == dias_task_analyzer)
            )
        else:
            files_processed = tuple()

        if start:
            start = self._ensure_time_unix(start)

            # return files after start
            files = (
                File.select()
                .where(File.id.not_in(files_processed))
                .where(File.file_type_id << filetypes)
                .where(File.exists)
                .where((File.start_time < end), (File.end_time > start))
                .order_by(File.start_time.asc())
            )

        # Return files before end_time
        else:
            files = (
                File.select()
                .where(File.id.not_in(files_processed))
                .where(File.file_type_id << filetypes)
                .where(File.exists)
                .where((File.start_time < end))
                .order_by(File.start_time.asc())
            )

        return [f.filepath for f in files]

    def get_acquisitions(self, file_list):
        """
        Group files in file_list by acquisition.

        Paramaters
        ----------
        file_list : List of strings
            list of filenames

        Returns
        -------
        defaultdict : key acquisition directory, value list of filenames
            Filenames are grouped by acquisition
        """
        acq_file_list = defaultdict(list)

        for f in file_list:
            acq_file_list[str(Path(f).parent)].append(f)

        return acq_file_list

    def register_done(self, dias_task_name, list_of_files):
        """
        Register files in list_of_files as processed by dias_task_name.

        Parameters
        ----------
        dias_task_name : String
            name of the analyzer that you want to GET and SET file tracking information for.
        list_of_files : List of Strings
            List of filepaths to be registered as processed. They need to already exist in the database File.
        """
        if not self.write:
            raise DiasUsageError(
                "Tracker is in read-only mode. You will not be able to update the database."
            )

        files_processed = File.select().where(File.filepath.in_(list_of_files))

        analyzer_id, _ = Analyzer.get_or_create(name=dias_task_name)

        record = namedtuple("record", "file_id analyzer_id")

        records = [record(f, analyzer_id) for f in files_processed]

        with db:
            # SQLite limits bulk inserts to 100 at a time
            for batch in chunked(records, 100):
                Processed.insert_many(batch).on_conflict_replace().execute()

        self.logger.debug(
            "Added {0} processed files for {1} analyzer".format(
                len(records), dias_task_name
            )
        )

    def add_output_file(self, dias_task_name, start, end, filepath):
        """Add row to file index table.

        Update the file index table with a row that
        contains a span of time, the absolute path to the
        output file and and the span of time the file contains.

        Parameters
        ----------
        dias_task_name : String
            Name of task whose files you wish to track.
        start : datetime.datetime or Float
            Earliest time contained in the file.
        end : datetime.datetime or Float
            Latest time contained in the file.
        filepath : str
            Absolute path to the file
        """
        if not self.write:
            self.logger.info(
                "Tracker is in read-only mode. You will not be able to update the database."
            )
            return

        analyzer_id, _ = FileType.get_or_create(name=dias_task_name)

        start = self._ensure_time_unix(start)
        end = self._ensure_time_unix(end)

        with db:
            File.insert(
                {
                    "file_type_id": analyzer_id,
                    "filepath": filepath,
                    "start_time": start,
                    "end_time": end,
                    "exists": True,
                }
            ).on_conflict_replace().execute()

        self.logger.debug("Added 1 output file for {0} to db.".format(dias_task_name))

    def get_output_files(self, dias_task_name, start, end=None, update=True):
        """Return list of filepaths for dias_task_name that include data between start and end.

        Parameters
        ----------
        dias_task_name : String
            Name of task whose output files you wish.
        start : datetime.datetime or Float
            Earliest time contained in the file.
        end : datetime.datetime or Float (optional)
            Latest time contained in the file.
        update : bool
            If true, update the File table before querying
        """
        if update and self.write:
            self.update_file_table

        try:
            dias_task_analyzer = FileType.get(FileType.name == dias_task_name)
        except FileType.DoesNotExist:
            raise DiasUsageError(
                "{0} is not registered as an analyzer with output files.".format(
                    dias_task_name
                )
            )

        if not end:
            end = datetime.utcnow()

        start = self._ensure_time_unix(start)
        end = self._ensure_time_unix(end)

        files = (
            File.select()
            .where(File.exists)
            .where(File.file_type_id == dias_task_analyzer)
            .where((File.start_time < end), (File.end_time > start))
            .order_by(File.start_time.asc())
        )

        return [f.filepath for f in files]

    @staticmethod
    def _ensure_time_unix(ts):
        """Convert to ts to unix time, if it is datetime.

        Paramters
        ----------
        ts : Float or datetime.datetime
            timestamp to ensure is in unix time.

        Returns
        -------
        Float
            timestamp, in Unix time.
        """
        if isinstance(ts, datetime):
            return ctime.datetime_to_unix(ts)
        return ts
