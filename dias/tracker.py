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
client = dias.tracker.Tracker('/mnt/gong/staging/', '/var/lib/dias/file_index.db')
client.add_analyzer_ine('my_analyzer')
my_todo = client.new_files('my_analyzer', filetypes="chimecal")
self.do_things(my_todo)
client.register_done('my_analyzer', my_todo)
"""

import re
import os
import glob
import h5py
import yaml
import logging

from collections import namedtuple

from ch_util import ephemeris
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

FILETYPES = ["chimestack", "chimecal", "chimetiming"]


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
    start_time : datetime.datetime
        timestamp of earliest datapoint in file
    end_time : datetime.datetime
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

    Attributes
    ----------
    path : String
        the location of the staging directory that the user is interested in

    Methods
    -------
    __init__
    create_tables
    _get_filetypes
    insert_files
    add_analyzer_ine
    remove_files
    update_file_table
    new_files
    register_done
    """

    def __init__(
        self, path="/mnt/gong/staging", file_tracking_db="/var/lib/dias/file_index.db"
    ):
        """Construct the Tracker client, and create SQLite tables, if they do not exist."""
        self.base_path = path

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel("INFO")

        self.file_tracking_db = file_tracking_db
        # create connection to sqlite database
        db.init(os.path.expanduser(self.file_tracking_db))
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
        return {ft: FileType.get(name=ft) for ft in FILETYPES}

    def insert_files(self, files):
        """
        Insert files into File Table of SQL db. For them sets exists = True.

        Files that already exist in db are ignored.

        Parameters
        ----------
        files : List of String
            list of filepaths
        """
        # pattern used to extract filetype of file from its name
        pattern = re.compile(".*Z_(.*)_corr.*")

        file_type_rows = self._get_filetypes()

        self.logger.info("Adding {0} to database".format(len(files)))

        # remove already logged files from list
        files_in_table = [f.filepath for f in File.select()]

        files = [f for f in files if f not in files_in_table]

        for f in files:
            with h5py.File(f, "r") as source:
                file_type = pattern.search(f).group(1)
                File.insert(
                    {
                        "filepath": f,
                        "file_type_id": file_type_rows[file_type],
                        "start_time": source["index_map/time"][0]["ctime"],
                        "end_time": source["index_map/time"][-1]["ctime"],
                        "exists": True,
                    }
                ).execute()

    def remove_files(self, files):
        """
        Set exists = False for files in File.

        Parameters
        ----------
        files : List of String
            list of filepaths
        """
        self.logger.info(
            "Settings exists = False for {0} in database".format(len(files))
        )

        with db:
            for f in files:
                file_row = File.get(File.filepath == f)
                file_row.exists = False
                file_row.save()

    def update_file_table(self):
        """Insert novel files that are in `self.path` into db, and set exists=False for files that are not at `self.path`, anymore."""
        files_on_disk = []
        for ft in FILETYPES:
            files_on_disk.extend(
                glob.glob(os.path.join(self.base_path, "*_{0}_corr".format(ft), "*.h5"))
            )

        files_in_db = [
            query.filepath for query in (File.select().where(File.exists == True))
        ]

        self.insert_files([f for f in files_on_disk if f not in files_in_db])
        self.remove_files([f for f in files_in_db if f not in files_on_disk])

    def add_analyzer_ine(self, analyzer):
        """
        Add analyzer to the Analyzer table, if it is not currently part of it.

        Parameters
        ----------
        analyzer : String
            name of the analyzer to add
        """
        if not Analyzer.select().where(Analyzer.name == analyzer).exists():
            Analyzer.create(name=analyzer)

    def new_files(self, dias_task_name, filetypes, update=True):
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
        """
        if update:
            self.update_file_table()

        self.add_analyzer_ine(dias_task_name)

        if isinstance(filetypes, str):
            filetypes = [FileType.get(FileType.name == filetypes)]
        elif isinstance(filetypes, (list, tuple)):
            filetypes = list(FileType.select().where(FileType.name << filetypes))

        dias_task_analyzer = Analyzer.get(Analyzer.name == dias_task_name)

        # Select files processed by analyzer
        files_processed = (
            File.select()
            .join(Processed)
            .join(Analyzer)
            .where(Processed.analyzer_id == dias_task_analyzer)
        )

        # SELECT files of analyzer.file_type_id and NOT Processed by Analyzer
        # TODO Filter by date
        files_unprocessed = (
            File.select()
            .where(File.id.not_in(files_processed))
            .where(File.file_type_id << filetypes)
            .where(File.exists)
            .order_by(File.start_time.asc())
        )

        return [f.filepath for f in files_unprocessed]

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
        self.add_analyzer_ine(dias_task_name)

        files_processed = File.select().where(File.filepath.in_(list_of_files))

        analyzer_id = Analyzer.get(Analyzer.name == dias_task_name)

        record = namedtuple("record", "file_id analyzer_id")

        records = [record(f, analyzer_id) for f in files_processed]

        with db:
            # SQLite limits bulk inserts to 100 at a time
            for batch in chunked(records, 100):
                Processed.insert_many(batch).on_conflict(action="IGNORE").execute()