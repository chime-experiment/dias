import ephemeris
import numpy as np
import os
import sqlite3
import copy

from .exception import DiasException


class FileTracker:
    """Keep an index of output files and start- and stop- times associated to them."""

    # name of the database files
    _DB_FILE = "data_index.db"
    _ARCHIVE_DB_FILE = "archive_index.db"

    _CREATE_ARCHIVE_DB_TABLE = """CREATE TABLE IF NOT EXISTS files(
                                start TIMESTAMP,
                                stop TIMESTAMP,
                                ntime INTEGER,
                                time_step REAL,
                                filename TEXT UNIQUE ON CONFLICT REPLACE)"""

    def __init__(self, analyzer, **kwargs):
        """
        Open connection to data index database.

        Creates table if it does not exist.

        Parameters
        ----------
        analyzer : :class:`dias.analyzer`
            The analyzer to track output data for.
        **kwargs
            Additional variables to add to the tracking. Keys are the names of the
            variables and values are default values. Supported value types (so far):
            only int.
        """
        # get all the info we need from the analyzer
        self.write_dir = analyzer.write_dir
        self.state_dir = analyzer.state_dir
        self.logger = analyzer.logger

        # save additional variable names and default values
        self._additional_vars = kwargs

        # prepare create and insert commands for additional variables
        self._create_cmd = "CREATE TABLE IF NOT EXISTS files(start TIMESTAMP, stop TIMESTAMP, "
        self._insert_cmd = "INSERT INTO files VALUES (?, ?, "
        for name, default in self._additional_vars.items():
            if type(default) is int:
                typename = "INTEGER"
            else:
                raise DiasException(
                    "Type of additional value {}: {} is not supported (yet).".format(
                        name, type(default)))
            self._create_cmd += "{} {}, ".format(name, typename)
            self._insert_cmd += "?, "
        self._create_cmd += "filename TEXT UNIQUE ON CONFLICT REPLACE)"
        self._insert_cmd += + "?)"

        # Open connection to data index database
        # and create table if it does not exist.
        db_file = os.path.join(self.state_dir, self._DB_FILE)
        db_types = sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES

        # The connection can only be used by one thread at a time.
        # This module does not serialize use of the connection.
        # Don't schedule multiple tasks at the same time.
        # At the moment dias doesn't support that anyways.
        self.data_index = sqlite3.connect(
            db_file, detect_types=db_types, check_same_thread=False
        )

        cursor = self.data_index.cursor()
        cursor.execute(self._create_cmd)
        self.data_index.commit()

    def __del__(self):
        self.data_index.close()

    def get_start_time(self):
        # Refresh the database
        self._refresh_data_index()
        cursor = self.data_index.cursor()
        query = "SELECT stop FROM files ORDER BY stop DESC LIMIT 1"
        results = list(cursor.execute(query))
        if results:
            return results[0][0]
        return None

    def update_data_index(self, start, stop, filename=None, **kwargs):
        """Add row to data index database.

        Update the data index database with a row that
        contains the name of the file and the span of time
        the file contains.

        Update the data index database with a row that
        contains a span of time, the relative path to the
        output file and any additional values defined in the constructor call.

        Parameters
        ----------
        start : unix time
            Earliest time contained in the file.
        stop : unix time
            Latest time contained in the file.
        filename : str
            Name of the file.
        **kwargs
            Additional values to save. The keys have to be identical to what has been
            defined in the constructor.

        """
        # check additional variables
        for k in kwargs:
            if k not in self._additional_vars.keys():
                raise DiasException("Found unexpected key in additional variables: {}".format(k))
        # start with default values and then update given additional values
        additional_vars = copy.copy(self._additional_vars)
        additional_vars.update(kwargs)

        # Parse arguments
        dt_start = ephemeris.unix_to_datetime(ephemeris.ensure_unix(start))
        dt_stop = ephemeris.unix_to_datetime(ephemeris.ensure_unix(stop))

        relpath = None
        if filename is not None:
            relpath = os.path.relpath(filename, self.write_dir)

        # Insert row for this file
        cursor = self.data_index.cursor()
        cursor.execute(
            self._insert_command, (dt_start, dt_stop, **kwargs, relpath)
        )

        self.data_index.commit()
        self.logger.info("Added %s to data index database." % relpath)

    def _refresh_data_index(self):
        """
        Remove expired files from the data index database.

        Find rows of the data index database that correspond
        to files that have been cleaned (removed) by dias manager.
        Replace the filename with None.
        """
        cursor = self.data_index.cursor()
        query = "SELECT filename FROM files ORDER BY start"
        all_files = list(cursor.execute(query))

        replace_command = "UPDATE files SET filename = ? WHERE filename = ?"

        for result in all_files:

            filename = result[0]

            if filename is None:
                continue

            if not os.path.isfile(os.path.join(self.write_dir, filename)):
                cursor = self.data_index.cursor()
                cursor.execute(replace_command, (None, filename))
                self.data_index.commit()
                self.logger.info("Removed %s from data index database." % filename)
