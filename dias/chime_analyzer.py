"""CHIME-specifc Analyzer Base Class."""

import os
import re
import glob

from datetime import datetime

from caput import config
from dias import Analyzer
from ch_util import finder



class CHIMEAnalyzer(Analyzer):
    """
    A base dias analyzer class with CHIME-specific extensions.

    The CHIMEAnalyzer class extends the base dias Analyzer class by adding
    CHIME-specific convenience methods.
    """

    archive_data_dir = config.Property(proptype=str)

    def __init__(self, name, write_dir, state_dir):
        """Construct the CHIME analyzer base class.

        Parameters
        ----------
        name : String
            The name of the task.
        write_dir : String
            The path to write output data to.
        state_dir : String
            The path to write state data to.
        """
        super().__init__(name, write_dir, state_dir)

    def Finder(self, acqs=()):
        """
        Get a ch_util Finder object.

        Parameters
        ----------
        acqs : list of :class:`ArchiveAcq` objects
            Acquisition filter. Default: `()`.

        Returns
        -------
        An instance of :class:`Finder`.
        """
        return finder.Finder(acqs=acqs, node_spoof={"gong": self.archive_data_dir})

    def find_all_archive(self, data_product="*"):
        """
        Return a list of unlocked files located in archive.

        Parameters
        ----------
        data_product : str
            Refers to the type of file being looked for. Should match file names for data types.
            Default: Returns files of all types.
            Data product types are located at the end of file names.

        Returns : list of str
            Where each str represents a path to a file.
        """

        glob_str = os.path.join(
                self.archive_data_dir, "*_{0}{1}".format(self.instrument, data_product), "*.h5"
                )
        return sorted(glob.glob(glob_str))

    def filter_files_by_time(self, files=[], start_time=datetime.min, stop_time=datetime.max):
        """
        Filter a list of files by time range. Date of file is inferred from the filename.

        Parameters
        ----------
        start_time : :class:datetime.datetime
            Inclusive of start_time.

        end_time : :class:datetime.datetime
            Inclusive of end_time.
        """

        for f in files:
            # obtain file's date from name
            # files have naming structure YYYYmmddT*
            file_time = re.search("(\d*)T.*", f).groups()[0]
            file_time = datetime.strptime(file_time, '%Y%m%d')

            if (file_time >= start_time) and (file_time <= stop_time):
                file_list.append(f)

        return file_list
