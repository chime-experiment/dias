"""Dataset Management Validation."""

from dias import CHIMEAnalyzer
from dias.utils.string_converter import str2timedelta
from dias.exception import DiasDataError

from caput import config
from chimedb import core
from chimedb import dataset as ds
import h5py
from chimedb import data_index
from ch_util import andata

import datetime
import numpy as np


class DatasetAnalyzer(CHIMEAnalyzer):
    """
    Validate dataset states.

    Metrics
    -------
    dias_task_<task_name>_failed_checks_total
    .....................................
    Number of datasets that failed a check.

    Labels
        check : string
            Type of check that failed. One of: `flags`.

    Output Data
    -----------
    None

    State Data
    ----------
    None

    Config
    ------
    instrument : str
        Search archive for corr acquisitions from this instrument. Default: "chimestack".
    flags_instrument : str
        Search archive for flaginput acquisitions from this instrument. Default: "chime".
    offset : str
        Process data this timedelta before current time. Default: "2h".
    freq_id : int
        Validate frames with this frequency ID. Default: 10

    Attributes
    ----------
    None

    """

    instrument = config.Property(proptype=str, default="chimestack")
    flags_instrument = config.Property(proptype=str, default="chime")
    freq_id = config.Property(proptype=int, default=10)
    offset = config.Property(proptype=str2timedelta, default="2h")

    def setup(self):
        """Set up analyzer."""
        self.failed_checks = self.add_data_metric(
            "failed_checks",
            "Number of datasets that failed the check specified by the label 'check'.",
            ["check"],
            "total",
        )

    def run(self):
        """Run analyzer."""
        # make chimedb connect
        core.connect()

        # pre-fetch most stuff to save queries later
        ds.get.index()

        # Use the tracker to get the chimestack files to analyze
        cs_file_list = self.new_files(filetypes=self.instrument + "_corr")

        # Determine the range of time being processed

        if not cs_file_list:
            raise DiasDataError("No new {} files found.".format(self.instrument))

        with h5py.File(cs_file_list[0], "r") as first_file:
            start_time = first_file["index_map/time"][0]["ctime"]
        with h5py.File(cs_file_list[-1], "r") as final_file:
            end_time = final_file["index_map/time"][-1]["ctime"]

        self.logger.info(
            "Analyzing data between {} and {}.".format(start_time, end_time)
        )

        cs_acqs = self.get_acquisitions(cs_file_list)

        # Loop over acquisitions
        for acq in cs_acqs.keys():

            # Loop over contiguous periods within this acquisition
            all_files = cs_acqs[acq]

            # Determine the range of time being processed
            with h5py.File(all_files[0], "r") as first_file:
                tstart = first_file["index_map/time"][0]["ctime"]
            with h5py.File(all_files[-1], "r") as final_file:
                tend = final_file["index_map/time"][-1]["ctime"]
            nfiles = len(all_files)

            self.logger.info(
                "Now processing acquisition %s (%d files)"
                % (acq.split("/")[-1], nfiles)
            )

            # Use Finder to get the matching flaginput files
            self.logger.info("Finding flags between {} and {}.".format(tstart, tend))
            flag_files = self.new_files("chime_flaginput", tstart, tend)
            flag_acqs = self.get_acquisitions(flag_files)

            self.logger.info(
                "Found {} acqws in flags files".format(len(flag_acqs.keys()))
            )
            if len(flag_acqs.keys()) < 1:
                raise DiasDataError(
                    "No flags found for {} files {}.".format(self.instrument, all_files)
                )

            # Loop over acquisitions
            for flag_acq in flag_acqs.keys():

                # Extract finder results within this acquisition
                all_flag_files = flag_acqs[flag_acq]

                # Loop over contiguous periods within this acquisition
                flg = list()

                # Determine the range of time being processed
                with h5py.File(all_flag_files[0], "r") as first_file:
                    flag_tstart = first_file["index_map/update_time"][0]
                with h5py.File(all_flag_files[-1], "r") as final_file:
                    flag_tend = final_file["index_map/update_time"][-1]

                nfiles = len(all_flag_files)

                if nfiles == 0:
                    continue

                self.logger.info(
                    "Now processing acquisition %s (%d files)" % (flag_acq, nfiles)
                )
                flg.append(andata.FlagInputData.from_acq_h5(all_flag_files))

            for _file in all_files:
                ad = andata.CorrData.from_acq_h5(
                    _file,
                    datasets=(
                        "flags/inputs",
                        "flags/dataset_id",
                        "flags/frac_lost",
                    ),
                )
                self.validate_flag_updates(ad, flg)
                self.validate_freqs(ad)

                self.register_done([_file])

    def validate_freqs(self, ad):
        """
        Compare number of frequencies with freq state.

        Parameters
        ----------
        ad : andata.CorrData
        """
        num_freqs = ad.nfreq

        # Get the unique dataset_ids in each file
        file_ds = ad.flags["dataset_id"][:]
        unique_ds = np.unique(file_ds)

        # Remove the null dataset
        unique_ds = unique_ds[unique_ds != "00000000000000000000000000000000"]

        # Find the freq state for each dataset
        states = {}
        for ds_id in unique_ds:
            state_id = (
                ds.Dataset.from_id(ds_id)
                .closest_ancestor_of_type("frequencies")
                .dataset_state.id
            )
            states[ds_id] = (
                ds.DatasetState.select(ds.DatasetState.data)
                .where(ds.DatasetState.id == state_id)
                .get()
                .data["data"]
            )

        for ds_id, freq_state in states.items():
            if len(freq_state) != num_freqs:
                self.failed_checks.labels(check="flags").inc()
                self.logger.warn(
                    "Number of frequencies don't match for corr file '{}' and 'freq' state {}.".format(
                        self.instrument, self.flags_instrument, ds_id.decode("UTF-8")
                    )
                )

    def validate_flag_updates(self, ad, flg):
        """
        Compare flagging dataset states with flaginput files.

        Parameters
        ----------
        ad : andata.CorrData
        flg : List(andata.FlagInputData)
        """
        # fmt: off
        extra_bad = [
            # These are non-CHIME feeds we want to exclude (26m, noise source channels, etc.)
            46, 142, 688, 944, 960, 1058, 1166, 1225, 1314, 1521, 2032, 2034,
            # Below are the last eight feeds on each cylinder, masked out because their beams are very
            # different
            0, 1, 2, 3, 4, 5, 6, 7, 248, 249, 250, 251, 252, 253, 254, 255,
            256, 257, 258, 259, 260, 261, 262, 263, 504, 505, 506, 507, 508, 509, 510,
            511,
            512, 513, 514, 515, 516, 517, 518, 519, 760, 761, 762, 763, 764, 765, 766,
            767,
            768, 769, 770, 771, 772, 773, 774, 775, 1016, 1017, 1018, 1019, 1020, 1021,
            1022, 1023,
            1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1272, 1273, 1274, 1275,
            1276, 1277, 1278, 1279,
            1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1528, 1529, 1530, 1531,
            1532, 1533, 1534, 1535,
            1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1784, 1785, 1786, 1787,
            1788, 1789, 1790, 1791,
            1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 2040, 2041, 2042, 2043,
            2044, 2045, 2046, 2047
        ]
        # fmt: on

        # Get the unique dataset_ids in each file
        file_ds = ad.flags["dataset_id"][:]
        unique_ds = np.unique(file_ds)

        # Remove the null dataset
        unique_ds = unique_ds[unique_ds != "00000000000000000000000000000000"]

        # Find the flagging update_id for each dataset
        states = {}
        for ds_id in unique_ds:
            state_id = (
                ds.Dataset.from_id(ds_id)
                .closest_ancestor_of_type("flags")
                .dataset_state.id
            )
            states[ds_id] = (
                ds.DatasetState.select(ds.DatasetState.data)
                .where(ds.DatasetState.id == state_id)
                .get()
                .data["data"]
                .encode()
            )

        for ds_id, update_id in states.items():

            # Get all non-missing frames at this frequency
            present = ad.flags["frac_lost"][self.freq_id] < 1.0

            # Select all present frames with the dataset id we want
            sel = (file_ds[self.freq_id] == ds_id) & present

            # Extract the input flags and mark any extra missing data
            # (i.e. that static mask list in baselineCompression)
            flags = ad.flags["inputs"][:, sel].astype(np.bool)
            flags[extra_bad, :] = False

            # Find the flag update from the files
            flagsfile = None
            for f in flg:
                update_ids = list(f.update_id)
                try:
                    flgind = update_ids.index(update_id)
                except ValueError as err:
                    self.logger.debug("Flags not found in file {}: {}".format(f, err))
                    continue
                flagsfile = f.flag[flgind]
            if flagsfile is None:
                raise DiasDataError(
                    "Flag ID for {} files {} not found.".format(self.instrument, ad)
                )
            flagsfile[extra_bad] = False

            # Test if all flag entries match the one from the flaginput file
            if (flags != flagsfile[:, np.newaxis]).all():
                self.failed_checks.labels(check="flags").inc()
                self.logger.warn(
                    "'{}' corr file and '{}' flaginput file: Flags don't match for dataset {}.".format(
                        self.instrument, self.flags_instrument, ds_id.decode("UTF-8")
                    )
                )
