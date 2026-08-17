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

# Widen the search for the flaginput file holding a given update by this much
# (in seconds) on either side. The tracker selects files with `start_time < end`
# and `end_time > start`, so an update that is the first or last one in its file
# is only found if the search window extends past it.
UPDATE_ID_SEARCH_MARGIN = 3600


def get_update_id_time(update_id):
    """
    Get the time an update was created from its update ID.

    ch_flag builds these as `<type>_<YYYYmmddTHHMMSS.ffffffZ>_<sources>`, e.g.
    `flaginput_20260717T202222.356129Z_raw`.

    Parameters
    ----------
    update_id : bytes or String
        The update ID.

    Returns
    -------
    float
        UNIX time at which the update was created.

    Raises
    ------
    ValueError
        If the update ID doesn't carry a timestamp in the expected format.
    """
    if isinstance(update_id, bytes):
        update_id = update_id.decode()

    parts = update_id.split("_")
    if len(parts) < 2:
        raise ValueError("No timestamp in update ID '{}'.".format(update_id))

    return (
        datetime.datetime.strptime(parts[1], "%Y%m%dT%H%M%S.%fZ")
        .replace(tzinfo=datetime.timezone.utc)
        .timestamp()
    )


class DatasetAnalyzer(CHIMEAnalyzer):
    """
    Validate dataset states.

    Metrics
    -------
    dias_data_<task_name>_flaginput_not_available_total
    ...................................................
    Number of times flaginput files have not been available for more than 24h.

    dias_data_<task_name>_failed_checks_total
    .....................................
    Number of datasets that failed a check.

    dias_data_<task_name>_updateid_not_found_total
    ..............................................
    Number of datasets whose update ids were not found in the expected flaginput files.

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
        self.flaginput_not_available = self.add_data_metric(
            "flaginput_not_available",
            "Number of times flaginput files have not been available for more than 24h.",
            unit="total",
        )
        self.updateid_not_found = self.add_data_metric(
            "updateid_not_found",
            "Number of times the update id was not found in expected flaginput files",
            unit="total",
        )

        # Paths of the flaginput files read for the acquisition being processed
        self._flag_files_read = set()

        # Initialized failed_checks metric
        self.failed_checks.labels(check="flags").set(0)
        self.failed_checks.labels(check="validnulls").set(0)
        self.flaginput_not_available.set(0)
        self.updateid_not_found.set(0)

    def run(self):
        """Run analyzer."""
        # make chimedb connect
        core.connect()

        # pre-fetch most stuff to save queries later
        ds.get.index()

        # Use the tracker to get the chimestack files to analyze
        end_time = datetime.datetime.utcnow() - self.offset

        self.logger.debug("Looking at files older than {0}".format(end_time))

        cs_file_list = self.new_files(filetypes=self.instrument + "_corr", end=end_time)

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

            # chimestack files may contain older flag updates, especially during hot periods
            # when flag updates are re-sent. This is only a guess at how far back to
            # pre-load; updates older than this are fetched on demand by find_flags.
            tstart -= 32 * 60 * 60

            # Use Finder to get the matching flaginput files
            self.logger.info("Finding flags between {} and {}.".format(tstart, tend))
            flag_files = self.new_files(
                "chime_flaginput", tstart, tend, only_unprocessed=False
            )

            if not flag_files:
                raise DiasDataError(
                    "No flags found for {} files {}.".format(self.instrument, all_files)
                )

            # new_files returns files oldest first, so the last one holds the most
            # recent update
            with h5py.File(flag_files[-1], "r") as final_file:
                flag_tend = final_file["index_map/update_time"][-1]

            self._flag_files_read = set()
            flg = dict()
            self.load_flags(flag_files, flg)

            for _file in all_files:

                # if flag files are not available yet for stackfile, do not process the stackfile
                with h5py.File(_file, "r") as f:
                    file_end = f["index_map/time"][-1]["ctime"]

                if flag_tend < file_end:  # flag not available
                    old = (
                        file_end
                        < (
                            datetime.datetime.utcnow() - datetime.timedelta(hours=24)
                        ).timestamp()
                    )

                    if old:
                        self.logger.warn(
                            "Flaginput files of older than 24h still not available."
                        )
                        self.flaginput_not_available.inc()

                    self.logger.info(
                        "Flags not available for {0}, yet. Skipping..".format(_file)
                    )
                    continue

                ad = andata.CorrData.from_acq_h5(
                    _file,
                    datasets=(
                        "flags/inputs",
                        "flags/dataset_id",
                        "flags/frac_lost",
                    ),
                )

                self.validate_null(_file, ad)
                self.validate_flag_updates(_file, ad, flg)
                self.validate_freqs(_file, ad)

                self.register_done([_file])

    def load_flags(self, flag_files, flg):
        """
        Read flaginput files into a mapping of update ID to input flags.

        Files that have already been read into `flg` are skipped.

        Parameters
        ----------
        flag_files : list of String
            Paths of the flaginput files to read.
        flg : dict -> update_id (bytes): (acquisition name, flags)
            Updated in place with the flags found in `flag_files`.
        """
        flag_files = [f for f in flag_files if f not in self._flag_files_read]

        for flag_acq, all_flag_files in self.get_acquisitions(flag_files).items():

            self.logger.info(
                "Now processing acquisition %s (%d files)"
                % (flag_acq, len(all_flag_files))
            )
            flg_ad = andata.FlagInputData.from_acq_h5(all_flag_files)

            for update_id, flag in zip(flg_ad.update_id[:], flg_ad.flag[:]):
                flg[update_id] = (flag_acq, flag)

            self._flag_files_read.update(all_flag_files)

    def find_flags(self, update_id, flg):
        """
        Get the flags for an update ID, reading more flaginput files if needed.

        A chimestack file can reference a flag update much older than its own data:
        when the flag broker restarts it re-sends the last update it knows about,
        keeping that update's original ID. Instead of guessing how far back to look,
        use the time carried by the update ID itself to find the file holding it.

        Parameters
        ----------
        update_id : bytes
            The update ID to look for.
        flg : dict -> update_id (bytes): (acquisition name, flags)
            Flags loaded so far. Updated in place with any file read here, including
            a `None` for `update_id` if it could not be found anywhere.

        Returns
        -------
        tuple or None
            (acquisition name, flags) for the update, or None if it was not found.
        """
        if update_id in flg:
            return flg[update_id]

        try:
            utime = get_update_id_time(update_id)
        except ValueError as err:
            self.logger.warn("Can't search for update {}: {}".format(update_id, err))
            flg[update_id] = None
            return None

        self.logger.info(
            "Update {} is not in the flags loaded so far, looking for it around {}.".format(
                update_id, utime
            )
        )
        flag_files = self.new_files(
            "chime_flaginput",
            utime - UPDATE_ID_SEARCH_MARGIN,
            utime + UPDATE_ID_SEARCH_MARGIN,
            only_unprocessed=False,
        )
        self.load_flags(flag_files, flg)

        if update_id not in flg:
            # Remember the miss so we don't search for the same update again
            flg[update_id] = None

        return flg[update_id]

    def validate_null(self, filename, ad):
        """
        Check that data that is considered valid (i.e. frac_lost < 1) does not have a null dataset ID.

        Paramaters
        ----------
        filename : String
            path to file loaded in ad
        ad : andata.CorrData
        """
        file_ds = ad.dataset_id[:] == "00000000000000000000000000000000"
        file_frac_lost = ad.flags["frac_lost"][:] < 1.0
        if np.logical_and(file_ds, file_frac_lost).any():
            self.failed_checks.labels(check="validnulls").inc()
            self.logger.warn(
                f"Corr file {filename} contains null dataset ids associated with valid frames (i.e. frac_lost < 1.0)"
            )

    def validate_freqs(self, filename, ad):
        """
        Compare number of frequencies with freq state.

        Parameters
        ----------
        filename : String
            path to file loaded in ad
        ad : andata.CorrData
        """
        num_freqs = ad.nfreq

        # Get the unique dataset_ids in each file
        file_ds = ad.dataset_id[:]
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
                    "Number of frequencies don't match for corr file '{}' and 'freq' state {} for dataset id {}.".format(
                        filename, self.flags_instrument, ds_id
                    )
                )

    def validate_flag_updates(self, filename, ad, flg):
        """
        Compare flagging dataset states with flaginput files.

        Parameters
        ----------
        filename: String
            Path to file loaded in ad
        ad : andata.CorrData
        flg : dict -> update_id (bytes): (acquisition name, flags)
        """
        # fmt: off
        # TODO: get them from a dataset state that should get registered by visCompression
        extra_bad = [
            # These are non-CHIME feeds we want to exclude (26m, noise source channels, etc.)
            46, 142, 688, 944, 960, 1058, 1141, 1166, 1225, 1314, 1521, 1642, 2032, 2034,
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

        # Get the unique dataset_ids in each file. Dataset IDs vary with frequency,
        # but the input flags don't, so only look at the frequency the comparison
        # below is made at.
        file_ds = ad.dataset_id[:]
        unique_ds = np.unique(file_ds[self.freq_id])

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

            # kotekan-recv wil use initial_flags until it gets an actual update
            # initial_flags will not be within the flaginput files
            if b"initial_flags" in update_id:
                self.logger.info("Skipping initial_flags update")
                continue

            # Get all non-missing frames at this frequency
            present = ad.flags["frac_lost"][self.freq_id] < 1.0

            # Select all present frames with the dataset id we want
            sel = (file_ds[self.freq_id] == ds_id) & present

            # Extract the input flags and mark any extra missing data
            # (i.e. that static mask list in baselineCompression)
            flags = ad.flags["inputs"][:, sel].astype(bool)
            flags[extra_bad, :] = False

            # Find the flag update from the files
            found = self.find_flags(update_id, flg)
            if found is None:
                self.updateid_not_found.inc()
                raise DiasDataError(
                    "Flag ID for {} file {} not found.".format(
                        self.instrument, filename
                    )
                )

            # Copy, because the flags are cached in flg and masked here
            flg_acq, flagsfile = found
            flagsfile = np.array(flagsfile)
            flagsfile[extra_bad] = False

            # Test if all flag entries match the one from the flaginput file
            if (flags != flagsfile[:, np.newaxis]).any():
                self.failed_checks.labels(check="flags").inc()
                self.logger.warn(
                    "Flags don't match for dataset {} ({}) between the following files:\n- '{}'\n- '{}/' ".format(
                        ds_id, update_id, filename, flg_acq
                    )
                )
                self.report_flags_diff(
                    flags, flagsfile[:, np.newaxis], ad.index_map["input"]
                )

    def report_flags_diff(self, flags_ds, flags_file, inputs):
        """Generate a detailed diff for flagging mismatches."""
        diff = flags_ds != flags_file
        self.logger.info(
            f"Flags in {diff.any(0).sum()}/{diff.shape[1]} frames mismatch."
        )
        self.logger.info(f"Flags for {diff.any(1).sum()} inputs mismatch:")
        for input_id, diff_input, flags_ds_input in zip(inputs, diff, flags_ds):
            num_mismatches = diff_input.sum()
            if num_mismatches != 0:
                self.logger.info(
                    f"Input {input_id} mismatches in {num_mismatches}/{diff_input.shape[0]} frames"
                )
                if np.all(flags_ds_input[diff_input] == True):
                    self.logger.info(f"The input is only flagged in the dataset state.")
                elif np.all(flags_ds_input[diff_input] == False):
                    self.logger.info(
                        f"The input is only flagged in the flaginput file."
                    )
                else:
                    self.logger.info(
                        "The input is flagged sometimes only in the flaginput file, sometimes only in the dataset state."
                    )
