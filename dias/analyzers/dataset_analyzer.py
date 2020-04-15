from dias import CHIMEAnalyzer
from dias.outfile_tracking import FileTracker
from dias.utils.string_converter import datetime2str,str2timedelta
from dias.exception import DiasDataError

from caput import config
from chimedb import core
from chimedb import dataset as ds
from chimedb import data_index
from ch_util import andata

import datetime
import numpy as np


class DatasetAnalyzer(CHIMEAnalyzer):
    """
    offset : str
        Process data this timedelta before current time.
    """
    instrument = config.Property(proptype=str, default="chimestack")
    flags_instrument = config.Property(proptype=str, default="chime")
    freq_id = config.Property(proptype=int, default=10)
    offset = config.Property(proptype=str2timedelta, default="2h")

    def setup(self):
        self.tracker = FileTracker(self)
        self.failed_checks = self.add_data_metric("failed_checks", "Number of datasets that failed the check specified by the label 'check'.", ["check"], "total")

    def run(self):
        # make chimedb connect
        core.connect()

        # pre-fetch most stuff to save queries later
        ds.get.index()

        # Determine the range of time to process
        end_time = datetime.datetime.utcnow() - self.offset

        tracker_start_time = self.tracker.get_start_time()
        start_time = (
            tracker_start_time if tracker_start_time else end_time - self.period
        )
        self.logger.info(
            "Analyzing data between {} and {}.".format(
                datetime2str(start_time), datetime2str(end_time)
            )
        )

        # Use Finder to get the chimestack files to analyze
        finder = self.Finder()
        finder.accept_all_global_flags()
        finder.only_corr()
        finder.filter_acqs(data_index.ArchiveInst.name == self.instrument)
        finder.set_time_range(start_time, end_time)

        # Loop over acquisitions
        for aa, acq in enumerate(finder.acqs):

            # Extract finder results within this acquisition
            acq_results = finder.get_results_acq(aa)

            # Loop over contiguous periods within this acquisition
            for all_files, (tstart, tend) in acq_results:
                print("all files: {}, start {} end {}".format(all_files, tstart, tend))

                nfiles = len(all_files)

                if nfiles == 0:
                    continue

                self.logger.info(
                    "Now processing acquisition %s (%d files)" % (acq.name, nfiles)
                )

                # Use another Finder to get the matching flaginput files
                self.logger.info(
                    "Finding flags between {} and {}.".format(
                        datetime2str(start_time), datetime2str(end_time)
                    )
                )
                flag_finder = self.Finder()
                flag_finder.accept_all_global_flags()
                flag_finder.only_flaginput()
                flag_finder.filter_acqs(
                    data_index.ArchiveInst.name == self.flags_instrument
                )
                flag_finder.set_time_range(tstart, tend)

                self.logger.info(
                    "Found {} acqws in flags files".format(len(flag_finder.acqs)))
                if len(flag_finder.acqs) < 1:
                    raise DiasDataError(
                        "No flags found for {} files {}.".format(self.instrument, file))

                # Loop over acquisitions
                for flag_aa, flag_acq in enumerate(flag_finder.acqs):

                    # Extract finder results within this acquisition
                    flag_acq_results = flag_finder.get_results_acq(flag_aa)

                    # Loop over contiguous periods within this acquisition
                    flg = list()
                    for all_flag_files, (flag_tstart, flag_tend) in flag_acq_results:
                        print("all files: {}, start {} end {}".format(all_flag_files,
                                                                      flag_tstart,
                                                                      flag_tend))
                        nfiles = len(all_flag_files)

                        if nfiles == 0:
                            continue

                        self.logger.info("Now processing acquisition %s (%d files)" % (
                        flag_acq.name, nfiles))
                        flg.append(andata.FlagInputData.from_acq_h5(all_flag_files))

                for _file in all_files:
                    ad = andata.CorrData.from_acq_h5(
                        _file,
                        datasets=("flags/inputs", "flags/dataset_id", "flags/frac_lost"),
                    )
                    self.validate_flag_updates(ad, flg)



    def validate_flag_updates(self, ad, flg):

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

        # Find the flagging update_id for each dataset
        states = {
            id: (
                ds.Dataset.from_id(bytes(id).decode())
                    .closest_ancestor_of_type("flags")
                    .dataset_state.data["data"]
                    .encode()
            )
            for id in unique_ds
        }

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
                dsid = states[ds_id]
                try:
                    flgind = update_ids.index(dsid)
                except ValueError as err:
                    self.logger.debug("Flags not found in file {}: {}".format(f, err))
                    continue
                flagsfile = f.flag[flgind]
            if flagsfile is None:
                raise DiasDataError("Flag ID for {} files {} not found.".format(self.instrument, ad))
            flagsfile[extra_bad] = False

            # Test if all flag entries match the one from the flaginput file
            if (flags != flagsfile[:, np.newaxis]).all():
                self.failed_checks.labels(check="flags").inc()
                self.logger.warn("'{}' corr file and '{}' flaginput file: Flags don't match for dataset {}.".format(self.instrument, self.flags_instrument, ds_id.decode('UTF-8')))
