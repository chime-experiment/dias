"""Generates RFI mask from the stacked autocorrelations.

.. currentmodule:: dias.analyzers.flag_rfi_analyzer

Classes
=======

.. autosummary::
    :toctree: generated/

    FlagRFIAnalyzer


Functions
=========

..  autosummary::
    :toctree: generated/

"""
import os
import time
import datetime
import subprocess
import sqlite3
import gc

import numpy as np
import h5py

from chimedb import data_index
from ch_util import tools, ephemeris, andata, rfi
from caput import config

from dias import chime_analyzer
from dias.utils.string_converter import str2timedelta, datetime2str
from dias import __version__ as dias_version_tag
from dias.outfile_tracking import FileTracker

__version__ = "0.1.1"

CYL_MAP = {xx: chr(63 + xx) for xx in range(2, 6)}
POL_MAP = {"S": "X", "E": "Y"}

###################################################
# auxiliary functions
###################################################


def _chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


def _fraction_flagged(mask, axis=None, logical_not=False):

    if axis is None:
        axis = np.arange(mask.ndim)
    else:
        axis = np.atleast_1d(axis)

    frac = np.sum(mask, axis=tuple(axis), dtype=np.float32) / float(
        np.prod([mask.shape[ax] for ax in axis])
    )

    if logical_not:
        frac = 1.0 - frac

    return frac


###################################################
# main analyzer task
###################################################


class FlagRFIAnalyzer(chime_analyzer.CHIMEAnalyzer):
    """Identifies data contaminated by RFI.

    Wrapper for `ch_util.rfi.number_deviations`.  Flags data as RFI if the
    stacked autocorrelations are greater than some number of local
    median absolute deviations (MAD) from the local median.

    `DocLib 777 <https://bao.chimenet.ca/doc/documents/777>`_ describes this analyzer and the
    associated theremin graph.

    Metrics
    -------
    dias_task_<task_name>_run_time_seconds
    ......................................
    Time to process single run.

    dias_task_<task_name>_files_total
    .................................
    Number of files processed.

    dias_task_<task_name>_file_time_seconds
    .......................................
    Time to process single file.

    dias_data_<task_name>_masked_missing_ratio
    ..........................................
    Fraction of data that is missing (e.g., dropped packets or down GPU nodes.)

    Labels
        stack : string of format `<POL>-<CYL>`
            The polarisation and cylinder of the feeds
            used to construct the stacked autocorrelation.
            The special value `ALL` indicates all feeds.

    dias_data_<task_name>_masked_before_ratio
    .........................................
    Fraction of data considered bad before applying MAD threshold.  Includes
    missing data and static frequency mask from `ch_util.rfi.frequency_mask`.

    Labels
        stack : string of format `<POL>-<CYL>`
            The polarisation and cylinder of the feeds
            used to construct the stacked autocorrelation.
            The special value `ALL` indicates all feeds.

    dias_data_<task_name>_masked_after_ratio
    ........................................
    Fraction of data considered bad after applying MAD threshold.  Includes
    missing data, static frequency mask from `ch_util.rfi.frequency_mask`,
    and MAD threshold mask.

    Labels
        stack : string of format `<POL>-<CYL>`
            The polarisation and cylinder of the feeds
            used to construct the stacked autocorrelation.
            The special value `ALL` indicates all feeds.

    Output Data
    -----------

    File naming
    ...........
    `<YYYYMMDD>T<HHMMSS>Z_chimestack_rfimask/<SSSSSSSS>.h5`
        YYYYMMDD and HHMMSS are the date and time (in UTC) of the start of
        the underlying chimestack data acquisition from which the RFI
        mask was derived. SSSSSSSS is the number of seconds elapsed
        between the start of the file and the start of the acquisition.

    Indexes
    .......
    freq
        1D structured array containing the `centre` and `width` of the
        frequency channels in MHz.
    stack
        1D array containing strings of format `<POL>-<CYL>` that indicate
        the polarisation and cylinder of the feeds used to construct the
        stacked autocorrelation. The special value `ALL` indicates all feeds.
    time
        1D array contaning the unix timestamps of the centre of the
        integrations.

    Datasets
    ........
    auto
        3D array of type `float` with axes [`freq`, `stack`, `time`] that
        contains the calibrated autocorrelations stacked over inputs.
        Units are Jansky.
    ndev
        3D array of type `float` with axes [`freq`, `stack`, `time`] that
        contains the number of local median absolute deviations of the
        autocorrelations from the local median.
    mask
        3D array of type `bool` with axes [`freq`, `stack`, `time`] that
        indicates data that is likely contamined by RFI.


    State Data
    ----------
    None

    Config Variables
    ----------------

    Attributes
    ----------
    offset : str
        Process data this timedelta before current time.
    instrument : str
        Search archive for corr acquisitions from this instrument.
    max_num_file : int
        Maximum number of files to load into memory at once.
    output_suffix :  str
        Name for the output acquisition type.
    freq_low : float
        Generate RFI flags for all frequencies above this threshold.
    freq_high : float
        Generate RFI flags for all frequencies below this threshold.
    separate_cyl_pol : bool
        Construct a mask for each cylinder and polarisation in addition
        to the mask for the entire array.
    apply_static_mask : bool
        Apply static mask obtained from `ch_util.rfi.frequency_mask`
        before computing statistics.
    freq_width : float
        Frequency interval in *MHz* for computing local statistics.
    time_width : float
        Time interval in *seconds* for computing local statistics.
    rolling : bool
        Use a rolling window instead of distinct blocks.
    threshold_mad : float
        Flag data as RFI if greater than this number of
        median absolute deviations.
    """

    # Config parameters related to scheduling
    offset = config.Property(proptype=str2timedelta, default="2h")

    # Config parameters defining data file selection
    instrument = config.Property(proptype=str, default="chimestack")
    max_num_file = config.Property(proptype=int, default=1)

    # Config parameters defining output data product
    output_suffix = config.Property(proptype=str, default="rfimask")

    # Config parameters defining frequency selection
    freq_low = config.Property(proptype=float, default=400.0)
    freq_high = config.Property(proptype=float, default=800.0)

    # Config parameters for the rfi masking algorithm
    separate_cyl_pol = config.Property(proptype=bool, default=False)
    apply_static_mask = config.Property(proptype=bool, default=True)
    freq_width = config.Property(proptype=float, default=10.0)
    time_width = config.Property(proptype=float, default=420.0)
    rolling = config.Property(proptype=bool, default=True)
    threshold_mad = config.Property(proptype=float, default=6.0)

    def setup(self):
        """Initialize data index database and Prometheus metrics."""
        self.logger.info(
            "Starting up. My name is %s and I am of type %s." % (self.name, __name__)
        )

        self.tracker = FileTracker(self)

        # Add task metrics
        self.run_timer = self.add_task_metric(
            "run_time", "Time to process single run.", unit="seconds"
        )

        self.file_counter = self.add_task_metric(
            "files", "Number of files processed.", unit="total"
        )

        self.file_timer = self.add_task_metric(
            "file_time", "Time to process single file.", unit="seconds"
        )

        # Add data metrics
        self.masked_missing = self.add_data_metric(
            "masked_missing",
            "Fraction of data that is "
            + "missing (e.g., dropped "
            + "packets or down GPU nodes).",
            labelnames=["stack"],
            unit="ratio",
        )

        self.masked_before = self.add_data_metric(
            "masked_before",
            "Fraction of data "
            + "considered bad before "
            + "applying MAD threshold.  "
            + "Includes missing data and "
            + "static frequency mask.",
            labelnames=["stack"],
            unit="ratio",
        )

        self.masked_after = self.add_data_metric(
            "masked_after",
            "Fraction of data "
            + "considered bad after "
            + "applying MAD threshold.  "
            + "Includes missing data, "
            + "static frequency mask, "
            + "and MAD threshold mask.",
            labelnames=["stack"],
            unit="ratio",
        )

        # Determine default achive attributes
        host = subprocess.check_output(["hostname"]).strip()
        user = subprocess.check_output(["id", "-u", "-n"]).strip()

        self.output_attrs = {}
        self.output_attrs["type"] = str(type(self))
        self.output_attrs["git_version_tag"] = dias_version_tag
        self.output_attrs["collection_server"] = host
        self.output_attrs["system_user"] = user
        self.output_attrs["instrument_name"] = self.instrument
        self.output_attrs["version"] = __version__

    def run(self):
        """Run the task.

        Load stacked autocorrelations from the last period,
        generate rfi mask, write to disk, and update the
        data index database.
        """
        self.run_start_time = time.time()

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

        # Use Finder to get the files to analyze
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

                nfiles = len(all_files)

                if nfiles == 0:
                    continue

                self.logger.info(
                    "Now processing acquisition %s (%d files)" % (acq.name, nfiles)
                )

                # Determine the output acquisition name and make directory
                epos = acq.name.find(self.instrument) + len(self.instrument)
                output_acq = "_".join([acq.name[:epos], self.output_suffix])
                output_dir = os.path.join(self.write_dir, output_acq)

                try:
                    os.makedirs(output_dir)
                except OSError:
                    if not os.path.isdir(output_dir):
                        raise

                acq_start = ephemeris.datetime_to_unix(
                    ephemeris.timestr_to_datetime(output_acq.split("_")[0])
                )

                # Get the correlator inputs active during this acquisition
                inputmap = tools.get_correlator_inputs(
                    ephemeris.unix_to_datetime(tstart), correlator="chime"
                )

                # Determine number of files to process at once
                if self.max_num_file is None:
                    chunk_size = nfiles
                else:
                    chunk_size = min(self.max_num_file, nfiles)

                # Loop over chunks of files
                for chnk, files in enumerate(_chunks(all_files, chunk_size)):

                    self.logger.info(
                        "Now processing chunk %d (%d files)" % (chnk, len(files))
                    )

                    self.file_start_time = time.time()

                    # Deteremine selections along the various axes
                    rdr = andata.CorrData.from_acq_h5(files, datasets=())

                    within_range = np.flatnonzero(
                        (rdr.freq >= self.freq_low) & (rdr.freq <= self.freq_high)
                    )
                    freq_sel = slice(within_range[0], within_range[-1] + 1)

                    stack_sel = [
                        ii
                        for ii, pp in enumerate(rdr.prod[rdr.stack["prod"]])
                        if pp[0] == pp[1]
                    ]

                    # Load autocorrelations
                    t0 = time.time()

                    data = andata.CorrData.from_acq_h5(
                        files,
                        datasets=["vis", "flags/vis_weight"],
                        freq_sel=freq_sel,
                        stack_sel=stack_sel,
                        apply_gain=False,
                        renormalize=False,
                    )

                    tspan = (time.time() - t0,)
                    self.logger.info(
                        "Took %0.1f seconds " % tspan + "to load autocorrelations."
                    )
                    t0 = time.time()

                    # Construct RFI mask for stacked incoherent beam
                    index, auto, ndev = rfi.number_deviations(
                        data,
                        stack=True,
                        apply_static_mask=self.apply_static_mask,
                        freq_width=self.freq_width,
                        time_width=self.time_width,
                        rolling=self.rolling,
                    )

                    stack = ["ALL"]

                    # Construct RFI mask for each cylinder/polarisation
                    if self.separate_cyl_pol:

                        cyl_index, cyl_auto, cyl_ndev = rfi.number_deviations(
                            data,
                            stack=False,
                            apply_static_mask=self.apply_static_mask,
                            freq_width=self.freq_width,
                            time_width=self.time_width,
                            rolling=self.rolling,
                        )

                        stack += [
                            POL_MAP[inputmap[ii].pol] + "-" + CYL_MAP[inputmap[ii].cyl]
                            for ii in cyl_index
                        ]

                        auto = np.concatenate((auto, cyl_auto), axis=1)
                        ndev = np.concatenate((ndev, cyl_ndev), axis=1)

                    tspan = (time.time() - t0,)
                    self.logger.info(
                        "Took %0.1f seconds " % tspan + "to generate mask."
                    )

                    # Construct various masks
                    mask_missing = auto > 0.0
                    mask_before = np.isfinite(ndev)
                    mask_after = ndev <= self.threshold_mad

                    # Determine the output file name. Note that we use the
                    # start of the first integration in the file to determine
                    # the filename, rather than the center of the first
                    # integration, so that the file names are consistent
                    # between the rfimask and corr data.
                    seconds_elapsed = data.index_map["time"]["ctime"][0] - acq_start

                    output_file = os.path.join(output_dir, "%08d.h5" % seconds_elapsed)

                    self.logger.info("Writing RFI mask to: %s" % output_file)

                    # Write to output file
                    with h5py.File(output_file, "w") as handler:

                        # Set the default archive attributes
                        handler.attrs["acquisition_name"] = output_acq
                        for key, val in self.output_attrs.items():
                            handler.attrs[key] = val

                        # Create an index map
                        index_map = handler.create_group("index_map")
                        index_map.create_dataset("freq", data=data.index_map["freq"])
                        index_map.create_dataset("stack", data=np.string_(stack))
                        index_map.create_dataset("time", data=data.time)

                        # Write 2D arrays containing snapshots of each jump
                        ax = np.string_(["freq", "stack", "time"])

                        dset = handler.create_dataset("mask", data=mask_after)
                        dset.attrs["axis"] = ax
                        dset.attrs["threshold"] = self.threshold_mad

                        dset = handler.create_dataset("auto", data=auto)
                        dset.attrs["axis"] = ax

                        dset = handler.create_dataset("ndev", data=ndev)
                        dset.attrs["axis"] = ax

                    # Update data index database
                    self.tracker.update_data_index(
                        data.time[0], data.time[-1], filename=output_file
                    )

                    # Update prometheus metrics
                    ndf = len(files)
                    self.file_counter.inc(ndf)

                    time_per_file = int(
                        (time.time() - self.file_start_time) / float(ndf)
                    )
                    self.file_timer.set(time_per_file)

                    for ss, lbl in enumerate(stack):
                        self.masked_missing.labels(stack=lbl).set(
                            _fraction_flagged(mask_missing[:, ss, :], logical_not=True)
                        )

                        self.masked_before.labels(stack=lbl).set(
                            _fraction_flagged(mask_before[:, ss, :], logical_not=True)
                        )

                        self.masked_after.labels(stack=lbl).set(
                            _fraction_flagged(mask_after[:, ss, :], logical_not=True)
                        )

                    # Garbage collect
                    del data
                    gc.collect()

        # Set run timer
        self.run_timer.set(int(time.time() - self.run_start_time))

    def finish(self):
        """Close connection to data index database."""
        self.logger.info("Shutting down.")
        del self.tracker
