"""Find jumps or steps in autocorrelations.

.. currentmodule:: dias.analyzers.find_jump_analyzer

Classes
=======

.. autosummary::
    :toctree: generated/

    FindJumpAnalyzer


Functions
=========

..  autosummary::
    :toctree: generated/

    mod_max_finder
    finger_finder

"""
import os
import time
import datetime
import subprocess
import glob
import sqlite3
import requests
import gc
from collections import Counter

import numpy as np
import h5py
import pywt

from ch_util import tools, ephemeris, andata, rfi

from caput import config

from dias import chime_analyzer
from dias.utils.string_converter import str2timedelta, datetime2str
from dias import __version__ as dias_version_tag

__version__ = "0.1.0"

DB_FILE = "data_index.db"
CREATE_DB_TABLE = """CREATE TABLE IF NOT EXISTS files(
                            start TIMESTAMP,
                            stop TIMESTAMP,
                            njump INTEGER,
                            filename TEXT UNIQUE ON CONFLICT REPLACE)"""

ARCHIVE_DB_FILE = "archive_index.db"
CREATE_ARCHIVE_DB_TABLE = """CREATE TABLE IF NOT EXISTS files(
                            start TIMESTAMP,
                            stop TIMESTAMP,
                            ntime INTEGER,
                            time_step REAL,
                            filename TEXT UNIQUE ON CONFLICT REPLACE)"""


###################################################
# auxiliary functions
###################################################


def _flag_transit(name, timestamp, window=900.0):
    """Flag times near the transit of a source."""
    extend = 24.0 * 3600.0

    flag = np.zeros(timestamp.size, dtype=np.bool)

    if name.lower() == "sun":
        ttrans = ephemeris.solar_transit(timestamp[0] - extend, timestamp[-1] + extend)
    else:
        ttrans = ephemeris.transit_times(
            ephemeris.source_dictionary[name],
            timestamp[0] - extend,
            timestamp[-1] + extend,
        )

    # Flag +/- window around each transit
    for tt in ttrans:
        flag |= (timestamp >= (tt - window)) & (timestamp <= (tt + window))

    return flag


def _chunks(l, n, m=0):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[max(0, i - m) : i + n]


def _sliding_window(arr, window):
    """Create rolling window view into the last axis of arr."""
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def mod_max_finder(scale, coeff, search_span=0.5, threshold=None):
    """Find local maxima of the modulus of the wavelet transform.

    Parameters
    ----------
    scale : np.ndarray
        1D array of type `float` that contains the scale parameter of
        the wavelet in units of number of time samples.
    coeff : np.ndarray
        2D array of type `float` with axes `[scale, time]` that contains
        the wavelet transform of a 1D time series.
    search_span : float
        A time sample is considered a local maxima if the modulus of the
        wavelet transform is greater than all neighboring time samples
        within `search_span * scale` (note the search is done separately at
        each scale).
    threshold : float
        The modulus of the wavelet transform must be greater than this
        threshold in order to be considered a local maxima.

    Returns
    -------
    flag : np.ndarray
        2D array of type `bool` with axes `[scale, time]` that indicates
        the location of local maxima of the modulus of the wavelet transform.
    mod_max :
        2D array of type `float` with axes `[scale, time]` that contains
        the value of the modulus of the wavelet transform at the location
        of local maxima and zero elsewhere.
    """
    # Parse input
    nscale, ntime = coeff.shape

    # Calculate modulus of wavelet transform
    mod = np.abs(coeff)

    # Flag local maxima of the modulus for each wavelet scale
    flg_mod_max = np.zeros(mod.shape, dtype=np.bool)

    for ss, sca in enumerate(scale):

        srch = max(int(search_span * sca), 1)

        slide_mod = _sliding_window(mod[ss, :], 2 * srch + 1)

        slc = slice(srch, ntime - srch)

        flg_mod_max[ss, slc] = np.all(mod[ss, slc, np.newaxis] >= slide_mod, axis=-1)

    # If requested, place threshold on modulus maxima
    if threshold is not None:
        flg_mod_max &= mod > threshold

    # Create array that only contains the modulus maxima (zero elsewhere)
    mod_max = mod * flg_mod_max.astype(np.float64)

    # Return flag and value arrays
    return flg_mod_max, mod_max


def finger_finder(scale, flag, mod_max, istart=3, do_fill=False):
    """Find local maxima that persist across scales.

    We call these fingers because of the way they look in images of the
    modulus of the wavelet transform versus time and scale.

    Parameters
    ----------
    scale : np.ndarray
        1D array of type `float` that contains the scale parameter of
        the wavelet in units of number of time samples.
    flag : np.ndarray
        2D array of type `bool` with axes `[scale, time]` that indicates
        the location of local maxima of the modulus of the wavelet transform.
    mod_max :
        2D array of type `float` with axes `[scale, time]` that contains
        the value of the modulus of the wavelet transform at the location
        of local maxima and zero elsewhere.
    istart : int
        Do not use scales with index less than this value when
        finding fingers.
    do_fill : bool
        After finding the fingers, fill in their values at scales
        with index less than `istart`.

    Returns
    -------
    candidates : np.ndarray
        2D array of type `int` with axes `[scale, candidate]` containing
        the index into the time axis where each candidate finger occurs
        at each scale.  The value -1 used to indicate that the finger does
        not exist at a particular scale.
    cmm : np.ndarray
        2D array of type `float` with axes `[scale, candidate]` containing
        the modulus maximum of the wavelet transform for each candidate
        finger at each scale.
    pdrift : np.ndarray
        1D array of type `float` with axis `[candidate]` containing the
        RMS over scales of the location of the local maximum relative too
        the location at the `istart` scale.  Units of number of time samples.
    start : np.ndarray
        1D array of type `int` with axis `[candidate]` containing the index
        of the minimum scale for each candidate finger.
    stop : np.ndarray
        1D array of type `int` with axis `[candidate]` containing the index
        of the maximum scale for each candidate finger.
    lbl : np.ndarray
        2D array of type `int` with axes `[scale, time]` that labels the
        candidate fingers with a unique integer that is the index into
        the candidate axis of the other returned arrays.  Positions that
        do not correspond to a finger are given the value -1.
    """
    nscale, ntime = flag.shape

    icandidate = np.flatnonzero(flag[istart, :])
    ncandidate = icandidate.size

    if ncandidate == 0:
        return [None] * 6

    candidates = np.zeros((nscale, ncandidate), dtype=np.int) - 1
    candidates[istart, :] = icandidate

    isort = np.argsort(mod_max[istart, icandidate])[::-1]

    ss = istart + 1
    keep_iter = True
    while (ss < nscale) and keep_iter:

        wsearch = max(int(0.25 * scale[ss]), 1)

        ipc = list(np.flatnonzero(flag[ss, :]))

        for cc in isort:

            cand = candidates[ss - 1, cc]

            if len(ipc) > 0:

                diff = [np.abs(ii - cand) for ii in ipc]

                best_match = ipc[np.argmin(diff)]

                if diff[np.argmin(diff)] <= wsearch:

                    candidates[ss, cc] = best_match

                    ipc.remove(best_match)

        iremain = np.flatnonzero(candidates[ss, :] >= 0)

        if iremain.size > 0:
            isort = iremain[np.argsort(mod_max[ss, candidates[ss, iremain]])[::-1]]
        else:
            keep_iter = False

        ss += 1

    # Fill in values below istart
    if do_fill:
        candidates[0:istart, :] = candidates[istart, np.newaxis, :]

    # Create ancillarly information
    start = np.zeros(ncandidate, dtype=np.int)
    stop = np.zeros(ncandidate, dtype=np.int)
    pdrift = np.zeros(ncandidate, dtype=np.float32)

    cmm = np.zeros((nscale, ncandidate), dtype=mod_max.dtype) * np.nan

    lbl = np.zeros((nscale, ntime), dtype=np.int) * np.nan
    lbl[flag] = -1

    for cc, index in enumerate(candidates.T):

        good_scale = np.flatnonzero(index >= 0)
        start[cc] = good_scale.min()
        stop[cc] = good_scale.max()

        pdrift[cc] = np.sqrt(
            np.sum((index[good_scale] - index[good_scale.min()]) ** 2)
            / float(good_scale.size)
        )

        for gw, igw in zip(good_scale, index[good_scale]):

            lbl[gw, igw] = cc
            cmm[gw, cc] = mod_max[gw, igw]

    # Return all information
    return candidates, cmm, pdrift, start, stop, lbl


###################################################
# main analyzer task
###################################################


class FindJumpAnalyzer(chime_analyzer.CHIMEAnalyzer):
    """Finds jumps or steps in the autocorrelations.

    Searches the autocorrelation of each input at some (user specified)
    subset of frequencies for jumps or step-like features.  Calculates
    the continuous wavelet transform of the autocorrelations using an
    asymmetric wavelet and then finds times that are local maxima of
    the modulus of the wavelet transform over a range of time scales.
    See "Singularity Detection and Processing with Wavelets" by
    Stephane Mallat and Wen Liang Hwang for more information.

    `DocLib 788 <https://bao.chimenet.ca/doc/documents/788>`_ describes this analyzer and the
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

    dias_data_<task_name>_freq_processed_total
    ...........................................
    Number of frequencies processed.

    dias_data_<task_name>_input_processed_total
    ............................................
    Number of correlator inputs processed.

    dias_task_<task_name>_jumps_total
    .................................
    Number of jumps detected for each frequency and input.

    Labels
        freq : float
            Centre frequency in MHz.
        input : int
            Feed number in cylinder order (also known as `chan_id`).

    Output Data
    -----------

    File naming
    ...........
    `<YYYYMMDD>T<HHMMSS>Z_chimecal_jumps/<SSSSSSSS>.h5`
        YYYYMMDD and HHMMSS are the date and time (in UTC) of the start of
        the underlying chimecal data acquisition from which the jump list
        was derived. SSSSSSSS is the number of seconds elapsed between
        the start of the file and the start of the acquisition.

    Indexes
    .......
    jump
        1D array of type `int` that uniquely labels each jump.
    window
        1D array of type `int` that labels the time samples in the
        window around each jump.
    weather_station_time
        1D array of type `float` that contains the unix timestamp for
        datasets obtained from the weather station.

    Searched
    ........
    files
        1D array of `str` containing the absolute path of the files that
        were searched for jumps.
    freq
        1D structured array that contains the `centre` and `width` of the
        frequency channels that were searched for jumps.
    input
        1D structured array that contains the `chan_id` and
        `correlator_input` of the inputs that were searched for jumps.
    time
        1D array of type `float` that contains the unix timestamps
        that were searched for jumps.

    Datasets
    ........
    accumulated_rain_fall
        1D array with axis [`weather_station_time`] that contains
        the accumulated rain fall in millimeters between that time
        and `rain_window` hours before that time.
    freq
        1D structured array with axis [`jump`] that contains the `centre` and
        `width` of the frequency channels where each jump occured.
    input
        1D structured array with axis [`jump`] that contains the `chan_id` and
        `correlator_input` of the input where each jump occured.
    time
        1D array of type `float` with axis [`jump`] that contains the
        unix timestamp where each jump occured.
    time_index
        1D array of type `int` with axis [`jump`] that contains the index
        into the `time` axis where the jump occured.  Must use the
        `files` in the `searched` group for this index to be valid.
    jump_size
        1D array of type `float` with axis [`jump`] that contains the size of
        the jump in fractional units.
    jump_time
        2D array of type `float` with axes [`jump`, `window`] that contains the
        unix timestamp for a window around each jump.
    jump_auto
        2D array of type `float` with axes [`jump`, `window`] that contains the
        autocorrelations for a window around each jump.
    jump_flag
        2D array of type `bool` with axes [`jump`, `window`] that indicates
        valid data in the `jump_time` and `jump_auto` datasets.

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
    max_num_freq : int
        Maximum number of frequencies to load into memory at once.
    freq_collapse : bool
        Average the normalized autocorrelations over all frequencies
        before applying the jump finding algorithm.  If True, then the
        `freq_low`, `freq_high`, and `freq_step` config variables are
        used to determine the frequency selection.  If False, then the
        `freq_physical` config variable is used.
    freq_low : float
        Search all frequencies above this threshold for jumps.  Unit is MHz.
    freq_high : float
        Search all frequencies below this threshold for jumps.  Unit is MHz.
    freq_step : int
        Downsample frequencies between `freq_low` and `freq_high` by a factor
        `freq_step`.  Unit is number of frequency channels.
    freq_physical : list
        Apply the jump finding algorithm to these frequency channels.
        Units are MHz.
    ignore_sun : bool
        Ignore times near solar transit.
    ignore_point_sources : bool
        Ignore times near the transit of the two brightest point sources.
    transit_window : float
        Window in seconds around solar and point source transits to ignore
        if the `ignore_sun` or `ignore_point_sources` config variables are set.
    use_input_flag : bool
        Do not apply algorithm to inputs that are already flagged as bad.
    input_threshold : float
        Input is considered good if it was flagged as good for more than this
        fraction of the time.  Only used if `input_flag` is True.
    rain_window : int
        Query the prometheus server for the rain fall accumulated over this
        amount of time in hours.
    wavelet_name : str
        Name of `pywt` wavelet.  Must be asymmetric for the algorithm to work.
    threshold : float
        Find jumps that are greater than this threshold in fractional units.
        For example, setting this value to 0.25 will find jumps in the
        autocorrelations that are more than 25 percent.
    search_span : float
        A time sample is considered a local maxima if the modulus of the
        wavelet transform is greater than all neighboring time samples
        within `search_span * scale` (note this search is done separately at
        each scale).
    psigma_max : float
        Find jumps that occur at the same time across scales to within
        this value.  Units are number of time samples.
    min_rise : int
        Find jumps that exceed `threshold` in less than this number
        of time samples.
    log_scale : bool
        Perform wavelet transform for scales that are logarithmically spaced.
    min_scale : int
        Minimum scale to evaluate wavelet transform.  Should be less than
        or equal to `min_rise`.  Currently there is no reason to set this
        to anything other than the value of `min_rise`.
    max_scale : int
        Find jumps that exceed `threshold` for more than this number
        of time samples.  The wavelet transform will be evaluated for
        scales up to `max_scale + 1`.
    num_scale : int
        Number of scales to evaluate the wavelet transform.  This is only
        used if `log_scale` is True.
    edge_buffer : int
        Number of time samples at the edge that are considered invalid
        is set to `edge_buffer * max_scale`.
    """

    # Config parameters related to scheduling
    offset = config.Property(proptype=str2timedelta, default="2h")

    # Config parameters defining data file selection
    instrument = config.Property(proptype=str, default="chimecal")
    max_num_file = config.Property(proptype=int, default=50)

    # Config parameters defining output data product
    output_suffix = config.Property(proptype=str, default="jumps")

    # Config parameters defining frequency selection
    max_num_freq = config.Property(proptype=int, default=8)
    freq_collapse = config.Property(proptype=bool, default=False)
    freq_low = config.Property(proptype=float, default=600.0)
    freq_high = config.Property(proptype=float, default=700.0)
    freq_step = config.Property(proptype=int, default=1)
    freq_physical = config.Property(
        proptype=list, default=[758.203125, 665.625, 558.203125, 433.59375]
    )

    # Config parameters defining time selection
    ignore_sun = config.Property(proptype=bool, default=True)
    ignore_point_sources = config.Property(proptype=bool, default=False)
    transit_window = config.Property(proptype=float, default=900.0)

    # Config parameters defining valid correlator inputs
    use_input_flag = config.Property(proptype=bool, default=False)
    input_threshold = config.Property(proptype=float, default=0.70)

    # Config parameters related to weather data
    rain_window = config.Property(proptype=int, default=6)

    # Config parameters for the jump finding algorithm
    wavelet_name = config.Property(proptype=str, default="gaus5")
    threshold = config.Property(proptype=float, default=0.15)
    search_span = config.Property(proptype=float, default=0.5)
    psigma_max = config.Property(proptype=float, default=20.0)
    min_rise = config.Property(proptype=int, default=31)
    log_scale = config.Property(proptype=bool, default=False)
    min_scale = config.Property(proptype=int, default=31)
    max_scale = config.Property(proptype=int, default=201)
    num_scale = config.Property(proptype=int, default=100)
    edge_buffer = config.Property(proptype=int, default=3)

    def setup(self):
        """Set up the analyzer.

        Create connection to data index database and archive index
        database.  Initialize Prometheus metrics.
        """
        self.logger.info(
            "Starting up. My name is %s and I am of type %s." % (self.name, __name__)
        )

        # Open connection to data index database
        # and create table if it does not exist.
        db_file = os.path.join(self.write_dir, DB_FILE)
        db_types = sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        self.data_index = sqlite3.connect(
            db_file, detect_types=db_types, check_same_thread=False
        )

        cursor = self.data_index.cursor()
        cursor.execute(CREATE_DB_TABLE)
        self.data_index.commit()

        # Open connection to archive index database
        # and create table if it does not exist.
        adb_file = os.path.join(self.write_dir, ARCHIVE_DB_FILE)
        self.archive_index = sqlite3.connect(
            adb_file, detect_types=db_types, check_same_thread=False
        )

        cursor = self.archive_index.cursor()
        cursor.execute(CREATE_ARCHIVE_DB_TABLE)
        self.archive_index.commit()

        # Add task metrics.
        self.run_timer = self.add_task_metric(
            "run_time", "Time to process single run.", unit="seconds"
        )

        self.file_counter = self.add_task_metric(
            "files", "Number of files processed.", unit="total"
        )

        self.file_timer = self.add_task_metric(
            "file_time", "Time to process single file.", unit="seconds"
        )

        # Add data metrics.
        self.freq_processed_total = self.add_data_metric(
            "freq_processed", "Number of frequencies analyzed.", unit="total"
        )

        self.input_processed_total = self.add_data_metric(
            "input_processed", "Number of inputs being analyzed.", unit="total"
        )

        self.jumps_total = self.add_data_metric(
            "jumps",
            "Number of jumps detected " + "for each frequency and input.",
            labelnames=["freq", "input"],
            unit="total",
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

        # Save important algorithm parameters to output file attributes
        params = [
            "wavelet_name",
            "threshold",
            "search_span",
            "psigma_max",
            "min_rise",
            "log_scale",
            "min_scale",
            "max_scale",
            "num_scale",
        ]
        for par in params:
            self.output_attrs[par] = getattr(self, par)

        # Specify datasets to load
        self.datasets = ["vis"]
        if self.freq_collapse:
            self.datasets.append("flags/vis_weight")

        # Calculate additional parameters needed for wavelet transform
        # from config parameters
        self.nwin = 2 * self.max_scale + 1
        self.nhwin = self.nwin // 2

        if self.log_scale:
            self.scale = np.logspace(
                np.log10(self.min_scale),
                np.log10(self.max_scale + 2),
                num=self.num_scale,
                dtype=np.int,
            )
        else:
            self.scale = np.arange(self.min_scale, self.max_scale + 2, dtype=np.int)

        self.istart = max(self.min_rise - self.min_scale, 0)
        self.nedge = self.edge_buffer * self.max_scale

    def run(self):
        """Run the analyzer.

        Load autocorrelations from the last period.
        Loop over frequencies and inputs and find jumps.
        Write locations and snapshots of the jumps to disk.
        """
        run_start_time = time.time()

        # Refresh the databases
        self.refresh_data_index()
        self.refresh_archive_index()

        # Determine the range of time to process
        end_time = datetime.datetime.utcnow() - self.offset

        cursor = self.data_index.cursor()
        query = "SELECT stop FROM files ORDER BY stop DESC LIMIT 1"
        results = list(cursor.execute(query))
        start_time = results[0][0] if results else end_time - self.period

        self.logger.info(
            "Analyzing data between {} and {}.".format(
                datetime2str(start_time), datetime2str(end_time)
            )
        )

        valid_start = ephemeris.datetime_to_unix(start_time)

        # Pad the start time to handle edge effects for the CWT
        cursor = self.archive_index.cursor()
        query = "SELECT AVG(ntime), AVG(time_step) FROM files"
        avg_ntime, avg_time_step = list(cursor.execute(query))[0]

        offset_start = datetime.timedelta(seconds=self.nedge * avg_time_step)
        offset_nfile = int(np.ceil(2 * self.nedge / float(avg_ntime)))

        # Find files that have not been analyzed
        cursor = self.archive_index.cursor()
        query = """SELECT filename, start, stop FROM files
                   WHERE ((stop > ?) AND (start < ?))
                   ORDER BY start"""
        results = list(cursor.execute(query, (start_time - offset_start, end_time)))

        acquisitions = sorted(list(set([os.path.dirname(res[0]) for res in results])))

        # Loop over acquisitions
        for aa, acq in enumerate(acquisitions):

            # Extract results within this acquisition
            all_files = [
                os.path.join(self.archive_data_dir, res[0])
                for res in results
                if res[0].startswith(acq)
            ]

            nfiles = len(all_files)

            if nfiles == 0:
                continue

            self.logger.info("Now processing acquisition %s (%d files)" % (acq, nfiles))

            # Determine the output acquisition name and make directory
            epos = acq.find(self.instrument) + len(self.instrument)
            output_acq = "_".join([acq[:epos], self.output_suffix])
            output_dir = os.path.join(self.write_dir, output_acq)

            try:
                os.makedirs(output_dir)
            except OSError:
                if not os.path.isdir(output_dir):
                    raise

            acq_start = ephemeris.datetime_to_unix(
                ephemeris.timestr_to_datetime(output_acq.split("_")[0])
            )

            # Load meta-data for this acquisition
            dset = ["flags/inputs"] if self.use_input_flag else ()

            all_data = andata.CorrData.from_acq_h5(all_files, datasets=dset)

            # Do not process this acquisition if it is too short
            if all_data.ntime < (3 * self.nedge):
                continue

            # Extract good inputs
            inputmap = tools.get_correlator_inputs(
                ephemeris.unix_to_datetime(all_data.time[0]), correlator="chime"
            )

            if self.use_input_flag:
                ngt = np.sum(all_data.flags["inputs"][:], axis=-1, dtype=np.int)
                ifeed = np.flatnonzero(
                    (ngt / float(all_data.ntime)) > self.input_threshold
                )
            else:
                ifeed = np.array(
                    [ii for ii, inp in enumerate(inputmap) if tools.is_chime(inp)]
                )

            ninp = len(ifeed)

            self.logger.debug("Processing %d feeds." % ninp)

            # Deteremine selections along the various axes
            auto_sel = np.array(
                [ii for ii, pp in enumerate(all_data.prod) if pp[0] == pp[1]]
            )
            auto_sel = andata._convert_to_slice(auto_sel)

            if not self.freq_collapse and self.freq_physical:

                if hasattr(self.freq_physical, "__iter__"):
                    freq_physical = self.freq_physical
                else:
                    freq_physical = [self.freq_physical]

                freq_sel = np.array(
                    [np.argmin(np.abs(all_data.freq - ff)) for ff in freq_physical]
                )

            else:

                in_range = np.flatnonzero(
                    (all_data.freq > self.freq_low) & (all_data.freq <= self.freq_high)
                )

                findex = np.arange(in_range[0], in_range[-1] + 1, self.freq_step)

                freq_sel = [
                    andata._convert_to_slice(find)
                    for find in _chunks(findex, self.max_num_freq)
                ]

                freq_mask = [
                    ~rfi.frequency_mask(all_data.freq[find]) for find in freq_sel
                ]

            # Determine number of files to process at once
            if self.max_num_file is None:
                chunk_size = nfiles
            else:
                chunk_size = min(self.max_num_file, nfiles)

            chunk_files = _chunks(all_files, chunk_size, m=offset_nfile)

            # Loop over chunks of files
            for chnk, files in enumerate(chunk_files):

                self.logger.info(
                    "Now processing chunk %d (%d files)" % (chnk, len(files))
                )

                self.file_start_time = time.time()

                ccursor = self.data_index.cursor()
                cquery = "SELECT stop FROM files ORDER BY stop DESC LIMIT 1"
                cresults = list(ccursor.execute(cquery))
                if cresults:
                    valid_start = ephemeris.datetime_to_unix(cresults[0][0])

                # Create list of candidates
                cfreq, cinput, ctime, cindex, csize = [], [], [], [], []
                jump_flag, jump_time, jump_auto = [], [], []
                ncandidate = 0

                # Load autocorrelations
                t0 = time.time()

                if self.freq_collapse:
                    # Since freq_collapse was requested, we will average
                    # over a large number of frequencies prior to applying
                    # the jump detection algorithm.  We load the frequencies
                    # in blocks to limit memory usage.
                    for ff, fsel in enumerate(freq_sel):

                        self.logger.debug(
                            "Loading freq block %d of %d. %s"
                            % (ff + 1, len(freq_sel), str(fsel))
                        )

                        t1 = time.time()

                        # Load this block of frequencies
                        data = andata.CorrData.from_acq_h5(
                            files,
                            datasets=self.datasets,
                            freq_sel=fsel,
                            prod_sel=auto_sel,
                            apply_gain=False,
                            renormalize=False,
                        )

                        self.logger.debug("Took %0.1f sec." % (time.time() - t1,))

                        # Save the index map
                        if not ff:
                            this_freq = data.index_map["freq"].copy()
                            this_input = data.index_map["input"].copy()
                            this_time = data.time.copy()

                            tstart = data.index_map["time"]["ctime"][0]

                            ntime = this_time.size
                            ninput = this_input.size
                            shp = (1, ninput, ntime)

                            # Initialize accumulators
                            autonorm = np.zeros(shp, dtype=np.float32)
                            weight = np.zeros(shp, dtype=np.float32)

                        else:
                            this_freq = np.concatenate(
                                (this_freq, data.index_map["freq"].copy())
                            )

                        # Normalize the autocorrelations for each frequency and
                        # input by the median value over the full time span.
                        auto = data.vis[:].real
                        inv_med_auto = tools.invert_no_zero(
                            np.median(auto, axis=-1, keepdims=True)
                        )

                        dtmp = auto * inv_med_auto - 1.0

                        # Do not include missing data or known rfi bands
                        # in the average
                        wtmp = (data.weight[:] > 0.0) & freq_mask[ff][
                            :, np.newaxis, np.newaxis
                        ]

                        # Add these frequencies to the accumulators
                        autonorm += np.sum(wtmp * dtmp, axis=0, keepdims=True)
                        weight += np.sum(wtmp, axis=0, keepdims=True)

                        # Garbage collect
                        del data, auto, inv_med_auto, dtmp, wtmp
                        gc.collect()

                    # Divide accumulated autos by accumulated weights
                    # to complete the average
                    autonorm *= tools.invert_no_zero(weight)

                    # Create a frequency axis representative of the average
                    dfreq = np.array(
                        [(np.mean(this_freq["centre"]), np.sum(this_freq["width"]))],
                        dtype=this_freq.dtype,
                    )

                else:
                    # Since freq_collapse was NOT requested, we will apply
                    # the jump detection algorithm to each frequency
                    # individually
                    data = andata.CorrData.from_acq_h5(
                        files,
                        datasets=self.datasets,
                        freq_sel=freq_sel,
                        prod_sel=auto_sel,
                        apply_gain=False,
                        renormalize=False,
                    )

                    # Save the index map
                    this_freq = data.index_map["freq"].copy()
                    this_input = data.index_map["input"].copy()
                    this_time = data.time.copy()

                    tstart = data.index_map["time"]["ctime"][0]

                    ntime = this_time.size
                    ninput = this_input.size

                    # Normalize the autocorrelations for each frequency and
                    # input by the median value over the full time span.
                    auto = data.vis[:].real
                    inv_med_auto = tools.invert_no_zero(
                        np.median(auto, axis=-1, keepdims=True)
                    )

                    autonorm = auto * inv_med_auto - 1.0

                    dfreq = this_freq

                    # Garbage collect
                    del data, auto, inv_med_auto
                    gc.collect()

                self.logger.info(
                    "Took %0.1f sec to load autocorr." % (time.time() - t0,)
                )

                t0 = time.time()

                # Create a flag indicating valid times
                valid = this_time > valid_start
                valid[: self.nedge] = False
                valid[-self.nedge :] = False

                # If requested, ignore jumps that occur during solar transit
                # or near bright source transits
                if self.ignore_sun:
                    valid &= ~_flag_transit(
                        "sun", this_time, window=self.transit_window
                    )

                if self.ignore_point_sources:
                    for ss in ["CYG_A", "CAS_A"]:
                        valid &= ~_flag_transit(
                            ss, this_time, window=self.transit_window
                        )

                # Check to make sure we have valid times to process
                if not np.any(valid):
                    self.logger.info("No valid times.  Skipping.")
                    continue

                valid_time = this_time[valid]

                # Loop over frequencies
                for ff, freq in enumerate(dfreq):

                    print_cnt = -1
                    msg = (
                        "Processing frequency "
                        + "%d (centre=%0.2f MHz, width=%0.2f MHz)"
                        % (ff, freq["centre"], freq["width"])
                    )
                    self.logger.debug(msg)

                    # Loop over inputs
                    for ii in ifeed:

                        print_cnt += 1
                        do_print = not (print_cnt % 256)

                        if do_print:
                            self.logger.debug("Processing input %d" % ii)
                            t2 = time.time()

                        signal = autonorm[ff, ii, :]

                        # Perform wavelet transform
                        coef, wv = pywt.cwt(signal, self.scale, self.wavelet_name)

                        if do_print:
                            tspan = time.time() - t2
                            self.logger.debug(
                                "Time to perform wavelet "
                                + "transform: %0.1f sec" % tspan
                            )
                            t2 = time.time()

                        # Find local modulus maxima
                        flg_mod_max, mod_max = mod_max_finder(
                            self.scale,
                            coef,
                            threshold=self.threshold,
                            search_span=self.search_span,
                        )

                        if do_print:
                            tspan = time.time() - t2
                            self.logger.debug(
                                "Time to find local modulus "
                                + "maxima: %0.1f sec" % tspan
                            )
                            t2 = time.time()

                        # Find persisent modulus maxima across scales (fingers)
                        fingers = finger_finder(
                            self.scale,
                            flg_mod_max,
                            mod_max,
                            istart=self.istart,
                            do_fill=False,
                        )
                        candidates, cmm, pdrift, start, stop, lbl = fingers

                        if do_print:
                            tspan = time.time() - t2
                            self.logger.debug(
                                "Time to find fingers: " + "%0.1f sec " % tspan
                            )

                        if candidates is None:
                            continue

                        # Cut bad candidates
                        index_good_candidates = np.flatnonzero(
                            (self.scale[stop] >= self.max_scale)
                            & valid[candidates[start, np.arange(start.size)]]
                            & (pdrift <= self.psigma_max)
                        )

                        ngood = index_good_candidates.size

                        if ngood == 0:
                            continue

                        self.logger.debug("Input %d has %d jumps" % (ii, ngood))

                        # Add remaining candidates to list
                        ncandidate += ngood

                        cfreq += [freq] * ngood
                        cinput += [this_input[ii]] * ngood

                        for igc in index_good_candidates:

                            icenter = candidates[start[igc], igc]

                            cindex.append(icenter)
                            ctime.append(this_time[icenter])

                            aa = max(0, icenter - self.nhwin)
                            bb = min(ntime, icenter + self.nhwin + 1)

                            ncut = bb - aa

                            csize.append(
                                np.median(signal[icenter + 1 : bb])
                                - np.median(signal[aa:icenter])
                            )

                            temp_var = np.zeros(self.nwin, dtype=np.bool)
                            temp_var[0:ncut] = True
                            jump_flag.append(temp_var)

                            temp_var = np.zeros(self.nwin, dtype=this_time.dtype)
                            temp_var[0:ncut] = this_time[aa:bb].copy()
                            jump_time.append(temp_var)

                            temp_var = np.zeros(self.nwin, dtype=signal.dtype)
                            temp_var[0:ncut] = signal[aa:bb].copy()
                            jump_auto.append(temp_var)

                # Get the accumulated rainfall for this period of time
                rain = self.load_rain(this_time[0], this_time[-1])

                # Create output file
                seconds_elapsed = tstart - acq_start
                output_file = os.path.join(output_dir, "%08d.h5" % seconds_elapsed)

                self.logger.info(
                    "Took %0.1f sec to process autocorr." % (time.time() - t0,)
                )

                self.logger.info("Writing %d jumps to: %s" % (ncandidate, output_file))

                # Write to output file
                with h5py.File(output_file, "w") as handler:

                    # Set the default archive attributes
                    handler.attrs["acquisition_name"] = output_acq
                    handler.attrs["njump"] = ncandidate
                    for key, val in self.output_attrs.items():
                        handler.attrs[key] = val

                    # Create a group that describes the data
                    # that was searched
                    srchd = handler.create_group("searched")
                    srchd.create_dataset("files", data=np.string_(files))
                    srchd.create_dataset("freq", data=this_freq)
                    srchd.create_dataset("input", data=this_input[ifeed])
                    srchd.create_dataset("time", data=valid_time)

                    # Create weather related index map
                    index_map = handler.create_group("index_map")
                    index_map.create_dataset("weather_station_time", data=rain["time"])

                    # Create a dataset with accumulated rain fall
                    ax = np.string_(["weather_station_time"])
                    dset = handler.create_dataset(
                        "accumulated_rain_fall", data=rain["accum"]
                    )
                    dset.attrs["axis"] = ax
                    dset.attrs["window"] = self.rain_window * 3600.0

                    # If we found any jumps, write them to the file.
                    if ncandidate > 0:

                        arr_cfreq = np.array(cfreq, dtype=this_freq.dtype)
                        arr_cinput = np.array(cinput, dtype=this_input.dtype)

                        jump_counter = Counter(
                            zip(arr_cfreq["centre"], arr_cinput["chan_id"])
                        )

                        # Add jump related index maps
                        index_map.create_dataset("jump", data=np.arange(ncandidate))
                        index_map.create_dataset("window", data=np.arange(self.nwin))

                        # Write 1D arrays containing info about each jump
                        ax = np.string_(["jump"])

                        dset = handler.create_dataset("freq", data=arr_cfreq)
                        dset.attrs["axis"] = ax

                        dset = handler.create_dataset("input", data=arr_cinput)
                        dset.attrs["axis"] = ax

                        dset = handler.create_dataset("time", data=np.array(ctime))
                        dset.attrs["axis"] = ax

                        dset = handler.create_dataset(
                            "time_index", data=np.array(cindex)
                        )
                        dset.attrs["axis"] = ax

                        dset = handler.create_dataset("jump_size", data=np.array(csize))
                        dset.attrs["axis"] = ax

                        # Write 2D arrays containing snapshots of each jump
                        ax = np.string_(["jump", "window"])

                        dset = handler.create_dataset(
                            "jump_time", data=np.array(jump_time)
                        )
                        dset.attrs["axis"] = ax

                        dset = handler.create_dataset(
                            "jump_auto", data=np.array(jump_auto)
                        )
                        dset.attrs["axis"] = ax

                        dset = handler.create_dataset(
                            "jump_flag", data=np.array(jump_flag)
                        )
                        dset.attrs["axis"] = ax

                    else:
                        jump_counter = {}

                # Update data index database
                self.update_data_index(
                    valid_time[0],
                    valid_time[-1],
                    njump=ncandidate,
                    filename=output_file,
                )

                # Update prometheus metrics
                ndf = len(files)
                self.file_counter.inc(ndf)

                time_span = time.time() - self.file_start_time
                self.file_timer.set(int(time_span / float(ndf)))

                for (lbl_freq, lbl_inp), jumps_total in jump_counter.items():
                    self.jumps_total.labels(freq=lbl_freq, input=lbl_inp).set(
                        jumps_total
                    )

                self.freq_processed_total.set(dfreq.size)

                self.input_processed_total.set(ifeed.size)

        # Set run timer
        self.run_timer.set(int(time.time() - run_start_time))

    def load_rain(self, start_time, end_time):
        """Query prometheus server for rainfall metrics.

        Parameters
        ----------
        start_time : datetime.datetime
            Load all data after this datetime (in UTC).
        end_time : datetime.datetime
            Load all data before this datetime (in UTC).

        Returns
        -------
        output : dict
            Dictionary containing a 'time' and 'accum' key that
            provide the unix timestamp and accumulated rain fall
            in millimeters between that timestamp and `rain_window`
            hours before that timestamp.
        """
        output = {}
        output["time"] = []
        output["accum"] = []

        start_datetime = ephemeris.unix_to_datetime(start_time)
        end_datetime = ephemeris.unix_to_datetime(end_time)

        # Format parameters for prometheus query
        step = (self.rain_window / 10.0) * 60.0
        sts = start_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
        ets = end_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Query prometheus running on hk-east
        # Note that our definition of start is flipped to prometheus
        prom_query = "sum_over_time(weather_rain[%dh])" % self.rain_window
        http_query = (
            "http://hk-east:9090/api/v1/query_range?query=%s&start="
            "%s&end=%s&step=%dm" % (prom_query, sts, ets, step)
        )

        resp = requests.get(http_query)
        jresp = resp.json()

        for res in jresp["data"]["result"]:
            timestamp, rain = zip(*res["values"])
            output["time"] += timestamp
            output["accum"] += [float(rr) for rr in rain]

        output["time"] = np.array(output["time"])
        output["accum"] = np.array(output["accum"])

        return output

    def update_data_index(self, start, stop, njump=0, filename=None):
        """Add row to data index database.

        Update the data index database with a row that
        contains a span of time, the number of jumps found
        during that time, and the relative path to the
        output file that contains information on the jumps found.

        Parameters
        ----------
        start : unix time
            Earliest time processed.
        stop : unix time
            Latest time processed.
        njump : int
            Number of jumps found between `start` and `stop`.
        filename : str or None
            Name of the file containing information on the jumps.
            If no jumps were found, then this will be set to None.
        """
        # Parse arguments
        dt_start = ephemeris.unix_to_datetime(ephemeris.ensure_unix(start))
        dt_stop = ephemeris.unix_to_datetime(ephemeris.ensure_unix(stop))

        relpath = None
        if filename is not None:
            relpath = os.path.relpath(filename, self.write_dir)

        # Insert row for this file
        cursor = self.data_index.cursor()
        cursor.execute(
            "INSERT INTO files VALUES (?, ?, ?, ?)", (dt_start, dt_stop, njump, relpath)
        )
        self.data_index.commit()
        self.logger.info("Added %s to data index database." % relpath)

    def refresh_data_index(self):
        """Remove expired files from the data index database.

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

    def refresh_archive_index(self):
        """Remove expired rows from the archive index database.

        Remove any rows of the archive index database
        that correspond to files that have been cleaned
        (removed) by arclink.  This has to be done by the
        analzer since full time coverage autocorrelation
        data is not archived or registered in alpenhorn.
        """
        # Find all files on archive node
        glob_str = os.path.join(
            self.archive_data_dir, "*_%s_corr" % self.instrument, "*.h5"
        )
        all_files = sorted(glob.glob(glob_str))

        # Find all files in database
        cursor = self.archive_index.cursor()
        query = "SELECT filename FROM files ORDER BY start"
        db_files = list(cursor.execute(query))

        for result in db_files:

            relpath = result[0]
            filename = os.path.join(self.archive_data_dir, relpath)

            if filename in all_files:
                # File exists on node.
                all_files.remove(filename)

            else:
                # File no longer exists on node, remove from database.
                cursor = self.archive_index.cursor()
                cursor.execute("DELETE FROM files WHERE (filename = ?)", (relpath,))
                self.archive_index.commit()

        # Add all remaining files to the database
        for filename in all_files:

            relpath = os.path.relpath(filename, self.archive_data_dir)

            try:
                with h5py.File(filename, "r") as handler:
                    ftime = handler["index_map"]["time"]["ctime"][:]
            except Exception as ex:
                self.logger.info("Could not load %s:  %s" % (filename, ex))
                continue

            ntime = ftime.size
            if ntime < 2:
                self.logger.info(
                    "%s only contains %d samples. Skipping." % (filename, ntime)
                )
                continue

            delta_t = np.median(np.abs(np.diff(ftime)))

            ftime += 0.5 * delta_t
            time_start = ftime[0]
            time_stop = ftime[-1]

            dt_start = ephemeris.unix_to_datetime(time_start)
            dt_stop = ephemeris.unix_to_datetime(time_stop)

            # Insert row for this file
            cursor = self.archive_index.cursor()
            cursor.execute(
                "INSERT INTO files VALUES (?, ?, ?, ?, ?)",
                (dt_start, dt_stop, ntime, delta_t, relpath),
            )
            self.archive_index.commit()

    def finish(self):
        """Close connection to data index and archive index databases."""
        self.logger.info("Shutting down.")
        self.data_index.close()
        self.archive_index.close()
