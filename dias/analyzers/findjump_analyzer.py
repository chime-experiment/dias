"""
=============================================
Find jumps or steps in autocorrelations
(:mod:`~dias.analyzers.findjump_analyzer`)
=============================================

.. currentmodule:: dias.analyzers.findjump_analyzer

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
# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

import os
import time
import datetime
import sqlite3
import gc

import numpy as np
import h5py
import pywt

from ch_util import tools, ephemeris, andata, data_index
from ch_util.fluxcat import FluxCatalog

from caput import config

from dias import chime_analyzer
from datetime import datetime
from dias.utils.time_strings import str2timedelta, datetime2str

DB_FILE = "data_index.db"
CREATE_DB_TABLE = '''CREATE TABLE IF NOT EXISTS files(start TIMESTAMP, stop TIMESTAMP,
                                                      njump INTEGER, filename TEXT UNIQUE ON CONFLICT REPLACE)'''

###################################################
# auxiliary functions
###################################################

def _flag_transit(name, timestamp, window=900.0):

    extend = 24.0 * 3600.0

    flag = np.zeros(timestamp.size, dtype=np.bool)

    if name.lower() == 'sun':
        ttrans = ephemeris.solar_transit(timestamp[0] - extend,
                                         timestamp[-1] + extend)
    else:
        ttrans = ephemeris.transit_times(FluxCatalog[name].skyfield,
                                         timestamp[0] - extend,
                                         timestamp[-1] + extend)

    # Flag +/- window around each transit
    for tt in ttrans:
        flag |= ((timestamp >= (tt-window)) & (timestamp <= (tt + window)))

    return flag


def _chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def _sliding_window(arr, window):

    # Advanced numpy tricks
    shape = arr.shape[:-1] + (arr.shape[-1]-window+1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def mod_max_finder(scale, coeff, search_span=0.5, threshold=None):

    # Parse input
    nscale, ntime = coeff.shape

    # Calculate modulus of wavelet transform
    mod = np.abs(coeff)

    # Flag local maxima of the modulus for each wavelet scale
    flg_mod_max = np.zeros(mod.shape, dtype=np.bool)

    for ss, sca in enumerate(scale):

        srch = max(int(search_span * sca), 1)

        slide_mod = _sliding_window(mod[ss, :], 2*srch + 1)

        slc = slice(srch, ntime-srch)

        flg_mod_max[ss, slc] = np.all(mod[ss, slc, np.newaxis] >= slide_mod, axis=-1)

    # If requested, place threshold on modulus maxima
    if threshold is not None:
        flg_mod_max &= (mod > threshold)

    # Create array that only contains the modulus maxima (zero elsewhere)
    mod_max = mod * flg_mod_max.astype(np.float64)

    # Return flag and value arrays
    return flg_mod_max, mod_max


def finger_finder(scale, flag, mod_max, istart=3, do_fill=False):

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

            cand = candidates[ss-1, cc]

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

        pdrift[cc] = np.sqrt(np.sum((index[good_scale] - index[good_scale.min()])**2) / float(good_scale.size))

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
    """

    # Config parameters related to scheduling
    offset = config.Property(proptype=str2timedelta, default='10h')
    period = config.Property(proptype=str2timedelta, default='6h')

    # Config parameters defining data file selection
    instrument = config.Property(proptype=str, default='chimecal')
    max_num_file = config.Property(proptype=int, default=20)

    # Config parameters defining output data product
    output_suffix = config.Property(proptype=str, default='jumps')

    # Config parameters defining frequency selection
    freq_collapse = config.Property(proptype=bool, default=False)
    freq_low = config.Property(proptype=float, default=600.0)
    freq_high = config.Property(proptype=float, default=700.0)
    freq_step = config.Property(proptype=float, default=0.390625)
    freq_physical = config.Property(proptype=list,
                    default=[758.203125, 665.625, 558.203125, 433.59375])

    # Config parameters defining time selection
    ignore_daytime = config.Property(proptype=bool, default=False)
    ignore_sun = config.Property(proptype=bool, default=True)
    ignore_point_sources = config.Property(proptype=bool, default=False)
    transit_window = config.Property(proptype=float, default=900.0)

    # Config parameters defining valid correlator inputs
    use_input_flag = config.Property(proptype=bool, default=False)
    input_threshold = config.Property(proptype=float, default=0.70)

    # Config parameters for the jump finding algorithm
    wavelet_name = config.Property(proptype=str, default='gaus5')
    threshold = config.Property(proptype=float, default=0.50)
    search_span = config.Property(proptype=float, default=0.5)
    psigma_max = config.Property(proptype=float, default=20.0)
    min_rise = config.Property(proptype=float, default=31)
    min_scale = config.Property(proptype=float, default=31)
    max_scale = config.Property(proptype=float, default=201)
    log_scale = config.Property(proptype=bool, default=False)
    num_scale = config.Property(proptype=int, default=100)

    def setup(self):
        """Create connection to data index database
        and initialize Prometheus metrics.
        """
        self.logger.info('Starting up. My name is %s and I am of type %s.' % (self.name, __name__))

        # Open connection to data index database and create table if it does not exist
        self.data_index = sqlite3.connect(os.path.join(self.write_dir, DB_FILE))

        cursor = self.data_index.cursor()
        cursor.execute(CREATE_DB_TABLE)
        self.data_index.commit()

        # Add a task metric that counts how often this task ran and how long it took to complete.
        # It will be exported as dias_task_findjump_analyzer_runs_total.
        self.run_counter = self.add_task_metric("runs", "Number of times the task ran.", unit="total")
        self.run_timer = self.add_task_metric("run_time", "Time to process single run.", unit="seconds")

        # Add additional metrics
        self.file_counter = self.add_task_metric("files", "Number of files processed.", unit="total")
        self.file_timer = self.add_task_metric("file_time", "Time to process single file.", unit="seconds")

        self.nfreq_processed = self.add_task_metric("number_freq_processed", "Number of frequencies being analyzed.")
        self.nfreq_detected = self.add_task_metric("number_freq_detected", "Number of unique frequencies with a jump detected.")

        self.ninput_processed = self.add_task_metric("number_input_processed", "Number of inputs being analyzed.")
        self.ninput_detected = self.add_task_metric("number_input_detected", "Number of unique inputs with a jump detected.")

        self.njump_detected = self.add_task_metric("number_jump_detected", "Number of jumps detected per frequency.")


    def run(self):
        """Load autocorrelations from the last period,
        find jumps or steps, and write location to disk.
        """

        # Calculate the start and end of the passed period.
        end_time = datetime.now() - self.offset
        start_time = end_time - self.period

        self.logger.info('Analyzing data between {} and {}.'.format(datetime2str(start_time),
                                                                    datetime2str(end_time)))

        self.run_start_time = time.time()

        # Calculate the wavelet transform for the following scales
        nwin = 2 * self.max_scale + 1
        nhwin = nwin // 2

        if self.log_scale:
            self.logger.info("Using log scale.")
            scale = np.logspace(np.log10(self.min_scale), np.log10(nwin), num=self.num_scale, dtype=np.int)
        else:
            self.logger.info("Using linear scale.")
            scale = np.arange(self.min_scale, nwin, dtype=np.int)

        # Use Finder to get the files to analyze
        finder = self.Finder()
        finder.accept_all_global_flags()
        finder.only_corr()
        finder.filter_acqs(data_index.ArchiveInst.name == self.instrument)
        finder.set_time_range(start_time, end_time)

        if self.ignore_daytime:
            finder.exclude_daytime()

        # Loop over acquisitions
        for aa, acq in enumerate(finder.acqs):

            # Extract finder results within this acquisition
            acq_results = finder.get_results_acq(aa)

            # Loop over contiguous periods within this acquisition
            for all_data_files, (tstart, tend) in acq_results:

                nfiles = len(all_data_files)

                if nfiles == 0:
                    continue

                self.logger.info("Now processing acquisition %s (%d files)" % (acq.name, nfiles))

                # Determine list of feeds to examine
                dset = ['flags/inputs'] if self.use_input_flag else ()

                rdr = andata.CorrReader(all_data_files)
                rdr.select_time_range(tstart, tend)
                rdr.dataset_sel = dset

                all_data = rdr.read()

                inputmap = tools.get_correlator_inputs(ephemeris.unix_to_datetime(all_data.time[0]),
                                                       correlator='chime')

                # Extract good inputs
                if self.use_input_flag:
                    ifeed = np.flatnonzero((np.sum(all_data.flags['inputs'][:], axis=-1, dtype=np.int) /
                                             float(all_data.flags['inputs'].shape[-1])) > self.input_threshold)
                else:
                    ifeed = np.array([ii for ii, inp in enumerate(inputmap) if tools.is_chime(inp)])

                ninp = len(ifeed)

                self.logger.info("Processing %d feeds." % ninp)

                # Determine number of files to process at once
                if self.max_num_file is None:
                    chunk_size = nfiles
                else:
                    chunk_size = min(self.max_num_file, nfiles)

                # Loop over chunks of files
                for chnk, data_files in enumerate(_chunks(all_data_files, chunk_size)):

                    self.logger.info("Now processing chunk %d (%d files)" % (chnk, len(data_files)))

                    self.file_start_time = time.time()

                    # Create list of candidates
                    cfreq, cinput, ctime, cindex, csize = [], [], [], [], []
                    jump_flag, jump_time, jump_auto = [], [], []
                    ncandidate = 0

                    # Deteremine selections along the various axes
                    rdr = andata.CorrReader(data_files)

                    rdr.select_time_range(tstart, tend)
                    datasets = ['vis']
                    if self.freq_collapse:
                        datasets.append('flags/vis_weight')
                    rdr.dataset_sel = datasets

                    auto_sel = np.array([ii for ii, pp in enumerate(rdr.prod) if pp[0] == pp[1]])
                    auto_sel = andata._convert_to_slice(auto_sel)
                    rdr.prod_sel = auto_sel

                    if not self.freq_collapse and self.freq_physical:

                        if hasattr(self.freq_physical, '__iter__'):
                            freq_physical = self.freq_physical
                        else:
                            freq_physical = [self.freq_physical]

                        rdr.select_freq_physical(freq_physical)

                    else:

                        rdr.select_freq_range(freq_low=self.freq_low,
                                              freq_high=self.freq_high,
                                              freq_step=self.freq_step)

                    rdr.apply_gain = False
                    rdr.renormalize = False

                    start, stop = rdr.time_sel

                    # Load autocorrelations
                    t0 = time.time()

                    data = rdr.read()

                    self.logger.debug("Took %0.1f seconds to load autocorrelations." % (time.time() - t0,))

                    # Save the index map
                    this_time = data.time.copy()
                    this_freq = data.index_map['freq'].copy()
                    this_input = data.index_map['input'].copy()

                    ntime = this_time.size

                    # If requested, ignore jumps during solar transit or near bright source transits
                    flag_quiet = np.ones(ntime, dtype=np.bool)
                    if self.ignore_sun:
                        flag_quiet &= ~_flag_transit('sun', this_time, window=self.transit_window)

                    if self.ignore_point_sources:
                        for ss in ["CYG_A", "CAS_A", "TAU_A", "VIR_A"]:
                            flag_quiet &= ~_flag_transit(ss, this_time, window=self.transit_window)

                    # If requested, collapse over frequency axis
                    auto = data.vis[:].real
                    fractional_auto = auto * tools.invert_no_zero(np.median(auto, axis=-1, keepdims=True)) - 1.0

                    if self.freq_collapse:
                        weight = (data.weight > 0.0).astype(np.float32)
                        inv_norm = tools.invert_no_zero(np.sum(weight, axis=0, keepdims=True))

                        fractional_auto = np.sum(weight * fractional_auto, axis=0, keepdims=True) * inv_norm

                        dfreq = np.array([(np.mean(this_freq['centre']), np.sum(this_freq['width']))],
                                         dtype=[('centre', '<f8'), ('width', '<f8')])

                    else:

                        dfreq = this_freq

                    # Garbage collect
                    del data
                    gc.collect()

                    # Loop over frequencies
                    for ff, freq in enumerate(dfreq):

                        print_cnt = 0
                        self.logger.info("Processing frequency %d (centre %0.2f MHz, width %0.2f MHz)" %
                                         (ff, freq['centre'], freq['width']))

                        # Loop over inputs
                        for ii in ifeed:

                            print_cnt += 1
                            do_print = not (print_cnt % 256)

                            if do_print:
                                self.logger.debug("Processing input %d" % ii)
                                t0 = time.time()

                            signal = fractional_auto[ff, ii, :]

                            # Perform wavelet transform
                            coef, wv = pywt.cwt(signal, scale, self.wavelet_name)

                            if do_print:
                                self.logger.debug("Took %0.1f seconds to perform wavelet transform." % (time.time() - t0,))
                                t0 = time.time()

                            # Find local modulus maxima
                            flg_mod_max, mod_max = mod_max_finder(scale, coef, threshold=self.threshold, search_span=self.search_span)

                            if do_print:
                                self.logger.debug("Took %0.1f seconds to find modulus maxima." % (time.time() - t0,))
                                t0 = time.time()

                            # Find persisent modulus maxima across scales
                            candidates, cmm, pdrift, start, stop, lbl = finger_finder(scale, flg_mod_max, mod_max,
                                                                                      istart=max(self.min_rise - self.min_scale, 0),
                                                                                      do_fill=False)

                            if do_print:
                                self.logger.debug("Took %0.1f seconds to find fingers." % (time.time() - t0,))
                                t0 = time.time()

                            if candidates is None:
                                continue

                            # Cut bad candidates
                            index_good_candidates = np.flatnonzero((scale[stop] >= self.max_scale) &
                                                                    flag_quiet[candidates[start, np.arange(start.size)]] &
                                                                    (pdrift <= self.psigma_max))

                            ngood = index_good_candidates.size

                            if ngood == 0:
                                continue

                            self.logger.info("Input %d has %d jumps" % (ii, ngood))

                            # Add remaining candidates to list
                            ncandidate += ngood

                            cfreq += [freq] * ngood
                            cinput += [this_input[ii]] * ngood

                            for igc in index_good_candidates:

                                icenter = candidates[start[igc], igc]

                                cindex.append(start + icenter)
                                ctime.append(this_time[icenter])

                                aa = max(0, icenter - nhwin)
                                bb = min(ntime, icenter + nhwin + 1)

                                ncut = bb - aa

                                csize.append(np.median(signal[icenter+1:bb]) -
                                             np.median(signal[aa:icenter]))

                                temp_var = np.zeros(nwin, dtype=np.bool)
                                temp_var[0:ncut] = True
                                jump_flag.append(temp_var)

                                temp_var = np.zeros(nwin, dtype=this_time.dtype)
                                temp_var[0:ncut] = this_time[aa:bb].copy()
                                jump_time.append(temp_var)

                                temp_var = np.zeros(nwin, dtype=signal.dtype)
                                temp_var[0:ncut] = signal[aa:bb].copy()
                                jump_auto.append(temp_var)


                    # If we found any jumps, write them to a file.
                    if ncandidate > 0:

                        output_acq = '_'.join([acq.name[:acq.name.find(self.instrument)+len(self.instrument)],
                                               self.output_suffix])

                        output_dir = os.path.join(self.write_dir, output_acq)

                        try:
                            os.makedirs(output_dir)
                        except OSError:
                            if not os.path.isdir(output_dir):
                                raise

                        acq_start = ephemeris.datetime_to_unix(ephemeris.timestr_to_datetime(output_acq.split('_')[0]))

                        seconds_elapsed = this_time[0] - acq_start
                        output_file = os.path.join(output_dir, "%08d.h5" % seconds_elapsed)

                        self.logger.info("Writing %d jumps to: %s" % (ncandidate, output_file))

                        # Write to output file
                        with h5py.File(output_file, 'w') as handler:

                            # Set the default archive attributes
                            handler.attrs['type'] = str(type(self))
                            handler.attrs['git_version_tag'] = subprocess.check_output(["git", "-C", os.path.dirname(__file__),
                                                                                        "describe", "--always"]).strip()
                            handler.attrs['collection_server'] = subprocess.check_output(["hostname"]).strip()
                            handler.attrs['system_user'] = subprocess.check_output(["id", "-u", "-n"]).strip()

                            # Create a dataset that indicates the data that was searched
                            searched = handler.create_group('searched')
                            searched.create_dataset('files', data=data_files)
                            searched.create_dataset('freq', data=this_freq)
                            searched.create_dataset('input', data=this_input[ifeed])
                            searched.create_dataset('time', data=this_time)

                            # Create an index map
                            index_map = handler.create_group('index_map')
                            index_map.create_dataset('jump', data=np.arange(ncandidate))
                            index_map.create_dataset('window', data=np.arange(nwin))

                            # Write 1D arrays containing info about each jump
                            ax = np.array(['jump'])

                            dset = handler.create_dataset('freq', data=np.array(cfreq, ))
                            dset.attrs['axis'] = ax

                            dset = handler.create_dataset('input', data=np.array(cinput, dtype=[('chan_id', '<u2'),
                                                                                         ('correlator_input', 'S32')]))
                            dset.attrs['axis'] = ax

                            dset = handler.create_dataset('time', data=np.array(ctime))
                            dset.attrs['axis'] = ax

                            dset = handler.create_dataset('time_index', data=np.array(cindex))
                            dset.attrs['axis'] = ax

                            dset = handler.create_dataset('jump_size', data=np.array(csize))
                            dset.attrs['axis'] = ax

                            # Write 2D arrays containing snapshots of each jump
                            ax = np.array(['jump', 'window'])

                            dset = handler.create_dataset('jump_flag', data=np.array(jump_flag))
                            dset.attrs['axis'] = ax

                            dset = handler.create_dataset('jump_time', data=np.array(jump_time))
                            dset.attrs['axis'] = ax

                            dset = handler.create_dataset('jump_auto', data=np.array(jump_auto))
                            dset.attrs['axis'] = ax

                    else:
                        self.logger.info("No jumps found for %s acquisition, chunk %d." % (acq.name, chnk))

                        output_file = None

                    # Update data index database
                    self.update_data_index(this_time[0], this_time[-1], njump=ncandidate, filename=output_file)

                    # Update prometheus metrics
                    ndf = len(data_files)
                    self.file_counter.inc(ndf)
                    self.file_timer.set(int((time.time() - self.file_start_time) / float(ndf)))

                    self.nfreq_processed.set(dfreq.size)
                    self.nfreq_detected.set(np.unique(np.array(cfreq)['centre']).size)

                    self.ninput_processed.set(ifeed.size)
                    self.ninput_detected.set(np.unique(np.array(cinput)['chan_id']).size)

                    self.njump_detected.set(ncandidate)


        # Increment run counter.
        self.run_counter.inc()
        self.run_timer.set(int(time.time() - self.run_start_time))


    def update_data_index(self, start, stop, njump=0, filename=None):

        # Parse arguments
        dt_start = ephemeris.unix_to_datetime(ephemeris.ensure_unix(start))
        dt_stop = ephemeris.unix_to_datetime(ephemeris.ensure_unix(stop))

        relpath = None
        if filename is not None:
            relpath = os.path.relpath(filename, self.write_dir)

        # Insert row for this file
        try:
            cursor = self.data_index.cursor()
            cursor.execute("INSERT INTO files VALUES (?, ?, ?, ?)", (dt_start, dt_stop, njump, relpath))

        except Exception as ex:
            self.log.error("Could not perform database insert: %s" % ex)

        else:
            self.data_index.commit()


    def finish(self):
        """Close connection to data index database."""
        self.logger.info('Shutting down.')

        # Close connection to database
        self.data_index.close()

