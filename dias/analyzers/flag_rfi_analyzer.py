"""
=====================================================
Generates RFI mask from the stacked autocorrelations.
(:mod:`~dias.analyzers.flag_rfi_analyzer`)
=====================================================

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
# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

import os
import time
import datetime
import subprocess
import sqlite3
import gc

import numpy as np
import h5py

from ch_util import tools, ephemeris, andata, data_index, rfi
from caput import config

from dias import chime_analyzer
from dias.utils.string_converter import str2timedelta, datetime2str

DB_FILE = "data_index.db"
CREATE_DB_TABLE = '''CREATE TABLE IF NOT EXISTS files(start TIMESTAMP, stop TIMESTAMP,
                                                      filename TEXT UNIQUE ON CONFLICT REPLACE)'''

CYL_MAP = {xx:chr(63+xx) for xx in range(2, 6)}
POL_MAP = {'S': 'X', 'E': 'Y'}

###################################################
# auxiliary functions
###################################################

def _chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def _fraction_flagged(mask, axis=None, logical_not=False):

    if axis is None:
        axis = np.arange(mask.ndim)
    else:
        axis = np.atleast_1d(axis)

    frac = (np.sum(mask, axis=tuple(axis), dtype=np.float32) /
            float(np.prod([mask.shape[ax] for ax in axis])))

    if logical_not:
        frac = 1.0 - frac

    return frac

###################################################
# main analyzer task
###################################################

class FlagRFIAnalyzer(chime_analyzer.CHIMEAnalyzer):
    """Identifies data contaminated by RFI.

    Attributes
    -----------
    offset : str
        Process data this timedelta before current time.
    period : str
        Cadence at which this analyzer is run.
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
    offset = config.Property(proptype=str2timedelta, default='10h')
    period = config.Property(proptype=str2timedelta, default='6h')

    # Config parameters defining data file selection
    instrument = config.Property(proptype=str, default='chimestack')
    max_num_file = config.Property(proptype=int, default=1)

    # Config parameters defining output data product
    output_suffix = config.Property(proptype=str, default='rfimask')

    # Config parameters defining frequency selection
    freq_low = config.Property(proptype=float, default=400.0)
    freq_high = config.Property(proptype=float, default=800.0)

    # Config parameters for the rfi masking algorithm
    apply_static_mask = config.Property(proptype=bool, default=True)
    freq_width = config.Property(proptype=float, default=10.0)
    time_width = config.Property(proptype=float, default=420.0)
    rolling = config.Property(proptype=bool, default=True)
    threshold_mad = config.Property(proptype=float, default=6.0)

    def setup(self):
        """Creates connection to data index database and initializes Prometheus metrics.
        """
        self.logger.info('Starting up. My name is %s and I am of type %s.' % (self.name, __name__))

        # Open connection to data index database and create table if it does not exist
        self.data_index = sqlite3.connect(os.path.join(self.write_dir, DB_FILE),
                                          detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)

        cursor = self.data_index.cursor()
        cursor.execute(CREATE_DB_TABLE)
        self.data_index.commit()

        # Add a task metric that counts how often this task ran and how long it took to complete.
        # It will be exported as dias_task_flag_rfi_analyzer_runs_total.
        self.run_counter = self.add_task_metric("runs", "Number of times the task ran.", unit="total")

        self.run_timer = self.add_task_metric("run_time", "Time to process single run.", unit="seconds")

        # Add additional metrics
        self.file_counter = self.add_task_metric("files", "Number of files processed.", unit="total")

        self.file_timer = self.add_task_metric("file_time", "Time to process single file.", unit="seconds")

        self.fraction_masked_missing = self.add_task_metric("fraction_masked_missing",
                                                            "Fraction of data that is missing " +
                                                            "(e.g., dropped packets or down GPU nodes).",
                                                            labelnames=['stack'])

        self.fraction_masked_before = self.add_task_metric("fraction_masked_before",
                                                            "Fraction of data considered bad before applying MAD threshold.  " +
                                                            "Includes missing data and static frequency mask.",
                                                            labelnames=['stack'])

        self.fraction_masked_after = self.add_task_metric("fraction_masked_after",
                                                            "Fraction of data considered bad after applying MAD threshold.  " +
                                                            "Includes missing data, static frequency mask, and MAD threshold mask.",
                                                            labelnames=['stack'])


    def run(self):
        """Loads stacked autocorrelations from the last period,
        generates rfi mask, writes it to disk, and updates the
        data index database.
        """

        self.run_start_time = time.time()

        # Refresh the database
        self.refresh_data_index()

        # Determine the range of time to process
        end_time = datetime.datetime.utcnow() - self.offset

        cursor = self.data_index.cursor()
        results = list(cursor.execute('SELECT stop FROM files ORDER BY stop DESC LIMIT 1'))
        start_time = results[0][0] if results else end_time - self.period

        self.logger.info('Analyzing data between {} and {}.'.format(datetime2str(start_time),
                                                                    datetime2str(end_time)))

        # Use Finder to get the files to analyze
        try:
            finder = self.Finder()
            finder.accept_all_global_flags()
            finder.only_corr()
            finder.filter_acqs(data_index.ArchiveInst.name == self.instrument)
            finder.set_time_range(start_time, end_time)

        except Exception as exc:
            self.logger.info('No data found: %s' % exc)
            return

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

                inputmap = tools.get_correlator_inputs(ephemeris.unix_to_datetime(tstart),
                                                       correlator='chime')

                # Determine number of files to process at once
                if self.max_num_file is None:
                    chunk_size = nfiles
                else:
                    chunk_size = min(self.max_num_file, nfiles)

                # Loop over chunks of files
                for chnk, data_files in enumerate(_chunks(all_data_files, chunk_size)):

                    self.logger.info("Now processing chunk %d (%d files)" % (chnk, len(data_files)))

                    self.file_start_time = time.time()

                    # Deteremine selections along the various axes
                    rdr = andata.CorrData.from_acq_h5(data_files, datasets=())

                    within_range = np.flatnonzero((rdr.freq >= self.freq_low) & (rdr.freq <= self.freq_high))
                    freq_sel = slice(within_range[0], within_range[-1]+1)

                    stack_sel = [ii for ii, pp in enumerate(rdr.prod[rdr.stack['prod']]) if pp[0] == pp[1]]

                    # Load autocorrelations
                    t0 = time.time()

                    data = andata.CorrData.from_acq_h5(data_files, datasets=['vis', 'flags/vis_weight'],
                                                                   freq_sel=freq_sel, stack_sel=stack_sel,
                                                                   apply_gain=False, renormalize=False)

                    self.logger.info("Took %0.1f seconds to load autocorrelations." % (time.time() - t0,))
                    t0 = time.time()

                    # Construct RFI mask for each cylinder/polarisation
                    cyl_index, cyl_auto, cyl_ndev = rfi.number_deviations(data, stack=False,
                                                                                apply_static_mask=self.apply_static_mask,
                                                                                freq_width=self.freq_width,
                                                                                time_width=self.time_width,
                                                                                rolling=self.rolling)

                    # Construct RFI mask for stacked incoherent beam
                    index, auto, ndev = rfi.number_deviations(data, stack=True,
                                                                    apply_static_mask=self.apply_static_mask,
                                                                    freq_width=self.freq_width,
                                                                    time_width=self.time_width,
                                                                    rolling=self.rolling)

                    self.logger.info("Took %0.1f seconds to generate mask." % (time.time() - t0,))

                    # Concatenate and define stack axis
                    auto = np.concatenate((cyl_auto, auto), axis=1)
                    ndev = np.concatenate((cyl_ndev, ndev), axis=1)

                    mask_missing = auto > 0.0
                    mask_before = np.isfinite(ndev)
                    mask_after = ndev <= self.threshold_mad

                    stack = [POL_MAP[inputmap[ii].pol] + '-' + CYL_MAP[inputmap[ii].cyl] for ii in cyl_index]
                    stack.append('ALL')

                    # Determine name of ouput file
                    output_acq = '_'.join([acq.name[:acq.name.find(self.instrument)+len(self.instrument)],
                                           self.output_suffix])

                    output_dir = os.path.join(self.write_dir, output_acq)

                    try:
                        os.makedirs(output_dir)
                    except OSError:
                        if not os.path.isdir(output_dir):
                            raise

                    acq_start = ephemeris.datetime_to_unix(ephemeris.timestr_to_datetime(output_acq.split('_')[0]))

                    seconds_elapsed = data.time[0] - acq_start
                    output_file = os.path.join(output_dir, "%08d.h5" % seconds_elapsed)

                    self.logger.info("Writing RFI mask to: %s" % output_file)

                    # Write to output file
                    with h5py.File(output_file, 'w') as handler:

                        # Set the default archive attributes
                        handler.attrs['type'] = str(type(self))
                        handler.attrs['git_version_tag'] = subprocess.check_output(["git", "-C", os.path.dirname(__file__),
                                                                                    "describe", "--always"]).strip()
                        handler.attrs['collection_server'] = subprocess.check_output(["hostname"]).strip()
                        handler.attrs['system_user'] = subprocess.check_output(["id", "-u", "-n"]).strip()

                        # Create an index map
                        index_map = handler.create_group('index_map')
                        index_map.create_dataset('freq', data=data.index_map['freq'])
                        index_map.create_dataset('stack', data=np.string_(stack))
                        index_map.create_dataset('time', data=data.time)

                        # Write 2D arrays containing snapshots of each jump
                        ax = np.string_(['freq', 'stack', 'time'])

                        dset = handler.create_dataset('mask', data=mask_after)
                        dset.attrs['axis'] = ax
                        dset.attrs['threshold'] = self.threshold_mad

                        dset = handler.create_dataset('auto', data=auto)
                        dset.attrs['axis'] = ax

                        dset = handler.create_dataset('ndev', data=ndev)
                        dset.attrs['axis'] = ax


                    # Update data index database
                    self.update_data_index(data.time[0], data.time[-1], filename=output_file)

                    # Update prometheus metrics
                    ndf = len(data_files)
                    self.file_counter.inc(ndf)
                    self.file_timer.set(int((time.time() - self.file_start_time) / float(ndf)))

                    for ss, lbl in enumerate(stack):
                        self.fraction_masked_missing.labels(stack=lbl).set(
                                        _fraction_flagged(mask_missing[:, ss, :], logical_not=True)
                                        )

                        self.fraction_masked_before.labels(stack=lbl).set(
                                        _fraction_flagged(mask_before[:, ss, :], logical_not=True)
                                        )

                        self.fraction_masked_after.labels(stack=lbl).set(
                                        _fraction_flagged(mask_after[:, ss, :], logical_not=True)
                                        )

                    # Garbage collect
                    del data
                    gc.collect()

        # Increment run counter.
        self.run_counter.inc()
        self.run_timer.set(int(time.time() - self.run_start_time))


    def update_data_index(self, start, stop, filename=None):
        """Update the data index database with a row that
        contains the name of the file and the span of time
        the file contains.

        Parameters
        ----------
        start : unix time
            Earliest time contained in the file.
        stop : unix time
            Latest time contanied in the file.
        filename : str
            Name of the file.
        """
        # Parse arguments
        dt_start = ephemeris.unix_to_datetime(ephemeris.ensure_unix(start))
        dt_stop = ephemeris.unix_to_datetime(ephemeris.ensure_unix(stop))

        relpath = None
        if filename is not None:
            relpath = os.path.relpath(filename, self.write_dir)

        # Insert row for this file
        try:
            cursor = self.data_index.cursor()
            cursor.execute("INSERT INTO files VALUES (?, ?, ?)", (dt_start, dt_stop, relpath))

        except Exception as ex:
            self.log.error("Could not perform database insert: %s" % ex)

        else:
            self.data_index.commit()


    def refresh_data_index(self):
        """Remove any rows of the data index database
        that correspond to files that have been cleaned
        (removed) by dias manager.
        """

        cursor = self.data_index.cursor()
        all_files = list(cursor.execute('SELECT filename FROM files ORDER BY start'))

        for result in all_files:

            filename = result[0]

            if not os.path.isfile(os.path.join(self.write_dir, filename)):

                try:
                    cursor = self.data_index.cursor()
                    cursor.execute('DELETE FROM files WHERE filename = ?', (filename,))

                except Exception as ex:
                    self.log.error("Could not perform database deletion: %s" % ex)

                else:
                    self.data_index.commit()
                    self.log.info("Removed %s from data index database." % filename)


    def finish(self):
        """Closes connection to data index database.
        """
        self.logger.info('Shutting down.')
        self.data_index.close()

