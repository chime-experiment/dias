"""
=====================================================
Generate RFI mask from the stacked autocorrelations.
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
import sqlite3
import gc

import numpy as np
import h5py

from ch_util import tools, ephemeris, andata, data_index, rfi

from caput import config

from dias import chime_analyzer
from datetime import datetime
from dias.utils.time_strings import str2timedelta, datetime2str

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
    
    frac = np.sum(mask, axis=tuple(axis), dtype=np.float32) / float(np.prod(mask.shape[list(axis)]))

    if logical_not:
        frac = 1.0 - frac
        
    return frac 

###################################################
# main analyzer task
###################################################

class FlagRFIAnalyzer(chime_analyzer.CHIMEAnalyzer):
    """Finds jumps or steps in the autocorrelations.
    """

    # Config parameters related to scheduling
    offset = config.Property(proptype=str2timedelta, default='10h')
    period = config.Property(proptype=str2timedelta, default='6h')

    # Config parameters defining data file selection
    instrument = config.Property(proptype=str, default='chimestack')
    max_num_file = config.Property(proptype=int, default=20)

    # Config parameters defining output data product
    output_suffix = config.Property(proptype=str, default='rfimask')

    # Config parameters defining frequency selection
    freq_low = config.Property(proptype=float, default=399.0)
    freq_high = config.Property(proptype=float, default=801.0)

    # Config parameters for the rfi masking algorithm
    apply_static_mask = config.Property(proptype=bool, default=True)
    freq_width = config.Property(proptype=float, default=10.0)
    time_width = config.Property(proptype=float, default=420.0)
    rolling = config.Property(proptype=bool, default=True)
    threshold_mad = config.Property(proptype=float, default=6.0)

    def setup(self):
        """Create connection to data index database and initialize Prometheus metrics.
        """
        self.logger.info('Starting up. My name is %s and I am of type %s.' % (self.name, __name__))

        # Open connection to data index database and create table if it does not exist
        self.data_index = sqlite3.connect(os.path.join(self.write_dir, DB_FILE))

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

        self.fraction_masked_before = self.add_task_metric("fraction_masked_before", 
                                            "Fraction of data considered bad before applying MAD threshold.",
                                            labelnames=['stack'])

        self.fraction_masked_after = self.add_task_metric("fraction_masked_after", 
                                            "Fraction of data considered bad after applying MAD threshold.",
                                            labelnames=['stack'])


    def run(self):
        """Load stacked autocorrelations from the last period, generate rfi mask,
        and write it to disk.
        """

        # Calculate the start and end of the passed period.
        end_time = datetime.now() - self.offset
        start_time = end_time - self.period

        self.logger.info('Analyzing data between {} and {}.'.format(datetime2str(start_time),
                                                                    datetime2str(end_time)))

        self.run_start_time = time.time()

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
                    rdr = andata.CorrReader(data_files)

                    rdr.select_time_range(tstart, tend)
                    rdr.dataset_sel = ['vis', 'flags/vis_weight']

                    auto_sel = np.array([ii for ii, pp in enumerate(rdr.prod) if pp[0] == pp[1]])
                    auto_sel = andata._convert_to_slice(auto_sel)
                    rdr.prod_sel = auto_sel

                    rdr.select_freq_range(freq_low=self.freq_low, freq_high=self.freq_high)

                    rdr.apply_gain = False
                    rdr.renormalize = False

                    start, stop = rdr.time_sel

                    # Load autocorrelations
                    t0 = time.time()

                    data = rdr.read()

                    self.logger.debug("Took %0.1f seconds to load autocorrelations." % (time.time() - t0,))
                    t0 = time.time()

                    # Construct RFI mask for each cylinder/polarisation
                    cyl_index, cyl_auto, cyl_ndev = rfi.number_deviations(data, stack=False,
                                                                                apply_static_mask=self.apply_static_mask,
                                                                                freq_width=self.freq_width,
                                                                                time_width=self.time_width,
                                                                                rolling=self.rolling)

                    cyl_mask_before = np.isfinite(cyl_ndev)
                    cyl_mask_after = cyl_ndev <= self.threshold_mad

                    # Construct RFI mask for stacked incoherent beam
                    index, auto, ndev = rfi.number_deviations(data, stack=True,
                                                                    apply_static_mask=self.apply_static_mask,
                                                                    freq_width=self.freq_width,
                                                                    time_width=self.time_width,
                                                                    rolling=self.rolling)

                    mask_before = np.isfinite(ndev)
                    mask_after = ndev <= self.threshold_mad

                    self.logger.debug("Took %0.1f seconds to generate mask." % (time.time() - t0,))

                    # Concatenate and define stack axis
                    auto = np.concatenate((cyl_auto, auto), axis=1)
                    mask_before = np.concatenate((cyl_mask_before, mask_before), axis=1)
                    mask_after = np.concatenate((cyl_mask_after, mask_after), axis=1)

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

                    seconds_elapsed = this_time[0] - acq_start
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
                        index_map.create_dataset('stack', data=np.array(stack))
                        index_map.create_dataset('time', data=data.time)

                        # Write 2D arrays containing snapshots of each jump
                        ax = np.array(['freq', 'stack', 'time'])

                        dset = handler.create_dataset('auto', data=auto)
                        dset.attrs['axis'] = ax

                        dset = handler.create_dataset('mask_before', data=mask_before)
                        dset.attrs['axis'] = ax

                        dset = handler.create_dataset('mask', data=mask_after)
                        dset.attrs['axis'] = ax


                    # Update data index database
                    self.update_data_index(data.time[0], data.time[-1], filename=output_file)

                    # Update prometheus metrics
                    ndf = len(data_files)
                    self.file_counter.inc(ndf)
                    self.file_timer.set(int((time.time() - self.file_start_time) / float(ndf)))

                    for ss, lbl in enumerate(stack):
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


    def finish(self):
        """Close connection to data index database."""
        self.logger.info('Shutting down.')

        # Close connection to database
        self.data_index.close()

