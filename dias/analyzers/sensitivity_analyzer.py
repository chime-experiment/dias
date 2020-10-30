"""Sensitivity analyzer.

Based on CHIMEAnalyzer class.
"""

from dias import CHIMEAnalyzer
from datetime import datetime
import calendar
from caput import config
from dias.utils.string_converter import str2timedelta
from chimedb import data_index
import sqlite3
from dias.utils.helpers import get_cyl

import os
import subprocess
import gc

import h5py
import numpy as np

from collections import Counter
from collections import defaultdict

from ch_util import andata
from ch_util import tools
from ch_util import ephemeris
from dias import exception
from dias import __version__ as dias_version_tag


class SensitivityAnalyzer(CHIMEAnalyzer):
    """SensitivityAnalyzer.

    Analyzer for telescope sensitivity.

    `DocLib 792 <https://bao.chimenet.ca/doc/documents/792>`_ describes this analyzer and the
    associated theremin graph.

    Metrics
    ----------
    dias_data_<task name>_average_sensitivity
    ................................................
    RMS of thermal noise, averaged over inter-cylinder
    baselines, all frequencies, and 1.5 hours.

    Labels
        pol : Polarization of the feeds averaged (EW/NS).

    Output Data
    -----------------
        h5 file, containing noise rms (Jy),
        averaged over all feeds for each polarization,
        as a function of frequency and time.
        The input file is the chime stacked dataset.

    File naming
    ..........................
        `<TIME>_<output_suffix>.h5`
        `TIME` is a unix timestamp of the first time record in each file and
        `output_suffix` is the value of the config variable with the same name.
        Output file is created for each input file read.

    Indexes
    .............
    freq
        Frequency indexes.
    pol
        Two polarizations for the feeds,
        index 0 is E-W polarization and
        index 1 is N-S polarization.
    time
        Unix time at which data is recorded.

    Datasets
    ...................
    rms
        Noise rms, computed from the fast cadence data,
        as a function of frequency and time.
    count
        Normalization to the summed and weighted noise variance.

    Attributes for output dataset
    .............................
    instrument_name
        Correlator for the acquired (chime).
    collection_server
        Machine at which the script is run.
    system_user
        User account running the script.
    git_version_tag
        Version of code used.

    State Data
    -----------
    None

    Config Variables
    -----------------------

    Attributes
    ----------
    correlator : str
        Source of the input data
    output_suffix : str
        Suffix for the output file
    acq_suffix : str
        Type of dataset to be read
    nfreq_per_block : int
        number of frequency channels to be run in one block
        loading all frequency channels at the same time leads to memory error
    include_auto : bool
        option to include autocorrelation
    include_intracyl : bool
        option in include intracylinder baselines
    include_crosspol : bool
        option to include crosspol data
    sep_cyl : bool
        option to preserve cylinder pairs
    cyl_start_char : int
        Starting character for the cylinders (ASCII)
    cyl_start_num : int
        Offset for CHIME cylinders
        (due to inital numbers allotted to pathfinder)
    lag : timedelta
        Number of hours before time of script execution for searching the files

    """

    correlator = config.Property(proptype=str, default="chime")
    instrument = config.Property(proptype=str, default="chimestack")
    output_suffix = config.Property(proptype=str, default="sensitivity")
    acq_suffix = config.Property(proptype=str, default="corr")

    nfreq_per_block = config.Property(proptype=int, default=16)
    include_auto = config.Property(proptype=bool, default=False)
    include_intracyl = config.Property(proptype=bool, default=False)
    include_crosspol = config.Property(proptype=bool, default=False)
    sep_cyl = config.Property(proptype=bool, default=False)
    cyl_start_char = config.Property(proptype=int, default=65)
    cyl_start_num = config.Property(proptype=int, default=2)
    lag = config.Property(proptype=str2timedelta, default="4h")

    def setup(self):
        """Open connection to data index database.

        Creates table if it does not exist.
        Further, it adds the data metric.
        """
        # Check for database

        # Add a data metric for sensitivity.
        self.sens = self.add_data_metric(
            "average_sensitivity",
            """Thermal noise estimate averaged over
            all frequencies, inter-cylinder baselines, and
            1.5 hours""",
            unit="ujy",
            labelnames=["pol"],
        )

    def run(self):
        """Task stage: analyzes data from the last period."""
        stop_time = datetime.utcnow() - self.lag

        # Find all calibration files
        file_list = self.new_files(filetypes=self.instrument + "_" + self.acq_suffix)

        if not file_list:
            err_msg = "No {}_{} files found from last {}.".format(
                self.instrument, self.acq_suffix, self.period
            )
            raise exception.DiasDataError(err_msg)

        with h5py.File(file_list[0], "r") as hf:
            start_time = ephemeris.unix_to_datetime(hf["index_map/time"][0]["ctime"])

        self.logger.info(
            "Calculating sensitivity from %s to %s" % (str(start_time), str(stop_time))
        )

        # Get Unix time for the start time for timestamp
        time_tuple = start_time.timetuple()
        start_time_unix = calendar.timegm(time_tuple)
        timestamp0 = start_time_unix

        # Look up inputmap
        inputmap = tools.get_correlator_inputs(
            ephemeris.unix_to_datetime(timestamp0), correlator=self.correlator
        )

        # Read a sample file for getting index map
        file_sample = file_list[0]
        data = andata.CorrData.from_acq_h5(
            file_sample,
            datasets=["reverse_map", "flags/inputs"],
            apply_gain=False,
            renormalize=False,
        )

        # Get baselines
        prod, prodmap, dist, conj, cyl, scale = self.get_baselines(
            data.index_map, inputmap
        )

        for files in file_list:

            # Load index map and reverse map
            data = andata.CorrData.from_acq_h5(
                files,
                datasets=["reverse_map", "flags/inputs"],
                apply_gain=False,
                renormalize=False,
            )

            flag_ind = data.flags["inputs"]

            # Determine axes
            nfreq = data.nfreq
            nblock = int(np.ceil(nfreq / float(self.nfreq_per_block)))

            # Also used in the output file name and database
            timestamp = data.time
            ntime = timestamp.size

            # Determine groups
            polstr = np.array(sorted(prod.keys()))
            npol = polstr.size

            # Calculate counts
            cnt = np.zeros((data.index_map["stack"].size, ntime), dtype=np.float32)

            if np.any(flag_ind[:]):
                for pp, ss in zip(
                    data.index_map["prod"][:], data.reverse_map["stack"]["stack"][:]
                ):
                    cnt[ss, :] += flag_ind[pp[0], :] * flag_ind[pp[1], :]
            else:
                for ss, val in Counter(
                    data.reverse_map["stack"]["stack"][:]
                ).iteritems():
                    cnt[ss, :] = val

            # Initialize arrays
            var = np.zeros((nfreq, npol, ntime), dtype=np.float32)
            counter = np.zeros((nfreq, npol, ntime), dtype=np.float32)

            # Loop over frequency blocks
            for index_0, block_number in enumerate(range(nblock)):

                fstart = block_number * self.nfreq_per_block
                fstop = min((block_number + 1) * self.nfreq_per_block, nfreq)
                freq_sel = slice(fstart, fstop)

                self.logger.debug(
                    "Processing block %d (of %d):  %d - %d"
                    % (block_number + 1, nblock, fstart, fstop)
                )

                bdata = andata.CorrData.from_acq_h5(
                    files,
                    freq_sel=freq_sel,
                    datasets=["flags/vis_weight"],
                    apply_gain=False,
                    renormalize=False,
                )

                bflag = (bdata.weight[:] > 0.0).astype(np.float32)
                bvar = tools.invert_no_zero(bdata.weight[:])

                # Loop over polarizations
                for ii, pol in enumerate(polstr):

                    pvar = bvar[:, prod[pol], :]
                    pflag = bflag[:, prod[pol], :]
                    pcnt = cnt[np.newaxis, prod[pol], :]
                    pscale = scale[pol][np.newaxis, :, np.newaxis]

                    var[freq_sel, ii, :] += np.sum(
                        (pscale * pcnt) ** 2 * pflag * pvar, axis=1
                    )
                    counter[freq_sel, ii, :] += np.sum(pscale * pcnt * pflag, axis=1)

                del bdata
                gc.collect()

            # Normalize
            inv_counter = tools.invert_no_zero(counter)
            var *= inv_counter ** 2

            # Compute metric to be exported
            self.sens.labels(pol="EW").set(
                1.0e6 * np.sqrt(1.0 / np.sum(tools.invert_no_zero(var[:, 0, :])))
            )
            self.sens.labels(pol="NS").set(
                1.0e6 * np.sqrt(1.0 / np.sum(tools.invert_no_zero(var[:, 1, :])))
            )

            # Write to file
            output_file = os.path.join(
                self.write_dir, "%d_%s.h5" % (timestamp[0], self.output_suffix)
            )
            self.logger.info("Writing output file...")

            with h5py.File(output_file, "w") as handler:

                index_map = handler.create_group("index_map")
                index_map.create_dataset("freq", data=data.index_map["freq"][:])
                index_map.create_dataset("pol", data=np.string_(polstr))
                index_map.create_dataset("time", data=data.time)

                dset = handler.create_dataset("rms", data=np.sqrt(var))
                dset.attrs["axis"] = np.array(["freq", "pol", "time"], dtype="S")

                dset = handler.create_dataset("count", data=counter.astype(np.int))
                dset.attrs["axis"] = np.array(["freq", "pol", "time"], dtype="S")

                handler.attrs["instrument_name"] = self.correlator
                handler.attrs["collection_server"] = subprocess.check_output(
                    ["hostname"]
                ).strip()
                handler.attrs["system_user"] = subprocess.check_output(
                    ["id", "-u", "-n"]
                ).strip()
                handler.attrs["git_version_tag"] = dias_version_tag

            self.add_output_file(data.time[0], data.time[-1], output_file)
            self.logger.info("File successfully written out.")

            self.register_done([files])

    def get_baselines(self, indexmap, inputmap):
        """Return baseline indices for averaging."""
        prod = defaultdict(list)
        prodmap = defaultdict(list)
        dist = defaultdict(list)
        conj = defaultdict(list)
        cyl = defaultdict(list)
        scale = defaultdict(list)

        feedpos = tools.get_feed_positions(inputmap)

        for pp, (this_prod, this_conj) in enumerate(indexmap["stack"]):

            if this_conj:
                bb, aa = indexmap["prod"][this_prod]
            else:
                aa, bb = indexmap["prod"][this_prod]

            inp_aa = inputmap[aa]
            inp_bb = inputmap[bb]

            if not tools.is_chime(inp_aa) or not tools.is_chime(inp_bb):
                continue

            if not self.include_auto and (aa == bb):
                continue

            if not self.include_intracyl and (inp_aa.cyl == inp_bb.cyl):
                continue

            this_dist = list(feedpos[aa, :] - feedpos[bb, :])

            if tools.is_array_x(inp_aa) and tools.is_array_x(inp_bb):
                key = "XX"

            elif tools.is_array_y(inp_aa) and tools.is_array_y(inp_bb):
                key = "YY"

            elif not self.include_crosspol:
                continue

            elif tools.is_array_x(inp_aa) and tools.is_array_y(inp_bb):
                key = "XY"

            elif tools.is_array_y(inp_aa) and tools.is_array_x(inp_bb):
                key = "YX"

            else:
                raise RuntimeError("CHIME feeds not polarized.")

            this_cyl = "%s%s" % (
                get_cyl(inp_aa.cyl, self.cyl_start_num, self.cyl_start_char),
                get_cyl(inp_bb.cyl, self.cyl_start_num, self.cyl_start_char),
            )
            if self.sep_cyl:
                key = key + "-" + this_cyl

            prod[key].append(pp)
            prodmap[key].append((aa, bb))
            conj[key].append(this_conj)
            dist[key].append(this_dist)
            cyl[key].append(this_cyl)

            if aa == bb:
                scale[key].append(0.5)
            else:
                scale[key].append(1.0)

        for key in prod.keys():
            prod[key] = np.array(prod[key])
            prodmap[key] = np.array(
                prodmap[key], dtype=[("input_a", "<u2"), ("input_b", "<u2")]
            )
            dist[key] = np.array(dist[key])
            conj[key] = np.nonzero(np.ravel(conj[key]))[0]
            cyl[key] = np.array(cyl[key])
            scale[key] = np.array(scale[key])

        tools.change_chime_location(default=True)

        return prod, prodmap, dist, conj, cyl, scale
