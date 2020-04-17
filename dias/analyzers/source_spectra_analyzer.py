"""Extracts flux of a given source, as a function of freq, when the source is right at the zenith.

.. currentmodule:: dias.analyzers.source_spectra_analyzer

Classes
=======

.. autosummary::
    :toctree: generated/

    SourceSpectraAnalyzer

Functions
=========

.. autosummary::
    :toctree: generated/

    fit_point_source_transit

"""

import os
import gc
import time
import subprocess
from collections import Counter, defaultdict

import h5py
import numpy as np

from caput import config
from chimedb import data_index
from ch_util import ephemeris, andata, tools, cal_utils, fluxcat
from dias import CHIMEAnalyzer, exception
from dias import __version__ as dias_version_tag
from dias.utils.string_converter import str2timedelta
from dias.utils.helpers import get_cyl

from dias import CHIMEAnalyzer
from datetime import datetime
from caput import config
from dias.utils.string_converter import str2timedelta
from chimedb import data_index
import sqlite3
from dias.utils.helpers import get_cyl

import os
import subprocess
import gc
import scipy.constants
import json

import h5py
import numpy as np

from ch_util import andata
from ch_util import tools
from ch_util import ephemeris
from dias import exception
from dias import __version__ as dias_version_tag


########################################################
# main analyzer task
########################################################

DB_FILE = "data_index_source.db"
CREATE_DB_TABLE = """CREATE TABLE IF NOT EXISTS files(
                     start TIMESTAMP, stop TIMESTAMP,
                     filename TEXT UNIQUE ON CONFLICT REPLACE)"""

class SourceSpectraAnalyzer(CHIMEAnalyzer):
    """Extracts flux of a given source transit, as a function of freq, when the source is right at the zenith.

    If the flux of the source is similar to what we expect from other measurements, our calibration is doing a good job.

    Metrics
    -------

    Output Data
    -----------
        h5 file, containing fringestopped visibility (Jy),
        averaged over all feeds for each polarization,
        as a function of frequency and time.
        The input file is the chime stacked dataset.

    File naming
    ...........
        `<src>_<csd>_<output_suffix>.h5`
        `src` is the source analyzed, `csd` (int) marks the day of transit and
        `output_suffix` is the value of the config variable with the same name.

        An output file is created for each input file read.

    Indexes
    .......
    freq
        1D array of type `float` representing an index of frequencies.
    pol
        1D array which references the polarization for the feed.
        There are two polarizations:
        index 0 is E-W polarization and
        index 1 is N-S polarization.
    time
        1D array of type `float` that contains the Unix timestamps at which data is recorded.
    ra
        1D array which contains the right ascension of the source.
    ha
        1D array which contains the hour angle covered in the output file.

    Datasets
    ........
    vis
        Fringestopped visibility as a function of frequency and time.
    weight
        Inverse variance, computed from the fast cadence data,
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
    lag : timedelta
        Number of hours before time of script execution for searching the files
    correlator : str
        Source of the input data
    acq_suffix : str
        Type of dataset to be read
    source_transits : list of str
        Source tranits that we want analyses for
    output_suffix : str
        Suffix for the output file
    cyl_start_char : int
        Starting character for the cylinders (ASCII)
    cyl_start_num : int
        Offset for CHIME cylinders
        (due to inital numbers allotted to pathfinder)
    sep_cyl : bool
        option to preserve cylinder pairs
    include_auto : bool
        option to include autocorrelation
    include_intracyl : bool
        option in include intracylinder baselines
    include_crosspol : bool
        option to include crosspol data
    nfreq_per_block : int
        number of frequency channels to be run in one block
        loading all frequency channels at the same time leads to memory error
    nsgima : float
        Span required from the peak of transit
        in terms of sigma
    process_daytime: int
        0 would only process sources that transit at night
        1 would process all sources that transit when the sun is outside of the primary beam
        2 would process all sources
    N2_freq: array, float
        List of frequencies to be exported to prometheus
    poly_deg_phi: int
        Degree of polynomial for fitting the transit
    peak_type: str
        Peak of the transit can be 0 hour angle or maximum flux

    """

    # Config parameters related to scheduling
    lag = config.Property(proptype=str2timedelta, default="4h")

    # Config parameters defining data file selection
    correlator = config.Property(proptype=str, default="chime")
    acq_suffix = config.Property(proptype=str, default="stack_corr")
    source_transits = config.Property(proptype=list, default=["CYG_A","CAS_A","TAU_A","VIR_A","HERCULES_A","3C_353","HYDRA_A","3C_123"])
    N2_freq = config.Property(proptype=list, default=[433.59375,558.203125,665.625,758.203125])

    # Config parameters defining output data product
    output_suffix = config.Property(proptype=str, default="spectrum")

    # Config parameters related to cylinders
    cyl_start_char = config.Property(proptype=int, default=65)
    cyl_start_num = config.Property(proptype=int, default=2)
    sep_cyl = config.Property(proptype=bool, default=False)

    # Config parameters related to optional inclusions
    include_auto = config.Property(proptype=bool, default=False)
    include_intracyl = config.Property(proptype=bool, default=False)
    include_crosspol = config.Property(proptype=bool, default=False)
    process_daytime = config.Property(proptype=int, default=2)

    # Config parameters related to the algorithm
    nfreq_per_block = config.Property(proptype=int, default=16)
    nsigma = config.Property(proptype=float, default=1.0)
    poly_deg_phi = config.Property(proptype=int, default=3)
    peak_type = config.Property(proptype=str, default='max_amp')

    def setup(self):
        """
        Set up the task.

        Creates metrics.
        """
        
        """Open connection to data index database.

        Creates table if it does not exist.
        Further, it adds the data metric.
        """
        # Check for database

        db_file = os.path.join(self.state_dir, DB_FILE)
        db_types = sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        self.data_index = sqlite3.connect(
            db_file, detect_types=db_types, check_same_thread=False
        )

        cursor = self.data_index.cursor()
        cursor.execute(CREATE_DB_TABLE)
        self.data_index.commit()

	# initalise run metric
        self.flux_metric = self.add_data_metric(
            "flux", labelnames=['frequency', 'source', 'pol'], unit='Jansky')

    def run(self):
        """Run the analyzer.

	 "Load stacked visibilities and beam form to the
	  location of the brightest sources."

        Write fluxes for a given time and frequency to disk.
        """
        lat = np.radians(ephemeris.CHIMELATITUDE)
        err_msg = ""
            
        # Create transit tracker
        for src in self.source_transits:
            try:
                src_body = fluxcat.FluxCatalog[src].skyfield
            except KeyError as e:
                raise DiasConfigError("Invalid source: {}".format(e))

            self.logger.info(
                "Initializing offline point source processing for {}.".format(src)
            )

            # Query files from now to period hours back
            query_stop_time = datetime.utcnow() - self.lag
            query_start_time = query_stop_time - self.period
        
            # Refresh the database
            self.refresh_data_index()

            cursor = self.data_index.cursor()
            query = "SELECT stop FROM files ORDER BY stop DESC LIMIT 1"
            results = list(cursor.execute(query))
            query_start_time = results[0][0] if results else query_stop_time - self.period

            self.logger.info(
                "Searching for transits from %s to %s"
                % (str(query_start_time), str(query_stop_time))
            )

            # nsigma distance in degree from transit from peak
            time_delta = 2*self.nsigma * cal_utils.guess_fwhm(
                400.0, pol="X", dec=src_body.dec.radians, sigma=True,
            seconds=True)

            # Find all calibration files that have transit of given source
            f = self.Finder()
            f.set_time_range(query_start_time, query_stop_time)
            f.accept_all_global_flags()
            f.only_corr()
            f.filter_acqs((data_index.ArchiveInst.name == self.acq_suffix))
            f.include_transits(src_body, time_delta=time_delta)
            file_list = f.get_results()
            nfile = len(file_list)
            times = [file_list[ii][1] for ii in range(0, nfile)]

            try:
                all_files = file_list[0][0]
                if not all_files:
                    raise IndexError()
            except IndexError:
                tmp = "No {} files found from last {} for source transit {}.\n".format(
                    self.acq_suffix, self.period, src
                )
                err_msg += tmp
                self.logger.info(tmp)
                continue

            # Loop over files
            for file_index, files in enumerate(all_files):
                # Read file time range
                with h5py.File(files, "r") as handler:
                    file_timestamp = handler["index_map"]["time"]["ctime"][:]

                csd = int(np.median(ephemeris.csd(file_timestamp)))

                # Compute source coordinates
                src_ra, src_dec = ephemeris.object_coords(
                    src_body, date=np.median(file_timestamp), deg=False
                )
                source_start_time, source_stop_time = file_list[file_index][1]

                is_daytime = 0.0
                # test if the source is transiting in the daytime
                solar_rise = ephemeris.solar_rising(
                    source_start_time - 24.0 * 3600.0, end_time=source_stop_time
                )
                for rr in solar_rise:
                    ss = ephemeris.solar_setting(rr)[0]
                    if (source_start_time <= ss) and (rr <= source_stop_time):
                        is_daytime += 1
                        tt = ephemeris.solar_transit(rr)[0]
                        # if Sun is in the beam
                        if source_start_time <= tt <= source_stop_time:
                            is_daytime += 1
                        break

                if self.process_daytime < is_daytime:
                    self.logger.info(
                        "Not processing %s as it does not meet daytime processing conditions" % (src)
                    )
                    continue

                self.logger.info("Now processing %s transit on CSD %d" % (src, csd))

                # Look up inputmap
                inputmap = tools.get_correlator_inputs(
                    ephemeris.unix_to_datetime(ephemeris.csd_to_unix(csd)),
                    correlator=self.correlator,
                )

                # Load index map and reverse map
                data = andata.CorrData.from_acq_h5(
                    files,
                    start=int(np.argmin(np.abs(file_timestamp - source_start_time))),
                    stop=int(np.argmin(np.abs(file_timestamp - source_stop_time))),
                    datasets=["reverse_map", "flags/inputs"],
                    apply_gain=False,
                    renormalize=False,
                )
                # Determine axes
                nfreq = data.nfreq
                nblock = int(np.ceil(nfreq / float(self.nfreq_per_block)))

                data_timestamp = data.time  # 1D array
                ntime = data_timestamp.size

                # Get baselines
                prod, _, dist, _, _, scale = self.get_baselines(
                    data.index_map, inputmap, data.reverse_map["stack"]
                )

                # Determine groups
                pols = np.array(sorted(prod.keys()))
                npol = pols.size

                # Calculate counts
                cnt = np.zeros((data.index_map["stack"].size, ntime), dtype=np.float32)

                if np.any(data.flags["inputs"][:]):
                    for pp, ss in zip(
                        data.index_map["prod"][:], data.reverse_map["stack"]["stack"][:]
                    ):
                        cnt[ss, :] += (
                            data.flags["inputs"][pp[0], :]
                            * data.flags["inputs"][pp[1], :]
                        )
                else:
                    for ss, val in Counter(
                        data.reverse_map["stack"]["stack"][:]
                    ).iteritems():
                        cnt[ss, :] = val

                # Calculate hour angle
                ra = np.radians(ephemeris.lsa(data_timestamp))
                ha = ra - src_ra
                ha = _correct_phase_wrap(ha, deg=False)
                ha = ha[np.newaxis, np.newaxis, :]

                # Initialize arrays
                vis = np.zeros((nfreq, npol, ntime), dtype=np.complex64)
                var = np.zeros((nfreq, npol, ntime), dtype=np.float32)
                counter = np.zeros((nfreq, npol, ntime), dtype=np.float32)

                # Loop over frequency blocks
                for block_number in range(nblock):

                    t0 = time.time()

                    fstart = block_number * self.nfreq_per_block
                    fstop = min((block_number + 1) * self.nfreq_per_block, nfreq)
                    freq_sel = slice(fstart, fstop)

                    self.logger.info(
                        "Processing block %d (of %d):  %d - %d"
                        % (block_number, nblock, fstart, fstop)
                    )

                    bdata = andata.CorrData.from_acq_h5(
                        files,
                        start=int(
                            np.argmin(np.abs(file_timestamp - source_start_time))
                        ),
                        stop=int(np.argmin(np.abs(file_timestamp - source_stop_time))),
                        freq_sel=freq_sel,
                        datasets=["vis", "flags/vis_weight"],
                        apply_gain=False,
                        renormalize=False,
                    )

                    bflag = (bdata.weight[:] > 0.0).astype(np.float32)
                    bvar = tools.invert_no_zero(bdata.weight[:])

                    lmbda = (
                        scipy.constants.c * 1e-6 / bdata.freq[:, np.newaxis, np.newaxis]
                    )

                    # Loop over polarizations
                    for ii, pol in enumerate(pols):

                        self.logger.info("Processing Pol %s" % pol)

                        pvis = bdata.vis[:, prod[pol], :]
                        pvar = bvar[:, prod[pol], :]
                        pflag = bflag[:, prod[pol], :]
                        pcnt = cnt[np.newaxis, prod[pol], :]
                        pscale = scale[pol][np.newaxis, :, np.newaxis]

                        fringestop_phase = tools.fringestop_phase(
                            ha,
                            lat,
                            src_dec,
                            dist[pol][np.newaxis, :, 0, np.newaxis] / lmbda,
                            dist[pol][np.newaxis, :, 1, np.newaxis] / lmbda,
                        )

                        vis[freq_sel, ii, :] += np.sum(
                            pscale * pcnt * pflag * pvis * fringestop_phase, axis=1
                        )
                        var[freq_sel, ii, :] += np.sum(
                            (pscale * pcnt) ** 2 * pflag * pvar, axis=1
                        )
                        counter[freq_sel, ii, :] += np.sum(
                            pscale * pcnt * pflag, axis=1
                        )

                    self.logger.info(
                        "Took %0.1f seconds to process this block."
                        % (time.time() - t0,)
                    )

                    del bdata
                    gc.collect()

                # Normalize
                inv_counter = tools.invert_no_zero(counter)
                vis *= inv_counter
                var *= inv_counter ** 2

                ra = np.degrees(np.unwrap(ra))
                ha = ra - np.degrees(src_ra)

                # Fit response to model
                fwhm = np.zeros((nfreq, npol), dtype=np.float32)
                for ii in range(npol):
                    fwhm[:, ii] = cal_utils.guess_fwhm(
                        data.freq, pol=pols[ii][0], dec=src_dec, sigma=True
                    )

                flag = counter > 0.0

                fitter = cal_utils.FitGaussAmpPolyPhase(poly_deg_phi=self.poly_deg_phi)
                fitter.fit(ha, vis.real, np.sqrt(var), width=fwhm, absolute_sigma=True)
                resid = vis.real - fitter.predict(ha)
                resid_rms = np.std(resid, axis=-1)
                
                if self.peak_type == 'max_amp':
                    ha_max = np.nanmedian(fitter.peak()) #To have one slice in time
                elif self.peak_type == 'zero_ha':
                    ha_max = 0.0
                else:
                    raise ValueError("Peak type not recognized")

                peak_flux = fitter.predict(ha_max) 
                
                N2_flux = np.zeros((len(self.N2_freq), npol))
                N2_freq_ind = find_freq(data.freq, self.N2_freq)
                N2_flux = peak_flux[N2_freq_ind].real
                
                for ii, freq_export in enumerate(data.freq[N2_freq_ind]):
                    for jj in range(npol):
                        self.flux_metric.labels(frequency=freq_export, source=src, pol=jj).set(N2_flux[ii,jj])

                # Write to file
                output_file = os.path.join(
                    self.write_dir,
                    "%s_csd_%d_%s.h5" % (src.lower(), csd, self.output_suffix),
                )
                self.logger.info("Writing output files...")
            
                self.update_data_index(data.time[0], data.time[-1], filename=output_file)

                with h5py.File(output_file, "w") as handler:

                    index_map = handler.create_group("index_map")
                    index_map.create_dataset("freq", data=data.index_map["freq"][:])
                    index_map.create_dataset("pol", data=np.string_(pols))
                    index_map.create_dataset("time", data=data.time)
                    index_map.create_dataset("ra", data=ra)
                    index_map.create_dataset("ha", data=ha)

                    dset = handler.create_dataset("vis", data=vis)
                    dset.attrs["axis"] = np.array(["freq", "pol", "ha"], dtype="S")
                    
                    dset = handler.create_dataset("peak_vis", data=peak_flux)
                    dset.attrs["axis"] = np.array(["freq", "pol"], dtype="S")

                    dset = handler.create_dataset(
                        "weight", data=tools.invert_no_zero(var)
                    )
                    dset.attrs["axis"] = np.array(["freq", "pol", "ha"], dtype="S")

                    dset = handler.create_dataset("count", data=counter.astype(np.int))
                    dset.attrs["axis"] = np.array(["freq", "pol", "ha"], dtype="S")


                    index_map.create_dataset("param", data="model_fitting")

                    dset = handler.create_dataset("residual_noise", data=resid_rms)
                    dset.attrs["axis"] = np.array(["freq", "pol"], dtype="S")

                    dset = handler.create_dataset("parameter", data=fitter.param)
                    dset.attrs["axis"] = np.array(
                        ["freq", "pol", "param"], dtype="S"
                    )

                    dset = handler.create_dataset(
                        "parameter_cov", data=fitter.param_cov
                    )
                    dset.attrs["axis"] = np.array(
                        ["freq", "pol", "param", "param"], dtype="S"
                    )

                    handler.attrs["source"] = src
                    handler.attrs["csd"] = csd
                    handler.attrs["chisq"] = fitter.chisq
                    handler.attrs["ndof"] = fitter.ndof
                    handler.attrs["model_kwargs"] = json.dumps(fitter.model_kwargs)
                    handler.attrs["model_class"] = ".".join(
                            [getattr(cal_utils.FitGaussAmpPolyPhase, key) for key in ["__module__", "__name__"]]
                            )
                    handler.attrs["is_daytime"] = is_daytime
                    handler.attrs["instrument_name"] = self.correlator
                    handler.attrs["collection_server"] = subprocess.check_output(
                            ["hostname"]).strip()
                    handler.attrs["system_user"] = subprocess.check_output(
                            ["id", "-u", "-n"]).strip()
                    handler.attrs["git_version_tag"] = dias_version_tag
                    self.logger.info("File for {} successfully written out.".format(src))

        if err_msg:
            raise exception.DiasDataError(err_msg)

    def get_baselines(self, indexmap, inputmap, reverse_stack):
        """Return baseline indices for averaging."""
        prod = defaultdict(list)
        prodmap = defaultdict(list)
        dist = defaultdict(list)
        conj = defaultdict(list)
        cyl = defaultdict(list)
        scale = defaultdict(list)

        # Compute feed positions with rotation
        feedpos = tools.get_feed_positions(inputmap)
        stack, stack_flag = tools.redefine_stack_index_map(
                inputmap, indexmap["prod"], indexmap["stack"], reverse_stack)
	
        if not np.all(stack_flag):
            self.logger.warning(
                    "There are %d stacked baselines that are masked "
                    "by the inputmap." % np.sum(~stack_flag))

        for pp, (this_prod, this_conj) in enumerate(stack):

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

    def update_data_index(self, start, stop, filename=None):
            """Add row to data index database.

            Update the data index database with a row that
            contains the name of the file and the span of time
            the file contains.

            Parameters
            ----------
            start : unix time
                Earliest time contained in the file.
            stop : unix time
                Latest time contained in the file.
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
            cursor = self.data_index.cursor()
            cursor.execute(
                "INSERT INTO files VALUES (?, ?, ?)", (dt_start, dt_stop, relpath)
            )

            self.data_index.commit()

    def refresh_data_index(self):
        """Remove expired rows from the data index database.

        Remove any rows of the data index database
        that correspond to files that have been cleaned
        (removed) by dias manager.
        """
        cursor = self.data_index.cursor()
        query = "SELECT filename FROM files ORDER BY start"
        all_files = list(cursor.execute(query))

        for result in all_files:

            filename = result[0]

            if not os.path.isfile(os.path.join(self.write_dir, filename)):

                cursor = self.data_index.cursor()
                cursor.execute("DELETE FROM files WHERE filename = ?", (filename,))
                self.data_index.commit()
                self.logger.info("Removed %s from data index database." % filename)

    def finish(self):
        """Close connection to data index database."""
        self.logger.info("Shutting down.")
        self.data_index.close()


###################################################
# auxiliary routines
###################################################

def _correct_phase_wrap(phi, deg=False):

    if deg:
        return ((phi + 180.0) % 360.0) - 180.0
    else:
        return ((phi + np.pi) % (2.0 * np.pi)) - np.pi

def find_freq(freq, freq_sel):
    ind = [np.argmin(np.abs(freq-freq_i)) for freq_i in freq_sel]
    return ind
    
