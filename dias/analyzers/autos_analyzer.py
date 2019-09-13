"""Calculates fit parameters from autocorrelations of most recent point source transit.

.. currentmodule:: dias.analyzers.autos_analyzer

Classes
=======

.. autosummary::
    :toctree: generated/

    AutosAnalyzer


Functions
=========

..  autosummary::
    :toctree: generated/

    gauss
    poly3
    get_flag_window

"""
import os
import time as tt
import datetime
from caput import config, time
import warnings
from scipy.optimize import OptimizeWarning

import numpy as np
import h5py
from scipy.optimize import curve_fit

from ch_util import data_index, ephemeris, fluxcat, andata

from dias import CHIMEAnalyzer

from dias.utils.string_converter import datetime2str

warnings.simplefilter("error", OptimizeWarning)

# Brightest sources.
SOURCES = {
    "CAS_A": ephemeris.CasA,
    "CYG_A": ephemeris.CygA,
    "TAU_A": ephemeris.TauA,
    "VIR_A": ephemeris.VirA,
}
NINPUT = 2048

# Fitting functions.


def gauss(x, A, mu, sigma):
    """Fits data to a Gaussian function."""
    return A * np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))


def poly3(x, p, q, r, s):
    """Fits data to a 3rd order polynomial function."""
    return (p * x ** 3) + (q * x ** 2) + r * x + s


def get_flag_window(ttrans, window, timestamp):
    """Flag indices for a specified window around transit."""
    time_start, time_end = ttrans - window, ttrans + window
    flag_window = np.flatnonzero((timestamp > time_start) & (timestamp < time_end))
    return flag_window


class AutosAnalyzer(CHIMEAnalyzer):
    """
    A CHIME analyzer to fit point source transits in autocorrelations.

    Metrics
    -------
    dias_task_<task name>_run_time
    ...........................................................
    Time to process single transit in seconds.

    dias_task_<task name>_transits_found
    ...........................................................
    Number of total transits found in search window.

    dias_data_<task name>_bad_fits
    ............................................
    Counter for any time an attempt to fit a transit failed.

    Output Data
    -----------

    File naming
    ...........
    `<YYYYMMDD>_<SOURCE>_fitparams.h5`
        YYYYMMDD is the date of the beginning of the night, data was analyzed
        from and SOURCE is the name of the source transit (TAU_A, CYG_A, VIR_A, or
        CAS_A).

    Indexes
    .......
    freq
        Telescope frequency indices.
    input
        Telescope channel IDs.

    Datasets
    .........
    centroid_wander
        Difference (in seconds) between centroid and expected time of transit.
    FWHM
        Full width half max (in seconds) of transit.
    gain
        ADC-to-Jy calibration factor.
    SEFD
        System equivalent flux density (Jy) at the declination of point source.
    rms_noise
        Root mean square noise (Jy) around the point source.

    State Data
    ----------
    None

    Config Variables
    ----------------

    Attributes
    ----------
    transit_window : float
        Window (seconds) around which transit fitting is done Default : 600.0.
    freq_sel: list
        List of frequency indices. Allows the processing of a subset of frequencies.
        Default is the full CHIME band.
    search_window : integer
        Number of hours to search back from current time. Default : 48.
    source : string
        Name of point source to analyze. One of ('TAU_A', 'CYG_A', 'VIR_A', or
        'CAS_A'). Default : 'CYG_A'.

    """

    source = config.Property(proptype=str, default="CYG_A")

    transit_window = config.Property(proptype=float, default=600.0)
    freq_sel = config.Property(proptype=list, default=range(1024))
    search_window = config.Property(proptype=int, default=48)

    def setup(self):
        """
        Set up the task.

        Creates metrics.
        """
        # Add task metrics.
        self.run_timer = self.add_task_metric(
            "run_time", "Time to process transit.", unit="seconds"
        )

        self.transits_found = self.add_task_metric(
            "transits_found", "Number of transits found in search window.", unit="total"
        )

        # Add data metrics.
        self.bad_fits = self.add_data_metric(
            "bad_fits", "Number of channels for which fitting failed.", unit="total"
        )

        # Select the configured sources from the hardcoded dictionary of
        # ephemeris sources.
        try:
            self.sel_source = SOURCES[self.source]
        except KeyError as e:
            raise DiasConfigError("Invalid source: {}".format(e))

    def run(self):
        """Run the analyzer.

        Define time range to look for transits.
        Write fit parameters to disk.
        """
        run_start_time = tt.time()

        end_time = datetime.datetime.utcnow()
        start_time = end_time - datetime.timedelta(
            hours=self.search_window
        )  # period is 24h

        self.logger.info(
            "Analyzing data between UTC times "
            + datetime2str(start_time)
            + " and "
            + datetime2str(end_time)
            + "."
        )

        # Convert current datetime to str and keep only date
        time_str = time.datetime_to_timestr(start_time)[:8]

        # calculate fit parameters
        self.logger.info("Processing source " + self.source)
        centroid_wander, width, gain, sefd, rms_noise, freq = self.fit_autos(
            start_time, end_time
        )

        with h5py.File(
            os.path.join(
                self.write_dir, time_str + "_" + self.source + "_fitparams.h5"
            ),
            "w",
        ) as f:
            f.create_dataset("centroid_wander", data=centroid_wander, dtype=float)
            f.create_dataset("FWHM", data=width, dtype=float)
            f.create_dataset("gain", data=gain, dtype=float)
            f.create_dataset("SEFD", data=sefd, dtype=float)
            f.create_dataset("rms_noise", data=rms_noise, dtype=float)
            f.create_dataset("axis/freq", data=freq, dtype=float)
            f.create_dataset("axis/input", data=np.arange(NINPUT), dtype=int)
            f.close()

        self.run_timer.set(int(tt.time() - run_start_time))

    def fit_autos(self, start_time, end_time):
        """Load autocorrelations from the most recent transit.

        Loop over frequencies and inputs and get fit paramters.

        Parameters
        ----------
        start_time : datetime.datetime
            Search for transits datetime (in UTC).
        end_time : datetime.datetime
            Search for transits before this datetime (in UTC).

        Returns
        -------
        centroid_wander : np.ndarray
            2D array of type `float` with axes `[freq, input]` containing the
            difference of the transit centroid from the expected transit time in seconds.
        width : np.ndarray
            2D array of type `float` with axes `[freq, input]` containing the FWHM in seconds.
        gain : np.ndarray
            2D array of type `float` with axes `[freq, input]` containing the ADC-to-JY calibration
            factor calculated by dividing the expected source flux by the fitted Gaussian peak.
        sefd : np.ndarray
            2D array of type `float` with axes `[freq, input]` containing the SEFD in Jy during the
            transit. Calculated by fitting and calibrating the background around transit.
        rms_noise : np.ndarray
            2D array of type `float` with axes `[freq, input]` containing the root mean
            square noise in Jy.
        freq : list
            Centre frequencies in MHz.
        """
        # Set a Finder object
        f = self.Finder()
        f.set_time_range(start_time, end_time)

        # get autos during for particular transit
        f.filter_acqs((data_index.ArchiveInst.name == "chimecal"))
        f.accept_all_global_flags()
        f.include_transits(self.sel_source, time_delta=800.0)

        results_list = f.get_results()

        if not results_list:
            self.logger.warn(
                "Did not find any data in the archive for source " + self.source
            )
            return (None, None)
        else:
            result = results_list[0]
            self.logger.info("Found " + str(len(results_list)) + " transits.")
        self.transits_found.set(len(results_list))

        result = results_list[-1]

        nfreq = len(self.freq_sel)
        self.logger.info("Loading autocorrelations at " + str(nfreq) + " frequencies.")
        data = andata.CorrData.from_acq_h5(result[0], freq_sel=self.freq_sel)

        times = data.index_map["time"]["ctime"]
        freq = data.index_map["freq"]["centre"]
        input_list = data.index_map["input"]["chan_id"]
        freq_width = data.index_map["freq"]["width"]

        transit_time = ephemeris.transit_times(self.sel_source, times[0], times[-1])[0]

        local_time = np.array(
            [
                ephemeris.unix_to_datetime(tt) - datetime.timedelta(hours=8)
                for tt in times
            ]
        )
        local_ttrans = ephemeris.unix_to_datetime(transit_time) - datetime.timedelta(
            hours=8
        )

        flag_window = get_flag_window(transit_time, self.transit_window, times)
        timestamp = times[flag_window]
        t0 = timestamp[0]
        tau = np.median(np.diff(timestamp))

        # get expected flux from catalog
        fluxes = fluxcat.FluxCatalog[self.source].predict_flux(freq)

        self.logger.info(
            "Fitting "
            + self.source
            + " transit from local time "
            + datetime2str(local_ttrans)
            + "."
        )

        # initialize parameters
        centroid_wander = np.zeros((nfreq, NINPUT))
        width = np.zeros((nfreq, NINPUT))
        gain = np.zeros((nfreq, NINPUT))
        sefd = np.zeros((nfreq, NINPUT))
        rms_noise = np.zeros((nfreq, NINPUT))
        radiometer_rms = np.zeros((nfreq, NINPUT))

        no_fit = 0
        for fbin in range(nfreq):
            for i in input_list:
                v = data["vis"][fbin, i, flag_window]
                flux = fluxes[fbin]

                try:
                    (
                        centroid_wander[fbin, i],
                        width[fbin, i],
                        gain[fbin, i],
                        sefd[fbin, i],
                        rms_noise[fbin, i],
                    ) = self.fit_params(v, timestamp, flux)
                except (OptimizeWarning, RuntimeError):
                    no_fit += 1
                    continue
        self.bad_fits.set(no_fit)

        return centroid_wander, width, gain, sefd, rms_noise, freq

    def fit_params(self, v, timestamp, flux):
        """Fit autos to Gaussian + background and return fit parameters and other data metrics.

        Parameters
        ----------
        v : list
           Autocorrelations of a given input and frequency around transit.
        timestamp : list
           Unix timestamps in seconds.
        flux : float
           Expected sourc flux in Jy.

        Returns
        -------
        centroid_wander : float
            Difference of the transit centroid from the expected transit time in seconds.
        width : float
            FWHM in seconds.
        gain : float
            ADC-to-JY calibration factor calculated by dividing the expected source flux
            by the fitted Gaussian peak.
        sefd : float
            SEFD in Jy during the transit. Calculated by fitting and calibrating the
            background around transit.
        rms_noise : float
            Root mean square noise in Jy.
        """
        # subtract an overall offset to help with fitting
        auto_vec = v.real
        offset = np.min(auto_vec)
        vis = auto_vec - offset
        n = len(timestamp)

        # first fit a Gaussian
        mean = transit_time - t0
        sigma = self.transit_window
        popt, pcov = curve_fit(gauss, (timestamp - t0), vis, p0=[max(vis), mean, sigma])
        # background fit to 3rd degree polynomial
        popt2, pcov2 = curve_fit(
            poly3,
            (timestamp - t0),
            (vis + offset - gauss(timestamp - t0, *popt)),
            p0=[0, 0, 0, np.median(auto_vec)],
        )

        def baseline(tt):
            bl = poly3(tt, *popt2)
            return bl

        def best_fit(tt):
            fit = gauss(tt - t0, *popt) + baseline(tt - t0)
            return fit

        height = popt[0]  # in ADC units
        gain = flux / (height)  # Jy per AC unit
        width = 2.0 * np.sqrt(2 * np.log(2)) * popt[2]  # seconds
        centroid_wander = popt[1] + t0 - transit_time  # unix seconds

        radiometer_rms = gain * popt2[-1] / np.sqrt(tau * freq_width)

        sefd = gain * baseline(popt[1])

        # find and remove RFI spikes before calculating RMS noise
        residuals = np.diff((vis + offset - best_fit(timestamp)))
        no_spikes = np.where(
            np.absolute(residuals) < 3 * np.median(np.absolute(residuals))
        )[0]
        rms_noise = np.sqrt(
            np.sum((residuals[no_spikes] * gain) ** 2) / (2 * len(no_spikes) - 2)
        )

        return centroid_wander, width, gain, sefd, rms_noise
