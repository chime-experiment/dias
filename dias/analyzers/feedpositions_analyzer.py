from dias import CHIMEAnalyzer
from datetime import datetime
from caput import config, time
from dias.utils.string_converter import datetime2str

from ch_util import data_index, ephemeris, fluxcat
import numpy as np
import os

import scipy.linalg as la

import h5py


# Choose 10 good frequencies. I chose the same ones that we used when writing
# full N2 data for 10 frequencies.
freq_sel = [
        758.203125,  716.406250,  697.656250,  665.625000,  633.984375,
        597.265625,  558.203125,  516.406250,  497.265625,  433.593750,
        ]
# Brightest sources. VirA does not have enough S/N.
sources = {
        'CAS_A': ephemeris.CasA,
        'CYG_A': ephemeris.CygA,
        'TAU_A': ephemeris.TauA,
        }

# number of inputs in CHIME
NINPUT = 2048

# number of inputs per cylinder per polarization
NCYLPOL = 256
# number of polarizations
NPOL = 2
# number of cylinders
NCYL = 4
# number of eigenvalues to keep
N_EVAL = 2
# Standard deviation of feed position residuals derived from
# data analysis of 11 files with 10 frequencies each
STD = 0.3
# Threshold for bad feeds
N_SIGMA = 5


class FeedpositionsAnalyzer(CHIMEAnalyzer):
    """A CHIME analyzer to calculate the East-West feed positions from the
    fringe rates in eigenvectors. The eigenvectors of the visibility matrix are
    found in the archive, then orhtogonalized.  To get the feed-positions in
    the UV plane we fourier transform the eigenvectors over the time axis.
    At the moment this Analyzer is supposed to run during the day to check for
    night transit data.

    Attributes
    ----------
    ref_feed_P1 : integer
        The feed we reference the polarisation 1 data to. Default: 2.
    ref_feed_P2: integer
        The feed we reference the polarisation 1 data to. Default : 258.
    pad_fac_EW : integer
        By which factor we pad the data before performing the fourier
        transform.  Default : 256.
    """

    ref_feed_P1 = config.Property(proptype=int, default=2)
    ref_feed_P2 = config.Property(proptype=int, default=258)
    pad_fac_EW = config.Property(proptype=int, default=256)

    def setup(self):
        self.logger.info(
                'Starting up. My name is ' + self.name +
                ' and I am of type ' + __name__ + '.')
        self.resid_metric = self.add_task_metric(
            "ew_pos_residuals_analyzer_run",
            "feedposition task run counter for specific source",
            labelnames=['source'], unit='total')
        self.freq_metric = self.add_data_metric(
            "ew_pos_good_freq",
            "how many frequencies out of 10 were good (EV ratio on vs off "
            "source smaller than 2)", labelnames=['source'], unit='total')
        self.percent_metric = self.add_data_metric(
            "bad_feeds",
            "how many feeds in percent are bad(position residuals are greater "
            + "than {} sigma / {} m)".format(N_SIGMA, N_SIGMA * STD),
            labelnames=['freq'], unit='percent')

        # initialize resid source metric
        for source in sources:
            self.resid_metric.labels(source=source).set(0)

    def run(self):

        end_time = datetime.utcnow()
        start_time = end_time - self.period  # period is 24h
        # self.logger.info(
        #        'Analyzer period: starttime '
        #        + datetime2str(start_time) + ', endtime '
        #        + datetime2str(end_time) + ', period ' + str(self.period))
        self.end_time_night = time.unix_to_datetime(
                ephemeris.solar_rising(start_time, end_time))[0]
        self.start_time_night = time.unix_to_datetime(
                ephemeris.solar_setting(start_time, end_time))[0]

        # In case the analyzer runs at night figure out when was sunset and
        # take one hour off.
        if self.start_time_night > self.end_time_night:
            # Instead of now we set the end time to one hour before sunset
            end_time = self.start_time_night - timedelta(hours=1)
            start_time = end_time - self.period
            # Overwrite start time night, end time night
            self.end_time_night = time.unix_to_datetime(
                ephemeris.solar_rising(start_time, end_time))[0]
            self.start_time_night = time.unix_to_datetime(
                ephemeris.solar_setting(start_time, end_time))[0]

        self.logger.info(
                'Analyzing night data between UTC times '
                + datetime2str(self.start_time_night)
                + ' and ' + datetime2str(self.end_time_night) + '.')

        night_transits = []

        # Check which of these sources transit at night
        for src in sources.keys():
            transit = ephemeris.transit_times(
                    sources[src], self.start_time_night, self.end_time_night)
            src_ra, src_dec = ephemeris.object_coords(
                    fluxcat.FluxCatalog[src].skyfield,
                    date=self.start_time_night, deg=True)
            if transit:
                night_transits.append(src)

        self.logger.info('Found night transits:\n{}'.format(night_transits))

        # Convert current datetime to str and keep only date
        time_str = time.datetime_to_timestr(self.start_time_night)[:8]

        # for each source in night_transits calculate the East-West positions
        for night_source in night_transits:
            self.logger.info(
                    'Processing source ' + night_source
                    + ' to find feed positions...')
            ew_positions, resolution = self.east_west_positions(night_source)

            if ew_positions is None:
                self.logger.info('Moving on.')
                continue

            # Calculate the median for each cylinder/ polarisation pair.
            ew_offsets = np.ones_like(ew_positions)
            for i in range(0, NCYL*NPOL, NPOL):
                ew_offsets[:, i * NCYLPOL:(i + 1) * NCYLPOL] *= \
                    np.median(ew_positions[
                        :, i * NCYLPOL:(i + 1) * NCYLPOL
                        ], axis=1)[:, np.newaxis]
                ew_offsets[:, (i + 1) * NCYLPOL:(i + 2) * NCYLPOL] *= \
                    np.median(ew_positions[
                        :, (i + 1) * NCYLPOL:(i + 2) * NCYLPOL
                        ], axis=1)[:, np.newaxis]

            # Subtract median from East-West positions to get residuals.
            residuals = ew_positions - ew_offsets

            # Calculate percentage of bad feeds, from analysis we know
            # that when excluding large outliers the standard deviation in the
            # residuals range around 0.3m. In a single feedposition analysis
            # normally no more than 2 percent of feeds lie outside of 5 sigma.
            for i in range(len(freq_sel)):
                nbad_feeds = np.sum(np.logical_or(residuals[i, :] > N_SIGMA*STD,
                                                  residuals[i, :] < - N_SIGMA*STD))
                percent_bad_feeds = nbad_feeds / float(NINPUT)
                self.logger.info('{} percent ({}) of the feeds are outside of {}'
                                 + 'sigma ({} * {}) around the expected feedpositions.'.format(
                                    percent_bad_feeds, nbad_feeds, N_SIGMA, N_SIGMA, STD))
                # Export bad feeds percentage to prometheus.
                self.precent_metric.labels(
                    freq=np.round(freq_sel[i], 0)).set(percent_bad_feeds)

            with h5py.File(os.path.join(
                               self.write_dir,
                               time_str + '_' + night_source
                               + '_positions.h5'),
                           'w') as f:
                f.create_dataset(
                        'east_west_pos', data=ew_positions, dtype=float)
                f.create_dataset(
                        'east_west_resid', data=residuals, dtype=float)
                f.create_dataset('axis/freq', data=freq_sel, dtype=float)
                f.create_dataset(
                        'axis/input', data=np.arange(NINPUT), dtype=int)
                f.close()

                self.logger.info(
                        'Fourier transform resolution in [m] from source: '
                        + night_source + " : " + str(resolution[0][0]))
                self.logger.info(
                        'Writing positions from ' + night_source + ' data to '
                        + self.write_dir)
                self.logger.debug(
                        'Incrementing prometheus metric, indicating '
                        'that feedposition analyzer has run on source'
                        ' ' + night_source)
                self.resid_metric.labels(source=night_source).inc()

    def east_west_positions(self, src):
        # src : list item of sources transiting in the night

        # Set a Finder object
        f = self.Finder()
        # f = data_index.Finder({'
        f.set_time_range(self.start_time_night, self.end_time_night)
        f.filter_acqs((data_index.ArchiveInst.name == 'chimecal'))
        f.accept_all_global_flags()
        f.include_transits(sources[src], time_delta=800.)

        results_list = f.get_results()

        if not results_list:
            self.logger.warn(
                    'Did not find any data in the archive for source ' + src)
            return (None, None)

        reader = results_list[0].as_reader()
        reader.select_freq_physical(freq_sel)

        # Read the data
        data = reader.read()

        # Get the timestamps
        time = data.index_map['time']['ctime'][:]
        # Get the frequencies
        freq = data.freq

        # Check here the eigenvalues on versus off source and see if we are
        # not dominated by RFI.
        # Get the source RA and DEC
        ra, dec = ephemeris.object_coords(
                fluxcat.FluxCatalog[src].skyfield,
                date=self.start_time_night, deg=True)
        ra_time = ephemeris.lsa(data.time)
        ha = ra_time - ra
        # In case we are dealing with CasA...
        ha = ha - (ha > 180.0) * 360.0 + (ha < -180.0) * 360.0
        ha = np.radians(ha)
        # This is the index of the transit
        transit_time = np.argmin(np.abs(ha))

        # Find ratio of 2 largest eigenvalues on source versus off source
        count = 0
        for i in range(len(freq_sel)):
            eval_offsource = data['eval'][i, :N_EVAL, 0]
            eval_onsource = data['eval'][i, :N_EVAL, transit_time]
            ratio = eval_onsource / eval_offsource
            if np.all(ratio < 2):
                count += 1
                self.logger.warn(
                        "Eigenvalue ratio on source versus off source smaller "
                        "than 2. Suspecting RFI contamination for this "
                        "frequency " + str(freq_sel[i]))

        # Determine the number of good frequencies out of 10 for this analyzer
        # and send to prometheus
        num_good_freq = len(freq_sel) - count
        self.freq_metric.labels(source=src).set(num_good_freq)
        self.logger.info(
                'Exporting number of good frequencies (non RFI contaminated) '
                'in this data to Prometheus')
        tshape = data['evec'].shape[-1]

        # Make some empty arrays for the orthogonalized eigenvectors
        vx_vec = np.zeros((len(freq_sel), NINPUT, tshape), dtype=complex)
        vy_vec = np.zeros((len(freq_sel), NINPUT, tshape), dtype=complex)

        for f in range(len(freq_sel)):
            for i in range(tshape):
                vx, vy = self.orthogonalize(data, f, i)
                vx_vec[f, :, i] = vx
                vy_vec[f, :, i] = vy

        # Combine the two polarisations into one vector evec
        evec = np.zeros((len(freq_sel), NINPUT, len(time)), dtype=complex)

        # Reference eigenvector to the first good feed for
        # the NS(P1) and EW(P2) polarisation
        for i in range(0, NCYL*NPOL, NPOL):
            evec[:, i * NCYLPOL:(i + 1) * NCYLPOL, :] = \
                    vy_vec[:, i * NCYLPOL:(i + 1) * NCYLPOL, :] / np.exp(
                            1J * np.angle(vy_vec[:, self.ref_feed_P1, :])
                            )[:, np.newaxis, :]
            evec[:, (i + 1) * NCYLPOL:(i + 2) * NCYLPOL, :] = \
                vx_vec[:, (i + 1) * NCYLPOL:(i + 2) * NCYLPOL, :] / np.exp(
                        1J * np.angle(vx_vec[:, self.ref_feed_P2, :])
                        )[:, np.newaxis, :]

        # Create empty arrays for East-West positions and residuals
        ew_positions = np.zeros((len(freq_sel), NINPUT), dtype=float)
        resolution = np.zeros((len(freq_sel), NINPUT), dtype=float)

        # Get the source RA and DEC
        ra, dec = ephemeris.object_coords(
                fluxcat.FluxCatalog[src].skyfield,
                date=self.start_time_night, deg=True)

        # Loop over frequencies and then inputs to get the EW-positions
        for f in range(len(freq_sel)):
            for i in range(NINPUT):
                ew_positions[f, i], resolution[f, i] = self.get_ew_pos_fft(
                        time, evec[f, i, :], freq[f], np.radians(dec),
                        pad_fac=self.pad_fac_EW)

        return ew_positions, resolution

    # Orthogonalization routine
    def orthogonalize(self, data, fsel, time_index):
        # If we did not write data for that frequency because of a node crash
        # skip that frequency
        # and return a vector with zeros.
        if all(np.abs(data['evec'][fsel, 0, :, time_index]) == 0):
            vx = np.zeros((NINPUT), dtype=complex)
            vy = np.zeros((NINPUT), dtype=complex)

            return vx, vy

        # Construct masks for the X and Y polarisations
        Ax = (((np.arange(NINPUT, dtype=np.int) // NCYLPOL) % 2) == 1)\
            .astype(np.float64)
        Ay = (((np.arange(NINPUT, dtype=np.int) // NCYLPOL) % 2) == 0)\
            .astype(np.float64)

        U = data['evec'][fsel, :2, :, time_index].T
        Lh = (data['eval'][fsel, :2, time_index])**0.5

        Vtx = Lh[:, np.newaxis] * np.dot(U.T.conj(), Ax[:, np.newaxis] * U) \
            * Lh[np.newaxis, :]
        m, vv = la.eigh(Vtx)
        vx = Ax[:, np.newaxis] * np.dot(U, vv / Lh[:, np.newaxis])
        vx /= np.dot(vx.T.conj(), vx).diagonal()**0.5

        Vty = Lh[:, np.newaxis] * np.dot(U.T.conj(), Ay[:, np.newaxis] * U) \
            * Lh[np.newaxis, :]
        m, vv = la.eigh(Vty)
        vy = Ay[:, np.newaxis] * np.dot(U, vv / Lh[:, np.newaxis])
        vy /= np.dot(vy.T.conj(), vy).diagonal()**0.5

        return vx[:, -1], vy[:, -1]

    def get_ew_pos_fft(self, times, evec_stream, f, dec, pad_fac=pad_fac_EW):
        """Routine that gets feed positions from the eigenvector data via an
        FFT.  The eigenvector is first apodized with ahannings window function
        and then fourier transformed along the time axis.


        Parameters
        ---------
        times : np.ndarray
            Unix time of the data
        evec_stream : np.ndarray
            The eigenvector data for a single frequency as a function of time.
        f : float
            The selected frequency
        dec : float
            The declination of the source in radians.
        pad_fac: integer
            The multiplicative factor by which we want to pad the data.

        Returns
        -------
        positions: np.ndarray(ninput)
            The East-West positions referenced to 2 feeds on the first
            cylinder.

        position_resolution: float
            Position resolution, determined by number of time samples times
            padding factor.
        """
        n = len(times)

        # Use a hannings window function to apodize the data
        apod = np.hanning(n)
        # Time resolution in radians
        dt = np.radians(ephemeris.lsa(times[1]) - ephemeris.lsa(times[0]))
        # Calculate the fourier transform of the apodized eigenvector data
        spec = np.fft.fft(apod * evec_stream, n=n * pad_fac)
        freq = np.fft.fftfreq(n * pad_fac, dt)

        # Find the maximum power in the spectrum
        x_loc = freq[np.argmax(np.abs(spec))]
        # The conjugate to time is the baseline vector in units of wavelength
        # divided by the cos(declination)
        conv_fac = -2.99792e2 / f / np.cos(dec)

        position = x_loc * conv_fac
        position_resolution = np.abs(freq[1] - freq[0]) * np.abs(conv_fac)

        return position, position_resolution
