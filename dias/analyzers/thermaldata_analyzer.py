"""Analyzers to check the integrity of data related to the thermal modeling of
CHIME complex gain.

"""


from dias import CHIMEAnalyzer
from datetime import datetime
from caput import config
from dias.utils import str2timedelta
import numpy as np


class ThermalDataAnalyzer(CHIMEAnalyzer):
    """Analyzer to check the integrity of data related to the thermal modeling
    of CHIME complex gain.

    For now it only checks for cable loop data.
    """

    # Config parameter for this anlyzer can be specified by assigning class
    # attributes a caput.config.Property
    offset = config.Property(proptype=str2timedelta, default='12h')
    trange = config.Property(proptype=str2timedelta, default='1h')
    # TODO: In the future, could figure out the loop ids from the database.
    loop_ids = config.Property(proptype=list, default=[944, 1314, 2034])
    ref_ids = config.Property(proptype=list, default=[688, 1058, 2032])

    SLOPE_TO_SECONDS = 1./2./np.pi/1E6  # Convert slope to seconds
    nchecks = 1  # Number of time bins to check.
    checkoffset = 20  # Start checking from this time bin.

    def setup(self):
        """Setup stage: this is called when dias starts up."""
        # Add a data metric for the computed cable loop delays.
        self.delay = self.add_data_metric(
                            "delay",
                            "delays computed for each cable loop",
                            unit='nanoseconds',
                            labelnames=['chan_id'])

    def run(self):
        """Loads chimetiming data. 
        Fits for delay of cable loops and exports delays to prometheus.
        """

        ncables = len(self.loop_ids)  # Number of cable loops

        # Calculate the start and end the data to be loaded.
        # TODO: Could instead monitor for new chimetiming files.
        start_time = datetime.now() - self.offset
        end_time = start_time + self.trange

        from ch_util import data_index
        f = self.Finder()
        f.set_time_range(start_time, end_time)
        f.filter_acqs((data_index.ArchiveInst.name == 'chimetiming'))
        # f.print_results_summary()

        results_list = f.get_results()
        # I only use the first acquisition found
        result = results_list[0]
        read = result.as_reader()
        prods = read.prod
        freq = read.freq['centre']
        ntimes = len(read.time)
        time_indices = np.linspace(self.checkoffset, ntimes, self.nchecks,
                                   endpoint=False, dtype=int)

        # Determine prod_sel
        prod_sel = []
        for ii in range(ncables):
            chan_id, ref_id = self.loop_ids[ii], self.ref_ids[ii]
            pidx = np.where(
                      np.logical_or(
                         np.logical_and(prods['input_a'] == ref_id,
                                        prods['input_b'] == chan_id),
                         np.logical_and(prods['input_a'] == chan_id,
                                        prods['input_b'] == ref_id)))[0][0]
            prod_sel.append(pidx)

        # Load data
        data = result.as_loaded_data(prod_sel=np.array(prod_sel))
        phases = np.angle(data.vis)

        # Perform the fits
        for cc in range(ncables):
            prms = self._get_fits(time_indices, phases[:, cc, :], freq)
            for tt in range(len(prms)):
                # First parameter is the slope
                delay_temp = prms[tt][0]*self.SLOPE_TO_SECONDS
                self.delay.labels(chan_id=self.loop_ids[cc]).set(delay_temp)

    def _find_longest_stretch(self, phase, freq, step=None, tol=0.2):
        """Find the longest stretch of frequencies without phase wrapping.
        Step is the expected phase change between frequencies.
        Relies on the step being a quite good guess and small compared to 2 pi.

        This is slow, but it is only done for a few time points.

        Parameters
        ----------

        phase : array of floats
            phase as a function of frequency.
        freq : array of floats
            frequencies corresponding to `phase`
        step : float
            Expected phase difference between points in `phase`
            (Initial guess)
        tol : float
            Fractional tolerance in the phase step for a phase
            point to be considered good.

        Returns
        -------

        stt_idx : int
            Index of start of longest uninterrupted stretch found.
        length : int
            Length of longest uninterrupted stretch found (number of points).
        """

        speed_factor = 0.84
        stt_idx = 0
        length = 0
        current_stt_idx = 0
        current_length = 1
        n = len(phase)
        if step is None:
            L = 100  # Initial estimate of loop length in meters.
            thz = 1E6  # Convert MHz to Hz
            c = 3E8  # Poor man's speed of light.
            phase_fact = 2.*np.pi*L/(speed_factor*c)
            step = phase_fact * abs(freq[1]-freq[0])*thz
        for ii in np.arange(1, n):
            bad_step = np.logical_or(
                                abs(phase[ii]-phase[ii-1]) > step*(1.+tol),
                                abs(phase[ii]-phase[ii-1]) < step*(1.-tol))
            if not bad_step:
                current_length += 1
            else:
                if current_length > length:
                    length = current_length
                    stt_idx = current_stt_idx
                current_stt_idx = ii
                current_length = 1

        if current_length > length:
            length = current_length
            stt_idx = current_stt_idx

        return stt_idx, length

    def _get_fits(self, tmidxs, allphase, freq):
        """Find parameters for linnear fit phase vs frequency.

        Parameters
        ----------

        tmidxs : array of int
            Perform one fit for each point in tmidxs.
        allphase : array of float
            Phases to fit. Shape: (#freq, #time)
        freq : array of float
            Frequencies corresponding to first axis of `allphase`.
            (in MHz).

        Returns
        -------

        prms_list : list of array of float
            Optimal parameters of the fit.
            List is of length equal to the length of tmidxs.
            Each entry is an array containing the two parameters for each fit.
        """
        # When looking for the longest uninterrupted stretch, restrict search
        # to the centre of the range, between these frequency indices:
        fit_stt, fit_stp = 400, 600  # Frequency indices.

        prms_list = []
        tmidxs = np.arange(allphase.shape[1])[tmidxs]
        for tm_idx in tmidxs:
            # Phases
            phase = allphase[:, tm_idx]
            # Find longest uninterrupted stretch
            stt_idx, length = self._find_longest_stretch(
                            phase[fit_stt:fit_stp], freq[fit_stt:fit_stp])
            stt_idx = stt_idx+fit_stt
            fitslc = np.s_[stt_idx:stt_idx+length]
            # Fit for delay
            prms = np.polyfit(x=freq[fitslc], y=phase[fitslc], deg=1)
            prms_list.append(prms)

        return prms_list
