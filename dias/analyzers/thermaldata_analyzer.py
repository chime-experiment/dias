"""Analyzers to check the integrity of data related to the thermal modeling of
CHIME complex gain.

"""


from dias import CHIMEAnalyzer
from datetime import datetime
from caput import config
from dias.utils import str2timedelta, datetime2str
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


    # For now, the inputs corresponding to cable delays are hard-coded
    # TODO: Make these inputs a config parameter
    # or better: figure out the loops from the database
    # (These need to be ordered.)
    loop_ids = [944, 1314, 2034]  # IDs of cable loops
    ref_ids = [688, 1058, 2032]  # IDs of reference NS signal for each loop
    ncables = len(loop_ids)  # Number of cable loops
    tns = 1./2./np.pi/1E6*1E9  # To convert to nano-seconds
    nchecks = 1  # Number of times to check.
    checkoffset = 20  # Start checking from this time bin.


    def setup(self):
        """Setup stage: this is called when dias starts up."""
        self.logger.info('Starting up. My name is {} and I am of type {}.'
                         .format(self.name, __name__))

        # Add a task metric that counts how often this task ran.
        # It will be exported as dias_task_<task_name>_runs_total.
        self.run_counter = self.add_task_metric(
                                        "runs",
                                        "Number of times the task ran.",
                                        unit="total")

        # Add a data metric for the computed cable loop delays.
        self.delays = self.add_data_metric(
                            "delays", 
                            "delays computed for each cable loop", 
                            labelnames=['chan_id'])

    def run(self):
        """Main task stage: analyze data from the last period.
        """

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
        for ii in range(self.ncables):
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

        for cc in range(self.ncables):
            prms = self._get_fits(time_indices, phases[:, cc, :], freq)
            for tt in range(len(prms)):
                # First parameter is the slope
                delay = prms[tt][0]*self.tns  # Nanoseconds
                self.delays.labels(chan_id=self.loop_ids[cc]).set(delay)

        # Increment (+1).
        self.run_counter.inc()

    def _find_longest_stretch(self, phase, freq, step=None, tol=0.2):
        """Find the longest stretch of frequencies without phase wrapping. 
        Step is the expected phase change between frequencies.
        Relies on the step being a quite good guess and small compared to 2 pi.
        
        This is slow, but it is only done for a few time points."""
    
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
        else:
            if current_length > length:
                length = current_length
                stt_idx = current_stt_idx
        return stt_idx, length

    def _fitphase(self, freq, phaseunwrap, phase, 
                  mask, masktol=0.5, qualtol=0.5):
        """"""
        # Linear fit
        prms = np.polyfit(x=freq[mask], y=phaseunwrap[mask], deg=1)
        fitfunc = np.polyval(p=prms, x=freq)
        # Wrap fit result:
        fitwrap = (fitfunc+np.pi) % (2.*np.pi)-np.pi
        # Diff to fit:
        fitdiff = phase-fitwrap
        # Update freq mask.
        freqmask = abs(fitdiff) < masktol
        fitbreadth = float(np.sum(freqmask))/len(freq)
        fitquality = float(np.sum(abs(fitdiff) < qualtol))/len(freq)
        # Unwrap data (I am actualy just adding factors of 2pi):
        phaseunwrap = fitfunc+fitdiff 
    
        return prms, phaseunwrap, freqmask, fitbreadth, fitquality

    def _get_diffmask(self, phaseunwrap, prms, freq, freqmask, tol):
        """"""
        fitdiff = phaseunwrap - np.polyval(p=prms, x=freq)
        freqtrace = freq[freqmask]
        trace = fitdiff[freqmask]
        tracediff = abs(np.diff(trace))
        difftol = tracediff < tol
        # Make sure to remove 1st and last points if they have large diff 
        # on the one side they face the data:
        difftol = np.insert(difftol, 0, False)
        difftol = np.append(difftol, False)
        # Create a mask
        difftol = np.array([difftol[0:-1], difftol[1:]])
        diffmask = np.all(difftol, axis=0)
        # Combine freqmask and diffmask:
        indexmask = np.arange(len(freqmask))[freqmask][diffmask]
        mask = np.zeros(len(freqmask), dtype=bool)
        mask[indexmask] = True
    
        return mask

    def _get_fits(self, tmidxs, allphase, freq, tol=0.08):
        """tol : in radians
        """
        freq_ends = [415, 785]  # Beyond these frequencies, phase is weird.
        phasetol = 0.5  # Tolerance. Remove phases closer to zero than this.
        # When looking for the longest uninterrupted stretch, restrict ourselves
        # to the centre of the range, between these frequency indices:
        fit_stt, fit_stp = 400, 600  # Frequency indices.
        # When fitting for the slope use only the centre 100 bins of the band.
        fitlength = 100
        # Stopping critetium for the fit quality differences 
        fitqualdiff_tol = 1E-4 
        # Maximum number of iterations before stopping
        max_iter = 5
        # Minimum number of remaining frequency points after masking 
        # (only there to catch cases where all points are masked out)
        minfreqpoints = 10
        prms_test, phaseunwrap_test, freq_test = [], [], []
        fitqual_test, tmidx_test, phase_test = [], [], []
        freqmask_test = []
        nf = len(freq)
        remove_freq_ends = np.logical_and(freq > freq_ends[0], freq < freq_ends[1])  # Mask
   
        tmidxs = np.arange(allphase.shape[1])[tmidxs]
        for tm_idx in tmidxs:
    
            # Phases
            phase = allphase[:, tm_idx]
            # Remove bad freqs:
            freqmask = np.logical_or(phase > phasetol,
                                     phase < -phasetol)
            
            # Find longest uninterrupted stretch
            stt_idx, length = self._find_longest_stretch(
                            phase[fit_stt:fit_stp], freq[fit_stt:fit_stp])
            stt_idx = stt_idx+fit_stt
            fitslc = np.s_[stt_idx:stt_idx+length]
            
            # First fit
            (prms, phaseunwrap, freqmask, 
             fitbreadth, fitquality) = self._fitphase(
                                    freq, phase, phase, fitslc, masktol=1)
    
            # Second fit
            # Fit only in the centre of the range:
            fitmask = np.logical_and(
                            np.arange(nf) > stt_idx+int(length/2.)-fitlength,
                            np.arange(nf) < stt_idx+int(length/2.)+fitlength)
            fitmask = np.logical_and(freqmask, fitmask)
            fitmask = np.logical_and(fitmask, remove_freq_ends)
            (prms, phaseunwrap, freqmask, 
             fitbreadth, fitquality) = self._fitphase(
                                freq, phaseunwrap, phase, fitmask)
    
            # Final fits (do at least one!)
            nfits = 0
            prev_fitqual = fitquality
            fitqualdiff = abs(fitquality - prev_fitqual)
            while fitqualdiff > fitqualdiff_tol or nfits == 0:
                if nfits == max_iter:
                    self.logger.warning('Fits did not converge')
                    break
                nfits += 1
                prev_fitqual = fitquality
                freqmask = np.logical_and(freqmask, remove_freq_ends)
                (prms, phaseunwrap, freqmask, 
                 fitbreadth, fitquality) = self._fitphase(
                            freq, phaseunwrap, phase, freqmask)
                freqmask = np.logical_and(freqmask, remove_freq_ends)
                fitqualdiff = abs(fitquality - prev_fitqual)
            else:
                # Only if it converged. Do one last bit of cleaning:
                freqmask = self._get_diffmask(
                    phaseunwrap, prms, freq, freqmask, tol)
                freqmask = np.logical_and(freqmask, remove_freq_ends)
                # Sometimes the mask erases everything
                if len(freq[freqmask]) > minfreqpoints:
                    # One last fit
                    prms, phaseunwrap, freqmask, _, _ = self._fitphase(
                        freq, phaseunwrap, phase, freqmask)
                    # Final mask
                    freqmask = self._get_diffmask(
                        phaseunwrap, prms, freq, freqmask, tol)
                    freqmask = np.logical_and(freqmask, remove_freq_ends)
                    
                    # Save results
                    phase_test.append(phase)
                    prms_test.append(prms)
                    freqmask_test.append(freqmask)
                    phaseunwrap_test.append(phaseunwrap)
                    freq_test.append(freq)
                    fitqual_test.append(fitquality)
                    tmidx_test.append(tm_idx)
    
                else:
                    self.logger.warning('Diffmask obliterates data.'
                                        ' Skipping this time point.')
            
        return prms_test

