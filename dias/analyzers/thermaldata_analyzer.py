"""Example dias analyzer.

This is a basic example for how to write an analyzer for dias.
"""


from dias import CHIMEAnalyzer
from datetime import datetime
from caput import config
from dias.utils import str2timedelta, datetime2str
import numpy as np


class ThermalDataAnalyzer(CHIMEAnalyzer):
    """Sample Analyzer for dias.
    This subclass of dias.analyzer.Analyzer describes the new analyzer.
    """

    # Config parameter for this anlyzer can be specified by assigning class
    # attributes a caput.config.Property
    offset = config.Property(proptype=str2timedelta, default='12h')
    trange = config.Property(proptype=str2timedelta, default='1h')

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

        # For now, the inputs corresponding to cable delays are hard-coded
        # TODO: Make these inputs a config parameter
        # or better: figure out the lopps from the database
        # These need to be ordered.
        loop_ids = [944, 1314, 2034] 
        ref_ids = [688, 1058, 2032]
        ncables = len(loop_ids)
        tns = 1./2./np.pi/1E6*1E9  # To convert to nano-seconds

        # TODO: there must be a better way than loading data_index here
        from ch_util import data_index
        f = self.Finder()
        f.set_time_range(start_time, end_time)
        f.filter_acqs((data_index.ArchiveInst.name == 'chimetiming'))
        # f.print_results_summary()

        results_list = f.get_results()
        result = results_list[0]
        read = result.as_reader()
        prods = read.prod
        freq = read.freq['centre']
        ntimes = len(read.time)
        nchecks = 1  # Number of times to check
        time_indices = np.linspace(20, ntimes, nchecks,
                                   endpoint=False, dtype=int)

        # Determine prod_sel
        prod_sel = []
        for ii in range(ncables):
            chan_id, ref_id = loop_ids[ii], ref_ids[ii]
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

        for cc in range(ncables):
            prms = self._get_fits(time_indices, phases[:, cc, :], freq)
            for tt in range(len(prms)):
                delay = prms[tt][0]*tns  # Nanoseconds
                self.delays.labels(chan_id=loop_ids[cc]).set(delay)

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
            L = 100
            phase_fact = 2.*np.pi*L/(speed_factor*3E8)
            step = phase_fact * abs(freq[1]-freq[0])*1E6
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
        # Unwrap data (Brilliant! I am actualy just adding factors of 2pi):
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
        prms_test, phaseunwrap_test, freq_test = [], [], []
        fitqual_test, tmidx_test, phase_test = [], [], []
        freqmask_test = []
        nf = len(freq)
        remove_freq_ends = np.logical_and(freq > 415, freq < 785)  # Mask
   
        tmidxs = np.arange(allphase.shape[1])[tmidxs]
        for tm_idx in tmidxs:
    
            # Phases
            phase = allphase[:, tm_idx]
            
            # Remove bad freqs:
            phasetol = 0.5
            freqmask = np.logical_or(phase > phasetol,
                                     phase < -phasetol)
            
            # Find longest uninterrupted stretch
            # Give it only the centre of the range
            fit_stt, fit_stp = 400, 600
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
            fitmask = np.arange(nf) > stt_idx+int(length/2.)-100
            fitmask = np.logical_and(
                        fitmask, np.arange(nf) < stt_idx+int(length/2.)+100)
            fitmask = np.logical_and(freqmask, fitmask)
            fitmask = np.logical_and(fitmask, remove_freq_ends)
            (prms, phaseunwrap, freqmask, 
             fitbreadth, fitquality) = self._fitphase(
                                freq, phaseunwrap, phase, fitmask, masktol=0.5)
    
            # Final fits (do at least one!)
            nfits = 0
            prev_fitqual = fitquality
            fitqualdiff = abs(fitquality - prev_fitqual)
            while fitqualdiff > 0.0001 or nfits == 0:
                if nfits == 5:
                    self.logger.warning('Fits did not converge')
                    break
                nfits += 1
                prev_fitqual = fitquality
                freqmask = np.logical_and(freqmask, remove_freq_ends)
                (prms, phaseunwrap, freqmask, 
                 fitbreadth, fitquality) = self._fitphase(
                            freq, phaseunwrap, phase, freqmask, masktol=0.5)
                freqmask = np.logical_and(freqmask, remove_freq_ends)
                fitqualdiff = abs(fitquality - prev_fitqual)
            else:
                # Only if it converged. Do one last bit of cleaning:
                freqmask = self._get_diffmask(
                    phaseunwrap, prms, freq, freqmask, tol)
                freqmask = np.logical_and(freqmask, remove_freq_ends)
                # Sometimes the mask erases everything
                if len(freq[freqmask]) > 10:
                    # One last fit
                    prms, phaseunwrap, freqmask, _, _ = self._fitphase(
                        freq, phaseunwrap, phase, freqmask, masktol=0.5)
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

    def finish(self):
        """Final stage: this is called when dias shuts down."""
        self.logger.info('Shutting down.')
        self.logger.debug('I could save some stuff I would like to keep until '
                          'next setup in {}.'.format(self.state_dir))
