import numpy as np
from scipy import interpolate
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import chi2

class PowerSpectralDensity():

    def __init__(self, time, data, flag=None, overlap_factor=0.5, fmin=None, tmax=None, censor=None, nsig_flag=None, speedy=False, one_sided=False):
        """Calculate the PSD of a timeseries using Welch's method.

        Break the timeseries up into overlapping groups of size `tmax`.  For each group,
        subtract the mean, apply a hanning window, and calculate the PSD.  Then calculate the
        average PSD over all groups.

        Parameters
        ----------
        time : np.ndarray[ntime,] or float
            Either the time axis or the sampling period.  Units should be seconds.
        data : np.ndarray[..., ntime]
            The timeseries. The PSD will be calculated over the last axis.
        flag : np.ndarray[..., ntime]
            Flag that indicates bad or missing data.  If a group contains only
            flagged data, then it will be discarded from the final average,
            but otherwise this quantity is ignored.
        overlap_factor : float
            Fraction of a group that will overlap with its neighbors.
        fmin : float
            Minimum frequency to calculate the PSD.
        tmax : float
            Maximum timescale to calculate the PSD.
            Either specify `tmax` or `fmin`, not both.
            The `fmin` parameter takes precedence
            (and in this case the resulting `tmax` will be `1 / fmin`).
            If neither `fmin` or `tmax` are provided,
            then the group size will be set to the total span of time.
            Recommend setting `tmax` to roughly 10 percent of the total span of time.
        censor : float
            Do not include groups in the average if they are in the top
            `censor` fraction of all groups at a given frequency.  (EXPERIMENTAL)
        nsig_flag : float
            Do not include group in the average if they are more than nsigma
            away from the average at a given frequency (EXPERIMENTAL).
        speedy : bool
            Force the number of samples to be a power of 2 for
            fast fourier transform (recommended).
        one_sided : bool
            Only return the PSD for postive frequencies (recommended).
        
        Attributes
        ----------
        freq : np.ndarray[nfreq,]
            Frequency in Hz.
        group_psd : np.ndarray[..., nfreq, ngroup]
            The PSD of each group.
        group_flag : np.ndarray[..., nfreq, ngroup]
            Boolean flag indicating if the group was included in the average.
        mean_psd : np.ndarray[..., nfreq]
            The average PSD.
        err_psd : np.ndarray[..., nfreq]
            The expected error on the PSD, given by mean_psd / sqrt(N) where
            N is the number of groups that were included in the average.
        err_psd_emp : np.ndarray[..., nfreq]
            Emperical estimate of the error in the mean_psd, determined from
            the scatter over groups.
        """
        
        # Check for flags
        if flag is not None:
            if flag.shape != data.shape:
                raise ValueError("Input flag must be the same size as data.")
            self.flag = flag
        else:
            self.flag = np.ones_like(data, dtype=np.int8)
        
        # Determine the sampling frequency
        if np.size(time) > 1:
            fs = 1.0 / np.median(np.diff(time))
        else:
            fs = 1.0 / time
            
        nsamples = data.shape[-1]
        
        
        # Determine keywords
        if fmin is None:
            if tmax is None:
                tmax = nsamples / fs
            fmin = 1.0/tmax
        else:
            fmin = fmin
        
        if (censor is not None) and (censor > 0.0) and (censor < 1.0):
            self.censor = censor
        else:
            self.censor = 0.0
            
        self.one_sided = one_sided
        self.nsig = nsig_flag
            
        # Determine number of samples per group
        fmin = max([fmin, fs/nsamples])
        
        nsamples_per_group = np.round(fs/fmin)

        if speedy:
		    nsamples_per_group = 2**(np.floor(np.log(nsamples_per_group)/np.log(2.0)))
        else:   
            if (nsamples_per_group % 2) == 0:
                nsamples_per_group -= 1
            
        nsamples_per_group = int(nsamples_per_group)
        
            
        # Determine the groups of data for which we will calculate the fft
        fmin = fs/nsamples_per_group
        
        navg = int(np.floor((float(nsamples)/float(nsamples_per_group) - overlap_factor)/(1.0 - overlap_factor)))
        
        noverlap = int(np.ceil(overlap_factor*nsamples_per_group))
        
        self.istart = np.arange(0,navg)*(nsamples_per_group - noverlap)
        
        self.freq = np.fft.fftfreq(nsamples_per_group, d=1.0/fs)
        
        
        # Create Hanning window        
        h = np.arange(0.0, nsamples_per_group)
        while (np.size(h.shape) != np.size(data.shape)) and (np.size(h.shape) < 5):
            h = np.expand_dims(h,0)
            
        self.window = 0.5*(1.0 - np.cos(2.0*np.pi*(h + 1.0)/nsamples_per_group))
        
        self.window_norm = np.squeeze((1.0/nsamples_per_group)*np.sum(self.window**2, axis=-1))
        
        
        # Place variables in object
        self.data = data
        
        self.fs = fs
        self.deltaf = fs / nsamples_per_group
        
        self.navg = navg
        self.nsamples_per_group = nsamples_per_group
                
                
        # Calculate the power spectral density
        self.calculate_psd()
        
        
    def calculate_psd(self):
        
        sz = self.data.shape[:-1] + (self.nsamples_per_group, self.navg)
        
        # Create arrays to hold fft
        data_fft = np.zeros(sz, dtype=np.complex128)
        
        nflag = np.zeros(self.data.shape[:-1] + (self.navg,), dtype=np.int32)
        
        # Loop through groups and calculate fft
        for gg in range(self.navg):
            
            gstart = self.istart[gg]
            gstop = self.istart[gg] + self.nsamples_per_group
            
            # Subtract mean
            mean_data = np.sum(self.flag[...,gstart:gstop]*self.data[...,gstart:gstop], axis=-1) / np.sum(self.flag[...,gstart:gstop], axis=-1)
            mean_data = np.nan_to_num(mean_data)
            
            # Calculate fft
            data_fft[...,gg] = (np.fft.fft((self.data[...,gstart:gstop] - mean_data[...,np.newaxis])*self.window, axis=-1) /      
                               (np.sqrt(self.window_norm*self.deltaf)*self.nsamples_per_group))
                               
            # Calculate number of samples flagged
            nflag[...,gg] = np.sum(self.flag[...,gstart:gstop] == 0, axis=-1)
            
            
        # Calculate power spectrum
        group_psd = np.absolute(data_fft)**2
        
        # If one-sided then toss the negative frequencies
        if self.one_sided:
            if (np.sum(self.data.imag != 0) == 0):
                 ipos = self.freq > 0.0
                 group_psd = 2.0*group_psd[...,ipos,:]
            else:
                if (self.nsamples_per_group % 2) == 1:
                    ineg = np.where(self.freq < 0)[0][::-1]
                    ipos = np.where(self.freq > 0)[0]
                    group_psd = group_psd[...,ipos,:] + group_psd[...,ineg,:]
                else:
                    ineg = np.where(self.freq < 0)[0][::-1]
                    ipos = np.where(self.freq > 0)[0]
                    group_psd = (group_psd[...,ipos,:] + 
                                 interpolate.interp1d(np.abs(self.freq[ineg]), group_psd[...,ineg,:], axis=-2, assume_sorted=True)(self.freq[ipos]))
                    
            self.freq = self.freq[ipos]
            
        self.nfreq = np.size(self.freq)
            
        # Flag groups with flagged data and anomalously high powers
        group_flag = (nflag == 0)
        group_flag = np.expand_dims(group_flag, axis=-2)
        group_flag = group_flag[...,np.zeros(self.nfreq, dtype=np.int8),:]
        
        if self.nsig is not None:
            
            log_psd = np.log(group_psd)
            
            med_log_psd = np.median(log_psd, axis=-1)[...,np.newaxis]
            mad_log_psd = 1.48265*np.median(np.abs(log_psd - med_log_psd), axis=-1)[...,np.newaxis]      
            
            group_flag = group_flag & (np.abs(log_psd - med_log_psd) < (self.nsig*mad_log_psd))
        
        # Censor data
        if self.censor > 0.0:
            
            isort = list(np.ix_(*[np.arange(i) for i in group_psd.shape]))
            isort[-1] = group_psd.argsort(-1)
            
            group_flag_sorted = group_flag[isort]
            
            ncount = np.cumsum(group_flag_sorted, axis=-1)
            
            ncount_max = ncount.max(axis=-1)[...,np.newaxis]
            
            ncount_cut = np.round(ncount_max*(1.0 - self.censor)).astype('int')
                        
            weights = np.where(ncount*group_flag_sorted == ncount_cut, ncount_max + 1 - ncount_cut, np.ones_like(group_flag_sorted))
            
            group_flag_sorted = group_flag_sorted & (ncount <= ncount_cut)
            
            ngroup = np.sum(group_flag_sorted, axis=-1)
            
            weights = weights.astype('float')*group_flag_sorted / ngroup[...,np.newaxis]
                                    
            irevsort = list(np.ix_(*[np.arange(i) for i in group_psd.shape]))
            irevsort[-1] = np.argsort(isort[-1], axis=-1)
            
            weights = weights[irevsort]
            group_flag = group_flag_sorted[irevsort]
            
        else:
            
            ngroup = np.sum(group_flag, axis=-1)
            
            weights = group_flag.astype('float') / ngroup[...,np.newaxis]
        
        # Calculate the mean and error
                        
        self.mean_psd = np.sum(weights*group_psd, axis=-1)
        
        self.err_psd = self.mean_psd / np.sqrt(ngroup)
        
        self.err_psd_emp = np.sqrt(np.sum(group_flag*(group_psd - self.mean_psd[...,np.newaxis])**2, axis=-1) / (ngroup*(ngroup-1)))
        
        self.group_psd = group_psd
        
        self.group_flag = group_flag
        
        
    def fit_psd_poly(self):
        
        nparams = 3
            
        fit_shape = self.mean_psd.shape[:-1] + (nparams,)
    
        self.params = np.zeros(fit_shape, dtype=np.float64)*float('NaN')
        self.model = np.zeros(self.mean_psd.shape, dtype=np.float64)*float('NaN')
    
        for index, junk in np.ndenumerate(self.mean_psd[...,0]):
            
            err = np.squeeze(self.err_psd[index])
            weight = np.where(err > 0.0, 1.0 / err, np.zeros_like(err))
            
            par = np.polynomial.polynomial.polyfit(1.0/self.freq, np.squeeze(self.mean_psd[index]), nparams-1, w=weight)
        
            self.params[index] = par
            self.model[index] = np.polynomial.polynomial.polyval(1.0/self.freq, par)
        
        
    def fit_psd_wls(self, verbose=False, model_name='standard'):
        
        if model_name == 'standard':
            func = psd_model_curve_fit
            ifit = np.array([0, 1, 2])
        elif model_name == 'one_over_f':
            func = psd_model_curve_fit_one_over_f
            ifit = np.array([0, 1])
        elif model_name == 'drift':
            func = psd_model_curve_fit_drift
            ifit = np.array([0, 2])
        else:
            ValueError("Do not recognize model %s" % model_name)
    
        npar = len(ifit)
        nparams = 3
            
        fit_shape = self.mean_psd.shape[:-1] + (nparams,)
    
        self.params = np.zeros(fit_shape, dtype=np.float64)*float('NaN')
        self.model = np.zeros(self.mean_psd.shape, dtype=np.float64)*float('NaN')
        
        bounds = ((0, None), (0, None), (0, None))
            
        for index, junk in np.ndenumerate(self.mean_psd[...,0]):
            
            # Errors
            err = np.squeeze(self.err_psd[index])
            weight = np.where(err > 0.0, 1.0 / err, np.zeros_like(err))
            
            # Data
            data = np.squeeze(self.mean_psd[index])
            
            # Starting parameters
            par0 = np.zeros(nparams) + data.min()/npar
            for ii,pp in enumerate(par0):
                par0[ii] = pp*self.freq[data.argmin()]**ii
                
            par0 = par0[ifit]
                            
            # Find best fit
            #par = leastsq(chisq_psd_model, par0, args=(self.freq, data, weight), method='TNC', bounds=bounds, options={'maxiter':1000})
            try:
                popt, pcov = curve_fit(func, self.freq, data, p0=par0, sigma=err)
                
            except:
                print("Error - curve_fit failed")
            
            
            #if verbose and not par['success']:
             #   print par['message']
             
            popt_s = np.zeros(nparams)
            popt_s[ifit] = popt**2
                    
            self.params[index] = popt_s
            self.model[index] = psd_model(popt_s, self.freq)
        
        
    def fit_psd_mean(self, verbose=False):
        
        if not hasattr(self, 'params'):
            self.fit_psd_wls()
            
        bounds = ((0, None), (0, None), (0, None))
                
        for index, junk in np.ndenumerate(self.mean_psd[...,0]):
        
            # Data
            data = np.squeeze(self.mean_psd[index])
            
            # ndof
            ndof = np.squeeze(np.sum(self.group_flag[index], axis=-1))
            
            # Starting parameters
            par0 = np.squeeze(self.params[index])
            
            # Find best fit
            par = minimize(nloglike_psd_model_mean, par0, args=(self.freq, data, ndof), method='TNC', bounds=bounds, options={'maxiter':1000})
            
            if verbose and not par['success']:
                print par['message']
                    
            self.params[index] = par['x']
            self.model[index] = psd_model(par['x'], self.freq)
            
            
            
    def fit_psd(self, verbose=False):
        
        if not hasattr(self, 'params'):
            self.fit_psd_wls()
            
        bounds = ((0, None), (0, None), (0, None))
                
        for index, junk in np.ndenumerate(self.mean_psd[...,0]):
        
            flag = np.squeeze(self.group_flag[index])
            data = np.squeeze(self.group_psd[index])
            
            freq = self.freq[...,np.newaxis]
            freq = freq[...,np.zeros(np.size(data,axis=-1), dtype=np.int8)]
            
            fit_index = np.where(flag)
            
            data = data[fit_index]
            freq = freq[fit_index]
            
            # Starting parameters
            par0 = np.squeeze(self.params[index])
            
            # Find best fit
            par = minimize(nloglike_psd_model, par0, args=(freq, data), method='TNC', bounds=bounds, options={'maxiter':1000})
            
            if verbose and not par['success']:
                print par['message']
                    
            self.params[index] = par['x']
            self.model[index] = psd_model(par['x'], self.freq)
            
            
    # def fit_psd(self):
    #
    #     if not hasattr(self, 'params'):
    #         self.fit_psd_wls()
    #
    #     for index, junk in np.ndenumerate(self.mean_psd[...,0]):
    #
    #         flag = np.squeeze(self.group_flag[index])
    #         data = np.squeeze(self.group_psd[index])
    #
    #         freq = self.freq[...,np.newaxis]
    #         freq = freq[...,np.zeros(np.size(data,axis=-1), dtype=np.int8)]
    #
    #         fit_index = np.where(flag)
    #
    #         data = data[fit_index]
    #         freq = freq[fit_index]
    #
    #         model = PsdLikelihoodModel(data, freq)
    #         results = model.fit(start_params=np.squeeze(self.params[index]))
    #
    #         self.params[index] = results.params
    #         self.model[index] = psd_model(results.params, self.freq)
            
            
            
    def plot_psd(self, index, xlin=False, ylin=False, xlabel=None, ylabel=None, 
                              color='b', label='data', fontsize=None,
                              no_model=False, no_model_label=False, no_legend=False,
                              do_line=False, **kwargs):
        
        import matplotlib.pyplot as plt
        
        if not xlin:
            plt.xscale('log', nonposx='clip')   
        if not ylin:
            plt.yscale('log', nonposy='clip')
            
        if fontsize is None:
            fontsize = plt.rcParams['font.size']
        
        pcolor = ['g','r','m','c']
        pdescription = ['white', '1/f', '1/f$^2$']
        
        xrng = (self.freq.min(), self.freq.max())
        ymin = self.mean_psd[index].min()
        ymax = self.mean_psd[index].max()
        
        if not ylin:
            ymin = 10.0**(np.floor(np.log10(ymin))-1)
            ymax = 10.0**(np.ceil(np.log10(ymax)))
            
        yrng = (ymin, ymax)
        xrng = (self.freq.min(), self.freq.max())
        xrng = (xrng[0] - 0.1*(xrng[1] - xrng[0]), xrng[1] + 0.1*(xrng[1] - xrng[0]))
        
        if do_line:
            plt.plot(self.freq, np.squeeze(self.mean_psd[index]), linestyle='-', color=color, label=label, **kwargs)
        else:
            plt.errorbar(self.freq, np.squeeze(self.mean_psd[index]), yerr=np.squeeze(self.err_psd[index]), 
                                marker='d', linestyle='None', color=color, label=label, **kwargs)
            
        if (not no_model) & hasattr(self, 'params'):
            params = np.squeeze(self.params[index])
            nparams = np.size(params)
            for ii, pp in enumerate(params):
                eparams = np.zeros_like(params)
                eparams[ii] = pp
                model = psd_model(eparams, self.freq)
                if not no_model_label:
                    lbl = pdescription[ii % len(pdescription)]
                else:
                    lbl = ''
                plt.plot(self.freq, model, color=pcolor[ii % len(pcolor)], 
                            linestyle='--', label=lbl, linewidth=2.0)
                
        if (not no_model) & hasattr(self, 'model'):
            plt.plot(self.freq, np.squeeze(self.model[index]), color='k', linewidth=2.0)
                
        plt.ylim(yrng)
        plt.xlim(xrng)
        plt.grid(which='both', axis='both')
        if not no_legend:
            plt.legend()
        
        if xlabel is None:
            xlabel = 'Frequency [Hz]'
        
        if ylabel is None:
            ylabel = 'Power Spectral Density  [AU$^2$/ Hz]'
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
         
         
def psd_model_curve_fit(freq, a, b, c):
        
    model = (a**2) + (b**2)*np.power(freq, -1) + (c**2)*np.power(freq, -2)
    
    return model
    
    
def psd_model_curve_fit_one_over_f(freq, a, b):
    
    model = (a**2) + (b**2)*np.power(freq, -1)

    return model
    
    
def psd_model_curve_fit_drift(freq, a, c):
        
    model = (a**2) + (c**2)*np.power(freq, -2)
    
    return model
    
         
def psd_model(params, freq):
    
    model = np.zeros_like(freq)
    
    for ii,pp in enumerate(params):
        model += pp*np.power(freq, -ii)
    
    return model
    
    
def chisq_psd_model(params, freq, data, weight):
    
    if np.sum(params < 0.0) != 0:
        return np.nan_to_num(np.inf)
    
    #return np.sum((weight*(data - psd_model(params, freq)))**2)
    return weight*(data - psd_model(params, freq))
    
def nloglike_psd_model(params, freq, data):
    
    if np.sum(params < 0.0) != 0:
        return np.nan_to_num(np.inf)
    
    model = psd_model(params, freq)
        
    x = 2.0*data/model
                        
    return np.sum(-np.log(chi2.pdf(x, 2)))
    
    
def nloglike_psd_model_mean(params, freq, data, ndof):
    
    if np.sum(params < 0.0) != 0:
        return np.nan_to_num(np.inf)
    
    model = psd_model(params, freq)
        
    x = 2*ndof*data/model
                        
    return np.sum(-np.log(chi2.pdf(x, 2*ndof)))
    
    
# class PsdLikelihoodModel(GenericLikelihoodModel):
#
#     def __init__(self, endog, exog, **kwargs):
#
#         super(PsdLikelihoodModel, self).__init__(endog, exog, **kwargs)
#
#
#     def nloglikeobs(self, params):
#
#         if np.sum(params < 0) != 0:
#             return -np.log(np.zeros_like(self.endog))
#
#         mu = params[0] + params[1]*np.power(self.exog, -1.0) + params[2]*np.power(self.exog, -2.0)
#
#         x = 2.0*self.endog/np.squeeze(mu)
#
#         x = np.nan_to_num(x)
#
#         return -np.log(chi2.pdf(x, 2))
#
#
#     def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwargs):
#
#         if start_params is None:
#             raise ValueError("Please specify start parameters.")
#
#
#         return super(PsdLikelihoodModel, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs)
            
            
