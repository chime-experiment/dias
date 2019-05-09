from dias import CHIMEAnalyzer
from datetime import datetime
import calendar
from caput import config
from dias.utils.string_converter import str2timedelta
from ch_util import data_index

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
from ch_util import cal_utils
from dias import exception
from dias import __version__ as dias_version_tag

from ch_util import ephemeris
from datetime import datetime  
from ch_util import data_index 
from ch_util.fluxcat import FluxCatalog
from scipy.optimize import curve_fit
import time
import scipy.constants

class source_spectra_Analyzer(CHIMEAnalyzer):
    """Source_spectra analyzer.

    Based on CHIMEAnalyzer class.
    """

    """SourceSpectraAnalyzer.

    Analyzer for looking at source transits and spectra.

    Metrics
    ----------
    None

    Output Data
    -----------------
        h5 file, containing fringestopped visibility (Jy),
        averaged over all feeds for each polarization,
        as a function of frequency and time.
        The input file is the chime stacked dataset.

    File naming
    ..........................
        `<src>_<csd>_<output_suffix>.h5`
        `src` is the source analyzed, `csd` marks the day of transit and
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
    ra
        Right Ascension of the source
    ha
        Hour angle covered in the output file

    Datasets
    ...................
    visiblity
        Fringestopped visibility as a function of frequency and time 
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
    rot : float
        Rotation angle of the cylinder in degrees
    nsgima : float
        Span required from the peak of transit
        in terms of sigma
    perform_fit : boolean
        Performs a Gaussian fit to transit

    """

    correlator = config.Property(proptype=str, default='chime')
    output_suffix = config.Property(proptype=str, default='sensitivity')
    acq_suffix = config.Property(proptype=str, default='stack_corr')

    nfreq_per_block = config.Property(proptype=int, default=16)
    include_auto = config.Property(proptype=bool, default=False)
    include_intracyl = config.Property(proptype=bool, default=False)
    include_crosspol = config.Property(proptype=bool, default=False)
    sep_cyl = config.Property(proptype=bool, default=False)
    cyl_start_char = config.Property(proptype=int, default=65)
    cyl_start_num = config.Property(proptype=int, default=2)
    lag = config.Property(proptype=str2timedelta, default="4h")

    rot = config.Property(proptype=float, default=-0.088)
    nsigma = config.Property(proptype=float, default=1.0)
    perform_fit = config.Property(proptype=bool, default=False)    
        
    def run(self):
        
        lat = np.radians(ephemeris.CHIMELATITUDE)
        # Create transit tracker
        source_list = [] 
        src = 'cyg_a'
        src_body  = FluxCatalog[src].skyfield

        self.logger.info("Initializing offline point source processing.")
        
        stop_time = datetime.utcnow() - self.lag
        # Query files from now to period hours back
        start_time = stop_time - self.period
        
        self.logger.info('Searching for transits from %s to %s' %
                         (str(start_time), str(stop_time)))
        
        window = self.nsigma * cal_utils.guess_fwhm(400.0, pol='X', dec=src_body.dec.radians, sigma=True) 
        #nsigma distance in degree from transit from peak
        time_delta = 2 * (window/360.0) * 24 * 3600.0  #time window around transit in sec

        # Find all calibration files that have transit of given source
        f = self.Finder()
        f.set_time_range(start_time, stop_time)
        f.accept_all_global_flags()
        f.only_corr()
        f.filter_acqs((data_index.ArchiveInst.name == self.acq_suffix))
        f.include_transits(src_body, time_delta=time_delta) 
        file_list = f.get_results()
        nfile = len(file_list)
        times = [file_list[ii][1] for ii in range(0,nfile)]
        
        try:
            all_files = file_list[0][0]
            if not all_files:
                raise IndexError()
        except IndexError:
            err_msg = 'No {} files found from last {}.'.format(
                self.acq_suffix, self.period)
            raise exception.DiasDataError(err_msg)
            
        # Loop over files
        for f_ind, files in enumerate(all_files):
            # Read file time range
            with h5py.File(files, 'r') as handler:
                timestamp = handler['index_map']['time']['ctime'][:]
            
            csd = int(np.median(ephemeris.csd(timestamp)))
                
            # Compute source coordinates
            timestamp0 = np.median(timestamp)
            src_ra, src_dec = ephemeris.object_coords(src_body, date=timestamp0, deg=False)
            start_time, stop_time = file_list[f_ind][1] 
            start = int(np.argmin(np.abs(timestamp - start_time)))
            stop = int(np.argmin(np.abs(timestamp - stop_time)))
            
            is_daytime = 0.0
            #test if the source is transiting in the daytime
            solar_rise = ephemeris.solar_rising(start_time - 24.0*3600.0, end_time=stop_time)
            for rr in solar_rise:
                ss = ephemeris.solar_setting(rr)[0]
                if ((start_time <= ss) and (rr <= stop_time)):
                    is_daytime += 1
                    tt = ephemeris.solar_transit(rr)[0]
                    #if Sun is in the beam
                    if (start_time <= tt <= stop_time):
                        is_daytime += 1
                    break

            if is_daytime > 1:
                self.logger.info('Not processing %s as Sun is in the primary beam' % (src))
                continue
        
            self.logger.info("Now processing %s transit on CSD %d" % (src, csd))

            # Look up inputmap
            timestamp0 = ephemeris.csd_to_unix(csd)
            inputmap = tools.get_correlator_inputs(ephemeris.unix_to_datetime(timestamp0),
                                                   correlator=self.correlator)
            
            # Load index map and reverse map
            data = andata.CorrData.from_acq_h5(files, start=start, stop=stop,
                                               datasets=['reverse_map', 'flags/inputs'],
                                               apply_gain=False, renormalize=False)
            # Determine axes
            nfreq = data.nfreq
            nblock = int(np.ceil(nfreq / float(self.nfreq_per_block)))
            
            timestamp = data.time
            ntime = timestamp.size

            # Get baselines
            prod, prodmap, dist, conj, cyl, scale = self.get_baselines(data.index_map, inputmap)

            # Determine groups
            polstr = np.array(sorted(prod.keys()))
            npol = polstr.size
            
            # Calculate counts
            cnt = np.zeros((data.index_map['stack'].size, ntime), dtype=np.float32)
            
            if np.any(data.flags['inputs'][:]):
                for pp, ss in zip(data.index_map['prod'][:], data.reverse_map['stack']['stack'][:]):
                    cnt[ss, :] += (data.flags['inputs'][pp[0], :] * data.flags['inputs'][pp[1], :])
            else:
                for ss, val in Counter(data.reverse_map['stack']['stack'][:]).iteritems():
                    cnt[ss, :] = val

            # Calculate hour angle
            ra = np.radians(ephemeris.lsa(timestamp))
            ha = ra - src_ra
            ha = self.correct_phase_wrap(ha, deg=False)
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
        
                print("Processing block %d (of %d):  %d - %d" % (block_number, nblock, fstart, fstop))
                self.logger.info("Processing block %d (of %d):  %d - %d" % (block_number, nblock, fstart, fstop))

                bdata = andata.CorrData.from_acq_h5(files, start=start, stop=stop, freq_sel=freq_sel,
                                                    datasets=['vis', 'flags/vis_weight'],
                                                    apply_gain=False, renormalize=False)


                bflag = (bdata.weight[:] > 0.0).astype(np.float32)
                bvar = tools.invert_no_zero(bdata.weight[:])
                
                lmbda = scipy.constants.c * 1e-6 / bdata.freq[:, np.newaxis, np.newaxis]
                
                # Loop over polarizations
                for ii, pol in enumerate(polstr):

                    self.logger.info("Processing Pol %s" % pol)

                    pvis = bdata.vis[:, prod[pol], :]
                    pvar = bvar[:, prod[pol], :]
                    pflag = bflag[:, prod[pol], :]
                    pcnt = cnt[np.newaxis, prod[pol], :]
                    pscale = scale[pol][np.newaxis, :, np.newaxis]
                
                    fringestop_phase = tools.fringestop_phase(ha, lat, src_dec,
                                                              dist[pol][np.newaxis, :, 0, np.newaxis] / lmbda,
                                                              dist[pol][np.newaxis, :, 1, np.newaxis] / lmbda)

                    vis[freq_sel, ii, :] += np.sum(pscale * pcnt * pflag * pvis * fringestop_phase, axis=1)
                    var[freq_sel, ii, :] += np.sum((pscale * pcnt)**2 * pflag * pvar, axis=1)
                    counter[freq_sel, ii, :] += np.sum(pscale * pcnt * pflag, axis=1)

                self.logger.info("Took %0.1f seconds to process this block." % (time.time() - t0,))
                
                del bdata
                gc.collect()
            
            # Normalize
            inv_counter = tools.invert_no_zero(counter)
            vis *= inv_counter
            var *= inv_counter**2
            
            ra = np.degrees(np.unwrap(ra))
            ha = ra - np.degrees(src_ra)

            # Fit response to model
            if self.perform_fit:

                fwhm = np.zeros((nfreq, npol), dtype=np.float32)
                for ii in range(npol):
                    fwhm[:, ii] = cal_utils.guess_fwhm(data.freq, pol=polstr[ii][0], dec=src_dec, sigma=True)

                flag = counter > 0.0

                parameter, parameter_cov, resid_rms = self.fit_point_source_transit(ha, vis.real, np.sqrt(var),
                                                                                               flag=flag, fwhm=fwhm)
            # Write to file
            output_file = os.path.join(self.write_dir, "%s_csd_%d_%s.h5" % 
                                      (src.lower(), csd, self.output_suffix))
            self.logger.info("Writing output files...")
            
            with h5py.File(output_file, 'w') as handler:
        
                index_map = handler.create_group('index_map')
                index_map.create_dataset('freq', data=data.index_map['freq'][:])
                index_map.create_dataset('pol', data=np.string_(polstr))
                index_map.create_dataset('time', data=data.time)
                index_map.create_dataset('ra', data=ra)
                index_map.create_dataset('ha', data=ha)
        
                dset = handler.create_dataset('vis', data=vis)
                dset.attrs['axis'] = np.array(['freq', 'pol', 'ha'],dtype='S')
                
                dset = handler.create_dataset('weight', data=tools.invert_no_zero(var))
                dset.attrs['axis'] = np.array(['freq', 'pol', 'ha'], dtype='S')
                
                dset = handler.create_dataset('count', data=counter.astype(np.int))
                dset.attrs['axis'] = np.array(['freq', 'pol', 'ha'], dtype='S')
                
                if self.perform_fit:

                    index_map.create_dataset('param', data='model_fitting')

                    dset = handler.create_dataset('residual_noise', data=resid_rms)
                    dset.attrs['axis'] = np.array(['freq', 'pol'], dtype='S')

                    dset = handler.create_dataset('parameter', data=parameter)
                    dset.attrs['axis'] = np.array(['freq', 'pol', 'param'], dtype='S')
                
                    dset = handler.create_dataset('parameter_cov', data=parameter_cov)
                    dset.attrs['axis'] = np.array(['freq', 'pol', 'param', 'param'], dtype='S')
                    
                handler.attrs['source'] = src
                handler.attrs['csd'] = csd
                handler.attrs['instrument_name'] = self.correlator
                handler.attrs['collection_server'] = subprocess.check_output(
                    ["hostname"]).strip()
                handler.attrs['system_user'] = subprocess.check_output(
                    ["id", "-u", "-n"]).strip()
                handler.attrs['git_version_tag'] = dias_version_tag
            self.logger.info('File successfully written out.')

    ###################################################
    # auxiliary routines
    ###################################################

    def fit_point_source_transit(self, hour_angle, response, weight, flag=None, 
                                 fwhm=None):
        """ Fits the point source response to a model that
            consists of a gaussian in amplitude plus a polynomial background.

        .. math::
            g(hour_angle) = peak_amplitude * \exp{-4 \ln{2} [(hour_angle - centroid)/fwhm]^2} *
                    \exp{j [phase_intercept + phase_slope * (hour_angle - centroid)]}

        Parameters
        ----------
        hour_angle : np.ndarray[nha, ]
            Transit right ascension.
        response : np.ndarray[nfreq, nra]
            Complex array that contains point source response.
        response_error : np.ndarray[nfreq, nra]
            Real array that contains 1-sigma error on
            point source response.
        flag : np.ndarray[nfreq, nra]
            Boolean array that indicates which data points to fit.

        Returns
        -------
        parameter : np.ndarray[nfreq, nparam]
            Best-fit parameters for each frequency and input:
            [peak_amplitude, centroid, fwhm, phase_intercept, phase_slope].
        parameter_cov: np.ndarray[nfreq, nparam, nparam]
            Parameter covariance for each frequency and input.
        """

        # Check if boolean flag was input
        if flag is None:
            flag = np.ones(response.shape, dtype=np.bool)
        elif flag.dtype != np.bool:
            flag = flag.astype(np.bool)

        # Create arrays to hold best-fit parameters and
        # parameter covariance.  Initialize to NaN.
        nfreq = response.shape[0]
        npol = response.shape[1]
        nparam = 6 #magic number!! 

        parameter = np.full((nfreq, npol, nparam), np.nan, dtype=np.float32)
        parameter_cov = np.full((nfreq, npol, nparam, nparam), np.nan, dtype=np.float32)
        resid_rms = np.full((nfreq, npol), np.nan, dtype=np.float32)

        # Create initial guess at FWHM if one was not input
        if fwhm is None:
            fwhm = np.full((nfreq, npol), 3.0, dtype=np.float32)

        # Iterate over frequency/pol and fit point source transit
        for ff in range(nfreq):
            
            for pp in range(npol):

                this_flag = flag[ff, pp]

                # Only perform fit if there is enough data.
                # Otherwise, leave parameter values as NaN.
                if np.sum(this_flag) <= nparam:
                    continue

                # We will fit the complex data.  Break n-element complex array g(ra)
                # into 2n-element real array [Re{g(ra)}, Im{g(ra)}] for fit.
                x = hour_angle[this_flag]
            
                y = response[ff, pp, this_flag]
                yerr = np.sqrt(tools.invert_no_zero(weight[ff, pp, this_flag]))

                # Initial estimate of parameter values
                p0 = np.array([np.max(y) - np.median(y), np.median(x), fwhm[ff, pp],
                               np.asarray([np.median(y), 0.0, 0.0])])

                # Perform the fit.  If there is an error,
                # then we leave parameter values as NaN.
                try:
                    popt, pcov = curve_fit(fit_func1, x, y,
                                            p0=p0, sigma=yerr, absolute_sigma=False)
                except Exception as error:
                    continue
                
                # Save the results
                parameter[ff, pp] = popt
                parameter_cov[ff, pp] = pcov

                # Compute scatter in residuals
                resid = y - fit_func1(x, *popt)

                resid_rms[ff, pp] = np.std(resid_dict)

        # Return the best-fit parameters and parameter covariance
        return parameter, parameter_cov, resid_rms

    def fit_func1(self, x, peak_amplitude, p0): 

        centroid, fwhm, bpoly = p0
        dx = x - centroid
        return func_gauss(dx, peak_amplitude, fwhm) + func_background(dx, bpoly)

    def func_gauss(self, dx, peak_amplitude, fwhm):
            
        return peak_amplitude * np.exp(-4.0*np.log(2.0)*(dx/fwhm)**2)

    def func_background(self, dx, bck):

        model = np.polyval(bck, dx)
        return  model

    def correct_phase_wrap(self, phi, deg=False):

        if deg:
            return ((phi + 180.0) % 360.0) - 180.0
        else:
            return ((phi + np.pi) % (2.0 * np.pi)) - np.pi

    def get_baselines(self, indexmap, inputmap):
        
        prod = defaultdict(list)
        prodmap = defaultdict(list)
        dist = defaultdict(list)
        conj = defaultdict(list)
        cyl = defaultdict(list)
        scale = defaultdict(list)

        # Compute feed positions without rotation
        tools.change_chime_location(rotation=self.rot)
        feedpos = tools.get_feed_positions(inputmap)
        
        for pp, (this_prod, this_conj) in enumerate(indexmap['stack']):

            if this_conj:
                bb, aa = indexmap['prod'][this_prod]
            else:
                aa, bb = indexmap['prod'][this_prod]
            
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
                key = 'XX'

            elif tools.is_array_y(inp_aa) and tools.is_array_y(inp_bb):
                key = 'YY'

            elif not self.include_crosspol:
                continue
                
            elif tools.is_array_x(inp_aa) and tools.is_array_y(inp_bb):
                key = 'XY'

            elif tools.is_array_y(inp_aa) and tools.is_array_x(inp_bb):
                key = 'YX'

            else:
                raise RuntimeError("CHIME feeds not polarized.")

            this_cyl = '%s%s' % (self.get_cyl(inp_aa.cyl), self.get_cyl(inp_bb.cyl))
            if self.sep_cyl:
                key = key + '-' + this_cyl

            prod[key].append(pp)
            prodmap[key].append((aa, bb))
            conj[key].append(this_conj)
            dist[key].append(this_dist)
            cyl[key].append(this_cyl)

            if aa == bb:
                scale[key].append( 0.5 )
            else:
                scale[key].append( 1.0 )

        for key in prod.keys():
            prod[key] = np.array(prod[key])
            prodmap[key] = np.array(prodmap[key], dtype=[('input_a', '<u2'), ('input_b', '<u2')])
            dist[key] = np.array(dist[key])
            conj[key] = np.nonzero(np.ravel(conj[key]))[0] 
            cyl[key] = np.array(cyl[key])
            scale[key] = np.array(scale[key])
            
        tools.change_chime_location(default=True)
            
        return prod, prodmap, dist, conj, cyl, scale

    def get_cyl(self, cyl_num):
        """Return the cylinfer ID (char)."""
        return chr(cyl_num - self.cyl_start_num + self.cyl_start_char)
