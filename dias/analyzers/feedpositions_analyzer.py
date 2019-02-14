from dias.chime_analyzer import CHIMEAnalyzer
from datetime import datetime, timedelta
from caput import config, time
from dias.utils.time_strings import str2timedelta, datetime2str

from ch_util import andata, data_index, ephemeris, fluxcat
import numpy as np
import os

import scipy.constants as c
import scipy.linalg as la

import h5py


#from ch_util.fluxcat import FluxCatalog 


# Choose 10 good frequencies. I chose the same ones that we used when writing full N2 data for 10 frequencies.
frequencies = [ 758.203125,  716.40625 ,  697.65625 ,  665.625   ,  633.984375,
               597.265625,  558.203125,  516.40625 ,  497.265625,  433.59375]
# Brightest sources. VirA does not have enough S/N. 
sources = {'CAS_A' : ephemeris.CasA, 'CYG_A' : ephemeris.CygA, 'TAU_A' : ephemeris.TauA}


class FeedpositionsAnalyzer(CHIMEAnalyzer):
    """A CHIME analyzer to calculate the East-West feed positions from the fringe rates
    in eigenvectors. The eigenvectors of the visibility matrix are found in the archive, then orhtogonalized.
    To get the feed-positions in the UV plane we fourier transform the eigenvectors over the time axis.
    At the moment this Analyzer is supposed to run during the day to check for night transit data. 
   
    Attributes
    ----------
    ref_feed_P1 : integer
        The feed we reference the polarisation 1 data to. Default: 2.
    ref_feed_P2: integer
        The feed we reference the polarisation 1 data to. Default : 258.
    pad_fac_EW : integer
        By which factor we pad the data before performing the fourier transform. Default : 256.
    """
    
    ref_feed_P1 = config.Property(proptype=int, default=2)
    ref_feed_P2 = config.Property(proptype=int, default=258)
    pad_fac_EW = config.Property(proptype=int, default=256)
 

    def setup(self):
        self.logger.info('Starting up. My name is ' + self.name +
                            ' and I am of type ' + __name__ + '.')
        self.resid_metric = self.add_data_metric("ew_pos_residuals_analyzer_run", "if feedposition task has run or not for specific source", labelnames=['source'], unit='')
       
    def run(self):
        
        end_time = datetime.utcnow()
        start_time = end_time - self.period # period is 24h
        #self.logger.info('Analyzer period: starttime ' + datetime2str(start_time) + ', endtime ' + datetime2str(end_time) + ', period ' + str(self.period))
        self.end_time_night = time.unix_to_datetime(ephemeris.solar_rising(start_time, end_time))[0]
        self.start_time_night = time.unix_to_datetime(ephemeris.solar_setting(start_time, end_time))[0]

        self.logger.info('Analyzing night data between UTC times' + datetime2str(self.start_time_night) +
                         ' and ' + datetime2str(self.end_time_night) + '.')

        
        sources = {'CAS_A' : ephemeris.CasA, 'CYG_A' : ephemeris.CygA, 'TAU_A' : ephemeris.TauA}
        
        night_transits = []
        
        # Check which of these sources transit at night
        for src in sources.keys():
            transit = ephemeris.transit_times(sources[src], self.start_time_night, self.end_time_night)
            src_ra, src_dec = ephemeris.object_coords(fluxcat.FluxCatalog[src].skyfield, date=self.start_time_night, deg=True)
            if transit:
                night_transits.append(src)
        self.logger.info('All night transits found: ' + night_transits[0])
                
        # Convert current datetime to str and keep only date
        time_str = time.datetime_to_timestr(self.start_time_night)[:8]
        
        # for each source in night_transits calculate the East-West positions
        for night_source in night_transits:
            self.logger.info('Processing source ' + night_source + ' to find feed positions...')
            ew_positions, resolution = self.east_west_positions(night_source)
            
            if ew_positions is None:
                self.logger.info('Moving on.')
                continue     
            
            # Calculate the median for each cylinder/ polarisation pair.
            ew_offsets = np.ones_like(ew_positions)
            for i in range(0, 8, 2):
                ew_offsets[:, i*256:(i+1)*256] *= np.median(ew_positions[:, i*256:(i+1)*256], axis=1)[:, np.newaxis]
                ew_offsets[:, (i+1)*256:(i+2)*256] *= np.median(ew_positions[:, (i+1)*256:(i+2)*256], axis=1)[:, np.newaxis]
            
            # Subtract median from East-West positions to get residuals.
            residuals = ew_positions - ew_offsets
            
            with h5py.File(os.path.join(self.write_dir, time_str + '_' + night_source + '_positions.h5'), 'w') as f:
                f.create_dataset('east_west_pos', data=ew_positions, dtype=float)
                f.create_dataset('east_west_resid', data=residuals, dtype=float)
                f.create_dataset('axis/freq', data=frequencies, dtype=float)
                f.create_dataset('axis/input', data=np.arange(2048), dtype=int)
                f.close()
        
                self.logger.info('Fourier transform resolution in [m] from source: ' + night_source + " : " + str(resolution[0][0]))
                self.logger.info('Writing positions from ' + night_source + ' data to ' +
                                 self.write_dir)
                self.logger.info('Exporting East-West residuals to prometheus')
                
                # Export a task metric that gives ouput 1 when the task ran successfully
                self.resid_metric.labels(source=night_source).set(1)
        
        
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
            self.logger.warn('Did not find any data in the archive for source ' + src)
            return 
 
        reader = results_list[0].as_reader()
    
        # Choose 10 good frequencies. I chose the same ones that we used when writing full N2 data for 10 frequencies.
        freq_sel = [ 758.203125,  716.40625 ,  697.65625 ,  665.625   ,  633.984375,
                      597.265625,  558.203125,  516.40625 ,  497.265625,  433.59375]
        reader.select_freq_physical(freq_sel)
    
        # Read the data  
        data  = reader.read()
    
        # Get the timestamps
        time = data.index_map['time']['ctime'][:]
        # Get the frequencies
        freq = data.freq
    
        # TO DO: Check here the eigenvalues on versus off srouce and see if we are not dominated by RFI.
        #
    
        tshape = data['evec'].shape[-1]
    
        # Make some empty arrays for the orthogonalized eigenvectors
        vx_vec = np.zeros((len(freq_sel), 2048, tshape), dtype=complex)
        vy_vec = np.zeros((len(freq_sel), 2048, tshape), dtype=complex)
    
        for f in range(len(freq_sel)):
            for i in range(tshape):
                vx, vy = self.orthogonalize(data, f, i)
                vx_vec[f, :, i] = vx
                vy_vec[f, :, i] = vy

        # Combine the two polarisations into one vector evec
        evec = np.zeros((len(freq_sel), 2048, len(time)), dtype=complex)
    
        # Reference eigenvector to the first good feed for the NS(P1) and EW(P2) polarisation
        for i in range(0, 8, 2):
            evec[:, i*256:(i+1)*256, :] = vy_vec[:, i*256:(i+1)*256, :] / np.exp(1J* np.angle(vy_vec[:, self.ref_feed_P1, :]))[:, np.newaxis, :]
            evec[:, (i+1)*256:(i+2)*256, :] = vx_vec[:, (i+1)*256:(i+2)*256, :] / np.exp(1J* np.angle(vx_vec[:, self.ref_feed_P2, :]))[:, np.newaxis, :]
    
        # Create empty arrays for East-West positions and residuals
        ew_positions = np.zeros((len(freq_sel), 2048), dtype=float)
        resolution = np.zeros((len(freq_sel), 2048), dtype=float)
    
        # Get the source RA and DEC 
        ra, dec = ephemeris.object_coords(fluxcat.FluxCatalog[src].skyfield, date=self.start_time_night, deg=True) 
    
        # Loop over frequencies and then inputs to get the EW-positions
        for f in range(len(freq_sel)):
            for i in range(2048):
                ew_positions[f, i], resolution[f, i] = self.get_ew_pos_fft(time, evec[f, i, :], freq[f], np.radians(dec), pad_fac=self.pad_fac_EW)


        return ew_positions, resolution
    
    
    
    # Orthogonalization routine
    def orthogonalize(self, data, fsel, time_index):  
        # If we did not write data for that frequency because of a node crash skip that frequency 
        # and return a vector with zeros.
        if all(np.abs(data['evec'][fsel, 0, :, time_index]) == 0):
            vx = np.zeros((2048), dtype=complex)
            vy = np.zeros((2048), dtype=complex)

            return vx, vy

        # Construct masks for the X and Y polarisations
        Ax = (((np.arange(2048, dtype=np.int) / 256) % 2) == 1).astype(np.float64)
        Ay = (((np.arange(2048, dtype=np.int) / 256) % 2) == 0).astype(np.float64)

        U = data['evec'][fsel, :2, :, time_index].T
        Lh = (data['eval'][fsel, :2, time_index])**0.5

        Vtx = Lh[:, np.newaxis] * np.dot(U.T.conj(), Ax[:, np.newaxis] * U) * Lh[np.newaxis, :]
        m, vv = la.eigh(Vtx)
        vx = Ax[:, np.newaxis] * np.dot(U, vv / Lh[:, np.newaxis])
        vx /= np.dot(vx.T.conj(), vx).diagonal()**0.5

        Vty = Lh[:, np.newaxis] * np.dot(U.T.conj(), Ay[:, np.newaxis] * U) * Lh[np.newaxis, :]
        m, vv = la.eigh(Vty)
        vy = Ay[:, np.newaxis] * np.dot(U, vv / Lh[:, np.newaxis])
        vy /= np.dot(vy.T.conj(), vy).diagonal()**0.5

        return vx[:, -1], vy[:, -1]

    
    def get_ew_pos_fft(self, times, evec_stream, f, dec, pad_fac=pad_fac_EW):
        """Routine that gets feed positions from the eigenvector data via an FFT. 
        The eigenvector is first apodized with ahannings window function and then 
        fourier transformed along the time axis.


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
            The East-West positions referenced to 2 feeds on the first cylinder. 

        position_resolution: float
            Position resolution, determined by number of time samples times padding factor.
        """
        n = len(times)

        # Use a hannings window function to apodize the data
        apod = np.hanning(n)
        # Time resolution in radians
        dt = np.radians(ephemeris.lsa(times[1])
                - ephemeris.lsa(times[0]))
        # Calculate the fourier transform of the apodized eigenvector data
        spec = np.fft.fft(apod*evec_stream, n = n*pad_fac)
        freq = np.fft.fftfreq(n*pad_fac, dt)
    
        # Find the maximum power in the spectrum
        x_loc = freq[np.argmax(np.abs(spec))]
        # The conjugate to time is the baseline vector in units of wavelength divided by the cos(declination)
        conv_fac = -2.99792e2 / f / np.cos(dec)

        position = x_loc * conv_fac
        position_resolution =  np.abs(freq[1] - freq[0]) * np.abs(conv_fac)

        return position, position_resolution