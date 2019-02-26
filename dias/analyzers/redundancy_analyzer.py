from dias import CHIMEAnalyzer
from datetime import datetime
from caput import config
from dias.utils import str2timedelta, datetime2str
import pickle
import numpy as np
import time
import h5py
import glob
import datetime


class RedundancyAnalyzer(CHIMEAnalyzer):
    """This analyzer interates through all the redundant baseline sets that comprise the CHIME array. It checks whether
        the visibility during a transit exceeds the MAD of the visibility within a timestep and flags it if it
        continues the exceed the MAD at a certain threshold for a given number of consequtive timesteps.
   
    Attributes
    ----------
    time_thrsld : integer
        The number of consequtive timesteps the visibility exceeds a threshold. Default: 15.
    res_mad_thrsld: integer
        The threshold to check when flagging the visibility. Default : 5.
    """

    time_thrsld = config.Property(proptype=int, default=15)
    res_mad_thrsld = config.Property(proptype=int, default=5)
    
    def setup(self):
        """Setup stage: this is called when dias starts up."""
        self.logger.info('Starting up. My name is {} and I am of type {}.'
                         .format(self.name, __name__))

    def run(self):
        """Main task stage: Check all redundant baselines for non-redundant outliers.
        """
        
        file_idx = 0
        folder_stack = np.sort(glob.glob('/mnt/gong/archive/*_chimestack_corr/'))
        folder_N2 = np.sort(glob.glob('/mnt/gong/archive/*_chimeN2_corr/'))

        fhlist = [h5py.File(fname, 'r') for fname in
                  np.sort(glob.glob(folder_stack[-1] + str('*.h5')))[:1]]

        fhlist_chimeN2 = [h5py.File(fname, 'r') for fname in
              np.sort(glob.glob(folder_N2[-1] + str('*.h5')))]

        rstack = fhlist[0]['reverse_map/stack'][:]

        #all prods to search
        ssel_search = rstack[:]['stack']

        ssel = rstack[:]['stack']

        #list of stack indexes for each redundant set of prods
        u_ssel = np.unique(ssel)

        #check redundancy using yesterday's date 
        transit_dt = (datetime.date.today() - datetime.timedelta(days = 1)).strftime("%Y-%m-%d")
        
        found = 0
        for t_idx in range(len(fhlist_chimeN2),-1,-1):
            times = fhlist_chimeN2[t_idx]['index_map/time']['ctime']
            t_wall_dt = np.array([datetime.datetime.fromtimestamp(a) for a in times])
            t_wall_del_idx = np.where(t_wall_dt == datetime.datetime(1970,1,1,0,0))[0]

            if len(t_wall_del_idx) > 0: 
                t_wall_dt = np.delete(t_wall_dt, t_wall_del_idx, 0)

            #get the cyga transit for this timestream
            bnd = 28
            src_idx, tran_peak_val, full_transit = cygA_transit_keys_grp(t_wall_dt, bnd)

            #confirm yesterdays cyga transit is in this file
            for y in range(len(src_idx)):
                if tran_peak_val[y].strftime("%Y-%m-%d") == transit_dt and full_transit == True:
                    transit_idx = src_idx[y]
                    file_idx = t_idx
                    found = 1
            
            if found == 1:
                break
             
        if found == 1:
            self.logger.info('Cyga transit found. Check array redundancy using cyga transit on {}.'
                 .format(datetime2str(transit_dt)))
            rd_tran_dt = t_wall_dt[transit_idx]
            rd_tran_dt = np.asarray(rd_tran_dt)

            build_ar_freq_set = []

            #loop through all the stack indexes and check every redundant baseline in each stack
            idx = 0
            for st_idx in u_ssel:
                redun_prods = np.where(ssel_search == st_idx)[0]

                build_ar_freq = redun_thrshld_flagger_cygA(redun_prods, fhlist_chimeN2[file_idx], transit_idx, time_thrsld, res_mad_thrsld)

                build_ar_freq[build_ar_freq == None] = NaN

                build_prod_freq_flag = []
                for freq_idx in range(build_ar_freq.shape[0]):
                    build_prod_flag = []
                    for prod_idx in range(build_ar_freq.shape[1]):
                        if np.nansum(build_ar_freq[freq_idx][prod_idx][:]) > 0:
                            build_prod_flag.append(1)
                        else:
                            build_prod_flag.append(0)
                    build_prod_flag = np.asarray(build_prod_flag)
                    build_prod_freq_flag.append(build_prod_flag)
                build_prod_freq_flag = np.asarray(build_prod_freq_flag)

                build_ar_freq_set.append(build_prod_freq_flag)

                if idx % 2 == 0:
                    np.savez('/scratch-slow/tretyako/non_redun_feeds.npz', *build_ar_freq_set)
                idx += 1
        
    def finish(self):
        """Final stage: this is called when dias shuts down."""
        self.logger.info('Shutting down.')
        
    def normalize_complex(x):
        max_amp = np.nanmax(np.abs(x))
        amp = 1/(max_amp**0.5)
        norm = []
        for i in range(len(x)):
            norm.append(amp*x[i])
            
        return np.asarray(norm)
    
    def mad(data, axis=None):
        return np.nanmedian(np.abs(data - np.nanmedian(data, axis)), axis)
    
    def redun_thrshld_flagger_cygA(redun_prods, fhlist_chimeN2, transit_idx, time_thrsld = 15, res_mad_thrsld = 5): 

        freq_N2_lst = fhlist_chimeN2[0]['index_map/freq'][:]
        freq_plot = []
        rd_tran_freq = []
        rd_tran_dt_freq = []

        for f_idx, cur_freq in enumerate(freq_N2_lst):
            freq_plot.append(cur_freq[0])
            vis = np.concatenate([fh['vis'][f_idx, redun_prods, :] for fh in fhlist_chimeN2], axis=1)
            rd_tran_ar = []
            rd_tran_dt_ar = []

            for i, vis_prod in enumerate(vis):
                rd_tran = normalize_complex(np.abs(vis_prod[transit_idx]))
                rd_tran_ar.append(rd_tran)
            rd_tran_ar = np.asarray(rd_tran_ar)
            rd_tran_freq.append(rd_tran_ar)
        freq_plot = np.asarray(freq_plot)
        rd_tran_freq = np.asarray(rd_tran_freq)

        res_vis_t_freq = []
        mad_vis_freq = []
        vis_med_freq = []

        for f in range(freq_plot.shape[0]):
            mad_vis = []
            vis_med = []
            for t in range(rd_tran_freq.shape[2]):
                mad_vis.append(mad(np.abs(rd_tran_freq[f,:,t])))
                vis_med.append(np.median(np.abs(rd_tran_freq[f,:,t])))
            vis_med_freq.append(vis_med)
            mad_vis_freq.append(mad_vis)
        mad_vis_freq = np.asarray(mad_vis_freq)
        vis_med_freq = np.asarray(vis_med_freq)

        for f in range(freq_plot.shape[0]):
            vis_ar = []
            for v in range(rd_tran_freq.shape[1]):
                vis_time = []
                for t in range(rd_tran_freq.shape[2]):
                    if np.abs(np.abs(rd_tran_freq[f,v,t]) - vis_med_freq[f,t])/mad_vis_freq[f,t] > res_mad_thrsld:
                        vis_time.append(1)
                    else:
                        vis_time.append(0)
                vis_time = np.asarray(vis_time)
                vis_ar.append(vis_time)
            vis_ar = np.asarray(vis_ar)
            res_vis_t_freq.append(vis_ar)
        res_vis_t_freq = np.asarray(res_vis_t_freq)

        build_ar_freq = []

        for f in range(freq_plot.shape[0]):
            plt_vis = []
            build_ar = []
            for v in range(rd_tran_freq.shape[1]):
                row = [None] * rd_tran_freq.shape[2]
                build_ar.append(row)
                if np.sum(res_vis_t_freq[f,v]) > time_thrsld:
                    for t in range(rd_tran_freq.shape[2]):
                        if res_vis_t_freq[f,v,t] == 1:
                            plt_vis.append((f,v,t)) #freq, vis,time
            plt_vis = np.asarray(plt_vis)
            build_ar = np.asarray(build_ar)

            for i in plt_vis:
                build_ar[i[1],i[2]] = np.abs(rd_tran_freq[i[0],i[1],i[2]])

            build_ar_freq.append(build_ar)
        build_ar_freq = np.asarray(build_ar_freq)

        return build_ar_freq#, t_wall_dt #(freq, vis, time), cygA_transit_time