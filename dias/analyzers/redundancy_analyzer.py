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
        for t_idx in range(len(fhlist_chimeN2)-1,-1,-1):
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
             
        if found == 0:
            self.logger.warn('Did not find any data in the archive for CygA on ' + str(transit_dt)
            return 
        elif found == 1:
            self.logger.info('Cyga transit found. Check array redundancy using cyga transit on {}.'
                 .format(datetime2str(transit_dt)))

            build_prod_freq_flag = []
            
            vis = fhlist_chimeN2[file_idx]['vis'][:, :, transit_idx]
                             
            freq_N2_lst = fhlist_chimeN2[file_idx]['index_map/freq'][:]

            #loop through all the stack indexes and check every redundant baseline in each stack
            for f_idx in range(vis.shape[0]):
                build_prod_set = []
                redun_prods_idx_ar = []
                for st_idx in u_ssel:
                    redun_prods = np.where(ssel_search == st_idx)[0]
                    redun_prods_idx_ar.append(redun_prods)

                    build_ar = redun_thrshld_flagger_cygA(vis[f_idx,redun_prods])

                    build_ar[build_ar == None] = NaN

                    build_prod_flag = []
                    for prod_idx in range(build_ar.shape[0]):
                        if np.nansum(build_ar[prod_idx][:]) > 0:
                            build_prod_flag.append(1)
                        else:
                            build_prod_flag.append(0)
                    build_prod_flag = np.asarray(build_prod_flag)
                    build_prod_set.append(build_prod_flag)          
                build_prod_freq_flag.append(build_prod_set) #4 freq, stack IDX, prods for stack idx - flag (1 or 0)
            redun_prods_idx_ar = np.asarray(redun_prods_idx_ar)

            with h5py.File(os.path.join(self.write_dir, 'redundancy_check_' + str(transit_dt) + '.h5'), 'w') as f:
                f.create_dataset('redund_prod_flags', data=build_prod_freq_flag, dtype=int)
                f.create_dataset('axis/stack_idx', data=u_ssel, dtype=int)
                f.create_dataset('axis/redun_prod_idx', data=redun_prods_idx_ar, dtype=int)
                f.create_dataset('axis/freq', data=freq_N2_lst, dtype=float)
                f.close()
        
                self.logger.info('Redundancy flags written for CygA transit on ' + str(transit_dt))

    def normalize_complex(x):
        max_amp = np.nanmax(np.abs(x))
        amp = 1/(max_amp**0.5)
        norm = []
        for i in range(len(x)):
            norm.append(amp*x[i])
            
        return np.asarray(norm)
    
    def mad(data, axis=None):
        return np.nanmedian(np.abs(data - np.nanmedian(data, axis)), axis)    
    
    def redun_thrshld_flagger_cygA(vis, time_thrsld = 15, res_mad_thrsld = 5): #Vis should be [[redun_prods],[times]]
        rd_tran_ar = []

        for prod_idx in range(vis.shape[0]): #prod, time
            rd_tran = normalize_complex(np.abs(vis[prod_idx]))
            rd_tran_ar.append(rd_tran)
        rd_tran_ar = np.asarray(rd_tran_ar)

        res_vis_t = []
        mad_vis_set = []
        vis_med_set = []

        mad_vis = []
        vis_med = []
        for t in range(rd_tran_ar.shape[1]):
            mad_vis.append(mad(np.abs(rd_tran_ar[:,t])))
            vis_med.append(np.median(np.abs(rd_tran_ar[:,t])))
        mad_vis = np.asarray(mad_vis)
        vis_med = np.asarray(vis_med)

        vis_ar = []
        for v in range(rd_tran_ar.shape[0]):
            vis_time = []
            for t in range(rd_tran_ar.shape[1]):
                if np.abs(np.abs(rd_tran_ar[v,t]) - vis_med[t])/mad_vis[t] > res_mad_thrsld:
                    vis_time.append(1)
                else:
                    vis_time.append(0)
            vis_time = np.asarray(vis_time)
            vis_ar.append(vis_time)        
        vis_ar = np.asarray(vis_ar)

        plt_vis = []
        build_ar = []
        for v in range(rd_tran_ar.shape[0]):
            row = [None] * rd_tran_ar.shape[1]
            build_ar.append(row)
            if np.sum(vis_ar[v]) > time_thrsld:
                for t in range(rd_tran_ar.shape[1]):
                    if vis_ar[v,t] == 1:
                        plt_vis.append((v,t)) #vis,time
        plt_vis = np.asarray(plt_vis)
        build_ar = np.asarray(build_ar)

        for i in plt_vis:
            build_ar[i[0],i[1]] = np.abs(rd_tran_ar[i[0],i[1]])

        return build_ar#(vis, time)
