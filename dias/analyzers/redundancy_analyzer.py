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
from ch_util import data_index, ephemeris

class RedundancyAnalyzer(CHIMEAnalyzer):
    """This analyzer iterates through all the redundant baseline sets that comprise the CHIME array. It checks whether
        the visibility during a transit exceeds the MAD of the visibility within a timestep and flags it if
        continues to exceed the MAD at a certain threshold for a given number of consecutive timesteps.
   
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
        self.cyg_a_found = self.add_data_metric("cyg_a_found","Array Redundancy Check")
        
        end_time = datetime.datetime.utcnow() - datetime.timedelta(days=1)#datetime.datetime.utcnow()
        start_time = datetime.datetime.utcnow() - datetime.timedelta(days=2)#end_time - datetime.timedelta(minutes=1)
        finder = self.Finder()
        finder.filter_acqs((data_index.ArchiveInst.name == 'chimestack'))
        finder.accept_all_global_flags()
        finder.set_time_range(start_time, end_time)
        results_list = finder.get_results()
        fhlist = h5py.File(results_list[0][0][0], 'r')

        rstack = fhlist['reverse_map/stack'][:]

        #all prods to search
        self.ssel_search = rstack[:]['stack']

        ssel = rstack[:]['stack']

        #list of stack indexes for each redundant set of prods
        self.u_ssel = np.unique(ssel)
              
    def run(self):
        """Main task stage: Check all redundant baselines for non-redundant outliers.
        """
        finder = self.Finder()
        finder.filter_acqs((data_index.ArchiveInst.name == 'chimeN2'))
        finder.accept_all_global_flags()
        end_time = datetime.datetime.utcnow()
        start_time = end_time - datetime.timedelta(hours=30)
        #sunrise = time.unix_to_datetime(eph.solar_rising(start_time, end_time))[-1]
        #sunset = time.unix_to_datetime(eph.solar_setting(start_time, end_time))[0]
        finder.set_time_range(start_time, end_time)
        finder.include_transits(ephemeris.CygA, time_delta=800.)
        results_list = finder.get_results()

        #check redundancy using yesterday's date
        transit_dt = (datetime.datetime.utcnow() - datetime.timedelta(hours=30)).strftime("%Y-%m-%d")
        found = 0
        if len(results_list) > 0:
            fhlist_chimeN2 = h5py.File(results_list[0][0][0], 'r')

            times = fhlist_chimeN2['index_map/time']['ctime']
            t_wall_dt = np.array([datetime.datetime.fromtimestamp(a) for a in times])
            t_wall_del_idx = np.where(t_wall_dt == datetime.datetime(1970,1,1,0,0))[0]

            if len(t_wall_del_idx) > 0: 
                t_wall_dt = np.delete(t_wall_dt, t_wall_del_idx, 0)

            #get the cyga transit for this timestream
            bnd = 28
            src_idx, tran_peak_val, full_transit = self.cygA_transit_keys_grp(t_wall_dt, bnd)

            #confirm yesterdays cyga transit is in this file
            for y in range(len(src_idx)):
#                 if tran_peak_val[y].strftime("%Y-%m-%d") == transit_dt and full_transit == True:
                if full_transit == True:
                    transit_idx = src_idx[y]
                    found = 1
             
        if found == 0:
            self.logger.warn('Did not find any data in the archive for CygA on ' + transit_dt)
            self.cyg_a_found.set(0)
            return 
        elif found == 1:
            self.logger.info('Cyga transit found. Check array redundancy using cyga transit on ' + transit_dt)

            build_prod_freq_flag = []
            
            vis = fhlist_chimeN2['vis'][:, :, transit_idx]
                             
            freq_N2_lst = fhlist_chimeN2['index_map/freq'][:]

            #loop through all the stack indexes and check every redundant baseline in each stack
            for f_idx in range(vis.shape[0]):
                build_prod_set = []
                redun_prods_idx_ar = []
                for st_idx in self.u_ssel:
                    redun_prods = np.where(self.ssel_search == st_idx)[0]
                    redun_prods_idx_ar.append(redun_prods)

                    build_ar = self.redun_thrshld_flagger_cygA(vis[f_idx,redun_prods])

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

            with h5py.File(os.path.join(self.write_dir, 'redundancy_check_' + transit_dt + '.h5'), 'w') as f:
                f.create_dataset('redund_prod_flags', data=build_prod_freq_flag, dtype=int)
                f.create_dataset('axis/stack_idx', data=u_ssel, dtype=int)
                f.create_dataset('axis/redun_prod_idx', data=redun_prods_idx_ar, dtype=int)
                f.create_dataset('axis/freq', data=freq_N2_lst, dtype=float)
                f.close()
        
                self.logger.info('Redundancy flags written for CygA transit on ' + str(transit_dt))
                self.cyg_a_found.set(1)

    def normalize_complex(self,x):
        max_amp = np.nanmax(np.abs(x))
        amp = 1/(max_amp**0.5)
        norm = []
        for i in range(len(x)):
            norm.append(amp*x[i])
            
        return np.asarray(norm)
    
    def mad(self,data, axis=None):
        return np.nanmedian(np.abs(data - np.nanmedian(data, axis)), axis)    
    
    def redun_thrshld_flagger_cygA(self,vis, time_thrsld = 15, res_mad_thrsld = 5): #Vis should be [[redun_prods],[times]]
        rd_tran_ar = []

        for prod_idx in range(vis.shape[0]): #prod, time
            rd_tran = self.normalize_complex(np.abs(vis[prod_idx]))
            rd_tran_ar.append(rd_tran)
        rd_tran_ar = np.asarray(rd_tran_ar)

        mad_vis = []
        vis_med = []
        for t in range(rd_tran_ar.shape[1]):
            mad_vis.append(self.mad(np.abs(rd_tran_ar[:,t])))
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

    def cygA_transit_keys_grp(self, dates_ar, ln ):
        src_tran = []
        src_ar = []
        bnd = ln#26 #this*2-2 will give the index width of the transit. eg 40*2-2=78 
        full = False
        
        e = np.array([datetime.datetime.fromtimestamp(a) for a in ephemeris.transit_times(ephemeris.CygA, dates_ar[0], dates_ar[-1])])
        if len(e) > 0:
            dt_idx = np.where(dates_ar >= e[0])[0]
            if len(dt_idx) == 0:
                dt_idx = np.where(dates_ar < e[0])[0]
            merged_ar = np.sort(np.concatenate((dates_ar[dt_idx[0]:(dt_idx[0]+bnd)], dates_ar[(dt_idx[0]-1):(dt_idx[0]-bnd):-1]), axis=0))        
            for y in range(len(merged_ar)):
                src_tran.append(np.where(dates_ar == merged_ar[y])[0][0])
            src_tran = np.asarray(src_tran)
            if len(src_tran) == bnd*2-1:
                full = True
            else:
                full = False
            src_ar.append(src_tran)
            src_tran = []
            for x in range(1, e.shape[0]):
                dt_idx = np.where(dates_ar >= e[x])[0]
                if len(dt_idx) == 0:
                    dt_idx = np.where(dates_ar < e[x])[0]
                merged_ar = np.sort(np.concatenate((dates_ar[dt_idx[0]:(dt_idx[0]+bnd)], dates_ar[(dt_idx[0]-1):(dt_idx[0]-bnd):-1]), axis=0))
                for y in range(len(merged_ar)):
                    src_tran.append(np.where(dates_ar == merged_ar[y])[0][0])
                src_tran = np.asarray(src_tran)
                src_ar.append(src_tran)
                src_tran = []
            src_ar = np.asarray(src_ar)
        return src_ar, e, full
