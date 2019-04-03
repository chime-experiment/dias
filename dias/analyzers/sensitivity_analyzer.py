from dias import CHIMEAnalyzer
from datetime import datetime
import calendar
from caput import config
from dias.utils.string_converter import str2timedelta, datetime2str
from ch_util import data_index

import os
import sys
import glob
import time
import subprocess
import gc

import h5py
import numpy as np
import scipy.constants

from collections import Counter
from collections import defaultdict

from ch_util import andata
from ch_util import tools
from ch_util import ephemeris


class SensitivityAnalyzer(CHIMEAnalyzer):
    """SensitivityAnalyzer.
    This subclass of dias.analyzer.Analyzer describes the analyzer.
    """
    period          = config.Property(proptype=str,  default="24h")
    start_time      = config.Property(proptype=str,   default='2018-01-03 17:13:13')

    correlator      = config.Property(proptype=str,   default='chime')
    output_suffix   = config.Property(proptype=str,   default='sensitivity')
    acq_suffix      = config.Property(proptype=str,   default='stack_corr')

    nfreq_per_block = config.Property(proptype=int,   default=16)
    include_auto    = config.Property(proptype=bool,  default=False)
    include_intracyl= config.Property(proptype=bool,  default=False)
    include_crosspol= config.Property(proptype=bool,  default=False)
    sep_cyl         = config.Property(proptype=bool,  default=False)
    cyl_start_char  = config.Property(proptype=int,  default=65)
    cyl_start_num   = config.Property(proptype=int,  default=2)

    def setup(self):
        """Setup stage: this is called when dias starts up."""
        self.logger.info('Started computation of telescope sensitvity...')

    def run(self):
        """Main task stage: analyze data from the last period.
        """
        stop_time  = datetime.utcnow()  
        start_time = stop_time - str2timedelta(self.period) #Query files from now to period hours back 

        # Find all calibration files
        f = self.Finder()
        f.set_time_range(start_time, stop_time)
        f.accept_all_global_flags()
        f.only_corr()
        f.filter_acqs((data_index.ArchiveInst.name == self.acq_suffix))
        file_list = f.get_results()
        all_files = file_list[0][0]
        
        if not all_files:
            self.logger.error("No files match your search criteria. Exiting...")
            return

        #Get Unix time for the start time for timestamp
        time_tuple      = start_time.timetuple()
        start_time_unix = calendar.timegm(time_tuple)
        timestamp0      = start_time_unix
            
        for files in all_files:
        
            # Look up inputmap
            inputmap   = tools.get_correlator_inputs(ephemeris.unix_to_datetime(timestamp0),
                                                   correlator=self.correlator)
            # Load index map and reverse map
            data       = andata.CorrData.from_acq_h5(files, 
                                               datasets=['reverse_map', 'flags/inputs'],
                                               apply_gain=False, renormalize=False)
            
            # Determine axes
            nfreq     = data.nfreq
            nblock    = int(np.ceil(nfreq / float(self.nfreq_per_block)))
            
            timestamp = data.time
            ntime     = timestamp.size

            # Get baselines
            prod, prodmap, dist, conj, cyl, scale = self.get_baselines(data.index_map, inputmap)

            # Determine groups
            polstr = np.array(sorted(prod.keys()))
            npol   = polstr.size
            
            # Calculate counts
            cnt = np.zeros((data.index_map['stack'].size, ntime), dtype=np.float32)
            
            if np.any(data.flags['inputs'][:]):
                for pp, ss in zip(data.index_map['prod'][:], data.reverse_map['stack']['stack'][:]):
                    cnt[ss, :] += (data.flags['inputs'][pp[0], :] * data.flags['inputs'][pp[1], :])
            else:
                for ss, val in Counter(data.reverse_map['stack']['stack'][:]).iteritems():
                    cnt[ss, :] = val
            

            # Initialize arrays
            var 	= np.zeros((nfreq, npol, ntime), dtype=np.float32)
            counter = np.zeros((nfreq, npol, ntime), dtype=np.float32)

            # Loop over frequency blocks
            for block_number in range(nblock):
        
                fstart   = block_number * self.nfreq_per_block
                fstop    = min((block_number + 1) * self.nfreq_per_block, nfreq)
                freq_sel = slice(fstart, fstop)
        
                self.logger.info("Processing block %d (of %d):  %d - %d" % (block_number+1, nblock, fstart, fstop))

                bdata    = andata.CorrData.from_acq_h5(files, freq_sel=freq_sel,
                                                    datasets=['flags/vis_weight'],
                                                    apply_gain=False, renormalize=False)

                bflag    = (bdata.weight[:] > 0.0).astype(np.float32)
                bvar     = tools.invert_no_zero(bdata.weight[:])
                
                # Loop over polarizations
                for ii, pol in enumerate(polstr):

                    pvar   = bvar[:, prod[pol], :]
                    pflag  = bflag[:, prod[pol], :]
                    pcnt   = cnt[np.newaxis, prod[pol], :]
                    pscale = scale[pol][np.newaxis, :, np.newaxis]
                
                    var[freq_sel, ii, :]     += np.sum((pscale * pcnt)**2 * pflag * pvar, axis=1)
                    counter[freq_sel, ii, :] += np.sum(pscale * pcnt * pflag, axis=1)

                del bdata
                gc.collect()
            
            # Normalize
            inv_counter = tools.invert_no_zero(counter)
            var        *= inv_counter**2

            # Write to file
            output_file = os.path.join(self.write_dir, "%d_%s.h5" % 
                                      (timestamp0, self.output_suffix))
            self.logger.info("Writing output file...")
            
            with h5py.File(output_file, 'w') as handler:
        
                index_map = handler.create_group('index_map')
                index_map.create_dataset('freq', data=data.index_map['freq'][:])
                index_map.create_dataset('pol', data=np.string_(polstr))
                index_map.create_dataset('time', data=data.time)
        
                dset = handler.create_dataset('rms', data=np.sqrt(var))
                dset.attrs['axis'] = np.array(['freq', 'pol', 'time'], dtype='S')
                
                dset = handler.create_dataset('count', data=counter.astype(np.int))
                dset.attrs['axis'] = np.array(['freq', 'pol', 'time'], dtype='S')
                
                handler.attrs['instrument_name'] =self.correlator
                handler.attrs['collection_server'] = subprocess.check_output(["hostname"]).strip()
                handler.attrs['system_user'] = subprocess.check_output(["id", "-u", "-n"]).strip()
                handler.attrs['git_version_tag'] = subprocess.check_output(["git", "-C", 
                                                                            os.path.dirname(__file__), 
                                                                            "describe", "--always"]).strip()

    def get_baselines(self, indexmap, inputmap):
        
        prod = defaultdict(list)
        prodmap = defaultdict(list)
        dist = defaultdict(list)
        conj = defaultdict(list)
        cyl = defaultdict(list)
        scale = defaultdict(list)

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
            
            print("Pol %s:  %d (of %d) prod do not have unit scale factor." % 
                        (key, np.sum(scale[key] != 1.0), scale[key].size))
            
        tools.change_chime_location(default=True)
                
        return prod, prodmap, dist, conj, cyl, scale

    def get_cyl(self, cyl_num):
        return chr(cyl_num - self.cyl_start_num + self.cyl_start_char)

    def finish(self):
        """Final stage: this is called when dias shuts down."""
        self.logger.info('Shutting down.')
