from dias import analyzer
from datetime import datetime
from caput import config
from dias.utils.time_strings import str2timedelta, datetime2str

import os
import sys
import glob
import argparse
import time
import datetime
import inspect
import subprocess
import gc

sys.path.insert(0,'/home/saurabhs24/ch_acq')
sys.path.insert(0,'/home/saurabhs24/ch_util')
sys.path.insert(0,'/home/saurabhs24/ch_scripts_fringestop')

import h5py
import numpy as np
import scipy.constants

from collections import Counter
from collections import defaultdict

import log

from ch_util import andata
from ch_util import tools
from ch_util import ephemeris


class SensitivityAnalyzer(analyzer.Analyzer):
    """Sample Analyzer for dias.
    This subclass of dias.analyzer.Analyzer describes the new analyzer.
    """
    # Config parameter for this anlyzer can be specified by assigning class
    # attributes a caput.config.Property
    correlator      = config.Property(proptype=str,   default='chime')
    output_dir      = config.Property(proptype=str,   default='/home/saurabhs24/telescope_sensitivity')
    output_suffix   = config.Property(proptype=str,   default='sensitivity')
    acq_dir         = config.Property(proptype=str,   default='/mnt/gong/archive')
    acq_suffix      = config.Property(proptype=str,   default='stack_corr')
    start_time      = config.Property(proptype=int,   default=1545436800)
    stop_time       = config.Property(proptype=int,   default=1545437700)
    nfreq_per_block = config.Property(proptype=int,   default=16)
    rot             = config.Property(proptype=float, default=-0.088 )
    include_auto    = config.Property(proptype=bool,  default=False)
    include_intracyl= config.Property(proptype=bool,  default=False)
    include_crosspol= config.Property(proptype=bool,  default=False)
    sep_cyl         = config.Property(proptype=bool,  default=False)

    def setup(self):
        """Setup stage: this is called when dias starts up."""
        self.logger.info('Started computation of telescope sensitvity...')

    def run(self):
        """Main task stage: analyze data from the last period.
        """
        # Setup logging

        if self.start_time and self.stop_time:
            start_search = self.start_time
            stop_search  = self.stop_time

        else:
            start_search = 0
            stop_search  = 0
        
        # Find all calibration files
        all_files = sorted(glob.glob(os.path.join(self.acq_dir,'*' + self.correlator + self.acq_suffix, '*.h5')))
        if not all_files:
            return

        # Remove files whose last modified time is before the time of the most recent update
        all_files = [ff for ff in all_files if (os.path.getmtime(ff) > start_search) and 
                                               (os.path.getmtime(ff) < stop_search)]
        #print start_search, stop_search, all_files
        if not all_files:
            self.logger.error("No files match your search criteria. Exiting...")
            return

        timestamp0 = start_search
            
        for files in all_files:
        
            t0 = time.time()
            
            # Look up inputmap
            inputmap   = tools.get_correlator_inputs(ephemeris.unix_to_datetime(timestamp0),
                                                   correlator=self.correlator)
            # Load index map and reverse map
            data       = andata.CorrData.from_acq_h5(files, 
                                               datasets=['reverse_map', 'flags/inputs'],
                                               apply_gain=False, renormalize=False)
            
            self.logger.info("Took %0.1f seconds to load index and reverse map" % (time.time() - t0))
            
            # Determine axes
            nfreq     = data.nfreq
            nblock    = int(np.ceil(nfreq / float(self.nfreq_per_block)))
            
            timestamp = data.time
            ntime     = timestamp.size

            # Get baselines
            prod, prodmap, dist, conj, cyl, scale = get_baselines(data.index_map, inputmap)

            # Determine groups
            polstr = np.array(sorted(prod.keys()))
            npol   = polstr.size
            
            t0 = time.time()
            
            # Calculate counts
            cnt = np.zeros((data.index_map['stack'].size, ntime), dtype=np.float32)
            
            if np.any(data.flags['inputs'][:]):
                for pp, ss in zip(data.index_map['prod'][:], data.reverse_map['stack']['stack'][:]):
                    cnt[ss, :] += (data.flags['inputs'][pp[0], :] * data.flags['inputs'][pp[1], :])
            else:
                for ss, val in Counter(data.reverse_map['stack']['stack'][:]).iteritems():
                    cnt[ss, :] = val
            
            self.logger.info("Took %0.1f seconds to get the counts" % (time.time() - t0))

            # Initialize arrays
            var 	= np.zeros((nfreq, npol, ntime), dtype=np.float32)
            counter = np.zeros((nfreq, npol, ntime), dtype=np.float32)

            # Loop over frequency blocks
            for block_number in range(nblock):
        
                t0 = time.time()
        
                fstart   = block_number * self.nfreq_per_block
                fstop    = min((block_number + 1) * self.nfreq_per_block, nfreq)
                freq_sel = slice(fstart, fstop)
        
                self.logger.info("Processing block %d (of %d):  %d - %d" % (block_number+1, nblock, fstart, fstop))

                bdata    = andata.CorrData.from_acq_h5(files, freq_sel=freq_sel,
                                                    datasets=['flags/vis_weight'],
                                                    apply_gain=False, renormalize=False)

                self.logger.info("Took %0.1f seconds to load this block" % (time.time() - t0))
                
                t0       = time.time()
                bflag    = (bdata.weight[:] > 0.0).astype(np.float32)
                bvar     = tools.invert_no_zero(bdata.weight[:])
                
                # Loop over polarizations
                for ii, pol in enumerate(polstr):

                    self.logger.info("Processing Pol %s" % pol)

                    pvar   = bvar[:, prod[pol], :]
                    pflag  = bflag[:, prod[pol], :]
                    pcnt   = cnt[np.newaxis, prod[pol], :]
                    pscale = scale[pol][np.newaxis, :, np.newaxis]
                
                    var[freq_sel, ii, :]     += np.sum((pscale * pcnt)**2 * pflag * pvar, axis=1)
                    counter[freq_sel, ii, :] += np.sum(pscale * pcnt * pflag, axis=1)

                self.logger.info("Took %0.1f seconds to process this block." % (time.time() - t0))
                
                del bdata
                gc.collect()
            
            # Normalize
            inv_counter = tools.invert_no_zero(counter)
            var        *= inv_counter**2
            
            # Write to file
            output_file = os.path.join(self.output_dir, "%d_%s.h5" % 
                                      (timestamp0, self.output_suffix))
            self.logger.info("Writing output file...")
            
            with h5py.File(output_file, 'w') as handler:
        
                index_map = handler.create_group('index_map')
                index_map.create_dataset('freq', data=data.index_map['freq'][:])
                index_map.create_dataset('pol', data=polstr)
                index_map.create_dataset('time', data=data.time)
        
                dset = handler.create_dataset('rms', data=np.sqrt(var))
                dset.attrs['axis'] = np.array(['freq', 'pol', 'time'])
                
                dset = handler.create_dataset('count', data=counter.astype(np.int))
                dset.attrs['axis'] = np.array(['freq', 'pol', 'time'])
                
                handler.attrs['instrument_name'] =self.correlator
                handler.attrs['version'] = __version__
                handler.attrs['type'] = __name__
                handler.attrs['collection_server'] = subprocess.check_output(["hostname"]).strip()
                handler.attrs['system_user'] = subprocess.check_output(["id", "-u", "-n"]).strip()
                handler.attrs['git_version_tag'] = subprocess.check_output(["git", "-C", 
                                                                            os.path.dirname(__file__), 
                                                                            "describe", "--always"]).strip()

    ###################################################
    # auxiliary routines
    ###################################################

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

            this_cyl = '%s%s' % (get_cyl(inp_aa.cyl), get_cyl(inp_bb.cyl))
            if sep_cyl:
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
        return chr(cyl_num - 2 + 65)

    def finish(self):
        """Final stage: this is called when dias shuts down."""
        self.logger.info('Shutting down.')
