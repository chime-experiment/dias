from dias import chime_analyzer
from datetime import datetime
from caput import config
from dias.utils.time_strings import str2timedelta, datetime2str
import os


class RawAdcAnalyzer(chime_analyzer.CHIMEAnalyzer):
    """Dias task for the raw adc flagging broker.  This task will copy data products 
    output from the flagging broker to a location where they can be access by theremin.
    This subclass of dias.analyzer.Analyzer describes the new analyzer.
    """
    
    # Config parameter for this anlyzer can be specified by assigning class
    # attributes a caput.config.Property.
    # Currently, flagging broker writes data products to recv2 which is also
    # currently mounted on coconut.
    # In the long run, it might be better for the broker to write directly to gong.
    acq_dir = config.Property(proptype=str, default='/mnt/recv2/rawflag')
    write_dir = config.Property(proptype=str, default='/home/dwulf/test_dias')
    state_dir = config.Property(proptype=str, default='/home/dwulf/test_dias')

    def setup(self):
        """Setup stage: this is called when dias starts up.
        """
        self.logger.info('Starting {}'.format(self.name))
        
        state_file = open(os.path.join(self.state_dir,self.name+'.state'),'r')
        self.last_dir_copied, self.last_file_copied = state_file.read().splitlines()
        self.logger.debug('State Info Loaded From {}'.format(os.path.join(self.state_dir,self.name+'.state')))
        
        self.logger.debug('Last Directory Copied: {}'.format(os.path.join(self.acq_dir,self.last_dir_copied)))
        self.logger.debug('Last File Copied From Last Directory: {}'.format(self.last_file_copied))

        # Add a task metric that counts how often this task ran.
        self.run_counter = self.add_task_metric("runs",
                                                "Number of times the task ran.",
                                                unit="total")
        self.file_counter = self.add_task_metric("files",
                                                 "Number of files copied.",
                                                 unit="total")

    def copy_current_dir(self):
        """Copy all files in curr_path that were written after the last_file_copied
        """
        curr_path = os.path.join(self.acq_dir,self.last_dir_copied)
        for file in os.listdir(curr_path):
            if file>self.last_file_copied:
                self.logger.debug('Copying {} to {}'.format(os.path.join(curr_path,file),
                                                             os.path.join(self.write_dir,self.last_dir_copied,'')))
                os.system("rsync -az {} {}".format(os.path.join(curr_path,file),
                                                   os.path.join(self.write_dir,self.last_dir_copied,'')))
                self.last_file_copied = file
                self.file_counter.inc()
    
                      
    def next_dir(self):
        """Check whether additional directories need to be copied and return the next one
        """
        next_directory = None
        subdir = [dd for dd in os.listdir(self.acq_dir) 
                  if os.path.isdir(os.path.join(self.write_dir,dd)) 
                  and dd > self.last_dir_copied
                  and not dd.startswith(".")].sort()
        if subdir:
            next_directory = subdir[0]          
        return next_directory              

    
    def run(self):
        """Main task stage: analyze data from the last period.
        """
        self.copy_current_dir()
        next_directory = self.next_dir()             
        while next_directory is not None:
            self.logger.debug('Moving on to {}.'.format(os.path.join(self.acq_dir,next_directory)))
            self.last_dir_copied = next_directory
            self.last_file_copied = ""
            self.copy_current_dir()          
            next_directory = self.next_dir()
            
        self.run_counter.inc()

    def finish(self):
        """Final stage: this is called when dias shuts down."""
        state_file = open(os.path.join(self.state_dir,self.name+'.state'),'w')
        state_file.write(self.last_dir_copied+'\n')
        state_file.write(self.last_file_copied)
        state_file.close()
        self.logger.debug('State information written to {}'.format(os.path.join(self.state_dir,self.name+'.state')))
        self.logger.info('Shutting down.')
        
