"""Example dias analyzer.

This is a basic example for how to write an analyzer for dias.
"""


from dias import chime_analyzer
from datetime import datetime
from caput import config
from dias.utils.time_strings import str2timedelta, datetime2str
import os


class RawAdcAnalyzer(chime_analyzer.CHIMEAnalyzer):
    """Dias task for the raw adc flagging broker.
    This subclass of dias.analyzer.Analyzer describes the new analyzer.
    """
    cwd = os.getcwd()
    # Config parameter for this anlyzer can be specified by assigning class
    # attributes a caput.config.Property
    acq_dir = config.Property(proptype='str', default='/mnt/gong/ssiegel/rawflag')

    def setup(self):
        """Setup stage: this is called when dias starts up."""
        self.logger.info('Starting {}'.format(self.name))
        self.logger.debug('State Info Loaded From {}.'.format(self.state_dir))
        
        # Ultimately, these variables will be set from state data
        self.last_dir_copied = "20181007T031648Z_chime_rawflag"
        self.last_file_copied = ""
        self.logger.debug('Last Directory Copied: {}.'.format(os.path.join(self.acq_dir,self.last_dir_copied))
        self.logger.debug('Last File Copied From Last Directory: {}.'.format(self.last_file_copied))
        
        # Verify that desitnation directory exists
        self.make_dir()

        # Add a task metric that counts how often this task ran.
        # It will be exported as dias_task_<task_name>_runs_total.
        #self.run_counter = self.add_task_metric("runs",
        #                                        "Number of times the task ran.",
        #                                        unit="total")

    def copy_current_dir(self):
        """Copy all files in curr_path that were written after the last_file_copied
        """
        curr_path = os.path.join(self.acq_dir,self.last_dir_copied)
        for file in os.listdir(curr_path):
            if file>self.last_file_copied:
                self.logger.debug('Copying {} to {}.'.format(os.path.join(curr_path,file),
                                                             os.path.join(self.write_dir,self.last_dir_copied))
                os.system("cp -p {} {}".format(os.path.join(curr_path,file),
                                               os.path.join(self.write_dir,self.last_dir_copied))
                self.last_file_copied = file
    
    def make_dir(self):
        """If it doesn't already exist, make a directory in write_dir that corresponds to the 
        source directory in acq_dir
        """
        if self.last_dir_copied not in os.listdir(self.write_dir):
            os.system("mkdir {}".format(os.path.join(self.write_dir,self.last_dir_copied))
                      
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
            self.logger.debug('Moving on to {}.'.format(os.path.join(self.acq_dir,next_directory))
            self.last_dir_copied = next_directory
            self.last_file_copied = ""
            self.make_dir()
            self.copy_current_dir()          
            next_directory = self.next_dir()
                          
        # Calculate the start and end of the passed period, which in this
        # example is the time we want to analyze data of.
        #end_time = datetime.now() - self.offset
        #start_time = end_time - self.period
        #self.logger.info('Analyzing data between {} and {}.'
        #                 .format(datetime2str(start_time),
        #                         datetime2str(end_time)))
        #self.logger.info('If I had any data, I would probably throw stuff at '\
        #        '{}.'.format(self.write_dir))
        # Increment (+1).
        #self.run_counter.inc()

    def finish(self):
        """Final stage: this is called when dias shuts down."""
        self.logger.info('Shutting down.')
        #self.logger.debug('I could save some stuff I would like to keep until '
        #                  'next setup in {}.'.format(self.state_dir))
