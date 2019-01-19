from dias.analyzers.sample_analyzer import SampleAnalyzer
from dias.prometheus import Prometheus
import logging

WRITE_DIR = '/home/rick/tmp'
TASK_NAME = 'my_task'
config = {'start_time' : '2018-01-03 17:13:13', 'offset' : '10h',
          'period' : '5m'}


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(message)s')

sample = SampleAnalyzer(TASK_NAME, WRITE_DIR, Prometheus(444))
sample.read_config(config)

# Test my new task without actually using dias.
sample.setup()
sample.run()
sample.run()
sample.finish()
