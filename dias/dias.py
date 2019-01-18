import logging
from caput import config
import yaml
import os

# Set the module logger.
logger = logging.getLogger(__name__)

def run_tasks():
    pass

class config_global(config.Reader):
    task_config_dir = config.Property(default=os.getcwd(), proptype=str,
                                      key='task_config_dir')

class config_task(config.Reader):
    analyzer = config.Property(proptype=str, key='analyzer')
    
def load_analyzers():
    global_config = config_global()
    global_file = open("dias.conf","r")
    global_config.read_config(yaml.load(global_file))
    global_file.close()
    config_dir = global_config.task_config_dir
    for config_file in os.listdir(config_dir):
        if config_file.endswith("ad.conf"):
            task_file = open(os.path.join(config_dir, config_file),"r")
            task_config = config_task()
            task_config.read_config(yaml.load(task_file))
            print(task_config.analyzer)

if __name__=="__main__":
    load_analyzers()