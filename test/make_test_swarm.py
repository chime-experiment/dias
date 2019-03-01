#!/usr/bin/python3

# Makes a bunch of tasks using the test analyzer to
# test the scheduler

# Usage:
#
#  python3 make_test_swarm.py TEMPORARY_PATH_FOR_CONFIG NUM_ANALYZERS
#
# TEMPORARY_PATH_FOR_CONFIG must not exist already

import math
import random
import os
import sys

basedir = sys.argv[1]
taskdir = os.path.join(basedir, "tasks")

num_analyzers = int(sys.argv[2])

os.mkdir(basedir)
os.mkdir(taskdir)

# Make dias.conf
with open(os.path.join(basedir, "dias.conf"), "w") as dias_conf:
    dias_conf.write("log_level: DEBUG\n")
    dias_conf.write("task_write_dir: {0}\n".format(
        os.path.join(basedir, "output")))
    dias_conf.write("task_state_dir: {0}\n".format(
        os.path.join(basedir, "output")))
    dias_conf.write("prometheus_client_port: 0\n")

# task filename format: "task###.conf" where ### is a zero padded number
# number of appropriate length
taskname = "task{:0" + str(int(math.log10(num_analyzers))) + "d}.conf"

for i in range(num_analyzers):
    print("Writing " + taskname.format(i), end="\r")
    with open(os.path.join(taskdir, taskname.format(i)), "w") as task_conf:
        task_conf.write("wait_time: {0}\n".format(
            random.randint(1, 300)))
        task_conf.write("period: {0}\n".format(
            random.randint(1, 300)))
        task_conf.write(
            "analyzer: dias.analyzers.test_analyzer.TestAnalyzer\n")
print("\n")
