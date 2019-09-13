#!/usr/bin/python3
"""
dias, a data integrity analysis system.

``dias`` lives on
`GitHub <https://github.com/chime-experiment/dias>`_.
"""


import os
import setuptools
import shutil
import sys
import versioneer

# The path to install the task configuration files into, if any
task_dst = None

# Strip out a --task-conf option if given to the install command
found_install = False
for arg in sys.argv:
    if arg == "install":
        found_install = True
    elif found_install and arg.startswith("--task-conf="):
        task_dst = os.path.join(arg[arg.find("=") + 1 :], "tasks")
        sys.argv.remove(arg)
        break

# Installing task configuration, if requested
if task_dst:
    # Check for the source directory
    task_src = os.path.join(os.path.dirname(sys.argv[0]), "conf", "tasks")
    if not os.path.isdir(task_src):
        raise FileExistsError(
            "Task configruation directory {0} not found".format(task_src)
        )

    # We don't allow installing the task config directory over
    # top of an existing directory, because that would likely
    # result in a mix of old and new configuration
    try:
        os.mkdir(task_dst, 0o755)
    except FileExistsError:
        # Re-raise with an explanation
        raise FileExistsError(
            "Cannot install task configraution: " "{0} already exists.".format(task_dst)
        )

    # Now copy all the task configuration files
    print("Installing task configuration files to {0}".format(task_dst))
    for name in os.listdir(task_src):
        path = os.path.join(task_src, name)
        if name.endswith(".conf") and os.path.isfile(path):
            shutil.copy(path, task_dst)

# Now for the regular setuptools-y stuff
setuptools.setup(
    name="dias",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="The CHIME Collaboration",
    author_email="dvw@phas.ubc.ca",
    description="CHIME data integrity automation system",
    packages=["dias", "dias.analyzers", "dias.utils"],
    scripts=["scripts/dias"],
    install_requires=[
        "chimedb.data_index @ git+https://git@github.com/chime-experiment/chimedb_di.git",
        "caput @ git+https://github.com/radiocosmology/caput.git",
        "ch_util @ git+ssh://git@bitbucket.org/chime/ch_util.git",
        "prometheus_client>=0.5.0",
        "pyyaml",
        "scipy",
        "numpy",
        "h5py",
        "pywavelets",
    ],
    license="GPL v3.0",
    url="http://github.com/chime-experiment/dias",
)
