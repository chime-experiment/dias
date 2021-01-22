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
        "chimedb.data_index @ git+https://github.com/chime-experiment/chimedb_di.git",
        "chimedb.dataset @ git+https://github.com/chime-experiment/chimedb_dataset.git",
        "caput @ git+https://github.com/radiocosmology/caput.git",
        "draco @ git+https://github.com/radiocosmology/draco.git",
        "ch_util @ git+https://github.com/chime-experiment/ch_util.git",
        "prometheus_client>=0.5.0",
        "bitshuffle @ git+https://github.com/kiyo-masui/bitshuffle.git",
        "skyfield",
        "pyyaml",
        "scipy",
        "msgpack",
        "numpy",
        "h5py",
        "pywavelets",
    ],
    license="GPL v3.0",
    url="http://github.com/chime-experiment/dias",
)
