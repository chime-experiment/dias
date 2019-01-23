#!/usr/bin/python3

from distutils.core import setup
import dias

setup(
    name             = 'dias',
    version          = dias.__version__,
    author           = "The CHIME Collaboration",
    author_email     = "dvw@phas.ubc.ca",
    description      = "CHIME data integrity automation system",
    packages         = ['dias','dias.analyzers', 'dias.utils'],
    scripts          = ['scripts/dias'],
    requires         = ['caput', 'ch_util'],
    license          = "GPL v3.0",
    url              = "http://github.com/chime-experiment/dias"
)
