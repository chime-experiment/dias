#!/usr/bin/python3

from distutils.core import setup
from dias import __version__

setup(
    name         = 'dias',
    version      = __version__,
    author       = "The CHIME Collaboration",
    author_email = "dvw@phas.ubc.ca"
    description  = "CHIME data integrity automation system"
    license      = "GPL v3.0",
    url          = "http://github.com/chime-experiment/dias"
)
