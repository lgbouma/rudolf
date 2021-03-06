import os, socket
from rudolf import __path__

DATADIR = os.path.join(os.path.dirname(__path__[0]), 'data')
RESULTSDIR = os.path.join(os.path.dirname(__path__[0]), 'results')
TABLEDIR = os.path.join(RESULTSDIR, 'tables')
PHOTDIR = os.path.join(DATADIR, 'phot')
PAPERDIR = os.path.join(os.path.dirname(__path__[0]), 'paper')
DRIVERDIR = os.path.join(os.path.dirname(__path__[0]), 'drivers')

LOCALDIR = os.path.join(os.path.expanduser('~'), 'local', 'rudolf')
if not os.path.exists(LOCALDIR):
    os.mkdir(LOCALDIR)
