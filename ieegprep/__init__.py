# define directory as package

#
import sys

# flatten access
from ieegprep.version import __version__
from ieegprep.fileio.IeegDataReader import VALID_FORMAT_EXTENSIONS

# ensure minimum version
if sys.version_info < (3, 8, 0):
    sys.exit("Python 3.8 or later is required.")