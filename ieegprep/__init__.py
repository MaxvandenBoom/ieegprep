# define directory as package

# ensure minimum version
import sys
if sys.version_info < (3, 8, 0):
    sys.exit("Python 3.8 or later is required.")

# flatten access
from ieegprep.version import __version__
from ieegprep.bids.data_epoch import load_data_epochs, load_data_epochs_averages
from ieegprep.bids.data_structure import list_bids_datasets
from ieegprep.bids.rereferencing import RerefStruct
from ieegprep.bids.sidecars import load_event_info, load_channel_info, load_ieeg_sidecar
from ieegprep.fileio.IeegDataReader import VALID_FORMAT_EXTENSIONS, IeegDataReader
