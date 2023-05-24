import os

if os.environ.get('MAIA_DISABLE_TF', '').lower() != 'true':
    from .tf import *

from .proto import *
from .file_handling import *
from .pgn_handling import *
from .leela_board import *

from .utils import *
from .proto_utils import *
from .training_helpers import *
from .multiproc import *

__version__ = '0.1.0'
