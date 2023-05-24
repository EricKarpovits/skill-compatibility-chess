from .utils import *
from ._leela_board import LeelaBoard, max_clock
from ._uci_to_idx import uci_to_idx

idx_to_uci_wn = {v : k for k,v in uci_to_idx[0].items()}
idx_to_uci_wc = {v : k for k,v in uci_to_idx[1].items()}
idx_to_uci_bn = {v : k for k,v in uci_to_idx[2].items()}
idx_to_uci_bc = {v : k for k,v in uci_to_idx[3].items()}
