import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.config import LINEAR_BIAS_INIT

# def linear_weights_init(m):
#     if isinstance(m, nn.Linear):
#         stdv = 1. / math.sqrt(m.weight.size(1))
#         m.weight.data.uniform_(-stdv, stdv)
#         if m.bias is not None:
#             m.bias.data.uniform_(-stdv, stdv)

def linear_weights_init(m):
    if isinstance(m, nn.Linear):
        # Xavier initialization breaks symmetry better than simple uniform
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            # Small positive bias helps prevent "dead" neurons early on
            m.bias.data.fill_(LINEAR_BIAS_INIT)

            
def conv_weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
