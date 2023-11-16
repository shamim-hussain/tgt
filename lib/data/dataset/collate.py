
import torch
import numpy as np

from .stack_with_pad import stack_with_pad
from collections import defaultdict
from numba.typed import List

def padded_collate(batch):
    batch_data = defaultdict(List)
    for elem in batch:
        for k,v in elem.items():
            batch_data[k].append(v)
    
    out = {k:torch.from_numpy(stack_with_pad(dat)) 
                    for k, dat in batch_data.items()}
    return out
