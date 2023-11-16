import torch
from typing import Any, Callable, Dict, Union
from collections import OrderedDict


def cached_property(func: Callable[[Any], Any]) -> property:
    atrribute_name = f'_{func.__name__}_cached'
    def fget(self):
        try:
            return getattr(self, atrribute_name)
        except AttributeError:
            val = func(self)
            setattr(self, atrribute_name, val)
            return val
    
    def fset(self, val):
        setattr(self, atrribute_name, val)
        
    def fdel(self):
        try:
            delattr(self, atrribute_name)
        except AttributeError:
            pass
    
    return property(fget=fget, fset=fset, fdel=fdel)


def state_dict_to_cpu(state_dict):
    if isinstance(state_dict, OrderedDict):
        return OrderedDict([(k, state_dict_to_cpu(v)) for k, v in state_dict.items()])
    elif isinstance(state_dict, dict):
        return {k: state_dict_to_cpu(v) for k, v in state_dict.items()}
    elif isinstance(state_dict, torch.Tensor):
        return state_dict.cpu()
    else:
        return state_dict
    
