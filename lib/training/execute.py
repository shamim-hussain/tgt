import os
import numpy as np
import torch
import random
import importlib

from . import config as cfg

MASTER_ADDR_KEY = 'master_addr'
MASTER_PORT_KEY = 'master_port'
MASTER_ADDR_DEFAULT = 'localhost'
MASTER_PORT_DEFAULT = '12356'

SCHEME_LIB = 'lib.training_schemes'
SCHEME_DEFAULT = ''
SCHEME_CLS = 'SCHEME'

KEY_SCHEME = 'scheme'
KEY_DISTRIBUTED = 'distributed'

KEY_DDP_NODE = 'ddp_node'
KEY_DDP_NUM_NODES = 'ddp_nnodes'
KEY_MASTER_NODE = 'ddp_master'

COMMANDS = {
    'train': 'execute_training',
    'predict': 'make_predictions',
    'evaluate': 'do_evaluations',
}

DEFAULT_CONFIG_FILE = 'config.yaml'

def get_configs_from_args(args):
    config = {}
    if len(args)>1:
        args = args[1:].copy()
        
        if os.path.isfile(args[0]):
            config.update(cfg.read_config_from_file(args[0]))
            args = args[1:]
        elif os.path.isdir(args[0]):
            config_path = os.path.join(args[0], DEFAULT_CONFIG_FILE)
            config.update(cfg.read_config_from_file(config_path))
            args = args[1:]
        
        if len(args)>0:
            additional_configs = cfg.safe_load_str('\n'.join(args))
            config.update(additional_configs)
        
        if (not KEY_SCHEME in config) and (not SCHEME_DEFAULT):
            raise ValueError(f'"{KEY_SCHEME}" is not in config!')
    return config

def import_scheme(scheme_name):
    full_name = f'{SCHEME_LIB}.{scheme_name}.{SCHEME_CLS}'
    module_name, object_name = full_name.rsplit('.', 1)
    imported_module = importlib.import_module(module_name)
    return getattr(imported_module, object_name)


def execute_no_distributed(command, scheme_class, config):
    scheme = scheme_class(config=config, command=command)
    getattr(scheme, COMMANDS[command])()


def _worker_fn(rank, world_size, command, scheme_class, config,
               node, ddp_num_nodes):
    torch.cuda.set_device(rank)
    
    if ddp_num_nodes is not None:
        rank = rank+node*world_size
        world_size = ddp_num_nodes*world_size
        print(f'MULTIPROC: Trying to initialize rank: {rank}/{world_size}')
        
    torch.distributed.init_process_group(backend="nccl",
                                         rank=rank,
                                         world_size=world_size)
    
    print(f'MULTIPROC: Initiated rank: {rank}', flush=True)
    try:
        scheme = scheme_class(config = config, 
                              ddp_rank = rank,
                              ddp_world_size = world_size,
                              command = command)
        getattr(scheme, COMMANDS[command])()
    finally:
        torch.distributed.destroy_process_group()
        print(f'Rank {rank}:Destroyed process!', flush=True)


def execute_by_spawn(command, scheme_class, config):
    world_size = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = config.pop(MASTER_ADDR_KEY, MASTER_ADDR_DEFAULT)
    os.environ['MASTER_PORT'] = config.pop(MASTER_PORT_KEY, MASTER_PORT_DEFAULT)
    if KEY_DDP_NUM_NODES in config:
        ddp_node = config.pop(KEY_DDP_NODE)
        ddp_num_nodes = config.pop(KEY_DDP_NUM_NODES)
        os.environ['MASTER_ADDR'] = config.pop(KEY_MASTER_NODE)
        print(f'Communicating via: {os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}')
    else:
        ddp_node = None
        ddp_num_nodes = None
    torch.multiprocessing.spawn(fn = _worker_fn,
                                args = (world_size,command,scheme_class,config,
                                        ddp_node,ddp_num_nodes),
                                nprocs = world_size,
                                join = True)


def _detect_torchrun():
    return ('LOCAL_RANK' in os.environ and\
            'RANK' in os.environ and\
            'WORLD_SIZE' in os.environ and\
            'MASTER_ADDR' in os.environ and\
            'MASTER_PORT' in os.environ)


def execute_by_torchrun(command, scheme_class, config):
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    torch.cuda.set_device(local_rank)
    print(f'TORCHRUN: Trying to initialize rank: {rank}/{world_size}', flush=True)
    torch.distributed.init_process_group(backend="nccl",
                                         rank=rank,
                                         world_size=world_size)
    print(f'TORCHRUN: Initiated rank: {rank}', flush=True)
    
    scheme = scheme_class(config = config, 
                          ddp_rank = rank,
                          ddp_world_size = world_size,
                          command = command)
    getattr(scheme, COMMANDS[command])()
    

def execute(command, config):
    scheme_class = import_scheme(config.get(KEY_SCHEME, SCHEME_DEFAULT))
    
    if config.get(KEY_DISTRIBUTED, False):
        if _detect_torchrun():
            execute_by_torchrun(command, scheme_class, config)
        else:
            execute_by_spawn(command, scheme_class, config)
    else:
        execute_no_distributed(command, scheme_class, config)
        
