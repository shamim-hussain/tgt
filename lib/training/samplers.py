import torch
import numpy as np
from torch.utils.data import Sampler

class DistributedTestDataSampler(Sampler):
    def __init__(self, data_source, batch_size, rank, world_size):
        data_len = len(data_source)
        all_indices = np.arange(data_len, dtype=int)
        split_indices = np.array_split(all_indices, world_size)
        
        num_batches = (len(split_indices[0]) + batch_size -1) // batch_size
        self.batch_indices = [i.tolist() for i in np.array_split(split_indices[rank],
                                                                 num_batches)]
    
    def __iter__(self):
        return iter(self.batch_indices)
    
    def __len__(self):
        return len(self.batch_indices)


class DistributedTrainDataSampler(Sampler):
    @staticmethod
    def get_slice4len(length, rank=None, world_size=None, return_min_max=False):
        if rank is None:
            rank = torch.distributed.get_rank()
        if world_size is None:
            world_size = torch.distributed.get_world_size()
        
        min_rank_len, num_max_ranks = divmod(length, world_size)
        max_rank_len = min_rank_len + int(bool(num_max_ranks))
        start = rank * min_rank_len + min(num_max_ranks, rank)
        end = start + (max_rank_len if rank < num_max_ranks else min_rank_len)
        
        if return_min_max:
            return start, end, min_rank_len, max_rank_len
        else:
            return start, end
        
    def __init__(self, data_source, shuffle=True, drop_last=False, rank=None, world_size=None):
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        data_len = len(data_source)
        start, end, min_rank_len, max_rank_len = self.get_slice4len(data_len, rank,
                                                                    world_size, True)
        assert min_rank_len > 0, 'Not enough data for all ranks'
        self.index_start = start
        self.index_len = end - start
        
        if drop_last:
            self.each_rank_len = min_rank_len
        else:
            self.each_rank_len = max_rank_len
            self.pad_len = self.each_rank_len - self.index_len
    
    def __iter__(self):
        indices = self.index_start + np.random.permutation(self.index_len)
        if self.drop_last:
            indices = indices[:self.each_rank_len]
        else:
            indices = np.pad(indices, (0, self.pad_len), 'wrap')
        assert len(indices) == self.each_rank_len
        return iter(indices)
    
    def __len__(self):
        return self.each_rank_len
    
    def set_epoch(self, epoch):
        pass

