import numpy as np
from torch.utils.data.dataset import Dataset
import os
from tqdm import tqdm
import pyarrow.dataset as pds
from . import bin_ops as pbins


class Column:
    def __init__(self, path=None, verbose=None):
        self.path = path
        self.verbose = verbose
    
    def load_data(self, records, index_filter):
        raise NotImplementedError
    
    def get_row(self, records, row_id, row):
        raise NotImplementedError
    
    def _set_path_if_none(self, dataset_path):
        if self.path is None:
            self.path = dataset_path
    
    def _set_verbose_if_none(self, verbose):
        if self.verbose is None:
            self.verbose = verbose



class PCQM4Mv2Dataset(Dataset):
    def __init__(self, split, dataset_path,
                 include_node_mask = True,
                 load_data = True,
                 cache_range_fn = None,
                 return_idx = False,
                 cache_rows = True,
                 additional_columns = None,
                 transforms = None,
                 verbose = 0):
        self.split = split
        self.dataset_path = dataset_path
        self.include_node_mask = include_node_mask
        self.cache_range_fn = cache_range_fn
        self.verbose = verbose
        self.return_idx = return_idx
        self.cache_rows = cache_rows
        self.additional_columns = [] if additional_columns is None else additional_columns
        self.transforms = [] if transforms is None else transforms
        
        for col in self.additional_columns:
            assert isinstance(col, Column), 'Columns must be of type Column'
            col._set_path_if_none(self.dataset_path)
            col._set_verbose_if_none(self.verbose)
        
        if load_data:
            self.load_data()
    
    def transform(self, row):
        for func in self.transforms:
            row = func(row)
        return row

    def get_cache_range(self):
        if self.cache_range_fn is None:
            return 0, self.dataset_length
        return self.cache_range_fn(self.dataset_length)
    
    # Loader functions
    def load_indices(self):
        if self.verbose: print('Loading dataset info...')
        split_file = os.path.join(self.dataset_path, 'splits.npz')
        with np.load(split_file) as npz_file:
            if '+' in self.split:
                splits = self.split.split('+')
                indices = np.concatenate([npz_file[split] for split in splits])
            else:
                indices = npz_file[self.split]
        self.dataset_length = len(indices)
        
        range_start, range_end = self.get_cache_range()
        self.index_offset = range_start
        self.indices = indices[range_start:range_end]
        self.index_filter = pds.field('idx').isin(self.indices)

    def load_records(self):
        if self.verbose: print('Loading records...')
        records_file = os.path.join(self.dataset_path, 'records.parquet')
        self.records = pds.dataset(records_file).to_table(filter=self.index_filter)
        assert self.records.num_rows == len(self.indices), \
            f'Loaded {self.records.num_rows} records, expected {len(self.indices)}'
    
    def load_data(self):
        self.load_indices()
        self.load_records()
        
        for col in self.additional_columns:
            self.records = col.load_data(self.records, self.index_filter)
    
    # Dataset functions
    @property
    def max_nodes_index(self):
        num_nodes = self.records['num_nodes'].to_numpy()
        max_nodes_index = np.argmax(num_nodes) + self.index_offset
        return max_nodes_index
    
    def get_row(self, row_id):
        row = {}
        num_nodes = self.records['num_nodes'][row_id].as_py()
        row['num_nodes'] = num_nodes
        row['edges'] = self.records['edges'][row_id].values.to_numpy().reshape(-1, 2)
        row['node_features'] = self.records['node_features'][row_id].values.to_numpy()\
                                                   .reshape(-1, 9)
        row['edge_features'] = self.records['edge_features'][row_id].values.to_numpy()\
                                                    .reshape(-1, 3)
        target = self.records['target'][row_id].as_py()
        row['target'] = target if target is not None else np.nan
        
        if self.return_idx:
            row['idx'] = self.records['idx'][row_id].as_py()
        
        if self.include_node_mask:
            row['node_mask'] = np.ones(num_nodes, dtype=np.uint8)
        
        for col in self.additional_columns:
            row = col.get_row(self.records, row_id, row)
        
        return row
    
    def _cache_rows(self):
        num_rows = self.records.num_rows
        cached_rows = [None] * num_rows
        if self.verbose:
            for row_id in tqdm(range(num_rows), desc='Caching rows...'):
                cached_rows[row_id] = self.get_row(row_id)
        else:
            for row_id in range(num_rows):
                cached_rows[row_id] = self.get_row(row_id)
        self._cached_rows = cached_rows
    
    def __getitem__(self, index):
        row_id = index - self.index_offset
        
        if self.cache_rows:
            try:
                item = self._cached_rows[row_id].copy()
            except AttributeError:
                self._cache_rows()
                item = self._cached_rows[row_id].copy()
        else:
            item = self.get_row(row_id)
        
        # Apply transforms
        if self.transform:
            item = self.transform(item)
        
        return item
    
    def __len__(self):
        return self.dataset_length



class Coords(Column):
    def __init__(self, name, path=None, coords_file=None, verbose=None):
        self.name = name
        self.path = path
        self.coords_file = coords_file
        self.verbose = verbose
    
    def load_data(self, records, index_filter):
        if self.verbose: print(f'Loading {self.name} coordinates...')
        if self.coords_file is None:
            assert self.path is not None,\
                 'Either path or coords_file must be specified'
            coords_file = os.path.join(self.path, f'{self.name}_coords.parquet')
        else:
            coords_file = self.coords_file
        
        coords = pds.dataset(coords_file).to_table(filter=index_filter)
        assert coords['idx'].equals(records['idx']), \
            f'Index mismatch between records and {self.name}_coords'
        records = records.add_column(records.num_columns, f'{self.name}_coords',
                                     coords[f'{self.name}_coords'])
        return records
    
    def get_row(self, records, row_id, row):
        row[f'{self.name}_coords'] = records[f'{self.name}_coords'][row_id]\
                                                  .values.to_numpy().reshape(-1, 3)
        return row


class DistInput(Column):
    def __init__(self, path, verbose=None):
        self.path = path
        self.verbose = verbose
    
    def load_data(self, records, index_filter):
        if self.verbose: print('Loading distance matrix...')
        dist_mat = pds.dataset(self.path).to_table(filter=index_filter)
        dist_mat = dist_mat.sort_by('idx')
        assert dist_mat['idx'].equals(records['idx']), \
            'Index mismatch between records and distance matrix'
        records = records.add_column(records.num_columns, 'dist_input',
                                     dist_mat['dms'])
        return records
    
    def get_row(self, records, row_id, row):
        num_nodes = row['num_nodes']
        row['dist_input'] = records['dist_input'][row_id].values.to_numpy()\
                                                    .reshape(num_nodes, num_nodes)
        return row



class Bins(Column):
    def __init__(self, path, num_bin_samples, verbose=None):
        self.path = path
        self.num_bin_samples = num_bin_samples
        self.verbose = verbose
    
    def load_data(self, records, index_filter):
        if self.verbose: print('Loading bins...')
        bins = pds.dataset(self.path).to_table(filter=index_filter)
        bins = bins.sort_by('idx')
        assert bins['idx'].equals(records['idx']), \
            'Index mismatch between records and bins'
        
        records = records.add_column(records.num_columns, 'dist_bins',
                                     bins['bins'])
        return records
    
    def get_row(self, records, row_id, row):
        num_nodes = row['num_nodes']
        bins = records['dist_bins'][row_id].values.to_numpy()
        bins = bins.reshape(self.num_bin_samples, -1)
        bins = pbins.unpack_bins_multi(bins, num_nodes)
        bins = bins.astype(np.float32)
        row['dist_bins'] = bins
        return row
