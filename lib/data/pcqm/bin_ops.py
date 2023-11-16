import numba as nb
import numpy as np


@nb.njit(nogil=True)
def flat_triu_indices(n):
    indices = np.zeros((n*n-n)//2, dtype=np.int64)
    k = 0
    offset = 0
    for i in range(n):
        for j in range(i+1,n):
            indices[k] = offset+j
            k += 1
        offset += n
    return indices

def pack_bins(bins):
    num_nodes, _ = bins.shape
    bins = bins.reshape(num_nodes*num_nodes)
    indices = flat_triu_indices(num_nodes)
    bins = bins[indices]
    return bins

def unpack_bins(bins, num_nodes):
    matrix = np.zeros((num_nodes*num_nodes),
                      dtype=bins.dtype)
    indices = flat_triu_indices(num_nodes)
    matrix[indices] = bins
    matrix = matrix.reshape(num_nodes, num_nodes)
    return matrix

def pack_bins_multi(bins):
    batch_size, num_nodes, _ = bins.shape
    bins = bins.reshape(batch_size, num_nodes*num_nodes)
    indices = flat_triu_indices(num_nodes)
    bins = bins[:,indices]
    return bins

def unpack_bins_multi(bins, num_nodes):
    batch_size = bins.shape[0]
    matrix = np.zeros((batch_size, num_nodes*num_nodes),
                      dtype=bins.dtype)
    indices = flat_triu_indices(num_nodes)
    matrix[:,indices] = bins
    matrix = matrix.reshape(batch_size, num_nodes, num_nodes)
    return matrix


