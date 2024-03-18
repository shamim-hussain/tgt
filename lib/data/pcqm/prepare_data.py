
import requests
import tarfile
import os
from tqdm import tqdm
import argparse

from ogb.lsc import PCQM4Mv2Dataset
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

from sklearn.model_selection import train_test_split



# URL of the file to download
URL = 'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz'
# Path to save downloaded file
TAR_FILE_NAME = 'pcqm4m-v2-train.sdf.tar.gz'
SDF_FILE_NAME = 'pcqm4m-v2-train.sdf'
# Root directory to save the files
ROOT = 'data/PCQM'



def download_sdf_tar(download_path: str) -> str:
    """
    Download the SDF tar.gz file from the URL.

    Args:
        download_path (str): The path to save the downloaded file.

    Returns:
        str: The path to the downloaded tar file.
    """
    tar_path = os.path.join(download_path, TAR_FILE_NAME)

    # Download the file with progress bar
    print(f"Downloading SDF file from {URL}...")
    if os.path.exists(tar_path):
        print(f"File already exists at {tar_path}. Skipping download.")
        return tar_path

    response = requests.get(URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB

    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    try:
        with open(tar_path, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
    finally:
        progress_bar.close()

    print("Download complete.")
    return tar_path


def extract_tar(tar_path: str, output_dir: str) -> str:
    """
    Extract the SDF file from the downloaded tar.gz file.

    Args:
        tar_path (str): The path to the tar file.
        output_dir (str): The directory to extract the SDF file.

    Returns:
        str: The path to the extracted SDF file.
    """
    sdf_path = os.path.join(output_dir, SDF_FILE_NAME)

    # Extract the downloaded tar.gz file
    print("Extracting tar file...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extract(SDF_FILE_NAME, path=output_dir, filter='data')
    print("Extraction complete.")

    return sdf_path


def download_and_extract_sdf(root: str, keep_tar: bool = False) -> str:
    """
    Download and extract the SDF file.

    Args:
        root (str): The root directory to save the files.
        keep_tar (bool, optional): Whether to keep the downloaded tar file. Defaults to False.

    Returns:
        str: The path to the extracted SDF file.
    """
    tar_path = os.path.join(root, TAR_FILE_NAME)
    sdf_path = os.path.join(root, SDF_FILE_NAME)

    if not os.path.exists(sdf_path):
        if not os.path.exists(tar_path):
            download_path = download_sdf_tar(download_path=root)
        else:
            print("Using existing tar file...")
            download_path = tar_path

        sdf_path = extract_tar(download_path, output_dir=root)
    else:
        print("Using existing SDF file...")

    if not keep_tar and os.path.exists(tar_path):
        os.remove(tar_path)
        print("Cleaned up tar file.")

    return sdf_path


def mol2graph(mol: Chem.Mol) -> tuple:
    """
    Convert an RDKit molecule object to a graph representation.

    Args:
        mol (Chem.Mol): The RDKit molecule object.

    Returns:
        tuple: A tuple containing:
            - num_nodes (np.int16): The number of nodes in the graph.
            - edges (np.ndarray): The edges of the graph represented as an array of shape (2, num_edges).
            - node_features (np.ndarray): The node features represented as an array of shape (num_nodes, num_node_features).
            - edge_features (np.ndarray): The edge features represented as an array of shape (num_edges, num_edge_features).
    """
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    num_nodes = np.array(len(x), dtype=np.int16)
    edges = edge_index.T.astype(np.int16)
    edge_features = edge_attr.astype(np.int16)
    node_features = x.astype(np.int16)

    return num_nodes, edges, node_features, edge_features


def process_sdf(sdf_file: str, remove_sdf: bool = True) -> dict:
    """
    Process the SDF file and extract relevant information.

    Args:
        sdf_file (str): The path to the SDF file.
        remove_sdf (bool, optional): Whether to remove the SDF file after processing. Defaults to True.

    Returns:
        dict: A dictionary containing the processed data.
    """
    print('Opening SDF ...')
    suppl = Chem.SDMolSupplier(sdf_file)

    records = {
        'idx': [],
        'num_nodes': [],
        'edges': [],
        'node_features': [],
        'edge_features': [],
        'dft_coords': []
    }

    print('Processing SDF ...')
    for idx, mol in enumerate(tqdm(suppl)):
        mol = Chem.RemoveAllHs(mol)
        num_nodes, edges, node_features, edge_features = mol2graph(mol)
        dft_coords = mol.GetConformer().GetPositions().astype('float32')

        records['idx'].append(idx)
        records['num_nodes'].append(num_nodes)
        records['edges'].append(edges.ravel())
        records['node_features'].append(node_features.ravel())
        records['edge_features'].append(edge_features.ravel())
        records['dft_coords'].append(dft_coords.ravel())

    if remove_sdf:
        os.remove(sdf_file)
        print('Cleaned up SDF file')
    return records


def process_dataset(sdf_file: str, root: str, remove_sdf: bool = False) -> tuple:
    """
    Process the PCQM4Mv2 dataset.

    Args:
        sdf_file (str): The path to the SDF file.
        root (str): The root directory of the dataset.
        remove_sdf (bool, optional): Whether to remove the SDF file after processing. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - records (dict): A dictionary containing the processed data.
            - splits (dict): A dictionary containing the dataset splits.
    """
    print('Extracting training data from SDF before other splits')
    records = process_sdf(sdf_file, remove_sdf=remove_sdf)

    print('Loading PCQM4Mv2 dataset')
    dataset = PCQM4Mv2Dataset(root=root, only_smiles=True)
    train_idx, val_idx, test_idx = (dataset.get_idx_split()[k]
                                    for k in ['train', 'valid', 'test-dev'])
    assert np.all(train_idx == records['idx'])

    print('Adding training targets from PCQM4Mv2 dataset')
    records['target'] = []
    for idx in tqdm(train_idx):
        _, target = dataset[idx]
        target = np.array(target, dtype=np.float32)
        records['target'].append(target)

    def process_split(split: str, split_idx: np.ndarray):
        print(f'Processing {split} data from PCQM4Mv2 dataset')
        for idx in tqdm(split_idx):
            smiles, target = dataset[idx]
            target = np.array(target, dtype=np.float32)

            mol = Chem.MolFromSmiles(smiles)
            num_nodes, edges, node_features, edge_features = mol2graph(mol)

            records['idx'].append(idx)
            records['num_nodes'].append(num_nodes)
            records['edges'].append(edges.ravel())
            records['node_features'].append(node_features.ravel())
            records['edge_features'].append(edge_features.ravel())
            records['target'].append(target)

    process_split('valid', val_idx)
    process_split('test', test_idx)

    records['idx'] = np.stack(records['idx'])
    records['num_nodes'] = np.stack(records['num_nodes'])
    records['target'] = np.stack(records['target'])

    print('Creating train-3d and valid-3d splits')
    train_3d_idx, val_3d_idx = train_test_split(train_idx, test_size=78606,
                                                random_state=777777)
    train_3d_idx.sort()
    val_3d_idx.sort()

    splits = dataset.get_idx_split().copy()
    splits['train-3d'] = train_3d_idx
    splits['valid-3d'] = val_3d_idx

    return records, splits


def save_records(records: dict, save_path: str):
    """
    Save the processed records as a Parquet file.

    Args:
        records (dict): The dictionary containing the processed data.
        save_path (str): The directory to save the Parquet file.
    """
    save_path = os.path.join(save_path, 'records.parquet')
    print('Saving records.parquet')
    records_dict = {
        'idx': records['idx'],
        'num_nodes': records['num_nodes'],
        'edges': records['edges'],
        'node_features': records['node_features'],
        'edge_features': records['edge_features'],
        'target': records['target'],
    }

    records_table = pa.table(records_dict)
    pq.write_table(records_table, save_path)


def save_splits(splits: dict, save_path: str):
    """
    Save the dataset splits as an NPZ file.

    Args:
        splits (dict): The dictionary containing the dataset splits.
        save_path (str): The directory to save the NPZ file.
    """
    save_path = os.path.join(save_path, 'splits.npz')
    print('Saving splits.npz')
    np.savez(save_path, **splits)


def save_dft_coords(records: dict, save_path: str):
    """
    Save the DFT coordinates as a Parquet file.

    Args:
        records (dict): The dictionary containing the processed data.
        save_path (str): The directory to save the Parquet file.
    """
    save_path = os.path.join(save_path, 'dft_coords.parquet')
    print('Saving dft_coords.npz')
    dft_coords_dict = {
        'idx': records['idx'][:len(records['dft_coords'])],
        'dft_coords': records['dft_coords'],
    }
    dft_coords_table = pa.table(dft_coords_dict)
    pq.write_table(dft_coords_table, save_path)


def main():
    parser = argparse.ArgumentParser(description='Process PCQM4Mv2 dataset')
    parser.add_argument('--download_dir', type=str, default=ROOT,
                        help='Directory to download and extract files')
    parser.add_argument('--keep_tar', action='store_true',
                        help='Keep the downloaded tar file after extraction')
    parser.add_argument('--remove_sdf', action='store_true',
                        help='Keep the extracted SDF file after processing')

    args = parser.parse_args()

    os.makedirs(args.download_dir, exist_ok=True)

    sdf_file = download_and_extract_sdf(root=args.download_dir, keep_tar=args.keep_tar)

    records, splits = process_dataset(sdf_file, root=args.download_dir,
                                      remove_sdf=args.remove_sdf)

    save_dft_coords(records, args.download_dir)
    save_records(records, args.download_dir)
    save_splits(splits, args.download_dir)




if __name__ == '__main__':
    main()

