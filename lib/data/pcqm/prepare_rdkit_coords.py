
import requests
import tarfile
import os
from tqdm import tqdm
import argparse

from ogb.lsc import PCQM4Mv2Dataset
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from rdkit.Chem import AllChem
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
# Number of conformers to generate
NUM_CONFS = 40


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


def mol2coords(mol: Chem.Mol) -> np.ndarray:
    """
    Generate 3D RDKIT coordinates from the molecule.
    (Adapted from https://github.com/PaddlePaddle/PaddleHelix/blob/dev/apps/pretrained_compound/ChemRL/GEM-2/pahelix/utils/compound_tools.py)
    
    Args:
        mol (Chem.Mol): The molecule object.
    
    Returns:
        np.ndarray: The 3D coordinates of the molecule.
    """
    dtype = np.float32
    try:
        new_mol = Chem.AddHs(mol)
        res = AllChem.EmbedMultipleConfs(new_mol, numConfs=NUM_CONFS, numThreads=0)
        ### MMFF generates multiple conformations
        res = AllChem.MMFFOptimizeMoleculeConfs(new_mol, numThreads=0)
        new_mol = Chem.RemoveHs(new_mol)
        index, _ = min(enumerate(res), key=lambda x: x[1])
        conf = new_mol.GetConformer(id=index)
    except Exception:
        new_mol = mol
        AllChem.Compute2DCoords(new_mol)
        conf = new_mol.GetConformer()
    
    if new_mol.GetAtomWithIdx(0).GetAtomicNum() == 0:
        return np.zeros((new_mol.GetNumAtoms(), 3), dtype=dtype)
    coords = conf.GetPositions()
    coords = coords[:new_mol.GetNumAtoms()].astype(dtype)
    return coords


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
        'rdkit_coords': []
    }

    print('Processing SDF ...')
    for idx, mol in enumerate(tqdm(suppl)):
        mol = Chem.RemoveAllHs(mol)
        rdkit_coords = mol2coords(mol)

        records['idx'].append(idx)
        records['rdkit_coords'].append(rdkit_coords.ravel())

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
        records (dict): A dictionary containing the processed data.
    """
    print('Processing training split from SDF file ...')
    records = process_sdf(sdf_file, remove_sdf=remove_sdf)

    print('Loading PCQM4Mv2 dataset')
    dataset = PCQM4Mv2Dataset(root=root, only_smiles=True)
    train_idx, val_idx, test_idx = (dataset.get_idx_split()[k]
                                    for k in ['train', 'valid', 'test-dev'])
    assert np.all(train_idx == records['idx'])

    def process_split(split: str, split_idx: np.ndarray):
        print(f'Processing {split} data from PCQM4Mv2 dataset')
        for idx in tqdm(split_idx):
            smiles, _ = dataset[idx]

            mol = Chem.MolFromSmiles(smiles)
            rdkit_coords = mol2coords(mol)

            records['idx'].append(idx)
            records['rdkit_coords'].append(rdkit_coords.ravel())

    process_split('valid', val_idx)
    process_split('test', test_idx)

    records['idx'] = np.stack(records['idx'])

    return records



def save_rdkit_coords(records: dict, save_path: str):
    """
    Save the RDKIT coordinates as a Parquet file.

    Args:
        records (dict): The dictionary containing the processed data.
        save_path (str): The directory to save the Parquet file.
    """
    save_path = os.path.join(save_path, 'rdkit_coords.parquet')
    print('Saving rdkit_coords.npz')
    rdkit_coords_dict = {
        'idx': records['idx'],
        'rdkit_coords': records['rdkit_coords'],
    }
    rdkit_coords_table = pa.table(rdkit_coords_dict)
    pq.write_table(rdkit_coords_table, save_path)


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

    records = process_dataset(sdf_file, root=args.download_dir,
                                      remove_sdf=args.remove_sdf)

    save_rdkit_coords(records, args.download_dir)


if __name__ == '__main__':
    main()

