import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import pickle as pkl

############################################################
# -------- Helper Function ---------------------------------
############################################################

def scipy_sparse_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


############################################################
# -------- Planetoid (cora / citeseer / pubmed) ----------
############################################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_planetoid_adj(dataset_name, datapath):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(datapath, f'ind.{dataset_name}.{names[i]}'), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    _, _, _, _, _, _, graph = tuple(objects)
    adj = sp.csr_matrix(graph)
    return adj


############################################################
# -------- Airport (OpenFlights routes.dat) --------------
############################################################

def load_airport_adj(dataset_name, datapath):
    """加载 OpenFlights Airport routes.dat 数据"""
    dir_path = os.path.join(datapath, dataset_name)
    edge_file_path = os.path.join(dir_path, 'routes.dat')
    if not os.path.exists(edge_file_path):
        raise FileNotFoundError(f"Cannot find edge file for airport: {edge_file_path}")

    edges_df = pd.read_csv(
        edge_file_path,
        sep=',',
        header=None,
        usecols=[3, 5], 
        dtype=str
    ).replace(r'\\N', np.nan, regex=True).dropna()

    source_nodes = edges_df.iloc[:, 0].astype(int).add(-1).values
    target_nodes = edges_df.iloc[:, 1].astype(int).add(-1).values
    
    num_nodes = max(source_nodes.max(), target_nodes.max()) + 1
    adj = sp.coo_matrix(
        (np.ones(len(source_nodes)), (source_nodes, target_nodes)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    )
    return adj


############################################################
# -------- Disease LP (disease_lp.edges.csv) -------------
############################################################

import os
import pandas as pd
import numpy as np
import scipy.sparse as sp

def load_disease_lp_adj(dataset_name, datapath):
    dir_path = os.path.join(datapath, dataset_name)
    
    edge_file_path = os.path.join(dir_path, 'disease_lp.edges.csv')
    
    if not os.path.exists(edge_file_path):
        raise FileNotFoundError(f"Cannot find edge file for disease_lp: {edge_file_path}")

    edges_df = pd.read_csv(
        edge_file_path, 
        header=None, 
        names=['source', 'target']
    )
    
    source_nodes = edges_df['source'].values
    target_nodes = edges_df['target'].values

    feats_path = os.path.join(dir_path, 'disease_lp.feats.npz')
    if os.path.exists(feats_path):
        features = sp.load_npz(feats_path)
        num_nodes = features.shape[0]
    else:
        num_nodes = max(source_nodes.max(), target_nodes.max()) + 1

    adj = sp.coo_matrix(
        (np.ones(len(source_nodes)), (source_nodes, target_nodes)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    )
    return adj

############################################################
# -------- Main ------------------------------------
############################################################

def get_adj_only(dataset_name, datapath):
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        print(f"[INFO] Loading '{dataset_name}' with Planetoid loader...")
        adj = load_planetoid_adj(dataset_name, datapath)
    elif dataset_name == 'airport':
        print(f"[INFO] Loading 'airport' dataset...")
        adj = load_airport_adj(dataset_name, datapath)
    elif dataset_name == 'disease_lp':
        print(f"[INFO] Loading 'disease_lp' dataset...")
        adj = load_disease_lp_adj(dataset_name, datapath)
    else:
        raise RuntimeError(f"[ERROR] Unknown dataset: {dataset_name}")

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj.setdiag(0)
    adj.eliminate_zeros()

    return scipy_sparse_to_torch_sparse_tensor(adj)
