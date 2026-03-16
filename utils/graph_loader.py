import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import pickle as pkl

from utils.ogb_utils import canonicalize_ogb_link_dataset_name, resolve_ogb_raw_dir

############################################################
# -------- Helper Function ---------------------------------
############################################################

def scipy_sparse_to_torch_sparse_tensor(sparse_mx):
    """将一个 scipy 稀疏矩阵转换为 torch 稀疏张量。"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # 推荐使用 torch.sparse_coo_tensor
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
    """加载 Planetoid 格式的图数据 (cora, citeseer, pubmed)"""
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
        usecols=[3, 5], # 仅读取第4和第6列
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
    """加载 disease_lp 数据集"""
    dir_path = os.path.join(datapath, dataset_name)
    
    # 修正 1: 将文件名中的 '.' 修改为 '_'
    edge_file_path = os.path.join(dir_path, 'disease_lp.edges.csv')
    
    if not os.path.exists(edge_file_path):
        raise FileNotFoundError(f"Cannot find edge file for disease_lp: {edge_file_path}")

    # 修正 2: 由于文件没有表头，在读取时添加 header=None 和 names 参数
    edges_df = pd.read_csv(
        edge_file_path, 
        header=None, 
        names=['source', 'target']
    )
    
    source_nodes = edges_df['source'].values
    target_nodes = edges_df['target'].values

    # 加载特征文件以确定节点总数 (这部分逻辑保持不变)
    feats_path = os.path.join(dir_path, 'disease_lp.feats.npz')
    if os.path.exists(feats_path):
        features = sp.load_npz(feats_path)
        num_nodes = features.shape[0]
    else:
        num_nodes = max(source_nodes.max(), target_nodes.max()) + 1

    # 构建邻接矩阵 (这部分逻辑保持不变)
    adj = sp.coo_matrix(
        (np.ones(len(source_nodes)), (source_nodes, target_nodes)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    )
    return adj


############################################################
# -------- OGB LinkProp ------------------------------------
############################################################

def load_ogb_link_adj(dataset_name, datapath):
    """加载 OGB linkprop 数据集的训练图邻接矩阵。"""
    canonical_name = canonicalize_ogb_link_dataset_name(dataset_name)
    if canonical_name is None:
        raise RuntimeError(f"Unsupported OGB dataset: {dataset_name}")

    raw_dir = resolve_ogb_raw_dir(canonical_name, datapath)
    if raw_dir is None:
        raise FileNotFoundError(f"Cannot resolve raw directory for {canonical_name} under {datapath}")

    edge_path = os.path.join(raw_dir, 'edge.csv.gz')
    num_nodes_path = os.path.join(raw_dir, 'num-node-list.csv.gz')

    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Cannot find edge file for {canonical_name}: {edge_path}")
    if not os.path.exists(num_nodes_path):
        raise FileNotFoundError(f"Cannot find num-node-list file for {canonical_name}: {num_nodes_path}")

    edges_df = pd.read_csv(edge_path, header=None, compression='gzip')
    num_nodes_df = pd.read_csv(num_nodes_path, header=None, compression='gzip')

    source_nodes = edges_df.iloc[:, 0].astype(np.int64).values
    target_nodes = edges_df.iloc[:, 1].astype(np.int64).values
    num_nodes = int(num_nodes_df.iloc[:, 0].sum())

    adj = sp.coo_matrix(
        (np.ones(len(source_nodes)), (source_nodes, target_nodes)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    )
    return adj

############################################################
# -------- 统一入口函数 ------------------------------------
############################################################

def get_adj_only(dataset_name, datapath):
    """
    根据 dataset_name 加载邻接矩阵并返回一个 PyTorch 稀疏张量
    """
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        print(f"[INFO] Loading '{dataset_name}' with Planetoid loader...")
        adj = load_planetoid_adj(dataset_name, datapath)
    elif dataset_name == 'airport':
        print(f"[INFO] Loading 'airport' dataset...")
        adj = load_airport_adj(dataset_name, datapath)
    elif dataset_name == 'disease_lp':
        print(f"[INFO] Loading 'disease_lp' dataset...")
        adj = load_disease_lp_adj(dataset_name, datapath)
    elif dataset_name in ['ogbl-collab', 'ogbl_collab', 'collab', 'ogbl-citation2', 'ogbl_citation2', 'citation2', 'citation-v2']:
        print(f"[INFO] Loading '{dataset_name}' dataset...")
        adj = load_ogb_link_adj(dataset_name, datapath)
    else:
        raise RuntimeError(f"[ERROR] Unknown dataset: {dataset_name}")

    # ---- 统一后处理 ----
    # 建立无向图
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj.setdiag(0)
    adj.eliminate_zeros()

    # --- 关键修改: 将最终的 scipy 矩阵转换为 PyTorch 稀疏张量 ---
    return scipy_sparse_to_torch_sparse_tensor(adj)