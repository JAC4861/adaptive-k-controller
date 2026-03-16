"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch


def load_data(args, datapath):
    """
    根據給定的任務和資料集載入資料。
    此函式作為一個分派器，並在之後執行最終的處理步驟，如正規化和特徵增強。
    """
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    else:
        # 對於連結預測，我們先載入基礎資料，然後再切分邊。
        data = load_data_lp(args.dataset, args.use_feats, datapath)
        adj = data['adj_train']
        if args.task == 'lp':
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed
            )
            data['adj_train'] = adj_train
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
            
    # 對所有資料集和任務進行通用處理
    data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    if args.dataset == 'airport':
        data['features'] = augment(data['adj_train'], data['features'])
    return data


# ############### 特徵處理 ####################################


def process(adj, features, normalize_adj, normalize_feats):
    """
    處理鄰接矩陣和特徵。
    包含將特徵轉為密集的 numpy 陣列、正規化特徵和正規化鄰接矩陣。
    """
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        # 在正規化前添加自環 (self-loops)
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """對稀疏矩陣進行行正規化 (Row-normalize)"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """將 scipy 稀疏矩陣轉換為 torch 稀疏張量"""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    """
    為 'airport' 資料集的特徵添加節點度 (degree) 資訊以進行增強。
    """
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    # 將度大於 5 的值設為 5
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


def sample_negative_edges_sparse(adj, num_samples, seed):
    """在稀疏图上采样负边，避免对补图做稠密化。"""
    upper_adj = sp.triu(adj, k=1).tocoo()
    num_nodes = adj.shape[0]
    max_negatives = num_nodes * (num_nodes - 1) // 2 - upper_adj.nnz
    if num_samples > max_negatives:
        raise ValueError(
            f"請求的負邊數 {num_samples} 超過可用非邊數 {max_negatives}"
        )

    existing_edges = set(zip(upper_adj.row.tolist(), upper_adj.col.tolist()))
    sampled_edges = []
    sampled_set = set()
    rng = np.random.RandomState(seed)

    while len(sampled_edges) < num_samples:
        remaining = num_samples - len(sampled_edges)
        batch_size = max(remaining * 4, 4096)
        row = rng.randint(0, num_nodes, size=batch_size)
        col = rng.randint(0, num_nodes, size=batch_size)
        mask = row < col
        row = row[mask]
        col = col[mask]

        for src, dst in zip(row, col):
            edge = (int(src), int(dst))
            if edge in existing_edges or edge in sampled_set:
                continue
            sampled_set.add(edge)
            sampled_edges.append(edge)
            if len(sampled_edges) >= num_samples:
                break

    return np.asarray(sampled_edges, dtype=np.int64)


# ############### 資料切分 #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    """
    為連結預測任務，將邊切分為訓練、驗證和測試集。
    同時為每個集合採樣負邊 (negative edges)。
    """
    rng = np.random.RandomState(seed)
    x, y = sp.triu(adj, k=1).nonzero()
    pos_edges = np.column_stack((x, y))
    rng.shuffle(pos_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    n_train = m_pos - n_val - n_test

    neg_edges = sample_negative_edges_sparse(adj, m_pos, seed)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = neg_edges[n_val + n_test:n_val + n_test + n_train]
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)


def split_data(labels, val_prop, test_prop, seed):
    """將節點索引切分為訓練、驗證和測試集"""
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                 nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                 nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    """為 'airport' 資料集的標籤進行特徵二值化"""
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### 連結預測資料載入器 ####################################


def load_data_lp(dataset, use_feats, data_path):
    """
    為連結預測任務載入資料。
    現在為所有資料集提供快取機制。
    """
    ### 修改開始：為連結預測任務添加通用快取機制 ###
    processed_dir = os.path.join(data_path, "processed", dataset)
    os.makedirs(processed_dir, exist_ok=True)
    cache_file = os.path.join(processed_dir, f"lp_usefeats{use_feats}.pkl")

    if os.path.exists(cache_file):
        print(f"[*] 正在從 {cache_file} 載入 {dataset} 的連結預測已處理資料")
        with open(cache_file, "rb") as f:
            data = pkl.load(f)
        return data
    ### 修改結束 ###
        
    if dataset in ['cora', 'pubmed']:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'disease_lp':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    else:
        raise FileNotFoundError('不支援資料集: {}'.format(dataset))
    
    data = {'adj_train': adj, 'features': features}
    
    ### 修改開始：將處理好的資料儲存至快取檔案 ###
    with open(cache_file, "wb") as f:
        pkl.dump(data, f)
    print(f"[*] 已將 {dataset} 的連結預測已處理資料儲存至 {cache_file}")
    ### 修改結束 ###

    return data


# ############### 節點分類資料載入器 ####################################


def load_data_nc(dataset, use_feats, data_path, split_seed):
    """
    為節點分類任務載入資料。
    現在為所有資料集提供快取機制。
    """
    ### 修改開始：為節點分類任務添加通用快取機制 ###
    processed_dir = os.path.join(data_path, "processed", dataset)
    os.makedirs(processed_dir, exist_ok=True)
    # 使用 .pkl 格式，因為它能處理多種 Python 物件，如 scipy 稀疏矩陣
    cache_file = os.path.join(processed_dir, f"nc_seed{split_seed}_usefeats{use_feats}.pkl")

    if os.path.exists(cache_file):
        print(f"[*] 正在從 {cache_file} 載入 {dataset} 的節點分類已處理資料")
        with open(cache_file, "rb") as f:
            data = pkl.load(f)
        return data
    ### 修改結束 ###

    if dataset in ['cora', 'pubmed']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
                dataset, use_feats, data_path, split_seed
        )
    else:
        if dataset == 'disease_nc':
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        else:
            raise FileNotFoundError('不支援資料集: {}'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val,
            'idx_test': idx_test}
    
    ### 修改開始：將處理好的資料儲存至快取檔案 ###
    with open(cache_file, "wb") as f:
        pkl.dump(data, f)
    print(f"[*] 已將 {dataset} 的節點分類已處理資料儲存至 {cache_file}")
    ### 修改結束 ###

    return data


# ############### 資料集 ####################################


def _resolve_citation_data_path(dataset_str, data_path):
    candidates = []

    raw_under_current = os.path.join(data_path, "raw")
    if os.path.isdir(raw_under_current):
        candidates.append(raw_under_current)
    candidates.append(data_path)

    parent_dir = os.path.dirname(data_path.rstrip(os.sep))
    base_name = os.path.basename(data_path.rstrip(os.sep))
    alt_names = [dataset_str, dataset_str.lower(), dataset_str.upper(), dataset_str.capitalize()]
    if dataset_str.lower() == 'pubmed':
        alt_names.append('PubMed')

    seen = set()
    ordered_names = []
    for name in alt_names:
        if name not in seen:
            ordered_names.append(name)
            seen.add(name)

    for name in ordered_names:
        alt_root = os.path.join(parent_dir, name)
        alt_raw = os.path.join(alt_root, "raw")
        if alt_raw != raw_under_current and os.path.isdir(alt_raw):
            candidates.append(alt_raw)
        if alt_root != data_path and alt_root != base_name and os.path.isdir(alt_root):
            candidates.append(alt_root)

    required = f"ind.{dataset_str}.x"
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, required)):
            return candidate

    return raw_under_current if os.path.isdir(raw_under_current) else data_path


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    """
    載入引文網路資料 (cora, pubmed)。
    原先針對 pubmed 的特定快取邏輯已被移除，並在呼叫函式中進行了通用化處理。
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    citation_data_path = _resolve_citation_data_path(dataset_str, data_path)

    objects = []
    for i in range(len(names)):
        with open(os.path.join(citation_data_path, f"ind.{dataset_str}.{names[i]}"), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(citation_data_path, f"ind.{dataset_str}.test.index"))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])

    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    """解析索引檔案"""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    """載入合成資料集 (disease_nc, disease_lp)"""
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    """載入 airport 資料集"""
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features