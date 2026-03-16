import random
from collections import deque
import os

import numpy as np
import torch


@torch.no_grad()
def _build_adj_list(adj_sparse):
    coo = adj_sparse.coalesce()
    idx = coo.indices()
    rows, cols = idx[0].tolist(), idx[1].tolist()
    n = adj_sparse.size(0)
    adj = [[] for _ in range(n)]
    for row, col in zip(rows, cols):
        if row != col:
            adj[row].append(col)
    return adj


def _bfs_dist(adj_list, source, target, max_hop=8):
    if source == target:
        return 0
    visited = {source}
    queue = deque([(source, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_hop:
            continue
        for neighbor in adj_list[node]:
            if neighbor == target:
                return depth + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return None


def _load_precomputed_pairs(pairs_file, device, limit=None, shuffle=False, seed=None):
    if pairs_file.endswith('.npy'):
        pairs = torch.from_numpy(np.load(pairs_file))
    else:
        pairs = torch.load(pairs_file)
        if isinstance(pairs, np.ndarray):
            pairs = torch.from_numpy(pairs)

    if pairs.dim() != 2 or pairs.size(1) not in (2, 3):
        raise ValueError('precomputed distortion pairs must have shape (N,2) or (N,3)')

    if shuffle:
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
            perm = torch.randperm(pairs.size(0), generator=generator)
        else:
            perm = torch.randperm(pairs.size(0))
        pairs = pairs[perm]

    if limit is not None:
        pairs = pairs[:limit]

    return pairs.to(device=device, dtype=torch.long)


def get_distortion_cache_path(data_path, split_seed, val_prop, test_prop, max_hop, pair_count):
    def _fmt(value):
        return format(value, '.6g').replace('.', 'p')

    filename = 'dist_pairs_lp_seed{}_val{}_test{}_hop{}_n{}.npy'.format(
        int(split_seed),
        _fmt(val_prop),
        _fmt(test_prop),
        int(max_hop),
        int(pair_count),
    )
    return os.path.join(data_path, filename)


@torch.no_grad()
def create_precomputed_pairs(adj_sparse, output_path, max_pairs=8000, max_hop=8, seed=1234):
    rng = random.Random(seed)
    adj_list = _build_adj_list(adj_sparse.coalesce())
    num_nodes = len(adj_list)
    sampled = []
    sampled_set = set()
    trials = 0
    max_trials = max(20 * max_pairs, 1)

    while len(sampled) < max_pairs and trials < max_trials:
        src = rng.randint(0, num_nodes - 1)
        dst = rng.randint(0, num_nodes - 1)
        trials += 1
        if src == dst:
            continue
        pair = (min(src, dst), max(src, dst))
        if pair in sampled_set:
            continue
        graph_dist = _bfs_dist(adj_list, src, dst, max_hop=max_hop)
        if graph_dist is None or graph_dist == 0:
            continue
        sampled_set.add(pair)
        sampled.append((src, dst, graph_dist))

    if not sampled:
        raise RuntimeError('failed to generate any valid distortion pairs for the current graph split')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, np.asarray(sampled, dtype=np.int64))
    return output_path, len(sampled)


@torch.no_grad()
def compute_distortion(
    embeddings,
    adj_sparse,
    manifold_name,
    sample_pairs=2000,
    max_hop=8,
    use_precomputed=False,
    pairs_file=None,
    pairs_limit=None,
    precomputed_shuffle=False,
    allow_missing_hop=False,
    precomputed_seed=None,
    seed=None,
):
    device = embeddings.device
    num_nodes = embeddings.size(0)

    if use_precomputed:
        if not pairs_file:
            raise ValueError('distortion_use_precomputed=1 requires distortion_pairs_file')
        pairs = _load_precomputed_pairs(
            pairs_file,
            device=device,
            limit=pairs_limit,
            shuffle=precomputed_shuffle,
            seed=precomputed_seed,
        )
        valid_mask = (
            (pairs[:, 0] >= 0)
            & (pairs[:, 0] < num_nodes)
            & (pairs[:, 1] >= 0)
            & (pairs[:, 1] < num_nodes)
        )
        pairs = pairs[valid_mask]

        if pairs.numel() == 0:
            return {'distortion': float('nan'), 'pairs_used': 0, 'pairs_mode': 'precomputed'}

        if pairs.size(1) == 2:
            if not allow_missing_hop:
                raise ValueError('precomputed distortion pairs without hop require distortion_allow_missing_hop=1')
            adj_list = _build_adj_list(adj_sparse)
            hop_pairs = []
            for src, dst in pairs.tolist():
                if src == dst:
                    continue
                graph_dist = _bfs_dist(adj_list, src, dst, max_hop=max_hop)
                if graph_dist is None or graph_dist == 0:
                    continue
                hop_pairs.append((src, dst, graph_dist))
            if not hop_pairs:
                return {'distortion': float('nan'), 'pairs_used': 0, 'pairs_mode': 'precomputed_bfs'}
            pairs = torch.tensor(hop_pairs, device=device, dtype=torch.long)

        pairs = pairs[(pairs[:, 0] != pairs[:, 1]) & (pairs[:, 2] > 0) & (pairs[:, 2] <= max_hop)]
        if pairs.size(0) == 0:
            return {'distortion': float('nan'), 'pairs_used': 0, 'pairs_mode': 'precomputed'}
        selected = pairs
        mode = 'precomputed'
    else:
        if seed is not None:
            random.seed(seed)
        adj_list = _build_adj_list(adj_sparse)
        sampled = []
        trials = 0
        max_trials = max(8 * sample_pairs, 1)
        while len(sampled) < sample_pairs and trials < max_trials:
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            if src == dst:
                trials += 1
                continue
            graph_dist = _bfs_dist(adj_list, src, dst, max_hop=max_hop)
            trials += 1
            if graph_dist is None or graph_dist == 0:
                continue
            sampled.append((src, dst, graph_dist))

        if not sampled:
            return {'distortion': float('nan'), 'pairs_used': 0, 'pairs_mode': 'random'}
        selected = torch.tensor(sampled, device=device, dtype=torch.long)
        mode = 'random'

    left = embeddings[selected[:, 0]]
    right = embeddings[selected[:, 1]]
    graph_dist = selected[:, 2].float()

    if manifold_name.lower() == 'euclidean':
        manifold_dist = torch.norm(left - right, dim=1)
    elif manifold_name.lower().startswith('poincare'):
        eps = 1e-7
        left_sq = torch.clamp(torch.sum(left * left, dim=1), max=1 - 1e-5)
        right_sq = torch.clamp(torch.sum(right * right, dim=1), max=1 - 1e-5)
        diff_sq = torch.sum((left - right) ** 2, dim=1)
        denom = (1 - left_sq) * (1 - right_sq)
        arg = 1 + 2 * diff_sq / (denom + eps)
        manifold_dist = torch.acosh(torch.clamp(arg, min=1 + eps))
    else:
        manifold_dist = torch.norm(left - right, dim=1)

    rel_dist = (torch.abs(manifold_dist - graph_dist) / (graph_dist + 1e-6)).mean().item()
    return {
        'distortion': rel_dist,
        'pairs_used': int(selected.size(0)),
        'pairs_mode': mode,
        'dg_mean': graph_dist.mean().item(),
        'dh_mean': manifold_dist.mean().item(),
    }