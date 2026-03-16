import copy
import glob
import os
import zipfile
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
import torch

from utils.data_utils import process


OGB_LINK_DATASETS = {
    "ogbl-collab": {
        "aliases": ("ogbl-collab", "ogbl_collab", "collab"),
        "zip_candidates": ("collab.zip", "ogbl-collab.zip", "ogbl_collab.zip"),
        "dir_candidates": ("ogbl_collab", "collab", "ogbl-collab"),
        "metric": "hits@50",
        "metric_display": "Hits@50",
        "symmetric_message_passing": True,
    },
    "ogbl-citation2": {
        "aliases": ("ogbl-citation2", "ogbl_citation2", "citation2", "citation-v2"),
        "zip_candidates": ("citation-v2.zip", "ogbl-citation2.zip", "ogbl_citation2.zip"),
        "dir_candidates": ("citation-v2", "ogbl_citation2", "ogbl-citation2"),
        "metric": "mrr",
        "metric_display": "MRR",
        "symmetric_message_passing": True,
    },
}


def canonicalize_ogb_link_dataset_name(dataset_name: str) -> Optional[str]:
    if dataset_name is None:
        return None

    lowered = str(dataset_name).strip().lower()
    for canonical_name, spec in OGB_LINK_DATASETS.items():
        if lowered in spec["aliases"]:
            return canonical_name
    return None


def is_ogb_link_dataset_name(dataset_name: str) -> bool:
    return canonicalize_ogb_link_dataset_name(dataset_name) is not None


def get_ogb_dataset_spec(dataset_name: str) -> Dict[str, object]:
    canonical_name = canonicalize_ogb_link_dataset_name(dataset_name)
    if canonical_name is None:
        raise ValueError(f"Unsupported OGB link dataset: {dataset_name}")
    return OGB_LINK_DATASETS[canonical_name]


def _find_ogb_dataset_dir(dataset_name: str, data_root: str) -> Optional[str]:
    spec = get_ogb_dataset_spec(dataset_name)

    for dirname in spec["dir_candidates"]:
        candidate = os.path.join(data_root, dirname)
        if os.path.isdir(os.path.join(candidate, "raw")) or os.path.isdir(os.path.join(candidate, "processed")):
            return candidate

    for raw_dir in glob.glob(os.path.join(data_root, "*", "raw")):
        parent_dir = os.path.dirname(raw_dir)
        base_name = os.path.basename(parent_dir)
        if base_name in spec["dir_candidates"]:
            return parent_dir

    return None


def resolve_ogb_raw_dir(dataset_name: str, data_root: str) -> Optional[str]:
    dataset_dir = _find_ogb_dataset_dir(dataset_name, data_root)
    if dataset_dir is None:
        return None

    raw_dir = os.path.join(dataset_dir, "raw")
    if os.path.isdir(raw_dir):
        return raw_dir
    return None


def _ensure_ogb_extracted(dataset_name: str, data_root: str) -> None:
    os.makedirs(data_root, exist_ok=True)
    if resolve_ogb_raw_dir(dataset_name, data_root) is not None:
        return

    spec = get_ogb_dataset_spec(dataset_name)
    found_zip = next(
        (os.path.join(data_root, zip_name) for zip_name in spec["zip_candidates"] if os.path.isfile(os.path.join(data_root, zip_name))),
        None,
    )
    if found_zip is None:
        return

    print(f"[OGB] Extracting local zip once: {found_zip} -> {data_root}")
    with zipfile.ZipFile(found_zip, "r") as zf:
        zf.extractall(data_root)


def _ensure_ogbl_collab_extracted(data_root: str) -> None:
    _ensure_ogb_extracted("ogbl-collab", data_root)


def _build_train_edge_split(dataset_name: str, split_edge: dict) -> torch.Tensor:
    canonical_name = canonicalize_ogb_link_dataset_name(dataset_name)
    if canonical_name == "ogbl-collab":
        return split_edge["train"]["edge"].long()
    if canonical_name == "ogbl-citation2":
        return torch.stack(
            (
                split_edge["train"]["source_node"].long(),
                split_edge["train"]["target_node"].long(),
            ),
            dim=1,
        )
    raise ValueError(f"Unsupported OGB link dataset: {dataset_name}")


def _build_message_passing_edge_index(dataset_name: str,
                                      train_edge_index: torch.Tensor,
                                      num_nodes: int) -> torch.Tensor:
    from torch_geometric.utils import to_undirected

    spec = get_ogb_dataset_spec(dataset_name)
    if spec.get("symmetric_message_passing", False):
        return to_undirected(train_edge_index, num_nodes=num_nodes)
    return train_edge_index


def load_ogb_link_dataset(args, data_root: str, dataset_name: str) -> Dict[str, object]:
    canonical_name = canonicalize_ogb_link_dataset_name(dataset_name)
    if canonical_name is None:
        raise ValueError(f"Unsupported OGB link dataset: {dataset_name}")

    try:
        from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
    except Exception as exc:
        raise RuntimeError(
            f"Loading {canonical_name} requires packages 'ogb' and 'torch_geometric'."
        ) from exc

    _ensure_ogb_extracted(canonical_name, data_root)

    dataset = PygLinkPropPredDataset(name=canonical_name, root=data_root)
    split_edge = dataset.get_edge_split()
    pyg_data = dataset[0]
    evaluator = Evaluator(name=canonical_name)
    spec = get_ogb_dataset_spec(canonical_name)

    train_edge = _build_train_edge_split(canonical_name, split_edge)
    train_edge_index = train_edge.t().contiguous()
    message_passing_edge_index = _build_message_passing_edge_index(
        canonical_name,
        train_edge_index,
        num_nodes=pyg_data.num_nodes,
    )

    rows = message_passing_edge_index[0].cpu().numpy()
    cols = message_passing_edge_index[1].cpu().numpy()
    vals = np.ones(rows.shape[0], dtype=np.float32)
    adj_train = sp.csr_matrix((vals, (rows, cols)), shape=(pyg_data.num_nodes, pyg_data.num_nodes), dtype=np.float32)

    if getattr(pyg_data, "x", None) is None:
        features_np = np.eye(pyg_data.num_nodes, dtype=np.float32)
    else:
        features_np = pyg_data.x.cpu().numpy().astype(np.float32)

    adj_train_norm, features = process(
        adj_train,
        features_np,
        normalize_adj=getattr(args, "normalize_adj", 1),
        normalize_feats=getattr(args, "normalize_feats", 1),
    )

    data_dict = {
        "dataset_name": canonical_name,
        "official_metric": spec["metric"],
        "official_metric_display": spec["metric_display"],
        "data": pyg_data,
        "split_edge": split_edge,
        "evaluator": evaluator,
        "num_nodes": int(pyg_data.num_nodes),
        "x": features,
        "features": features,
        "adj_train": adj_train,
        "adj_train_norm": adj_train_norm,
        "train_edge_index": train_edge_index,
        "message_passing_edge_index": message_passing_edge_index,
        "train_edges": train_edge,
        "node_year": getattr(pyg_data, "node_year", None),
        "edge_weight": getattr(pyg_data, "edge_weight", None),
        "edge_year": getattr(pyg_data, "edge_year", None),
    }

    if canonical_name == "ogbl-collab":
        data_dict.update({
            "val_edges": split_edge["valid"]["edge"].long(),
            "val_edges_false": split_edge["valid"]["edge_neg"].long(),
            "test_edges": split_edge["test"]["edge"].long(),
            "test_edges_false": split_edge["test"]["edge_neg"].long(),
        })
    elif canonical_name == "ogbl-citation2":
        data_dict.update({
            "val_edges": torch.stack(
                (split_edge["valid"]["source_node"].long(), split_edge["valid"]["target_node"].long()),
                dim=1,
            ),
            "test_edges": torch.stack(
                (split_edge["test"]["source_node"].long(), split_edge["test"]["target_node"].long()),
                dim=1,
            ),
            "val_target_neg": split_edge["valid"]["target_node_neg"].long(),
            "test_target_neg": split_edge["test"]["target_node_neg"].long(),
        })

    return data_dict


def load_ogbl_collab(args, data_root: str) -> Dict[str, object]:
    return load_ogb_link_dataset(args, data_root=data_root, dataset_name="ogbl-collab")


def sample_negative_edges(train_edge_index: torch.Tensor,
                          num_nodes: int,
                          num_neg_samples: int,
                          device: torch.device) -> torch.Tensor:
    from torch_geometric.utils import negative_sampling

    neg_edge_index = negative_sampling(
        edge_index=train_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=int(num_neg_samples),
        method="sparse",
    )
    return neg_edge_index.t().contiguous().to(device)


@torch.no_grad()
def batch_predict_edges(model,
                        embeddings: torch.Tensor,
                        edge_index_2col: torch.Tensor,
                        batch_size: int = 65536) -> torch.Tensor:
    scores = []
    total = edge_index_2col.size(0)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_edge = edge_index_2col[start:end]
        batch_score = model.decode(embeddings, batch_edge)
        scores.append(batch_score.detach().view(-1).cpu())
    if not scores:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(scores, dim=0)


@torch.no_grad()
def _batch_predict_source_target_neg(model,
                                     embeddings: torch.Tensor,
                                     source_nodes: torch.Tensor,
                                     target_nodes_neg: torch.Tensor,
                                     batch_size: int = 65536) -> torch.Tensor:
    num_source = source_nodes.size(0)
    num_neg = target_nodes_neg.size(1)
    repeated_source = source_nodes.view(-1, 1).expand(-1, num_neg).reshape(-1)
    flat_target_neg = target_nodes_neg.reshape(-1)
    flat_edges = torch.stack((repeated_source, flat_target_neg), dim=1)
    flat_scores = batch_predict_edges(model, embeddings, flat_edges, batch_size=batch_size)
    return flat_scores.view(num_source, num_neg)


@torch.no_grad()
def evaluate_ogb_link_dataset(model,
                              embeddings: torch.Tensor,
                              split_edge: dict,
                              evaluator,
                              dataset_name: str,
                              batch_size_edge_eval: int = 65536,
                              include_valid: bool = True,
                              include_test: bool = True) -> Dict[str, float]:
    canonical_name = canonicalize_ogb_link_dataset_name(dataset_name)
    if canonical_name is None:
        raise ValueError(f"Unsupported OGB link dataset: {dataset_name}")

    if not include_valid and not include_test:
        return {}

    if canonical_name == "ogbl-collab":
        evaluator.K = 50

        out = {}
        if include_valid:
            pos_valid = split_edge["valid"]["edge"].to(embeddings.device)
            neg_valid = split_edge["valid"]["edge_neg"].to(embeddings.device)
            y_pos_valid = batch_predict_edges(model, embeddings, pos_valid, batch_size_edge_eval)
            y_neg_valid = batch_predict_edges(model, embeddings, neg_valid, batch_size_edge_eval)
            valid_out = evaluator.eval({"y_pred_pos": y_pos_valid, "y_pred_neg": y_neg_valid})
            out["valid_hits@50"] = float(valid_out["hits@50"])

        if include_test:
            pos_test = split_edge["test"]["edge"].to(embeddings.device)
            neg_test = split_edge["test"]["edge_neg"].to(embeddings.device)
            y_pos_test = batch_predict_edges(model, embeddings, pos_test, batch_size_edge_eval)
            y_neg_test = batch_predict_edges(model, embeddings, neg_test, batch_size_edge_eval)
            test_out = evaluator.eval({"y_pred_pos": y_pos_test, "y_pred_neg": y_neg_test})
            out["test_hits@50"] = float(test_out["hits@50"])

        return out

    if canonical_name == "ogbl-citation2":
        out = {}
        if include_valid:
            valid_source = split_edge["valid"]["source_node"].to(embeddings.device)
            valid_target = split_edge["valid"]["target_node"].to(embeddings.device)
            valid_target_neg = split_edge["valid"]["target_node_neg"].to(embeddings.device)
            valid_pos_edges = torch.stack((valid_source, valid_target), dim=1)
            y_pos_valid = batch_predict_edges(model, embeddings, valid_pos_edges, batch_size_edge_eval)
            y_neg_valid = _batch_predict_source_target_neg(
                model,
                embeddings,
                valid_source,
                valid_target_neg,
                batch_size=batch_size_edge_eval,
            )
            valid_out = evaluator.eval({"y_pred_pos": y_pos_valid, "y_pred_neg": y_neg_valid})
            out["valid_mrr"] = float(valid_out["mrr_list"].mean().item())

        if include_test:
            test_source = split_edge["test"]["source_node"].to(embeddings.device)
            test_target = split_edge["test"]["target_node"].to(embeddings.device)
            test_target_neg = split_edge["test"]["target_node_neg"].to(embeddings.device)
            test_pos_edges = torch.stack((test_source, test_target), dim=1)
            y_pos_test = batch_predict_edges(model, embeddings, test_pos_edges, batch_size_edge_eval)
            y_neg_test = _batch_predict_source_target_neg(
                model,
                embeddings,
                test_source,
                test_target_neg,
                batch_size=batch_size_edge_eval,
            )
            test_out = evaluator.eval({"y_pred_pos": y_pos_test, "y_pred_neg": y_neg_test})
            out["test_mrr"] = float(test_out["mrr_list"].mean().item())

        return out

    raise ValueError(f"Unsupported OGB link dataset: {dataset_name}")


@torch.no_grad()
def evaluate_ogbl_collab(model,
                         embeddings: torch.Tensor,
                         split_edge: dict,
                         evaluator,
                         batch_size_edge_eval: int = 65536) -> Dict[str, float]:
    return evaluate_ogb_link_dataset(
        model=model,
        embeddings=embeddings,
        split_edge=split_edge,
        evaluator=evaluator,
        dataset_name="ogbl-collab",
        batch_size_edge_eval=batch_size_edge_eval,
    )


def clone_state_dict_cpu(model) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
