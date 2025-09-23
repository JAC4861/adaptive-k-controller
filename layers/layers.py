"""Euclidean layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


def get_dim_act(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
    return dims, acts


# class GraphConvolution(Module):
#     """
#     Simple GCN layer.
#     """

#     def __init__(self, in_features, out_features, dropout, act, use_bias):
#         super(GraphConvolution, self).__init__()
#         self.dropout = dropout
#         self.linear = nn.Linear(in_features, out_features, use_bias)
#         self.act = act
#         self.in_features = in_features
#         self.out_features = out_features

#     def forward(self, input):
#         x, adj = input
#         hidden = self.linear.forward(x)
#         hidden = F.dropout(hidden, self.dropout, training=self.training)
#         if adj.is_sparse:
#             support = torch.spmm(adj, hidden)
#         else:
#             support = torch.mm(adj, hidden)
#         output = self.act(support), adj
#         return output

#     def extra_repr(self):
#         return 'input_dim={}, output_dim={}'.format(
#             self.in_features, self.out_features
#         )
class GraphConvolution(Module):
    """
    一个简单的GCN层。
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input_tuple):
        x, adj = input_tuple
        hidden = self.linear(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        if adj.is_sparse:
            support = torch.spmm(adj, hidden)
        else:
            support = torch.mm(adj, hidden)
        
        # 修正就在这里：只返回经过激活函数处理后的特征
        output = self.act(support)
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
            self.in_features, self.out_features
        )

class Linear(Module):
    """
    Simple Linear layer with dropout.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act

    def forward(self, x):
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        out = self.act(hidden)
        return out


class FermiDiracDecoder(Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        # 数值稳定处理
        scaled = (dist - self.r) / self.t
        scaled = torch.clamp(scaled, min=-50.0, max=50.0)  # 避免溢出
        probs = torch.sigmoid(-scaled)

        # Debug: 检测 NaN
        if torch.isnan(probs).any():
            print("[Warning] NaN detected in FermiDiracDecoder.forward()")
            print("dist min/max:", dist.min().item(), dist.max().item())
            print("scaled min/max:", scaled.min().item(), scaled.max().item())

        return probs
