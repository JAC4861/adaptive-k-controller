"""Hyperbolic layers (modified: curvature always trainable parameters)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt


def get_dim_act_curv(args):
    """
    Prepare per-layer dimensions, activations, and curvature parameters.

    改动要点：
    - 以前: 如果 args.c 不为 None，则用普通 tensor (不可训练)；这会导致自适应控制无法获取曲率参数。
    - 现在: 无论 args.c 是否给定，统一使用 nn.Parameter(shape=[1])。
      初值: args.c (给定) 或 1.0 (未给定)。
    - 如需恢复“固定曲率”行为，可在外部添加一个标志 (例如 args.fix_curvature) 并在此处根据其值设置 requires_grad=False。

    返回:
      dims: 每层输入输出维度列表
      acts: 每层激活函数列表
      curvatures: List[nn.Parameter] (长度 = n_curvatures)
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)

    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))

    if args.task in ['lp', 'rec']:
        # Link prediction / recommendation 多一个输出层
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1

    init_c_value = 1.0 if args.c is None else float(args.c)

    # 如果将来想支持“固定”，可以:
    # trainable = not getattr(args, "fix_curvature", False)
    trainable = True

    curvatures = []
    for _ in range(n_curvatures):
        param = nn.Parameter(torch.tensor([init_c_value], dtype=torch.float32), requires_grad=trainable)
        if not args.cuda == -1:
            param = nn.Parameter(param.to(args.device), requires_grad=trainable)
        curvatures.append(param)

    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer (generic).
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        # c 两端相同
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear(x)
        h = self.hyp_act(h)
        return h


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features,
                 c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        x = x.clone() 
        h = self.linear(x)
        h = self.agg(h, adj)
        h = self.hyp_act(h)
        return h, adj


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    c: nn.Parameter([curvature]) 现在是可训练参数。
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features

        # c 可能是 nn.Parameter
        self.c = c

        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        try:
            c_val = self.c.detach().cpu().item()
        except:
            c_val = self.c
        return f'in_features={self.in_features}, out_features={self.out_features}, c={c_val}'


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c  # nn.Parameter
        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        # 把点映射到切空间
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                # 局部注意力（可能较慢）
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent  # 未直接用，可调试删除
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
        else:
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        try:
            c_val = self.c.detach().cpu().item()
        except:
            c_val = self.c
        return f'c={c_val}'


class HypAct(Module):
    """
    Hyperbolic activation layer.
    这里不强制注册额外曲率参数，只使用传入引用。
    如果传入的是 nn.Parameter，同一个对象可被多个层引用。
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        # 保持引用，不重复注册新参数
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        try:
            cin = self.c_in.detach().cpu().item()
        except:
            cin = self.c_in
        try:
            cout = self.c_out.detach().cpu().item()
        except:
            cout = self.c_out
        return f'c_in={cin}, c_out={cout}'