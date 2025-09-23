import argparse
from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        # ---- Optimization & Runtime ----
        'lr': (0.01, 'learning rate'),
        'dropout': (0.0, 'dropout probability'),
        'cuda': (0, 'which cuda device to use (-1 for cpu)'),
        'epochs': (5000, 'maximum epochs'),
        'weight-decay': (0., 'l2 regularization strength'),
        'optimizer': ('Adam', 'optimizer: Adam|RiemannianAdam'),
        'momentum': (0.999, 'optimizer momentum/beta2'),
        'patience': (100, 'early stop patience'),
        'min-epochs': (100, 'do not early stop before this'),
        'seed': (1234, 'training seed'),
        'log-freq': (1, 'print train metrics every N epochs'),
        'eval-freq': (1, 'evaluate every N epochs'),
        'grad-clip': (None, 'max grad norm (None=disabled)'),
        'lr-reduce-freq': (None, 'stepLR frequency; None=once at end'),
        'gamma': (0.5, 'StepLR gamma'),
        'print-epoch': (True, 'internal'),

        # ---- Saving / Logging ----
        'save': (0, '1=save model & artifacts'),
        'save-dir': (None, 'explicit save directory'),
        'log-csv': (None, 'append one-line result to this CSV'),
        'save-curv-history': (0, '1=save curvature history JSON'),
        'curv-history-path': (None, 'explicit curvature history JSON path'),

        # ---- Curvature & Regularization ----
        'use-two-sided-reg': (0, 'legacy switch; prefer penalty-mode'),
        'penalty-mode': ('none', 'none|two-sided|high-only|low-only|random-mu'),
        'lambda-reg': (0.0, 'λ for curvature penalty'),
        'mu-reg': (0.0, 'μ for low-curvature term'),
        'epsilon-reg': (1e-6, 'stability ε'),
        'penalty-avg': (1, '1=average penalty, 0=sum'),
        'penalty-warmup-epochs': (0, 'skip penalty first N epochs'),
        'pilot-align': (0, '1=derive μ from pilot JSON'),
        'pilot-json': (None, 'path to pilot JSON'),
        'curv_lr_scale': (1.0, 'LR scale for curvature params'),

        # ---- Distortion (random mode) ----
        'distortion-sample-pairs': (8000, 'num sampled pairs (random mode)'),
        'distortion-max-hop': (8, 'BFS hop cutoff'),
        'distortion-seed': (1234, 'pair sampling seed'),
        'distortion-eval-every': (0, 'every N evals also compute distortion; 0=only final'),
        'compute-final-distortion': (1, '1=compute final distortion'),

        # ---- Distortion (precomputed mode) ----
        'distortion-use-precomputed': (0, '1=use precomputed pairs file'),
        'distortion-pairs-file': (None, 'path to precomputed pairs (.npy/.pt)'),
        # NOTE: limit 行已移除，改用下面单独 add_argument
        'distortion-pre-shuffle': (0, '1=shuffle precomputed pairs before limit'),
        'distortion-allow-missing-hop': (0, '1=allow (i,j) without hop and BFS'),
        # ---- Misc ----
        'sweep-c': (0, 'internal flag for curvature sweeping'),
    },

    'model_config': {
        'task': ('nc', 'task: lp|nc'),
        'model': ('GCN', 'encoder: Shallow|MLP|HNN|GCN|GAT|HyperGCN|HGCN'),
        'dim': (128, 'embedding dimension'),
        'manifold': ('Euclidean', 'Euclidean|Hyperboloid|PoincareBall'),
        'c': (1.0, 'hyperbolic curvature (radius); None=trainable'),
        'r': (2., 'fermi-dirac decoder param r'),
        't': (1., 'fermi-dirac decoder param t'),
        'pretrained-embeddings': (None, 'path to .npy (Shallow)'),
        'pos-weight': (0, 'positive class upweight (nc)'),
        'num-layers': (2, 'hidden layers'),
        'bias': (1, '1=use bias'),
        'act': ('relu', 'activation'),
        'n-heads': (4, 'attention heads (GAT)'),
        'alpha': (0.2, 'leakyrelu alpha (GAT)'),
        'double-precision': ('0', 'use float64'),
        'use-att': (0, 'hyperbolic attention'),
        'local-agg': (0, 'local tangent aggregation')
    },

    'data_config': {
        'dataset': ('cora', 'dataset name'),
        'val-prop': (0.05, 'validation proportion (lp)'),
        'test-prop': (0.1, 'test proportion (lp)'),
        'use-feats': (1, '1=use node features'),
        'normalize-feats': (1, '1=feature normalization'),
        'normalize-adj': (1, '1=row-normalize adjacency'),
        'split-seed': (1234, 'split seed'),
    }
}

parser = argparse.ArgumentParser()
for _, cfg in config_args.items():
    parser = add_flags_from_config(parser, cfg)

parser.add_argument('--distortion-pairs-limit', type=int, default=None,
                    help='truncate to first N precomputed pairs (int)')

def get_parser():
    return parser

if __name__ == "__main__":
    args = parser.parse_args()
    print("distortion_use_precomputed =", getattr(args, 'distortion_use_precomputed', None))
    print("distortion_pairs_file      =", getattr(args, 'distortion_pairs_file', None))
    print("distortion_pairs_limit     =", getattr(args, 'distortion_pairs_limit', None))
