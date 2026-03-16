from __future__ import division, print_function
import os, time, json, logging, datetime
import numpy as np
import torch

from config import parser
from models.base_models import NCModel, LPModel
from utils.train_utils import get_dir_name, format_metrics
from utils.curv_and_distortion import (
    extract_curvature_params,
    compute_distortion
)
from utils.curvature_control import CurvatureController
from utils.ogb_utils import (
    load_ogb_link_dataset,
    sample_negative_edges,
    evaluate_ogb_link_dataset,
    clone_state_dict_cpu,
    canonicalize_ogb_link_dataset_name,
    is_ogb_link_dataset_name,
)
import optimizers

# -------------------------------------------------
# 注入额外控制/监控参数
# -------------------------------------------------
def _maybe_add_args(p):
    new_flags = [
        ('--plateau-low', float, 0.785, 'curvature plateau lower'),
        ('--plateau-high', float, 0.800, 'curvature plateau upper'),
        ('--plateau-mean', float, 0.33, 'plateau mean distortion'),
        ('--plateau-slack', float, 0.01, 'mean slack'),
        ('--plateau-window', int, 10, 'window length'),
        ('--plateau-var-th', float, 0.003, 'variance threshold'),
        ('--plateau-min-epochs', int, 25, 'min epochs before freeze'),
        ('--curv-warmup', int, 5, 'warmup epochs'),
        ('--curv-penalty-square', int, 1, '1 square penalty'),
        ('--curv-freeze', int, 1, '1 enable freeze'),
        ('--adapt-fail-epoch', int, 50, 'adapt λ/μ epoch (0 disable)'),
        ('--adapt-lambda-scale', float, 1.5, 'scale factor for λ/μ'),
        ('--distortion-monitor-every', int, 1, 'monitor every N eval'),
        ('--monitor-sample-pairs', int, 2000, 'pairs for monitor distortion'),
        ('--min-curvature', float, 1e-5, 'minimum allowed trainable curvature'),
    ]
    existing = {a.dest for g in p._action_groups for a in g._group_actions}
    for flag, typ, default, help_text in new_flags:
        dest = flag.lstrip('-').replace('-', '_')
        if dest not in existing:
            p.add_argument(flag, type=typ, default=default, help=help_text)
    
    if 'curv_lr_scale' not in existing:
        p.add_argument('--curv_lr_scale', type=float, default=1.0,
                         help='LR scale for curvature params')
    
    # --- MODIFICATION: 添加 force_preprocess 参数 ---
    if 'force_preprocess' not in existing:
        p.add_argument('--force_preprocess', action='store_true',
                         help='If set, force data preprocessing even if cache exists.')
    
    return p

_maybe_add_args(parser)

# -------------------------------------------------
# 工具函数
# -------------------------------------------------
def clamp_curvature_parameters(curvature_params, min_value):
    if not curvature_params:
        return
    min_value = float(min_value)
    with torch.no_grad():
        for param in curvature_params:
            param.clamp_(min=min_value)


def current_curvature_value(model, default_c, last_known_value=None):
    try:
        curvs = extract_curvature_params(model, include_frozen=True)
        if curvs:
            return float(torch.clamp(curvs[0][1].detach().clone(), min=1e-5).item())
    except Exception:
        pass
    if last_known_value is not None:
        return float(last_known_value)
    try:
        if default_c is not None and str(default_c).lower() != "none":
            return float(default_c)
    except (ValueError, TypeError):
        pass
    return None


def get_peak_gpu_mem_mb(device_str):
    if not torch.cuda.is_available() or not str(device_str).startswith('cuda'):
        return None
    try:
        peak_bytes = torch.cuda.max_memory_allocated(device=device_str)
        return float(peak_bytes / (1024.0 * 1024.0))
    except Exception:
        return None

def pick_val_metric(metric_dict):
    for k in ['f1','acc','roc','ap']:
        if k in metric_dict:
            return float(metric_dict[k])
    for k,v in metric_dict.items():
        if k != 'loss':
            try:
                return float(v)
            except:
                pass
    return float(metric_dict.get('loss', 0.0))


def is_ogb_link_dataset(args):
    return args.task == 'lp' and is_ogb_link_dataset_name(args.dataset)


def _metric_token(metric_name):
    return str(metric_name).upper().replace('@', '').replace('-', '_')


def build_distortion_kwargs(args, sample_pairs=None):
    kwargs = {
        'max_hop': int(getattr(args, 'distortion_max_hop', 8)),
        'use_precomputed': bool(int(getattr(args, 'distortion_use_precomputed', 0))),
        'pairs_file': getattr(args, 'distortion_pairs_file', None),
        'pairs_limit': getattr(args, 'distortion_pairs_limit', None),
        'precomputed_shuffle': bool(int(getattr(args, 'distortion_pre_shuffle', 0))),
        'allow_missing_hop': bool(int(getattr(args, 'distortion_allow_missing_hop', 0))),
        'precomputed_seed': int(getattr(args, 'distortion_seed', 1234)),
        'seed': int(getattr(args, 'distortion_seed', 1234)),
    }
    if sample_pairs is not None:
        kwargs['sample_pairs'] = int(sample_pairs)
    return kwargs

def maybe_monitor_dist(args, embeddings, data, model):
    sample_pairs = int(getattr(args,'monitor_sample_pairs',2000))
    res = compute_distortion(
        embeddings,
        data['adj_train_norm'],
        manifold_name=args.manifold,
        model=model,
        **build_distortion_kwargs(args, sample_pairs=sample_pairs)
    )
    return res.get('distortion', None)


def validate_distortion_config(args):
    use_precomputed = bool(int(getattr(args, 'distortion_use_precomputed', 0)))
    if not use_precomputed:
        return

    pairs_file = getattr(args, 'distortion_pairs_file', None)
    if not pairs_file:
        raise ValueError('distortion_use_precomputed=1 时必须提供 distortion_pairs_file')
    if not os.path.isfile(pairs_file):
        raise FileNotFoundError(
            f"预生成失真 pairs 文件不存在: {pairs_file}. "
            "请先运行 scripts/gen_pairs.py 生成，或关闭 distortion_use_precomputed。"
        )


def train_ogb_link_adaptive(args, save_dir):
    dataset_name = canonicalize_ogb_link_dataset_name(args.dataset)
    if dataset_name is None:
        raise ValueError(f'Unsupported OGB dataset: {args.dataset}')

    if int(getattr(args, 'use_valedges_as_input', 0)) != 0:
        logging.warning('[OGB] use_valedges_as_input is not enabled in this clean protocol run.')

    data_root = getattr(args, 'data_root', './data')
    ogb = load_ogb_link_dataset(args, data_root=data_root, dataset_name=dataset_name)
    metric_name = ogb['official_metric']
    metric_token = _metric_token(metric_name)
    valid_metric_key = f'valid_{metric_name}'
    test_metric_key = f'test_{metric_name}'

    args.n_nodes, args.feat_dim = ogb['features'].shape
    args.nb_edges = int(ogb['train_edges'].size(0))
    args.nb_false_edges = max(1, args.nb_edges)

    model = LPModel(args)
    logging.info(str(model))

    curv_lr_scale = float(getattr(args, 'curv_lr_scale', 1.0))
    curv_pairs = extract_curvature_params(model, include_frozen=True)
    if curv_lr_scale != 1.0 and curv_pairs:
        base_params, curv_list = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            lname = name.lower()
            if lname == 'c' or lname.endswith('.c') or 'curv' in lname:
                curv_list.append(p)
            else:
                base_params.append(p)
        optimizer = getattr(optimizers, args.optimizer)(
            params=[
                {'params': base_params},
                {'params': curv_list, 'lr': args.lr * curv_lr_scale}
            ],
            lr=args.lr, weight_decay=args.weight_decay
        )
        curvature_params = curv_list
        logging.info(f"[Optimizer] curvature lr scale={curv_lr_scale}")
    else:
        optimizer = getattr(optimizers, args.optimizer)(
            params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        curvature_params = [p for _, p in curv_pairs] if curv_pairs else []

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )

    if int(args.cuda) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)

    features = ogb['features'].to(args.device)
    adj_train_norm = ogb['adj_train_norm'].to(args.device)
    train_edges = ogb['train_edges'].to(args.device)
    train_edge_index = ogb['train_edge_index'].to(args.device)

    controller = None
    if args.manifold == 'PoincareBall' and curvature_params and float(getattr(args, 'lambda_reg', 0.0)) > 0:
        adapt_fail_epoch = int(getattr(args, 'adapt_fail_epoch', 50))
        if adapt_fail_epoch <= 0:
            adapt_fail_epoch = None
        controller = CurvatureController(
            curvature_params=curvature_params,
            plateau_low=float(args.plateau_low),
            plateau_high=float(args.plateau_high),
            plateau_mean=float(args.plateau_mean),
            lambda_reg=float(args.lambda_reg),
            mu_reg=float(args.mu_reg),
            auto_mu=(float(args.mu_reg) == 0.0),
            warmup=int(args.curv_warmup),
            window=int(args.plateau_window),
            var_th=float(args.plateau_var_th),
            slack=float(args.plateau_slack),
            min_plateau_epochs=int(args.plateau_min_epochs),
            penalty_square=bool(int(args.curv_penalty_square)),
            enable_freeze=bool(int(args.curv_freeze)),
            adapt_fail_epoch=adapt_fail_epoch,
            adapt_lambda_scale=float(args.adapt_lambda_scale),
            device=args.device
        )
        logging.info(f"[CurvCtrl] λ={controller.lambda_reg} μ={controller.mu_reg} window=[{controller.L},{controller.H}] mean≈{controller.plateau_mean}")

    edge_batch_size = int(getattr(args, 'batch_size_edge_eval', 65536))
    num_neg_train = int(getattr(args, 'num_neg_train', 1))

    t0 = time.time()
    best_val_metrics = {metric_name: -1.0}
    best_test_metrics = {metric_name: -1.0}
    best_emb = None
    best_epoch = -1
    best_model_state = None
    counter = 0
    eval_counter = 0
    last_curv_value = None
    stress_at_best = None
    hist = {
        'epoch': [],
        'epoch_time_seconds': [],
        'train_loss': [],
        'val_loss': [],
        'val_metric_main': [],
        'K': [],
        'distortion_ctrl': [],
        valid_metric_key: [],
        test_metric_key: [],
    }

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()

        emb = model.encode(features, adj_train_norm)
        pos_total = train_edges.size(0)
        sample_size = min(pos_total, edge_batch_size)
        perm = torch.randperm(pos_total, device=train_edges.device)[:sample_size]
        pos_edge = train_edges[perm]
        neg_edge = sample_negative_edges(
            train_edge_index=train_edge_index,
            num_nodes=ogb['num_nodes'],
            num_neg_samples=sample_size * max(1, num_neg_train),
            device=torch.device(args.device),
        )

        pos_score = model.decode(emb, pos_edge)
        neg_score = model.decode(emb, neg_edge)
        base_loss = torch.nn.functional.binary_cross_entropy(pos_score, torch.ones_like(pos_score))
        base_loss = base_loss + torch.nn.functional.binary_cross_entropy(neg_score, torch.zeros_like(neg_score))
        loss = base_loss

        penalty_val = None
        pH = 0.0
        pL = 0.0
        if controller is not None:
            penalty_val, pH, pL = controller.penalty(epoch)
            if penalty_val is not None:
                loss = loss + penalty_val

        loss.backward()
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        scheduler.step()

        eval_now = (epoch + 1) % int(args.eval_freq) == 0

        if eval_now:
            eval_counter += 1
            test_eval_freq = int(getattr(args, 'eval_test_freq', 1))
            compute_test_now = test_eval_freq > 0 and (eval_counter % test_eval_freq == 0)
            model.eval()
            with torch.no_grad():
                emb_val = model.encode(features, adj_train_norm)
                eval_out = evaluate_ogb_link_dataset(
                    model=model,
                    embeddings=emb_val,
                    split_edge=ogb['split_edge'],
                    evaluator=ogb['evaluator'],
                    dataset_name=dataset_name,
                    batch_size_edge_eval=edge_batch_size,
                    include_valid=True,
                    include_test=compute_test_now,
                )

            val_hit = float(eval_out[valid_metric_key])
            test_hit = float(eval_out[test_metric_key]) if test_metric_key in eval_out else None
            val_m = {metric_name: val_hit}

            if (epoch + 1) % int(getattr(args, 'distortion_monitor_every', 1)) == 0:
                stress_val = maybe_monitor_dist(
                    args,
                    emb_val,
                    {'adj_train_norm': adj_train_norm},
                    model,
                )
            else:
                stress_val = None

            K_now = current_curvature_value(model, args.c, last_known_value=last_curv_value)
            if K_now is not None:
                last_curv_value = K_now

            print(
                f"EPOCH={epoch+1} CURV={(K_now if K_now is not None else -1.0):.6f} "
                f"VAL_{metric_token}={val_hit:.5f} STRESS={(stress_val if stress_val is not None else -1.0):.6f}"
            )

            if controller is not None and stress_val is not None:
                controller.update_distortion(stress_val, epoch)
                controller.maybe_adapt_lambda(epoch)
                controller.maybe_freeze(epoch)

            if (epoch + 1) % int(args.log_freq) == 0:
                parts = [
                    f"Ep:{epoch+1:04d}",
                    f"lr={scheduler.get_last_lr()[0]:.5e}",
                    f"base={base_loss.item():.4f}",
                    f"tot={loss.item():.4f}",
                ]
                if penalty_val is not None:
                    parts.append(f"pen={penalty_val.item():.4f}(H={pH:.4f},L={pL:.4f})")
                parts.append(f"K={(K_now if K_now is not None else -1):.4f}")
                if stress_val is not None:
                    parts.append(f"Dm={stress_val:.4f}")
                if controller is not None and controller.frozen:
                    parts.append(f"[FROZEN@{controller.freeze_epoch}]")
                logging.info(" ".join(parts))
                if test_hit is not None:
                    logging.info(f"Ep:{epoch+1:04d} val_{metric_name}: {val_hit:.6f} test_{metric_name}: {test_hit:.6f}")
                else:
                    logging.info(f"Ep:{epoch+1:04d} val_{metric_name}: {val_hit:.6f} test_{metric_name}: skipped")

            if val_hit > best_val_metrics[metric_name]:
                best_val_metrics = {metric_name: val_hit}
                if test_hit is not None:
                    best_test_metrics = {metric_name: test_hit}
                best_emb = emb_val.detach().cpu()
                best_epoch = epoch + 1
                stress_at_best = stress_val
                best_model_state = clone_state_dict_cpu(model)
                counter = 0
            else:
                counter += 1
                if counter >= args.patience and epoch > args.min_epochs:
                    logging.info('[EarlyStop] patience reached.')
                    break

            hist['epoch'].append(epoch + 1)
            hist['train_loss'].append(float(loss.detach().item()))
            hist['val_loss'].append(None)
            hist['val_metric_main'].append(val_hit)
            hist['K'].append(K_now)
            hist['distortion_ctrl'].append(stress_val)
            hist[valid_metric_key].append(val_hit)
            hist[test_metric_key].append(test_hit)

        if controller is not None:
            last_stress = hist['distortion_ctrl'][-1] if hist['distortion_ctrl'] else None
            controller.log_step(epoch, last_stress, penalty_val if penalty_val is not None else 0.0, pH, pL)

        epoch_time = time.time() - epoch_start
        if eval_now:
            hist['epoch_time_seconds'].append(round(epoch_time, 6))
            if (epoch + 1) % int(args.log_freq) == 0:
                logging.info(f"Ep:{epoch+1:04d} epoch_time={epoch_time:.2f}s")
            print(f"EPOCH_TIME={epoch+1}:{epoch_time:.4f}")
        elif (epoch + 1) % int(args.log_freq) == 0:
            logging.info(f"Ep:{epoch+1:04d} epoch_time={epoch_time:.2f}s")

    total_time = time.time() - t0
    avg_epoch_time = float(sum(hist['epoch_time_seconds']) / len(hist['epoch_time_seconds'])) if hist['epoch_time_seconds'] else 0.0
    logging.info(f"[Train] finished total_time={total_time:.2f}s best_epoch={best_epoch}")
    logging.info(f"[Train] avg_epoch_time={avg_epoch_time:.2f}s")

    final_distortion = None
    if int(getattr(args, 'compute_final_distortion', 1)) == 1:
        model.eval()
        with torch.no_grad():
            emb_to_use = best_emb.to(args.device) if best_emb is not None else model.encode(features, adj_train_norm)
        dres = compute_distortion(
            emb_to_use,
            adj_train_norm,
            manifold_name=args.manifold,
            model=model,
            **build_distortion_kwargs(args)
        )
        final_distortion = dres.get('distortion', None)
        logging.info(f"[FinalDist] d={final_distortion}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if best_model_state is not None and best_test_metrics[metric_name] < 0:
        model.eval()
        with torch.no_grad():
            emb_best = best_emb.to(args.device) if best_emb is not None else model.encode(features, adj_train_norm)
            final_test_out = evaluate_ogb_link_dataset(
                model=model,
                embeddings=emb_best,
                split_edge=ogb['split_edge'],
                evaluator=ogb['evaluator'],
                dataset_name=dataset_name,
                batch_size_edge_eval=edge_batch_size,
                include_valid=False,
                include_test=True,
            )
        best_test_metrics = {metric_name: float(final_test_out[test_metric_key])}

    logging.info(f"[OGB] dataset={dataset_name} metric={ogb['official_metric_display']}")
    logging.info(f"[OGB] best_valid_{metric_name}={best_val_metrics[metric_name]:.6f} at epoch={best_epoch}")
    logging.info(f"[OGB] test_{metric_name}(best_valid_ckpt)={best_test_metrics[metric_name]:.6f}")

    if args.save and save_dir:
        final_curv = current_curvature_value(model, args.c, last_known_value=last_curv_value)
        payload = {
            'dataset': dataset_name,
            'metric': metric_name,
            'final_curvature': final_curv,
            'best_epoch': best_epoch,
            'val_metrics': best_val_metrics,
            'test_metrics': best_test_metrics,
            'stress_final': final_distortion,
            'stress_at_best': stress_at_best,
            'controller_active': controller is not None,
            'runtime_seconds': round(total_time, 4),
            'avg_epoch_time_seconds': round(avg_epoch_time, 4),
            'seed': int(args.seed),
        }
        if controller:
            payload.update({
                'plateau_low': controller.L,
                'plateau_high': controller.H,
                'plateau_mean': controller.plateau_mean,
                'freeze_epoch': controller.freeze_epoch,
                'frozen': controller.frozen,
                'lambda_reg_final': controller.lambda_reg,
                'mu_reg_final': controller.mu_reg,
            })
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(payload, f, indent=2)
        with open(os.path.join(save_dir, 'history.json'), 'w') as f:
            json.dump(hist, f, indent=2)
        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        if best_model_state is not None:
            torch.save(best_model_state, os.path.join(save_dir, 'model.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        if best_emb is not None:
            np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.numpy())
        logging.info('[Save] artifacts saved.')

    print("\n" + "=" * 40)
    print('====== FINAL STANDARDIZED METRICS FOR PARSER ======')
    final_curv = current_curvature_value(model, args.c, last_known_value=last_curv_value)
    final_stress_for_print = stress_at_best if stress_at_best is not None else final_distortion
    print(f"DATASET={dataset_name}")
    print(f"OFFICIAL_METRIC={metric_name}")
    print(f"FINAL_C={final_curv if final_curv is not None else 'NA'}")
    print(f"VAL_METRIC={best_val_metrics[metric_name]:.5f}")
    print(f"STRESS={final_stress_for_print if final_stress_for_print is not None else 'NA'}")
    print(f"BEST_EPOCH={best_epoch}")
    print(f"TOTAL_TIME={total_time:.2f}")
    print(f"AVG_EPOCH_TIME={avg_epoch_time:.4f}")
    peak_gpu_mem_mb = get_peak_gpu_mem_mb(args.device)
    if peak_gpu_mem_mb is not None:
        print(f"PEAK_GPU_MEM_MB={peak_gpu_mem_mb:.2f}")
    print(f"TEST_{metric_token}={best_test_metrics[metric_name]:.5f}")
    if controller:
        print(f"PLATEAU_ENTER={controller.plateau_enter_epoch or 0}")
        print(f"FROZEN_AT={controller.freeze_epoch or 0}")
        print(f"ADAPT_FAIL={1 if getattr(controller, 'adapt_failed', False) else 0}")
    else:
        print('PLATEAU_ENTER=0')
        print('FROZEN_AT=0')
        print('ADAPT_FAIL=0')
    print("=" * 55 + "\n")

# -------------------------------------------------
# 主入口
# -------------------------------------------------
def main():
    args = parser.parse_args()
    torch.autograd.set_detect_anomaly(True)
    validate_distortion_config(args)

    # ====== Seed / Device ======
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else int(args.patience)

    # ====== Logging ======
    save_dir = None
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.environ.get('LOG_DIR','./logs'), args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(os.path.join(save_dir,'log.txt')),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO)

    logging.info(f"[Init] device={args.device} seed={args.seed}")

    if is_ogb_link_dataset(args):
        train_ogb_link_adaptive(args, save_dir)
        return

# ====== Data (Unified Logic) ======
    logging.info("[Data] Bypassing .pt cache, calling load_data directly for consistency.")
    datapath = os.environ.get('DATAPATH', './data')
    dataset_root = os.path.join(datapath, args.dataset)

# 为了确保万无一失，我们在这里强制开启特征归一化
# 这会覆盖掉任何命令行参数的默认值
    # args.normalize_feats = True 

# 直接调用核心数据加载函数，这将使用 data_utils.py 内部的 .pkl 缓存机制
    from utils.data_utils import load_data
    data = load_data(args, dataset_root)
    logging.info("[Data] Data loaded and processed via data_utils.py.")

    args.n_nodes, args.feat_dim = data['features'].shape

    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        Model = LPModel

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # ====== Model & Optimizer ======
    model = Model(args)
    logging.info(str(model))

    curv_lr_scale = float(getattr(args,'curv_lr_scale',1.0))
    curv_pairs = extract_curvature_params(model, include_frozen=True)
    if curv_lr_scale != 1.0 and curv_pairs:
        base_params, curv_list = [], []
        for name,p in model.named_parameters():
            if not p.requires_grad: 
                continue
            lname = name.lower()
            if lname == 'c' or lname.endswith('.c') or 'curv' in lname:
                curv_list.append(p)
            else:
                base_params.append(p)
        optimizer = getattr(optimizers, args.optimizer)(
            params=[
                {'params': base_params},
                {'params': curv_list, 'lr': args.lr * curv_lr_scale}
            ],
            lr=args.lr, weight_decay=args.weight_decay
        )
        curvature_params = curv_list
        logging.info(f"[Optimizer] curvature lr scale={curv_lr_scale}")
    else:
        optimizer = getattr(optimizers, args.optimizer)(
            params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        curvature_params = [p for _,p in curv_pairs] if curv_pairs else []

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )

    if int(args.cuda) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for k in list(data.keys()):
            if torch.is_tensor(data[k]):
                data[k] = data[k].to(args.device)

    # ====== Controller ======
    controller = None
    if args.manifold == 'PoincareBall' and curvature_params and float(getattr(args,'lambda_reg',0.0)) > 0:
        adapt_fail_epoch = int(getattr(args,'adapt_fail_epoch',50))
        if adapt_fail_epoch <= 0:
            adapt_fail_epoch = None
        controller = CurvatureController(
            curvature_params=curvature_params,
            plateau_low=float(args.plateau_low),
            plateau_high=float(args.plateau_high),
            plateau_mean=float(args.plateau_mean),
            lambda_reg=float(args.lambda_reg),
            mu_reg=float(args.mu_reg),
            auto_mu=(float(args.mu_reg)==0.0),
            warmup=int(args.curv_warmup),
            window=int(args.plateau_window),
            var_th=float(args.plateau_var_th),
            slack=float(args.plateau_slack),
            min_plateau_epochs=int(args.plateau_min_epochs),
            penalty_square=bool(int(args.curv_penalty_square)),
            enable_freeze=bool(int(args.curv_freeze)),
            adapt_fail_epoch=adapt_fail_epoch,
            adapt_lambda_scale=float(args.adapt_lambda_scale),
            device=args.device
        )
        logging.info(f"[CurvCtrl] λ={controller.lambda_reg} μ={controller.mu_reg} window=[{controller.L},{controller.H}] mean≈{controller.plateau_mean}")

    # ====== Training State & Loop ======
    t0 = time.time()
    best_val_metrics = model.init_metric_dict()
    best_emb = None
    best_epoch = -1
    best_model_state = None # <--- 在这里新增一行
    counter = 0
    last_curv_value = None
    stress_at_best = None
    hist = {"epoch":[], "epoch_time_seconds":[], "train_loss":[], "val_loss":[], "val_metric_main":[], "K":[], "distortion_ctrl":[]}
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        emb = model.encode(data['features'], data['adj_train_norm'])
        train_m = model.compute_metrics(emb, data, 'train')
        base_loss = train_m['loss']

        penalty_val = None; pH=0.0; pL=0.0
        if controller is not None:
            penalty_val, pH, pL = controller.penalty(epoch)
            if penalty_val is not None:
                train_m['loss'] = train_m['loss'] + penalty_val

        train_m['loss'].backward()
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        clamp_curvature_parameters(curvature_params, getattr(args, 'min_curvature', 1e-5))
        clamp_curvature_parameters(curvature_params, getattr(args, 'min_curvature', 1e-5))
        scheduler.step()

        eval_now = (epoch + 1) % int(args.eval_freq) == 0

        if eval_now:
            model.eval()
            with torch.no_grad():
                emb_val = model.encode(data['features'], data['adj_train_norm'])
                val_m = model.compute_metrics(emb_val, data, 'val')

            val_metric_val = pick_val_metric(val_m)

            if (epoch+1) % int(getattr(args,'distortion_monitor_every',1)) == 0:
                stress_val = maybe_monitor_dist(args, emb_val, data, model)
            else:
                stress_val = None

            K_now = current_curvature_value(model, args.c, last_known_value=last_curv_value)
            if K_now is not None:
                last_curv_value = K_now

            print(f"EPOCH={epoch+1} CURV={(K_now if K_now is not None else -1.0):.6f} VAL_ROC={val_metric_val:.5f} STRESS={(stress_val if stress_val is not None else -1.0):.6f}")

            if controller is not None and stress_val is not None:
                controller.update_distortion(stress_val, epoch)
                controller.maybe_adapt_lambda(epoch)
                controller.maybe_freeze(epoch)

            if (epoch+1) % int(args.log_freq) == 0:
                parts = [f"Ep:{epoch+1:04d}", f"lr={scheduler.get_last_lr()[0]:.5e}", f"base={base_loss.item():.4f}", f"tot={train_m['loss'].item():.4f}"]
                if penalty_val is not None:
                    parts.append(f"pen={penalty_val.item():.4f}(H={pH:.4f},L={pL:.4f})")
                parts.append(f"K={(K_now if K_now is not None else -1):.4f}")
                if stress_val is not None:
                    parts.append(f"Dm={stress_val:.4f}")
                if controller is not None and controller.frozen:
                    parts.append(f"[FROZEN@{controller.freeze_epoch}]")
                logging.info(" ".join(parts))
                logging.info(" ".join([f"Ep:{epoch+1:04d}", format_metrics(val_m,'val')]))

            if model.has_improved(best_val_metrics, val_m):
                best_val_metrics = val_m
                best_emb = emb_val.detach().cpu()
                best_epoch = epoch + 1
                stress_at_best = stress_val
                #保存最优
                best_model_state = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= args.patience and epoch > args.min_epochs:
                    logging.info("[EarlyStop] patience reached.")
                    break

            hist["epoch"].append(epoch+1)
            hist["train_loss"].append(float(train_m['loss']))
            hist["val_loss"].append(float(val_m['loss']))
            hist["val_metric_main"].append(val_metric_val)
            hist["K"].append(K_now)
            hist["distortion_ctrl"].append(stress_val)

        if controller is not None:
            last_stress = hist["distortion_ctrl"][-1] if hist["distortion_ctrl"] else None
            controller.log_step(epoch, last_stress, penalty_val if penalty_val is not None else 0.0, pH, pL)

        epoch_time = time.time() - epoch_start
        if eval_now:
            hist["epoch_time_seconds"].append(round(epoch_time, 6))
            if (epoch+1) % int(args.log_freq) == 0:
                logging.info(f"Ep:{epoch+1:04d} epoch_time={epoch_time:.2f}s")
            print(f"EPOCH_TIME={epoch+1}:{epoch_time:.4f}")
        elif (epoch+1) % int(args.log_freq) == 0:
            logging.info(f"Ep:{epoch+1:04d} epoch_time={epoch_time:.2f}s")

    total_time = time.time() - t0
    avg_epoch_time = float(sum(hist["epoch_time_seconds"]) / len(hist["epoch_time_seconds"])) if hist["epoch_time_seconds"] else 0.0
    logging.info(f"[Train] finished total_time={total_time:.2f}s best_epoch={best_epoch}")
    logging.info(f"[Train] avg_epoch_time={avg_epoch_time:.2f}s")

    # ====== Final Test & Distortion Evaluation ======
    test_metrics = {}
    model.eval()
    with torch.no_grad():
        if best_emb is not None:
            best_emb_dev = best_emb.to(args.device)
            test_metrics = model.compute_metrics(best_emb_dev, data, 'test')
        else:
            tmp_emb = model.encode(data['features'], data['adj_train_norm'])
            test_metrics = model.compute_metrics(tmp_emb, data, 'test')

    final_distortion = None
    if int(getattr(args,'compute_final_distortion',1)) == 1:
        # Re-use best embedding for final distortion calculation
        emb_to_use = best_emb.to(args.device) if best_emb is not None else model.encode(data['features'], data['adj_train_norm'])
        dres = compute_distortion(
            emb_to_use,
            data['adj_train_norm'],
            manifold_name=args.manifold,
            model=model,
            **build_distortion_kwargs(args)
        )
        final_distortion = dres.get('distortion', None)
        logging.info(f"[FinalDist] d={final_distortion}")

    logging.info(" ".join(["Val results:", format_metrics(best_val_metrics,'val')]))
    logging.info(" ".join(["Test results:", format_metrics(test_metrics,'test')]))
    
    # ====== Save artifacts ======
    if args.save and save_dir:
        final_curv = current_curvature_value(model, args.c, last_known_value=last_curv_value)
        payload = {
            "final_curvature": final_curv,
            "best_epoch": best_epoch,
            "val_metrics": {k: v.item() if torch.is_tensor(v) else v for k, v in best_val_metrics.items()},
            "test_metrics": {k: v.item() if torch.is_tensor(v) else v for k, v in test_metrics.items()},
            "stress_final": final_distortion, # Note: this is on final model state, not best
            "stress_at_best": stress_at_best,
            "controller_active": controller is not None,
            "runtime_seconds": round(total_time,4),
            "avg_epoch_time_seconds": round(avg_epoch_time,4),
            "seed": int(args.seed)
        }
        if controller:
            payload.update({
                "plateau_low": controller.L, "plateau_high": controller.H,
                "plateau_mean": controller.plateau_mean, "freeze_epoch": controller.freeze_epoch,
                "frozen": controller.frozen, "lambda_reg_final": controller.lambda_reg,
                "mu_reg_final": controller.mu_reg
            })
        with open(os.path.join(save_dir,"metrics.json"),'w') as f:
            json.dump(payload,f,indent=2)
        with open(os.path.join(save_dir,'history.json'),'w') as f:
            json.dump(hist,f,indent=2)
        json.dump(vars(args), open(os.path.join(save_dir,'config.json'),'w'))
        if best_model_state is not None:
            torch.save(best_model_state, os.path.join(save_dir, 'model.pth'))
        else: # Fallback in case no improvement was found
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        if best_emb is not None:
            np.save(os.path.join(save_dir,'embeddings.npy'), best_emb.numpy())
        logging.info("[Save] artifacts saved.")

    # ====== 最终标准化输出（供 bash 解析） ======
    print("\n" + "="*40)
    print("====== FINAL STANDARDIZED METRICS FOR PARSER ======")
    val_main = pick_val_metric(best_val_metrics)
    final_stress_for_print = stress_at_best if stress_at_best is not None else final_distortion
    final_curv = current_curvature_value(model, args.c, last_known_value=last_curv_value)
    
    print(f"FINAL_C={final_curv if final_curv is not None else 'NA'}")
    print(f"VAL_METRIC={val_main:.5f}")
    print(f"STRESS={final_stress_for_print if final_stress_for_print is not None else 'NA'}")
    print(f"BEST_EPOCH={best_epoch}")
    print(f"TOTAL_TIME={total_time:.2f}")
    print(f"AVG_EPOCH_TIME={avg_epoch_time:.4f}")
    peak_gpu_mem_mb = get_peak_gpu_mem_mb(args.device)
    if peak_gpu_mem_mb is not None:
        print(f"PEAK_GPU_MEM_MB={peak_gpu_mem_mb:.2f}")
    
    for key in ['roc','ap','acc','f1','loss']:
        if key in test_metrics:
            print(f"TEST_{key.upper()}={test_metrics[key]:.5f}")
    
    if controller:
        print(f"PLATEAU_ENTER={controller.plateau_enter_epoch or 0}")
        print(f"FROZEN_AT={controller.freeze_epoch or 0}")
        print(f"ADAPT_FAIL={1 if getattr(controller,'adapt_failed',False) else 0}")
    else:
        print("PLATEAU_ENTER=0")
        print("FROZEN_AT=0")
        print("ADAPT_FAIL=0")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()