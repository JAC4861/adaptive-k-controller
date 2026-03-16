from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time
import warnings

import numpy as np
import optimizers
import torch
import matplotlib.pyplot as plt
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data, sparse_mx_to_torch_sparse_tensor
from utils.distortion_utils import compute_distortion, create_precomputed_pairs, get_distortion_cache_path
from utils.train_utils import get_dir_name, format_metrics
from env import Env
from QLearning import *


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


def _as_float_list(values):
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        values = [values]
    converted = []
    for value in values:
        if torch.is_tensor(value):
            converted.append(float(value.detach().float().cpu().view(-1)[0].item()))
        else:
            converted.append(float(value))
    return converted


def _format_curvature_list(name, values):
    if not values:
        return '{}=[]'.format(name)
    return '{}=[{}]'.format(name, ', '.join('{:.6f}'.format(item) for item in values))


def _collect_curvature_snapshot(epoch, hgnn, ace, env):
    return {
        'epoch': int(epoch),
        'env_c1': _as_float_list(getattr(env, 'c1', [])),
        'env_c2': _as_float_list(getattr(env, 'c2', [])),
        'hgnn_c': _as_float_list(getattr(hgnn, 'c', [])),
        'ace_c': _as_float_list(getattr(ace, 'c', [])),
        'hgnn_encoder_curvatures': _as_float_list(getattr(getattr(hgnn, 'encoder', None), 'curvatures', [])),
        'ace_encoder_curvatures': _as_float_list(getattr(getattr(ace, 'encoder', None), 'curvatures', [])),
    }


def _log_curvature_snapshot(snapshot):
    logging.info(
        '[Curv] epoch=%04d %s %s %s %s %s %s'
        % (
            snapshot['epoch'],
            _format_curvature_list('env_c1', snapshot['env_c1']),
            _format_curvature_list('env_c2', snapshot['env_c2']),
            _format_curvature_list('hgnn_c', snapshot['hgnn_c']),
            _format_curvature_list('ace_c', snapshot['ace_c']),
            _format_curvature_list('hgnn_encoder', snapshot['hgnn_encoder_curvatures']),
            _format_curvature_list('ace_encoder', snapshot['ace_encoder_curvatures']),
        )
    )


def _safe_metric_value(metric_dict, key):
    if metric_dict is None or key not in metric_dict:
        return None
    value = metric_dict[key]
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def _maybe_prepare_distortion_pairs(args, datapath, data):
    if args.task != 'lp':
        return
    if not bool(int(getattr(args, 'distortion_use_precomputed', 1))):
        return
    if getattr(args, 'distortion_pairs_file', None):
        return

    pair_count = int(getattr(args, 'monitor_sample_pairs', 8000))
    max_hop = int(getattr(args, 'distortion_max_hop', 8))
    cache_path = get_distortion_cache_path(
        datapath,
        getattr(args, 'split_seed', 1234),
        getattr(args, 'val_prop', 0.05),
        getattr(args, 'test_prop', 0.1),
        max_hop,
        pair_count,
    )

    if not os.path.exists(cache_path):
        graph_adj = sparse_mx_to_torch_sparse_tensor(data['adj_train']).coalesce()
        created_path, created_pairs = create_precomputed_pairs(
            graph_adj,
            cache_path,
            max_pairs=pair_count,
            max_hop=max_hop,
            seed=int(getattr(args, 'distortion_seed', 1234)),
        )
        logging.info('[DistPairs] generated %s pairs -> %s' % (created_pairs, created_path))
    else:
        logging.info('[DistPairs] reusing %s' % cache_path)

    args.distortion_pairs_file = cache_path
    args.distortion_pairs_limit = pair_count

def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))
    warnings.filterwarnings(action='ignore')

    # Load data
    datapath = os.path.join(os.environ['DATAPATH'], args.dataset)
    data = load_data(args, datapath)
    _maybe_prepare_distortion_pairs(args, datapath, data)
    args.n_nodes, args.feat_dim = data['features'].shape
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    # Initialize RL environment
    lr_q = args.lr_q                        # RL Learning Rate
    action_space1 = ['reject', 'accept']    # HGNN action space
    action_space2 = [                       # ACE  action space
        'r, r', 'r, a',
        'a, r', 'a, a'
    ]
    joint_actions = []
    for i in range(len(action_space1)):     # Joint action space
        for j in range(len(action_space2)):
            joint_actions.append((i, j))
    env = Env(theta=args.theta, initial_c=args.c)
    Agent1 = QLearningTable(actions=list(range(len(action_space1))), joint=joint_actions, start=args.start_q, learning_rate=lr_q)
    Agent2 = QLearningTable(actions=list(range(len(action_space2))), joint=joint_actions, start=args.start_q, learning_rate=lr_q)

    hgnn = Model(args)          # Agent1 HGNN
    ace = Model(args)           # Agent2 ACE
    logging.info(str(hgnn))
    optimizer = getattr(optimizers, args.optimizer)(params=hgnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in hgnn.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0 :
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        hgnn = hgnn.to(args.device)
        ace = ace.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)

    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = hgnn.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    val_metric_record = []
    train_metric_record = []
    distortion_record = []
    curvature_record = []
    epoch_metrics_record = []
    best_distortion = None
    best_epoch = None
    for epoch in range(args.epochs):
        t = time.time()
        hgnn.train()
        ace.train()
        optimizer.zero_grad()

        # train model with RL and return Agent1's train metrics
        # Terminate mechanism
        if epoch > args.start_q + 30:
            r1 = np.array(env.c1_record)[-30:-1, 0]
            r2 = np.array(env.c1_record)[-30:-1, 1]
            if abs(max(r1) - min(r1)) <= 0.03 and not env.stop[0]:
                env.stop[0] = True
                print("Layer1 RL terminate at {:.3f}.".format(env.c1_record[-1][0]))
                counter = args.patience // 2
            if abs(max(r2) - min(r2)) <= 0.03 and not env.stop[1]:
                env.stop[1] = True
                print("Layer2 RL terminate at {:.3f}.".format(env.c1_record[-1][1]))
                counter = args.patience // 2

        train_metrics = hgnn.train_with_RL(env, Agent1, Agent2, data, epoch, ace)
        train_metric_record.append(train_metrics[hgnn.key_param])
        train_metrics['loss'].backward()
        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(hgnn.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        lr_scheduler.step()

        curvature_snapshot = _collect_curvature_snapshot(epoch + 1, hgnn, ace, env)
        curvature_record.append(curvature_snapshot)
        _log_curvature_snapshot(curvature_snapshot)

        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {}'.format(lr_scheduler.get_lr()[0]),
                                   format_metrics(train_metrics, 'train'),
                                   'time: {:.4f}s'.format(time.time() - t)
                                   ]))
        if (epoch + 1) % args.eval_freq == 0:
            hgnn.eval()
            embeddings = hgnn.encode(data['features'], data['adj_train_norm'])
            val_metrics = hgnn.compute_metrics(embeddings, data, 'val')
            val_metric_record.append(val_metrics[hgnn.key_param])
            distortion_value = None
            distortion_eval_every = int(getattr(args, 'distortion_eval_every', 1))
            if distortion_eval_every > 0 and ((epoch + 1) // args.eval_freq) % distortion_eval_every == 0:
                dist_start = time.time()
                distortion_info = compute_distortion(
                        embeddings.detach(),
                        data['adj_train_norm'],
                        manifold_name=args.manifold,
                        **build_distortion_kwargs(args, sample_pairs=int(getattr(args, 'monitor_sample_pairs', 2000)))
                )
                distortion_value = distortion_info.get('distortion', None)
                distortion_record.append([
                    epoch + 1,
                    distortion_value,
                    distortion_info.get('pairs_used', 0),
                    time.time() - dist_start,
                ])
                logging.info(
                        '[Dist] epoch=%04d distortion=%s pairs=%s mode=%s time=%.4fs'
                        % (
                            epoch + 1,
                            'nan' if distortion_value is None else '{:.6f}'.format(distortion_value),
                            distortion_info.get('pairs_used', 0),
                            distortion_info.get('pairs_mode', 'unknown'),
                            time.time() - dist_start,
                        )
                )

            epoch_metrics = {
                'epoch': int(epoch + 1),
                'train_loss': _safe_metric_value(train_metrics, 'loss'),
                'train_roc': _safe_metric_value(train_metrics, 'roc'),
                'train_ap': _safe_metric_value(train_metrics, 'ap'),
                'val_loss': _safe_metric_value(val_metrics, 'loss'),
                'val_roc': _safe_metric_value(val_metrics, 'roc'),
                'val_ap': _safe_metric_value(val_metrics, 'ap'),
                'distortion': distortion_value,
                'env_c1': curvature_snapshot['env_c1'],
                'env_c2': curvature_snapshot['env_c2'],
                'hgnn_c': curvature_snapshot['hgnn_c'],
                'ace_c': curvature_snapshot['ace_c'],
            }
            epoch_metrics_record.append(epoch_metrics)
            print(
                'EPOCH={epoch} TRAIN_LOSS={train_loss} TRAIN_ROC={train_roc} TRAIN_AP={train_ap} '
                'VAL_LOSS={val_loss} VAL_ROC={val_roc} VAL_AP={val_ap} STRESS={stress} '
                'C1L1={c1l1} C1L2={c1l2} C2L1={c2l1} C2L2={c2l2}'.format(
                    epoch=epoch_metrics['epoch'],
                    train_loss='NA' if epoch_metrics['train_loss'] is None else '{:.6f}'.format(epoch_metrics['train_loss']),
                    train_roc='NA' if epoch_metrics['train_roc'] is None else '{:.6f}'.format(epoch_metrics['train_roc']),
                    train_ap='NA' if epoch_metrics['train_ap'] is None else '{:.6f}'.format(epoch_metrics['train_ap']),
                    val_loss='NA' if epoch_metrics['val_loss'] is None else '{:.6f}'.format(epoch_metrics['val_loss']),
                    val_roc='NA' if epoch_metrics['val_roc'] is None else '{:.6f}'.format(epoch_metrics['val_roc']),
                    val_ap='NA' if epoch_metrics['val_ap'] is None else '{:.6f}'.format(epoch_metrics['val_ap']),
                    stress='NA' if epoch_metrics['distortion'] is None else '{:.6f}'.format(epoch_metrics['distortion']),
                    c1l1='NA' if len(epoch_metrics['env_c1']) < 1 else '{:.6f}'.format(epoch_metrics['env_c1'][0]),
                    c1l2='NA' if len(epoch_metrics['env_c1']) < 2 else '{:.6f}'.format(epoch_metrics['env_c1'][1]),
                    c2l1='NA' if len(epoch_metrics['env_c2']) < 1 else '{:.6f}'.format(epoch_metrics['env_c2'][0]),
                    c2l2='NA' if len(epoch_metrics['env_c2']) < 2 else '{:.6f}'.format(epoch_metrics['env_c2'][1]),
                )
            )
            
            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))

            if hgnn.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = hgnn.compute_metrics(embeddings, data, 'test')
                best_emb = embeddings.cpu()
                best_distortion = distortion_value
                best_epoch = epoch + 1
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter >= args.patience and epoch > args.min_epochs and all(env.stop):
                    logging.info("Early stopping")
                    break

    logging.info("Optimization Finished!")
    total_time = time.time() - t_total
    logging.info("Total time elapsed: {:.4f}s".format(total_time))
    if not best_test_metrics:
        hgnn.eval()
        best_emb = hgnn.encode(data['features'], data['adj_train_norm'])
        best_test_metrics = hgnn.compute_metrics(best_emb, data, 'test')
        best_epoch = args.epochs
    final_distortion = best_distortion
    if int(getattr(args, 'compute_final_distortion', 1)) == 1 and best_emb is not None:
        distortion_info = compute_distortion(
                best_emb.to(args.device),
                data['adj_train_norm'],
                manifold_name=args.manifold,
                **build_distortion_kwargs(args, sample_pairs=int(getattr(args, 'monitor_sample_pairs', 2000)))
        )
        final_distortion = distortion_info.get('distortion', final_distortion)
        logging.info(
                '[FinalDist] distortion=%s pairs=%s mode=%s'
                % (
                    'nan' if final_distortion is None else '{:.6f}'.format(final_distortion),
                    distortion_info.get('pairs_used', 0),
                    distortion_info.get('pairs_mode', 'unknown'),
                )
        )
    logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
    logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))
    metrics_payload = {
        'dataset': args.dataset,
        'task': args.task,
        'model': args.model,
        'manifold': args.manifold,
        'seed': int(args.seed),
        'split_seed': int(getattr(args, 'split_seed', args.seed)),
        'best_epoch': best_epoch,
        'total_time_seconds': round(float(total_time), 6),
        'best_val_loss': _safe_metric_value(best_val_metrics, 'loss'),
        'best_val_roc': _safe_metric_value(best_val_metrics, 'roc'),
        'best_val_ap': _safe_metric_value(best_val_metrics, 'ap'),
        'best_test_loss': _safe_metric_value(best_test_metrics, 'loss'),
        'best_test_roc': _safe_metric_value(best_test_metrics, 'roc'),
        'best_test_ap': _safe_metric_value(best_test_metrics, 'ap'),
        'final_distortion': None if final_distortion is None else float(final_distortion),
        'final_env_c1': _as_float_list(getattr(env, 'c1', [])),
        'final_env_c2': _as_float_list(getattr(env, 'c2', [])),
        'distortion_pairs_file': getattr(args, 'distortion_pairs_file', None),
        'distortion_pairs_limit': int(getattr(args, 'distortion_pairs_limit', 0)) if getattr(args, 'distortion_pairs_limit', None) is not None else None,
        'save_dir': save_dir,
    }
    print('BEST_EPOCH={}'.format(best_epoch if best_epoch is not None else 'NA'))
    print('VAL_ROC={}'.format('NA' if metrics_payload['best_val_roc'] is None else '{:.6f}'.format(metrics_payload['best_val_roc'])))
    print('VAL_AP={}'.format('NA' if metrics_payload['best_val_ap'] is None else '{:.6f}'.format(metrics_payload['best_val_ap'])))
    print('TEST_ROC={}'.format('NA' if metrics_payload['best_test_roc'] is None else '{:.6f}'.format(metrics_payload['best_test_roc'])))
    print('TEST_AP={}'.format('NA' if metrics_payload['best_test_ap'] is None else '{:.6f}'.format(metrics_payload['best_test_ap'])))
    print('STRESS={}'.format('NA' if final_distortion is None else '{:.6f}'.format(final_distortion)))
    print('FINAL_C1={}'.format(','.join('{:.6f}'.format(value) for value in metrics_payload['final_env_c1']) if metrics_payload['final_env_c1'] else 'NA'))
    print('FINAL_C2={}'.format(','.join('{:.6f}'.format(value) for value in metrics_payload['final_env_c2']) if metrics_payload['final_env_c2'] else 'NA'))
    print('TOTAL_TIME={:.6f}'.format(metrics_payload['total_time_seconds']))
    print('SAVE_DIR={}'.format(save_dir if save_dir is not None else 'NA'))
    if args.save:
        # Save embeddings and attentions
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
        if hasattr(hgnn.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(hgnn.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        # Save model
        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        torch.save(hgnn.state_dict(), os.path.join(save_dir, 'model.pth'))

        # Save curvature record and figures
        np.save(os.path.join(save_dir, 'curv1.npy'), np.array(env.c1_record))
        np.save(os.path.join(save_dir, 'curv2.npy'), np.array(env.c2_record))

        # Save acc record
        np.save(os.path.join(save_dir, 'metric_record.npy'), np.array([train_metric_record, val_metric_record]))
        if distortion_record:
            np.save(os.path.join(save_dir, 'distortion_record.npy'), np.array(distortion_record, dtype=np.float64))
        if curvature_record:
            with open(os.path.join(save_dir, 'curvature_record.json'), 'w') as handle:
                json.dump(curvature_record, handle)
        if epoch_metrics_record:
            with open(os.path.join(save_dir, 'epoch_metrics.json'), 'w') as handle:
                json.dump(epoch_metrics_record, handle)
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as handle:
            json.dump(metrics_payload, handle, indent=2)

        logging.info("Agent1: {}, Agent2: {}".format(env.c1, env.c2))
        logging.info(f"Saved model in {save_dir}")

if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
