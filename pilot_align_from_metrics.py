#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 pilot 阶段生成的 metrics.json 文件集合中自动估计最优曲率 K*，计算 μ，并输出 pilot_alignment.json。

核心功能：
1. 扫描 <root>/absK_*/metrics.json
2. 根据策略选出 K*：
   - metric-max                : 选择 val_metric 最大
   - loss+rho*stress-min       : 最小化 (val_loss + rho * stress)
   - val_loss-min              : 最小 val_loss
   - stress-min                : 最小 stress（常用于失真主导调试）
   - plateau-aware             : 先找 loss 最小的前 p% 再在其中择最小 stress
   - summary-json              : 直接读取 root/pilot_summary.json 的 absK_star 字段
3. 输出 pilot_alignment.json，字段包含：K_star_est, lambda_used, mu_recommend, eps_used, strategy, points 等
4. 选项 --auto-full 可在生成 alignment 后自动启动一次 full 训练（需要 train.py 支持 --pilot-align）

用法示例：
  基于 loss + stress：
    python scripts/pilot_align_from_metrics.py \
       --root runs/pilot_cora \
       --lambda 0.05 \
       --eps 1e-3 \
       --strategy loss+rho*stress-min \
       --rho 1.0

  基于 val_metric 最大：
    python scripts/pilot_align_from_metrics.py \
       --root runs/pilot_cora \
       --lambda 0.05 \
       --eps 1e-3 \
       --strategy metric-max

  直接用 aggregate_pilot.py 生成的 pilot_summary.json：
    python scripts/pilot_align_from_metrics.py \
       --root runs/pilot_cora \
       --lambda 0.05 \
       --eps 1e-3 \
       --strategy summary-json

  一键自动进入 full 训练（学习曲率）：
    python scripts/pilot_align_from_metrics.py \
       --root runs/pilot_cora \
       --lambda 0.05 \
       --eps 1e-3 \
       --strategy loss+rho*stress-min \
       --rho 1.0 \
       --auto-full \
       --full-save-dir runs/full_cora_reg \
       --full-extra "--task lp --dataset cora --model HGCN --manifold PoincareBall --dim 16 --num-layers 2 --lr 0.01 --dropout 0.5 --weight-decay 0.001 --epochs 200 --penalty-mode two-sided --c None"

注意：
- metrics.json 至少应包含：absK, val_loss, val_metric, stress（stress 可为空，取决于你的实现）
- 若某策略需要的字段缺失，会跳过对应点
- μ = λ (K* + ε)
"""

import os
import json
import glob
import time
import math
import argparse
import subprocess
from typing import List, Dict, Any

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='pilot 根目录，例如 runs/pilot_cora')
    ap.add_argument('--lambda', dest='lam', type=float, required=True, help='λ，用于计算 μ')
    ap.add_argument('--eps', dest='eps', type=float, default=1e-3, help='ε（平滑）')
    ap.add_argument('--strategy', default='loss+rho*stress-min',
                    choices=[
                        'metric-max',
                        'loss+rho*stress-min',
                        'val_loss-min',
                        'stress-min',
                        'plateau-aware',
                        'summary-json'
                    ],
                    help='选择 K* 的策略')
    ap.add_argument('--rho', type=float, default=1.0,
                    help='loss+rho*stress-min 策略中的 ρ')
    ap.add_argument('--plateau-top-pct', type=float, default=30.0,
                    help='plateau-aware 策略：取 val_loss 最低的前 p% 再选 stress 最低')
    ap.add_argument('--summary-name', default='pilot_summary.json',
                    help='strategy=summary-json 时读取的文件名')
    ap.add_argument('--out', default='pilot_alignment.json', help='输出 JSON 路径')
    ap.add_argument('--min-points', type=int, default=2, help='有效点少于此值则视为失败')
    ap.add_argument('--require-stress', type=int, default=0,
                    help='1=过滤掉没有 stress 的点（metric-max 等策略默认不必）')
    ap.add_argument('--verbose', action='store_true', help='打印中间详细信息')
    # 自动进入 full 训练相关
    ap.add_argument('--auto-full', action='store_true',
                    help='生成 alignment 后立即启动一次 full 训练')
    ap.add_argument('--full-script', default='python train.py',
                    help='执行 full 训练的命令前缀（可改为 torchrun 等）')
    ap.add_argument('--full-save-dir', default=None,
                    help='full 训练结果保存目录（若不传自动命名）')
    ap.add_argument('--full-seed', type=int, default=42, help='full 训练 seed')
    ap.add_argument('--full-extra', type=str, default='',
                    help='传给 full 训练的额外参数字符串（需包含最基本任务/模型/数据参数）')
    ap.add_argument('--mu-decimals', type=int, default=6, help='μ 输出保留小数位')
    return ap.parse_args()

def load_metrics_points(root: str, require_stress: bool, verbose: bool) -> List[Dict[str, Any]]:
    pattern = os.path.join(root, 'absK_*', 'metrics.json')
    files = sorted(glob.glob(pattern))
    points = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            k = d.get('absK') if d.get('absK') is not None else d.get('final_curvature')
            if k is None:
                if verbose:
                    print(f"[SKIP] {f} 无 absK/final_curvature")
                continue
            val_loss = d.get('val_loss')
            val_metric = d.get('val_metric')
            stress = d.get('stress')
            if require_stress and stress is None:
                if verbose:
                    print(f"[SKIP] {f} 无 stress (require_stress=1)")
                continue
            points.append({
                "absK": float(k),
                "val_loss": float(val_loss) if val_loss is not None else None,
                "val_metric": float(val_metric) if val_metric is not None else None,
                "stress": float(stress) if stress is not None else None,
                "path": f
            })
        except Exception as e:
            if verbose:
                print(f"[WARN] 读取失败 {f}: {e}")
    return points

def select_k_star(points: List[Dict[str, Any]], strategy: str, rho: float,
                  plateau_top_pct: float, verbose: bool):
    usable = []
    if strategy == 'metric-max':
        for p in points:
            if p['val_metric'] is not None:
                p['_score'] = p['val_metric']
                usable.append(p)
    elif strategy == 'val_loss-min':
        for p in points:
            if p['val_loss'] is not None:
                p['_score'] = -p['val_loss']  # 取负后统一用 max
                usable.append(p)
    elif strategy == 'stress-min':
        for p in points:
            if p['stress'] is not None:
                p['_score'] = -p['stress']
                usable.append(p)
    elif strategy == 'loss+rho*stress-min':
        for p in points:
            if p['val_loss'] is not None and p['stress'] is not None:
                p['_score'] = -(p['val_loss'] + rho * p['stress'])
                usable.append(p)
    elif strategy == 'plateau-aware':
        # 先按 val_loss 升序，再截取前 p%，再选 stress 最低
        pl = [p for p in points if p['val_loss'] is not None and p['stress'] is not None]
        pl.sort(key=lambda x: x['val_loss'])
        if not pl:
            return None, []
        k_cut = max(1, int(math.ceil(len(pl) * plateau_top_pct / 100.0)))
        subgroup = pl[:k_cut]
        # stress 最低
        subgroup.sort(key=lambda x: x['stress'])
        best = subgroup[0]
        best['_score'] = -best['stress']
        if verbose:
            print(f"[plateau-aware] 总点={len(pl)}, 截取前{k_cut}个, 选 stress 最低 absK={best['absK']}")
        return best, subgroup
    else:
        raise ValueError(f"Unsupported strategy {strategy}")

    if not usable:
        return None, []
    usable.sort(key=lambda x: x['_score'], reverse=True)
    return usable[0], usable

def write_alignment(out_path: str,
                    k_star: float,
                    lam: float,
                    eps: float,
                    strategy: str,
                    rho: float,
                    raw_points: List[Dict[str, Any]],
                    mu_decimals: int,
                    extra: dict):
    mu_rec = lam * (k_star + eps) if k_star is not None else None
    payload = {
        "K_star_est": k_star,
        "lambda_used": lam,
        "mu_recommend": round(mu_rec, mu_decimals) if mu_rec is not None else None,
        "eps_used": eps,
        "strategy": strategy,
        "rho": rho if 'rho' in extra.get('strategy_reqs', []) or strategy == 'loss+rho*stress-min' else None,
        "timestamp": time.time(),
        "points": [
            {
                "absK": p['absK'],
                "val_loss": p['val_loss'],
                "val_metric": p['val_metric'],
                "stress": p['stress'],
                "score": p.get('_score')
            } for p in raw_points
        ],
    }
    payload.update(extra.get('append', {}))
    tmp = out_path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_path)
    print(f"[pilot-align] 写出 {out_path}")
    if k_star is not None:
        print(f"[pilot-align] 选出 K*={k_star:.6f} => μ={payload['mu_recommend']}")
    else:
        print("[pilot-align] 未能估计 K*")

def auto_run_full(alignment_json: str,
                  cmd_prefix: str,
                  save_dir: str,
                  seed: int,
                  lam: float,
                  eps: float,
                  full_extra: str):
    # alignment_json 内含 mu_recommend
    with open(alignment_json) as f:
        align = json.load(f)
    k_star = align.get("K_star_est")
    mu_rec = align.get("mu_recommend")
    if k_star is None or mu_rec is None:
        print("[auto-full] 对齐文件缺少 K* 或 μ，跳过自动 full。")
        return
    if not save_dir:
        save_dir = f"runs/full_auto_{int(time.time())}"
    os.makedirs(save_dir, exist_ok=True)
    # 传参：--pilot-align 1 --pilot-json alignment_json --mu-reg 0 (让 train.py 自动用 μ)
    # 你也可以直接传 --mu-reg {mu_rec} 而不启用 pilot-align
    full_cmd = f"""{cmd_prefix} \
 {full_extra} \
 --seed {seed} \
 --lambda-reg {lam} \
 --mu-reg 0.0 \
 --epsilon-reg {eps} \
 --pilot-align 1 \
 --pilot-json {alignment_json} \
 --save 1 \
 --save-dir {save_dir}"""
    print("[auto-full] 执行命令：")
    print(full_cmd)
    ret = subprocess.call(full_cmd, shell=True)
    if ret != 0:
        print(f"[auto-full] Full 训练失败 (返回码={ret})")
    else:
        print(f"[auto-full] Full 训练完成，输出目录：{save_dir}")

def main():
    args = parse_args()

    # Strategy = summary-json
    if args.strategy == 'summary-json':
        summary_path = os.path.join(args.root, args.summary_name)
        if not os.path.exists(summary_path):
            print(f"[ERR] 未找到 {summary_path}")
            write_alignment(
                args.out, None, args.lam, args.eps, args.strategy, args.rho, [],
                args.mu_decimals,
                extra={"append": {"error": "summary_missing"}}
            )
            return
        sj = json.load(open(summary_path))
        k_star = sj.get('absK_star') or sj.get('K_star_est')
        if k_star is None:
            write_alignment(
                args.out, None, args.lam, args.eps, args.strategy, args.rho, [],
                args.mu_decimals,
                extra={"append": {"error": "absK_star_missing"}}
            )
            return
        write_alignment(
            args.out, float(k_star), args.lam, args.eps, args.strategy, args.rho, [],
            args.mu_decimals,
            extra={"append": {"source_summary": summary_path}}
        )
        if args.auto_full:
            auto_run_full(args.out, args.full_script, args.full_save_dir, args.full_seed,
                          args.lam, args.eps, args.full_extra)
        return

    # 普通策略：直接读 metrics.json
    points = load_metrics_points(args.root, require_stress=bool(args.require_stress), verbose=args.verbose)
    if len(points) < args.min_points:
        print(f"[ERR] 有效点数不足 {len(points)} < {args.min_points}")
        write_alignment(
            args.out, None, args.lam, args.eps, args.strategy, args.rho, points,
            args.mu_decimals,
            extra={"append": {"error": "insufficient_points", "count": len(points)}}
        )
        return

    best, usable = select_k_star(points, args.strategy, args.rho, args.plateau_top_pct, args.verbose)
    if best is None:
        write_alignment(
            args.out, None, args.lam, args.eps, args.strategy, args.rho, usable,
            args.mu_decimals,
            extra={"append": {"error": "selection_failed"}}
        )
    else:
        write_alignment(
            args.out, best['absK'], args.lam, args.eps, args.strategy, args.rho, usable,
            args.mu_decimals, extra={}
        )
        if args.auto_full:
            auto_run_full(args.out, args.full_script, args.full_save_dir, args.full_seed,
                          args.lam, args.eps, args.full_extra)

if __name__ == '__main__':
    main()