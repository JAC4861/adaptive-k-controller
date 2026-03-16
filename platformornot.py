import glob, json, os, math
import numpy as np

def load_points(root):
    pts = []
    for f in sorted(glob.glob(os.path.join(root, "absK_*", "metrics.json"))):
        with open(f) as fh:
            d = json.load(fh)
        k = d.get("absK") or d.get("final_curvature")
        s = d.get("stress")
        if k is not None and s is not None:
            pts.append((float(k), float(s)))
    pts.sort(key=lambda x: x[0])
    return pts

def simple_plateau(points, rel_threshold=0.05):
    # 1) 最后两点是否已满足平台
    if len(points) < 3: 
        return None, {}
    ks, ss = zip(*points)
    # 基本指标
    end_rel = (ss[-2] - ss[-1]) / ss[-2] if ss[-2] != 0 else 0
    # 2) 逐点相对下降率
    rel_drops = []
    for i in range(1, len(ss)):
        rel_drops.append((ss[i-1]-ss[i]) / ss[i-1])
    # 3) 连续两次 < 阈值
    plateau_start = None
    for i in range(1, len(rel_drops)):
        if rel_drops[i-1] < rel_threshold and rel_drops[i] < rel_threshold:
            plateau_start = ks[i+0]  # 对应第二个小的那个之后的 |K|
            break
    return plateau_start, {
        "ks": ks,
        "ss": ss,
        "rel_drops": rel_drops,
        "end_pair_rel_diff": end_rel
    }

def window_plateau(points, w=2, rel_threshold=0.05):
    ks, ss = zip(*points)
    windows = []
    for i in range(len(ss)-w+1):
        window_vals = ss[i:i+w]
        delta = (max(window_vals)-min(window_vals))/max(window_vals)
        windows.append((ks[i], delta))
    # 找最早持续满足的平台
    plateau_start = None
    for j,(k,delta) in enumerate(windows):
        if delta < rel_threshold:
            # 要求后续全部窗口都 < 阈值（或放宽允许最后一个例外）
            tail = [d for _,d in windows[j:]]
            if all(d < rel_threshold for d in tail):
                plateau_start = k
                break
    return plateau_start, {"windows": windows}

# 使用示例
root = "runs/pilot_cora"
pts = load_points(root)
print("Loaded points:", pts)

p1, info1 = simple_plateau(pts, rel_threshold=0.05)
print("[Simple] plateau_start =", p1)
print("Relative drops:", info1.get("rel_drops"))

p2, info2 = window_plateau(pts, w=2, rel_threshold=0.05)
print("[Window] plateau_start =", p2)
print("Window deltas:", info2.get("windows"))