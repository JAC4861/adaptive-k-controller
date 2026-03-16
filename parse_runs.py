#!/usr/bin/env python
import re, csv, argparse, pathlib

PATTERNS = {
    "final_c": re.compile(r"FINAL_C=([\d\.eE+-]+)"),
    "val_metric": re.compile(r"VAL_METRIC=([\d\.eE+-]+)"),
    "stress": re.compile(r"STRESS=([\d\.eE+-]+)"),
    "best_epoch": re.compile(r"BEST_EPOCH=([\d\.eE+-]+)"),
    "total_time": re.compile(r"TOTAL_TIME=([\d\.eE+-]+)"),
    "plateau": re.compile(r"PLATEAU_ENTER=([0-9]+)"),
    "frozen": re.compile(r"FROZEN_AT=([0-9]+)"),
    "fail": re.compile(r"ADAPT_FAIL=([0-9]+)")
}

def parse_log(path: pathlib.Path):
    txt = path.read_text(errors='ignore')
    def grab(name):
        m = PATTERNS[name].findall(txt)
        return m[-1] if m else ""
    return {
        "log_file": str(path),
        "final_c": grab("final_c"),
        "val_metric": grab("val_metric"),
        "stress": grab("stress"),
        "best_epoch": grab("best_epoch"),
        "total_time": grab("total_time"),
        "plateau": grab("plateau"),
        "frozen": grab("frozen"),
        "fail": grab("fail"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    rows=[]
    for p in pathlib.Path(args.log_dir).glob("*.log"):
        rows.append(parse_log(p))

    with open(args.out_csv,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["log_file"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[INFO] Parsed {len(rows)} logs -> {args.out_csv}")

if __name__ == "__main__":
    main()