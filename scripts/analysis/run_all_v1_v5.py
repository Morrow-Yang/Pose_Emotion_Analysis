#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_v1_v5.py — Run your EXACT v1–v5 analysis scripts (CLI versions)

This wrapper calls analysis_v1.py ... analysis_v5.py with the same --root/--json_name/--out_dir
so you can reproduce your original results on any dataset folder layout:

ROOT/
  emotionA/alphapose-results.json
  emotionB/alphapose-results.json
  ...

Example:
python run_all_v1_v5.py --root /path/to/alphapose_outputs_by_label --json_name alphapose-results.json --out_base out_pose_analysis

Outputs:
out_pose_analysis/
  v1/  (outputs of analysis_v1)
  v2/
  v3/
  v4/
  v5/
"""

import argparse
import subprocess
from pathlib import Path
import sys

SCRIPTS = [
    ("v1", "analysis_v1.py"),
    ("v2", "analysis_v2.py"),
    ("v3", "analysis_v3.py"),
    ("v4", "analysis_v4.py"),
    ("v5", "analysis_v5.py"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory containing per-emotion subfolders.")
    ap.add_argument("--json_name", default="alphapose-results.json", help="AlphaPose JSON filename in each emotion folder.")
    ap.add_argument("--out_base", required=True, help="Base output directory; creates subfolders v1..v5.")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use.")
    args = ap.parse_args()

    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).resolve().parent

    for tag, script in SCRIPTS:
        out_dir = out_base / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        script_path = here / script
        cmd = [args.python, str(script_path), "--root", args.root, "--json_name", args.json_name, "--out_dir", str(out_dir)]
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print("[OK] All v1–v5 finished. Outputs in:", out_base)

if __name__ == "__main__":
    main()
