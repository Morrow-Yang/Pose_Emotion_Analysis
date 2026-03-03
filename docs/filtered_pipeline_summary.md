# Filtered CAER Pipeline Status

Date: 2026-03-02

## What has been run
- YOLOv8 person prefilter (conf 0.35, min_area_ratio 0.01, min_height_ratio 0.1, max_ar 4.0, batch 16) → keep/drop manifests and hardlinked subset at `outputs/frames_filtered/CAER/train`.
- AlphaPose rerun on filtered frames (Res50 256x192, COCO, detbatch 1, posebatch 16, gpus 0). Command used (latest run):
  - In `data/external/AlphaPose-master`: `python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir outputs/frames_filtered/CAER/train/<EMOTION> --outdir outputs/alphapose_filtered/outputs/CAER/train/<EMOTION> --format open --posebatch 16 --detbatch 1 --qsize 256 --gpus 0`
- Top1 filtering attempted via `scripts/features/filter_top1_alphapose.py` (auto-run after AlphaPose) producing `alphapose-results.top1.json` per emotion.
- Temporal features: `scripts/innovation/temporal_motion_analysis.py --root outputs/alphapose_filtered/outputs/CAER/train --out outputs/analysis/temporal/caer_filtered --json_name alphapose-results.top1.json`.
- Geometry+temporal merge: `scripts/analysis/merge_geom_temporal.py --root outputs/alphapose_filtered/outputs/CAER/train --json_name alphapose-results.top1.json --temporal_csv outputs/analysis/temporal/caer_filtered/temporal_motion_features.csv --out_dir outputs/analysis/analysis/caer_v4_filtered`.
- Sliding windows: `scripts/innovation/window_temporal_features.py --input outputs/analysis/temporal/caer_filtered/temporal_motion_features.csv --output outputs/analysis/temporal/caer_filtered/temporal_motion_windows.csv --window_sec 1.0 --hop_sec 0.5 --fps 10 --min_count 1`.
- Attempted `run_all_v1_v5.py` on filtered outputs → **failed at v1** because no valid samples.

## Current outputs (filtered)
- AlphaPose (per emotion): `outputs/alphapose_filtered/outputs/CAER/train/<EMOTION>/`
- Temporal: `outputs/analysis/temporal/caer_filtered/` (temporal_motion_features.csv: 1,069 rows)
- Merged v4: `outputs/analysis/analysis/caer_v4_filtered/` (pose_features_v4_with_temporal.csv: 258 rows)

## Issues / blockers
- AlphaPose outputs from `data/external/AlphaPose-master` lack per-frame identifiers. Files under `sep-json` hold OpenPose-style dicts with only `pose_keypoints_2d` and no `image_id/imgname`; top-level `alphapose-results.json` is a dict of 130 keys (one per `sep-json` file). After top1, only ~130 samples per emotion remain, so downstream CSVs are tiny and `analysis_v1` aborts with "No valid samples".

## Recommended fix
- Re-run AlphaPose using the repository’s root `scripts/demo_inference.py` (which emits per-frame image_id/imgname like the original runs) on the filtered frames. Example (per emotion loop):
  ```powershell
  & "C:\ProgramData\anaconda3\shell\condabin\conda-hook.ps1"; conda activate aiemotion; $env:PYTHONUTF8=1;
  $emos = 'Anger','Disgust','Fear','Happy','Neutral','Sad','Surprise'
  foreach ($e in $emos) {
    $in  = "outputs/frames_filtered/CAER/train/$e"
    $out = "outputs/alphapose_filtered/outputs/CAER/train/$e"
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir $in --outdir $out --format open --posebatch 16 --detbatch 1 --qsize 256 --gpus 0
  }
  ```
- Then re-run:
  1) `scripts/features/filter_top1_alphapose.py` per emotion (or via a loop),
  2) `scripts/innovation/temporal_motion_analysis.py` → `outputs/analysis/temporal/caer_filtered`,
  3) `scripts/analysis/merge_geom_temporal.py` → `outputs/analysis/analysis/caer_v4_filtered`,
  4) `scripts/analysis/analysis_v4.py --precomputed_csv outputs/analysis/analysis/caer_v4_filtered/pose_features_v4_with_temporal.csv`,
  5) `scripts/innovation/window_temporal_features.py`,
  6) optional: `scripts/analysis/run_all_v1_v5.py` if full per-frame JSON is available.

## Notes
- The previous (unfiltered) AlphaPose outputs under `outputs/alphapose/outputs/CAER/train` still contain proper per-frame JSON and can be used as reference for expected structure.
- If storage allows, keep both filtered and unfiltered outputs for comparison; otherwise, ensure the filtered re-run writes to a separate folder before deleting the current aggregated JSONs.
