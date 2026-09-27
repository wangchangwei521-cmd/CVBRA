# Reproduction instructions

## Environment and primary method

The recorded environment used Python 3.12.13, PyTorch 2.12.0.dev20260408+cu128
and Ultralytics 8.4.53. This is a recorded development build of PyTorch, not a
claim that a stable 2.12.0 release was used. The smoke tests in this release were
run in that existing environment; a fresh installation and full GPU rerun were
not performed during packaging.

Confirmatory CVBRA-L10 uses 8 epochs, image size 1280, batch size 2 and SGD with
initial learning rate 0.00075. Each epoch has 900 original target, 900 fog-0.6
target, 900 fog-1.0 target and 900 source-replay exposures. Layers 0-9 parameters
and buffers are restored exactly from the source after training; layers 10-23
retain the adapted state. The last epoch is used without metric-based selection.
The implementation is `scripts/run_cvbra_v1_training.py`, including the exact
`_source_replay_rows`, `_target_labels` and `_combined_state` functions.

Headline COCO AP uses candidate floor 0.001, NMS IoU 0.70 and max_det 500.
Historical fixed-operating-point AP at confidence 0.25 is a different statistic.
Do not silently replace the headline metric with that statistic or AP50.

## Data and checkpoints

| Dataset | Provider | Study scope |
| --- | --- | --- |
| HazyDet | https://github.com/GrokCV/HazyDet | 8,000/1,000/2,000 train/validation/test images, with replay drawn from train |
| UAV-OBB v4 | https://doi.org/10.17632/6snrjwcpkh.4 | 900 adaptation and 483 reserve scenes; primary validation uses 167 of 218 official images, in 141 train-disjoint groups; 16 test images |
| AU-AIR | https://bozcani.github.io/auairdataset/ | 6,578 sampled frames from eight sequences, zero-tuning retention probe |
| DroneVehicle | https://github.com/VisDrone/DroneVehicle | 1,469 RGB validation images, zero-tuning retention probe |
| NVD | https://github.com/amrdev-pixel/Nordic-Vehicle-Dataset | 900 source-training, 2,526 retention, 900 adaptation, 1,000 validation and 2,027 test frames across five sequences |

Obtain images and labels from their providers under the applicable terms. The
CSV files in `splits/` are identifiers and role assignments, not dataset payloads.
The HazyDet replay selection is deterministically implemented in
`_source_replay_rows`. The main dataset class order is car, truck, bus; NVD is car
only. Fog generation is in `src/buse_uav/data/corruptions.py`; target-view
preparation retains its original `materialize_sava_uav_obb_development.py`
filename because CVBRA consumes the same prepared views. That filename is not a
claim that SAVA is the submitted method.

The historical scripts expect raw data under `data/raw/`, prepared views under
`data/processed/`, source checkpoints under `weights/hazydet/`, and outputs under
`runs/`. The paper's source checkpoints are domain-trained models, not generic
COCO-pretrained YOLO/RT-DETR weights. Checkpoint files and original execution
locks are not included. Consult the paper's corresponding author for retained
reproduction materials, subject to provider terms. Load only trusted PyTorch
checkpoints: the original runners use object deserialization.

## Entry points

From the repository root, inspect each interface with
`python -m scripts.<name_without_py> --help` before execution.

| Purpose | Script |
| --- | --- |
| Confirmatory adaptation | `run_cvbra_v1_training.py` |
| STF and component ablations | `run_cvbra_v1_matched_baseline_training.py` |
| Allocation sensitivity | `run_cvbra_v1_allocation_sensitivity_v1.py` |
| L5/L10 replication | `run_cvbra_v2_reviewer_closure_v1.py` |
| Standard low-floor evaluation | `run_cvbra_v3_metric_integrity_v1.py` |
| Six-decision replication | `run_cvbra_v3_full_grid_multiseed_v1.py` |
| Matched ALDI++-AF-HN | `run_cvbra_v4_aldi_hn_multitrajectory_v1.py` |
| RT-DETR-L confirmation | `run_cvbra_v4_rtdetr_l_full_budget_multitrajectory_v1.py` |
| Visibility stress | `run_cvbra_v2_visibility_stress_v1.py` |
| External probes | `run_cvbra_v1_auair_external.py`, `run_cvbra_v1_dronevehicle_validation.py` |
| NVD preparation and corrected runs | `prepare_nvd_real_snow_cvbra_v1.py`, `prepare_nvd_real_snow_cvbra_v1_1.py`, `run_nvd_real_snow_cvbra_v1_1.py` |

Full execution needs original datasets, checkpoints and historical dependencies.
The original scripts intentionally stop when an expected artifact or hash is
missing. Do not disable such a check and describe the resulting run as an exact
reproduction. `configs/settings/` contains scientific extracts with internal
authorization/audit fields removed; it is not the original `configs/experiment/`
lock chain. One malformed, superseded historical retention specification was not
exported; the later v2-v4 settings and original code remain available.

For NVD, ineffective seed-only repetitions were detected before target metrics.
The corrected runs use distinct deterministic orders of the same sample
multisets. The correction logic remains in the original preparation scripts.
One workstation-specific Ultralytics path was replaced by package discovery;
no training equations or numerical results were changed for release.

The public release excludes internal audit narratives and complete result
archives. This does not withdraw the paper's negative results or statistical
limitations: CVBRA retains a source loss, its NVD result is below STF, and its
difference from the NVD no-visibility control remains unresolved. The NVD test
has 32 temporal blocks from one video, not 32 independent sites.
