# CVBRA

Research code for **Cross-Visibility Balanced Replay Adaptation (CVBRA)** for
vehicle detection in UAV imagery. CVBRA combines original/fog target views,
source replay, and controlled layer updates while retaining single-pass inference.

The associated manuscript is *Visibility-aware single-inference domain adaptation
for vehicle detection in unmanned aerial vehicle imagery with source-domain
preservation*.

## Contents

- `scripts/`: training, data preparation, evaluation and analysis implementations.
- `src/buse_uav/`: required shared modules, retaining the original package namespace.
- `configs/settings/`: scientific settings extracted from the experiment protocols.
- `splits/`: sample identities and split roles, without images or annotations.
- `tests/`: four synthetic implementation test suites and their fixtures.
- `docs/`: dataset sources, environment and reproduction instructions.

This is a minimal code release. Internal reports, audit logs, aggregate result
archives, contact details, benchmark imagery, annotations, model weights and
prediction dumps are not included. Analysis scripts implement the original
procedures but require separately obtained inputs to execute those analyses.

## Installation and checks

The recorded experiments used Python 3.12.13, Ultralytics 8.4.53 and PyTorch
**2.12.0.dev20260408+cu128**, a development build. Install a compatible
PyTorch/torchvision pair for your system before installing this package.

```bash
python -m pip install -e ".[test]"
python tools/verify_release.py
python -m pytest -q
```

`verify_release.py` uses only the Python standard library. The tests use synthetic
fixtures and do not reproduce the paper's GPU training. Changes in framework
versions, hardware or training order can change results.

Read [Reproduction](docs/REPRODUCTION.md) before running any experiment script.
Some historical scripts default to a training stage and validate original local
registration files. This repository is not a self-contained one-command rerun;
it does not include those internal records or checkpoints. The settings extracts
are documentation, not replacements for the original hash-validated lock files.

## Scientific scope

The main evaluation is supervised HazyDet-to-UAV-OBB adaptation in a shared
car/truck/bus horizontal-box space. The NVD experiment is a supplementary,
car-only weather-boundary test, not a general claim of winter robustness.
Source-domain loss and unresolved statistical contrasts must not be interpreted
as guaranteed retention or equivalence. The NVD correction uses distinct training
orders after ineffective seed-only repeats were identified before metrics;
implementation details needed to understand it are retained in the code and docs.

## License

The original copyright notice in `LICENSE` is retained. No new open-source reuse
license has been selected; public code availability is not a new MIT/Apache
permission grant. Dependencies and benchmark assets retain their respective
terms; see [Third-party resources](THIRD_PARTY_LICENSES.md).
