# CVBRA

Code for **Cross-Visibility Balanced Replay Adaptation (CVBRA)**, a method for
vehicle detection in UAV imagery. CVBRA combines original and fogged target
views, source replay, and controlled layer updates with single-pass inference.

The associated manuscript is *Visibility-aware single-inference domain adaptation
for vehicle detection in unmanned aerial vehicle imagery with source-domain
preservation*.

## Contents

- `scripts/`: training, data preparation, evaluation and analysis.
- `src/buse_uav/`: shared modules under the original package namespace.
- `configs/settings/`: scientific settings extracted from the experiment protocols.
- `splits/`: sample identities and split roles, without images or annotations.
- `tests/`: four synthetic implementation test suites and their fixtures.
- `docs/`: dataset sources, environment and reproduction instructions.

The release contains code, settings and split identifiers. Dataset images,
annotations, model weights, predictions and complete result archives are not
included. Training and analysis require the inputs described in
[Reproduction](docs/REPRODUCTION.md).

## Installation and checks

The recorded experiments used Python 3.12.13, Ultralytics 8.4.53 and PyTorch
**2.12.0.dev20260408+cu128**, a development build. Install a compatible
PyTorch/torchvision pair for your system before installing this package.

```bash
python -m pip install -e ".[test]"
python tools/verify_release.py
python -m pytest -q
```

`verify_release.py` uses only the Python standard library. The tests check
implementation behavior with synthetic fixtures; full GPU training requires
separate data and checkpoints. Framework versions, hardware and training order
can affect results.

Experiment scripts validate the original hash-checked registration files, which
are not included. `configs/settings/` documents the scientific configuration;
exact reproduction also requires the original execution locks. Some scripts
default to training, so inspect their `--help` output before execution.

## Scientific scope

The main evaluation is supervised HazyDet-to-UAV-OBB adaptation with horizontal
boxes for car, truck and bus. The supplementary NVD experiment is a car-only
weather-boundary test. Its findings apply to the sampled sequences and do not
establish general winter robustness. Source-domain loss remains, and unresolved
statistical contrasts do not establish equivalence. Corrected NVD runs use
distinct training orders after ineffective seed-only repeats were identified
before metrics; see the reproduction notes for details.

## License

See [LICENSE](LICENSE) for the original copyright notice. No additional
open-source reuse license is provided. Dependencies and benchmark assets retain
their respective terms; see [Third-party resources](THIRD_PARTY_LICENSES.md).
