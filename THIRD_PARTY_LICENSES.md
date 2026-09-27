# Third-party resources

This release contains project-authored source, settings and sample identifiers.
It does not bundle upstream repository snapshots, dataset imagery, original box
annotations, or model weights. The original project copyright notice is retained
in `LICENSE`; no new permission grant is implied by the repository being public.

| Resource | Source and scope |
| --- | --- |
| Ultralytics | https://github.com/ultralytics/ultralytics ; external dependency. Follow its applicable AGPL-3.0 or enterprise terms. This release does not relicense it. |
| PyTorch and torchvision | https://github.com/pytorch/pytorch ; external dependencies |
| pycocotools | https://github.com/cocodataset/cocoapi ; external COCO evaluation dependency |
| MMDetection | https://github.com/open-mmlab/mmdetection ; optional legacy adapter dependency, not needed for the primary YOLO11n smoke tests |
| HazyDet | https://github.com/GrokCV/HazyDet ; obtain data and model-zoo assets from the provider under its terms |
| UAV-OBB | https://doi.org/10.17632/6snrjwcpkh.4 ; exact dataset version used in the paper |
| AU-AIR | https://bozcani.github.io/auairdataset/ ; provider terms govern data access and use |
| DroneVehicle | https://github.com/VisDrone/DroneVehicle ; only the RGB validation subset is used for the paper's external probe |
| NVD | https://github.com/amrdev-pixel/Nordic-Vehicle-Dataset ; provider distributes data under CC-BY-NC; images and derived frames are not included |

The ALDI++-AF-HN comparator is a project implementation of training principles on
YOLO11n, not the authors' original released detector. Internal consistency reports
are not part of this minimal public release.
Third-party license obligations must be considered before redistributing a
combined implementation or deploying it as a service; this inventory is not a
new license for dependencies or datasets.
