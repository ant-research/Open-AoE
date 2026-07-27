# AoE Retarget Lab

AoE Retarget Lab converts Open-AoE egocentric video into reconstructed
hand-object motion and robot-retargeting review assets. The supported fresh
path is:

```text
AoE RGB/annotation
  -> SAM3 target segmentation and tracking
  -> SAM3D Objects + object 6DoF
  -> EgoInfinity hand/object reconstruction
  -> Do-as-I-Do and/or SPIDER retargeting
  -> RGB overlay, pure-mesh HOI, plain robot video, triptych
```

Third-party backends, weights, datasets, caches, and generated experiments are
not vendored. Backend source trees remain upstream-owned; this repository
provides input conversion, orchestration, provenance checks, and review tools.

## Start here

1. [Installation and third-party configuration](docs/INSTALL.md)
2. [Runnable commands and artifact layout](docs/API.md)
3. [AoE dataset selection and debugging experience](docs/AOE_DATASET_PRACTICES.md)
4. [Clean-machine deployment and reproduction practices](docs/REPRODUCTION_PRACTICES.md)

Additional references:

- [Design principles](docs/PRINCIPLES.md)
- [Script directory layout](scripts/README.md)
