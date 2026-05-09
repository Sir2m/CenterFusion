# CenterFusion: Center-based Radar and Camera Fusion for 3D Object Detection

A PyTorch re-implementation of **CenterFusion**, as described in:

> Nabati, R., & Qi, H. (2020). *CenterFusion: Center-based Radar and Camera Fusion for 3D Object Detection*. arXiv:2011.04841.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [References](#references)
- [License](#license)

---

## Overview

CenterFusion is a middle-fusion approach for 3D object detection that combines radar and camera sensor data. The method addresses the radar-camera data association problem through a frustum-based mechanism, generating radar feature maps that complement image features for accurate estimation of object depth, rotation, velocity, and attributes.

Key contributions of the original paper:

- A frustum-based radar association method that uses preliminary 3D bounding box estimates to create a Region of Interest (RoI) frustum for each detected object in 3D space.
- A pillar expansion pre-processing step that converts sparse radar points into fixed-size 3D pillars to compensate for inaccurate radar height measurements.
- A middle-fusion architecture that concatenates radar-derived feature maps with image features and feeds them into secondary regression heads for refined prediction.
- State-of-the-art performance on the nuScenes benchmark, improving the NDS score of the camera-only baseline (CenterNet) by over 12% and reducing velocity estimation error by over 62%.

---

## Architecture

The CenterFusion pipeline consists of the following stages:

1. **Center Point Detection** — A CenterNet backbone (DLA-34) processes the input image and produces preliminary 3D bounding box estimates including depth, dimensions, and rotation.
2. **Frustum Association** — For each detected object, a RoI frustum is constructed in 3D space using the preliminary bounding box. Radar detections are associated to objects if their expanded pillar intersects the frustum.
3. **Radar Feature Extraction** — Associated radar detections generate three-channel heatmaps encoding depth and radial velocity components, centered on each object's 2D bounding box.
4. **Secondary Regression** — The concatenated image and radar feature maps are passed to secondary regression heads that refine depth and rotation estimates and produce velocity and attribute predictions.

---


## Dataset

This implementation uses the [nuScenes](https://www.nuscenes.org/) dataset.

1. Download the nuScenes dataset and place it under `data/nuscenes/`.
2. The expected directory structure is:

```
data/
  nuscenes/
    maps/
    samples/
    sweeps/
    v1.0-trainval/
    v1.0-test/
    v1.0-mini/
```

3. Run the data pre-processing script to generate annotation files:

```bash
python src/tools/convert_nuScenes.py
```

---

## Usage

### Training

```bash
python src/main.py ctdet \
  --dataset nuscenes \
  --exp_id centerfusion \
  --batch_size 26 \
  --num_epochs 60 \
  --lr 2.5e-4 \
  --load_model <path_to_centernet_pretrained_weights>
```

### Evaluation

```bash
python src/test.py ctdet \
  --dataset nuscenes \
  --exp_id centerfusion \
  --load_model <path_to_trained_model> \
  --flip_test
```

### Inference on a Single Scene

```bash
python src/demo.py ctdet \
  --dataset nuscenes \
  --load_model <path_to_trained_model> \
  --demo <path_to_sample>
```

---

## Results

Performance on the nuScenes validation split using the DLA-34 backbone.

| Method | Modality | NDS | mAP | mATE | mASE | mAOE | mAVE | mAAE |
|---|---|---|---|---|---|---|---|---|
| CenterNet (baseline) | Camera | 0.328 | 0.306 | 0.716 | 0.264 | 0.609 | 1.426 | 0.658 |
| CenterFusion (ours) | Camera + Radar | 0.453 | 0.332 | 0.649 | 0.263 | 0.535 | 0.540 | 0.142 |

Ablation results (nuScenes validation split):

| Configuration | NDS | mAP |
|---|---|---|
| Baseline (CenterNet) | 0.328 | 0.306 |
| + Pillar Expansion | +15.4% | +1.0% |
| + Frustum Association | +25.9% | +2.0% |
| + Pillar Expansion + Frustum Association | +34.5% | +4.3% |
| + Flip Test | +37.8% | +8.4% |

---

## References

```
@article{nabati2020centerfusion,
  title={CenterFusion: Center-based Radar and Camera Fusion for 3D Object Detection},
  author={Nabati, Ramin and Qi, Hairong},
  journal={arXiv preprint arXiv:2011.04841},
  year={2020}
}
```

- Zhou, X., Wang, D., & Krahenbuhl, P. (2019). Objects as Points. arXiv:1904.07850.
- Caesar, H., et al. (2019). nuScenes: A multimodal dataset for autonomous driving. arXiv:1903.11027.
- Yu, F., et al. (2018). Deep Layer Aggregation. CVPR 2018.

---

## License

This project is for academic and research purposes. Please refer to the [nuScenes terms of use](https://www.nuscenes.org/terms-of-use) when using the dataset.
