# MAPCrop: Multidimensional Aesthetic Perception for Image Cropping

PyTorch implementation of **MAPCrop**, an image cropping framework based on multidimensional aesthetic perception and cue coordination.

## Overview

MAPCrop captures complementary aesthetic cues and their relationships to build a more comprehensive representation of image aesthetics. It consists of two main components:

- **Multidimensional Aesthetic Cue Perception (MACP)** extracts aesthetic information from complementary perspectives to enrich image representations.
- **Aesthetic Cue Coordination and Evaluation (ACCE)** models the synergy among aesthetic cues to support crop quality assessment and selection.

## Installation

Use Linux with Python 3.10, an NVIDIA GPU, and a CUDA toolkit compatible with PyTorch 2.5.1.

```bash
bash setup_server.sh
source .venv/bin/activate
```

## Preparation

Prepare [GAICv2](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch) images, annotations, and train/validation/test splits. The expected annotation format is defined in [data/gaic.py](data/gaic.py).

## Usage

### Training

```bash
python train.py --config configs/server.yaml --stage cropping
```

### Evaluation

Evaluate a trained MAPCrop checkpoint on the configured GAICv2 test split:

```bash
python evaluate.py --checkpoint /path/to/mapcrop.pt \
  --config configs/server.yaml --device cuda --output test_metrics.json
```

### Inference

```bash
python infer.py --checkpoint /path/to/mapcrop.pt \
  --image /path/to/image.jpg --device cuda --top-k 4 --output predictions/example
```

The output contains ranked crop images and their original-image coordinates in `crops.json`.

