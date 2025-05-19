# UNet-Based Fluke Eggs Detection

This repository implements a bounding‑box detection and classification pipeline for four types of fluke eggs using a small, pretrained UNet (ResNet‑34 encoder) in PyTorch.

## Directory Structure

```
├─ data/                       # Root directory of training images and annotations
│   ├─ Clonorchis sinensis egg/
│   ├─ Fasciolopsis buski eggs/
│   ├─ Paragonimus westermani eggs/
│   └─ Schistosoma eggs/
└─ model/
    └─ model.py                # Training & evaluation script
```

## Setup

1. **Clone the repository**

   ```bash
   git clone https://your-repo-url.git
   cd Unet-Fluke-Eggs-Detection
   ```

2. **Create a virtual environment** (optional but recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   .\venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

* Place your training data under `data/` following the folder structure above. Each image file (`.jpg`, `.png`, etc.) must have a corresponding JSON annotation with the same basename:

  ```json
  {"labels": [{"name": "华支睾吸虫卵", "x1": 501, "y1": 315, "x2": 1371, "y2": 956}]}
  ```

* Run the training script:

  ```bash
  python model/model.py
  ```

  You should see an output like:

  ```plaintext
  Found 128 samples for training.
  Epoch 1/20: … loss=1.342
  …
  Class IoUs:
  Class 1: Mean IoU = 0.7421
  …
  Sample 0: True bbox = …, Pred bbox = …
  ```

## Model & Training Details

* **Backbone**: ResNet‑34 pretrained on ImageNet
* **UNet**: `segmentation_models_pytorch.Unet` with 5 output channels (background + 4 fluke classes)
* **Loss**: `torch.nn.CrossEntropyLoss`
* **Optimizer**: `torch.optim.Adam` (lr=1e-3)
* **Augmentations**: Resize to 256×256, horizontal flip, brightness/contrast, normalization (Albumentations)
* **Metrics**: Mean IoU per class on the training set
