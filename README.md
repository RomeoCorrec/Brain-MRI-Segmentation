# Brain MRI Segmentation

A deep learning project for segmenting brain tumors in MRI images using U-Net and YOLO architectures.

## Overview

This project implements brain tumor segmentation using state-of-the-art deep learning models. It includes a U-Net implementation for semantic segmentation and YOLO configuration for object detection-based approaches.

## Features

- **U-Net Architecture**: Custom implementation of U-Net with encoder-decoder structure and skip connections
- **Data Processing**: Jupyter notebook for data mapping and preprocessing
- **YOLO Integration**: Configuration for YOLO-based tumor detection
- **PyTorch Implementation**: Built with PyTorch for flexibility and performance

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include:
- PyTorch
- torchvision
- OpenCV
- albumentations
- segmentation_models_pytorch
- ultralytics (YOLO)
- matplotlib
- pandas

## Project Structure

```
.
├── model.py              # U-Net model implementation
├── data_mapping.ipynb    # Data preprocessing and visualization
├── brain_tumor.yaml      # YOLO dataset configuration
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Usage

### Model Architecture

The U-Net model (`model.py`) can be imported and used for training:

```python
from model import UNet

# Initialize model
model = UNet(n_channels=3, n_classes=1)

# For grayscale input
model = UNet(n_channels=1, n_classes=1)
```

### Data Processing

Use the `data_mapping.ipynb` notebook to:
- Load and visualize MRI images
- Prepare training data
- Apply data augmentation
- Create train/validation splits

### YOLO Configuration

The `brain_tumor.yaml` file contains the dataset configuration for YOLO-based training. Update the `path` field to point to your dataset location.

## Model Details

### U-Net Architecture
- **Input**: 3-channel RGB or 1-channel grayscale images
- **Output**: Single channel segmentation mask
- **Layers**: 5 encoder blocks and 4 decoder blocks
- **Features**: Batch normalization, ReLU activation, skip connections

## License

This project is available for educational and research purposes.
