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
