TinyML – Mosquito Species Classification System
This repository contains the data preprocessing pipeline for the Mosquito Species Classification project. The goal is to prepare a lightweight dataset suitable for training a TinyML model on Edge Impulse.
Project Overview
Target Platform: Edge Impulse (TinyML)
Input Data: Mosquito Dataset for Classification CNN
Classes: Aedes, Anopheles, Culex
Image Size: 160x160 pixels
Color Mode: Grayscale
Preprocessing Pipeline
The script preprocess_mosquito.py automates the entire workflow:
Auto-Download: Fetches the dataset directly from Kaggle Hub.
Grayscale Conversion: Reduces channel dimension (3 -> 1) to save memory for microcontroller deployment.
Resizing: Downscales images to 160x160 using INTER_AREA interpolation.
Data Augmentation: Applies a deterministic split to balance the dataset:
25%: Original images
25%: Rotated 90° (Random Clockwise/Counter-Clockwise)
25%: Flipped (Random Horizontal/Vertical)
25%: Both Rotated & Flipped
Getting Started
Prerequisites
Ensure you have Python 3.x installed. Install the required dependencies:
pip install -r requirements.txt


(Note: requirements.txt includes opencv-python, numpy, and kagglehub)
Option A: Use Pre-processed Data (Fastest)
If the dataset is already uploaded to this repository (e.g., as a .zip file):
Unzip mosquito_dataset_processed.zip.
Use the images directly for training.
Option B: Run the Pipeline (Generate from Scratch)
To download and process the data freshly on your machine:
python preprocess_mosquito.py


Note: The first time you run this, kagglehub might ask you to authenticate. Follow the terminal instructions to log in with your Kaggle account.
Folder Structure
After running the script (or unzipping), the directory structure will look like this:
mosquito_dataset_processed/
├── Aedes/
│   ├── image_001_orig.png
│   ├── image_002_rot.png
│   └── ...
├── Anopheles/
│   └── ...
└── Culex/
    └── ...


