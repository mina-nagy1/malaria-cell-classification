# Malaria Cell Classification using Deep Learning

This project implements a convolutional neural network (CNN) to classify microscopic blood smear images as parasitized or uninfected, addressing a real-world medical image classification problem related to malaria diagnosis.

The work spans model development, error analysis, threshold calibration, model quantization, and production deployment, reflecting real-world machine learning practice.


## Problem Overview

Malaria diagnosis traditionally relies on manual microscopic inspection, which is time-consuming and prone to human error. Automating this process using computer vision can assist medical professionals by providing fast and consistent preliminary screening.

This project formulates malaria detection as a binary image classification task.


## Dataset

- **Source:** TensorFlow Datasets (TFDS) – Malaria Dataset
- **Classes:**  
  - `Parasitized`  
  - `Uninfected`
- **Image Type:** RGB blood smear images
- **Data Handling:**  
  - Dataset is loaded dynamically using TFDS  
  - No raw data is stored in this repository  

Detailed dataset handling instructions are available in `data/README.md`.


## Model Architecture

The model is a custom Convolutional Neural Network designed for efficiency and clarity rather than excessive depth.

**Key characteristics:**
- Input shape: `(224, 224, 3)`
- Internal normalization using `Rescaling(1./255)`
- Convolution → Pooling → Regularization blocks
- Batch normalization for training stability
- Sigmoid output for binary classification

The architecture is defined in: src/model.py


## Training & Evaluation

Model training and evaluation were performed inside structured Jupyter notebooks to allow clear visualization, inspection, and analysis.

**Key points:**
- Explicit train / validation / test split
- Threshold optimization using ROC analysis
- Evaluation metrics include:
  - Accuracy
  - Precision
  - Recall
  - AUC
  - Confusion-matrix-based analysis

Rather than relying solely on accuracy, model performance was examined through:
- Confusion matrices
- ROC curves
- Threshold-dependent behavior analysis
This allows a transparent understanding of how classification behavior changes under different decision boundaries.

All experiments and metrics were tracked using Weights & Biases.


## Results & Reporting

A consolidated experiment tracking and evaluation report is available on Weights & Biases:

**Weights & Biases Report:**  
https://wandb.ai/Mina_team/Malaria_Detection/reports/Malaria-Detection-using-Convolutional-Neural-Networks--VmlldzoxNTQ4NzQwMQ

The report includes:
- Training and validation performance curves
- Confusion matrices at selected decision thresholds
- ROC curve analysis with threshold annotations
- Quantitative evaluation tables

Key exported figures used in the analysis are stored in:
`reports/figures/`



## Deployment
A complete inference service is implemented under:
deployment/service/

**Key Features**
- ONNX Runtime–based inference for efficient, framework-independent model execution
- Dynamically quantized ONNX model, reducing model size and improving inference latency while preserving accuracy
- Clear separation of concerns:
 -api/ → request handling and routing
 -core/logic/ → model loading and inference logic
 -ui.py → Streamlit-based user interface
 -Threshold-aware prediction handling, ensuring deployment behavior matches evaluation logic
 -CPU-optimized inference, suitable for edge devices and low-resource environments


## Quantization Strategy

The trained TensorFlow model is converted to ONNX format and then dynamically quantized.
Dynamic quantization is applied at inference time, allowing weight precision reduction without retraining and enabling faster execution on standard CPUs.

The conversion and quantization pipeline is documented in:
notebooks/Quantization.ipynb


## License

This project is released under the **MIT License**, allowing reuse with attribution.


## Author

**Mina Nagy**  
