# Matrix_Media_Assignment

# Project Overview

This repository contains various scripts and notebooks developed for handling different machine learning and data analysis tasks. Each file is designed to address specific requirements as outlined in the tasks.

## Table of Contents

- [Time Series Forecasting](#time-series-forecasting)
- [Anomaly Detection](#anomaly-detection)
- [Image Classification](#image-classification)
- [Logistic Regression](#logistic-regression)
- [Data Preprocessing](#data-preprocessing)

### Time Series Forecasting

**Objective**: Develop a script to perform time series forecasting using ARIMA models.

**Methodology**:
- Utilized `ARIMA` (AutoRegressive Integrated Moving Average) model from the `statsmodels` library to forecast future data points.
- Applied the model to temperature data to predict future values based on historical data.

### Anomaly Detection

**Objective**: Implement a basic anomaly detection algorithm from scratch.

**Methodology**:
- Created a simple algorithm using statistical methods (standard deviation) to detect outliers in a dataset.
- The algorithm identifies data points that are significantly different from the rest of the data, flagging potential anomalies.

### Image Classification

**Objective**: Write code to perform image classification using a pre-trained CNN architecture.

**Methodology**:
- Leveraged a pre-trained VGG16 model from the TensorFlow Keras applications for image classification.
- Employed image preprocessing techniques to prepare images for classification by resizing and normalizing them to match the input requirements of the model.

### Logistic Regression

**Objective**: Implement logistic regression from scratch using Python.

**Methodology**:
- Developed a logistic regression model using only NumPy to understand the underlying mechanics of the optimization algorithm.
- The script includes functions for model training (using gradient descent) and prediction.

### Data Preprocessing

**Objective**: Create a script to preprocess numerical data for machine learning tasks.

**Methodology**:
- Built a preprocessing pipeline using `scikit-learn` to handle tasks such as missing value imputation, feature scaling (standardization and normalization), and polynomial feature creation.
- The pipeline enhances model performance by preparing data through various transformations.

## Installation

To run the scripts, you will need Python installed along with the following libraries:
- NumPy
- pandas
- scikit-learn
- statsmodels
- TensorFlow

You can install these packages via pip:

```bash
pip install numpy pandas scikit-learn statsmodels tensorflow
