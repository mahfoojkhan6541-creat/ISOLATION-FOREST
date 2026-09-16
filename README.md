# Isolation Forest

## Overview

Isolation Forest is an unsupervised machine learning algorithm used for **anomaly detection**.

It identifies unusual or abnormal data points by isolating them from normal data.

## How It Works

Isolation Forest creates multiple random decision trees.

* Normal data points usually require more splits to be isolated.
* Anomalous data points are different from the majority, so they are isolated with fewer splits.
* The algorithm uses the average path length to determine whether a data point is normal or abnormal.

## Output

The model classifies data into two categories:

* **NORMAL** – data point follows the normal pattern.
* **ABNORMAL** – data point is detected as an anomaly.

## Advantages

* Does not require labelled data for training.
* Works well with high-dimensional data.
* Fast and efficient for anomaly detection.
* Can detect unusual patterns automatically.

## Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* Isolation Forest

## Basic Workflow

```text
Input Data
    ↓
Feature Selection
    ↓
Isolation Forest
    ↓
Anomaly Detection
    ↓
NORMAL / ABNORMAL
```

## Purpose

The purpose of this module is to detect abnormal patterns in component test data using the Isolation Forest algorithm.
