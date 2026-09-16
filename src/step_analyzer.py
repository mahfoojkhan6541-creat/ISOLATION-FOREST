import os, sys, joblib
import pandas as pd
import numpy as np

BASE_DIR = r"D:\WhiteBox\SIH2026_IF"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FEATURES_DIR = os.path.join(BASE_DIR, "features")

print('Running comprehensive analysis script...')
