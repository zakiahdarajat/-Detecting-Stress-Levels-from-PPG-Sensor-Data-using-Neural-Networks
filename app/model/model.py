import joblib
import re
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder as lnc
import pandas as pd
from keras.models import load_model

__version__ = "1.0.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

model = load_model(f'{BASE_DIR}/tfmodel.h5', compile=False)
model.compile(loss='categorical_crossentropy',optimizer='adam')

scaler = joblib.load(f'{BASE_DIR}/scaler.joblib')

label = ['condition']
features = ['MEAN_RR','RMSSD','pNN25','pNN50','LF','HF','LF_HF']


def predict_pipe(l):
    # Input validation for security and robustness
    if not isinstance(l, (list, np.ndarray)):
        raise ValueError("Input must be a list or numpy array")
    
    if len(l) != len(features):
        raise ValueError(f"Input must have exactly {len(features)} features")
    
    # Ensure all values are numeric
    try:
        l_numeric = [float(x) for x in l]
    except (ValueError, TypeError):
        raise ValueError("All input values must be numeric")
    
    ex = scaler.transform(np.array(l_numeric).reshape(1, -1))
    return model.predict(ex)

    




