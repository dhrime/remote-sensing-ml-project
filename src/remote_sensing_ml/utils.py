# utils.py

import numpy as np
from .features import extract_all_features

def build_features_from_hf_split(hf_split):
    X = []
    y = []

    for item in hf_split:
        img = item["image"]      
        label = item["label"]     
        feats = extract_all_features(img)
        X.append(feats)
        y.append(label)

    return np.stack(X), np.array(y)
