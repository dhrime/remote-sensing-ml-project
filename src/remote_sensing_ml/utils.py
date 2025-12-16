# utils.py

import numpy as np
from skimage.color import rgb2gray
from .features import extract_all_features, pil_to_np, extract_hog_features, extract_lbp_histogram, extract_color_histogram

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

def extract_hog_from_split(hf_split):
    """Extract only HOG features from a HuggingFace dataset split."""
    X = []
    y = []
    for item in hf_split:
        img = item["image"]
        label = item["label"]
        img_np = pil_to_np(img)
        gray = rgb2gray(img_np)
        hog_feats = extract_hog_features(gray)
        X.append(hog_feats)
        y.append(label)
    return np.stack(X), np.array(y)

def extract_lbp_from_split(hf_split):
    """Extract only LBP features from a HuggingFace dataset split."""
    X = []
    y = []
    for item in hf_split:
        img = item["image"]
        label = item["label"]
        img_np = pil_to_np(img)
        gray = rgb2gray(img_np)
        lbp_feats = extract_lbp_histogram(gray)
        X.append(lbp_feats)
        y.append(label)
    return np.stack(X), np.array(y)

def extract_color_from_split(hf_split):
    """Extract only color histogram features from a HuggingFace dataset split."""
    X = []
    y = []
    for item in hf_split:
        img = item["image"]
        label = item["label"]
        img_np = pil_to_np(img)
        color_feats = extract_color_histogram(img_np)
        X.append(color_feats)
        y.append(label)
    return np.stack(X), np.array(y)
