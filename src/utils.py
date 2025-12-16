import numpy as np
from features import extract_all_features

def build_features_from_hf_split(hf_split):
    X = []
    y = []

    print(f"Extracting features from {len(hf_split)} images...")

    # Iterate through the Hugging Face dataset object
    for i, item in enumerate(hf_split):
        if i % 100 == 0:
            print(f"Processing image {i}/{len(hf_split)}")

        img = item["image"]
        label = item["label"]
        feats = extract_all_features(img)
        X.append(feats)
        y.append(label)

    return np.stack(X), np.array(y)