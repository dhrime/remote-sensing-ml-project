import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern

def pil_to_np(img_pil):
    img = img_pil.convert("RGB").resize((128, 128)) 
    img = np.array(img).astype(np.float32)
    img /= 255.0
    return img

def extract_hog_features(gray):
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    ).astype(np.float32)

def extract_lbp_histogram(gray):
    P, R = 16, 2
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, n_bins+1), range=(0, n_bins), density=True
    )
    return hist.astype(np.float32)

def extract_color_histogram(rgb, bins=32):
    hists = []
    for c in range(3):
        hist, _ = np.histogram(
            rgb[..., c], bins=bins, range=(0.0, 1.0), density=True
        )
        hists.append(hist.astype(np.float32))
    return np.concatenate(hists)

def extract_all_features(pil_img):
    img = pil_to_np(pil_img)
    gray = rgb2gray(img)

    hog_vec = extract_hog_features(gray)
    lbp_vec = extract_lbp_histogram(gray)
    col_vec = extract_color_histogram(img)

    return np.concatenate([hog_vec, lbp_vec, col_vec])