#!/usr/bin/env python3


import argparse
import hashlib
import json
import os
import time

import joblib
import numpy as np
from skimage.io import imread
from skimage.segmentation import slic
from skimage.util import img_as_float
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from skimage.color import rgb2lab

from constants import (N_SEGMENTS, SQUARE_SIZE, COMPACTNESS, SLIC_SIGMA,
                       C, SVR_EPSILON, C_LIST, EPSILON_LIST,
                       MODEL, STANDARDIZE, USE_POSITION, USE_CONTEXT, USE_STATS,
                       CONTEXT_SIZE, CONTEXT_DOWN, HGB_MAX_ITER, HGB_LEARNING_RATE)

FEATURE_KEYS = [
    "n_segments", "compactness", "slic_sigma", "square_size",
    "position", "context", "stats", "context_size", "context_down",
    "histogram", "gradient"
]



def rgb_to_lab(img):
    """Convert RGB image to CIELAB."""
    return rgb2lab(img)



def load_rgb(path):
    img = imread(path)
    if img.ndim != 3:
        return None
    return img_as_float(img[..., :3])


def segment_gray(y, n_segments, compactness, sigma):
    seg = slic(y, n_segments=n_segments, compactness=compactness,
               sigma=sigma, channel_axis=None)
    _, seg = np.unique(seg, return_inverse=True)
    return seg.reshape(y.shape)

def superpixel_geometry(segments):
    n = segments.max() + 1
    flat = segments.ravel()
    counts = np.bincount(flat, minlength=n).astype(float)
    rows, cols = np.indices(segments.shape)
    cy = np.bincount(flat, weights=rows.ravel(), minlength=n) / counts
    cx = np.bincount(flat, weights=cols.ravel(), minlength=n) / counts
    return counts, cy, cx


def touching_pairs(segments):
    horiz = np.stack([segments[:, :-1].ravel(), segments[:, 1:].ravel()], axis=1)
    vert = np.stack([segments[:-1, :].ravel(), segments[1:, :].ravel()], axis=1)
    pairs = np.concatenate([horiz, vert])
    pairs = np.sort(pairs[pairs[:, 0] != pairs[:, 1]], axis=1)
    return np.unique(pairs, axis=0)


def superpixel_patches(y, segments, square, geom=None):
    h, w = y.shape
    _, cy, cx = geom or superpixel_geometry(segments)
    half = square // 2
    tops = np.clip(np.round(cy).astype(int) - half, 0, h - square)
    lefts = np.clip(np.round(cx).astype(int) - half, 0, w - square)
    patches = np.empty((len(cy), square, square))
    for k in range(len(cy)):
        patches[k] = y[tops[k]:tops[k] + square, lefts[k]:lefts[k] + square]
    return patches


def context_windows(y, cy, cx, size, down):
    half, block = size // 2, size // down
    ypad = np.pad(y, half, mode="edge")
    out = np.empty((len(cy), down * down))
    for k in range(len(cy)):
        r, c = int(round(cy[k])), int(round(cx[k]))
        win = ypad[r:r + size, c:c + size]
        out[k] = win.reshape(down, block, down, block).mean(axis=(1, 3)).ravel()
    return out


def neighbor_mean(values, pairs, n):
    total, count = np.zeros(n), np.zeros(n)
    np.add.at(total, pairs[:, 0], values[pairs[:, 1]])
    np.add.at(total, pairs[:, 1], values[pairs[:, 0]])
    np.add.at(count, pairs[:, 0], 1)
    np.add.at(count, pairs[:, 1], 1)
    return np.where(count > 0, total / np.maximum(count, 1), values)


def superpixel_features(y, segments, cfg):
    h, w = y.shape
    geom = superpixel_geometry(segments)
    counts, cy, cx = geom
    n = len(counts)

    patches = superpixel_patches(y, segments, cfg["square_size"], geom)
    parts = [np.abs(np.fft.fft2(patches)).reshape(n, -1)]
    if cfg.get("histogram", False):
        hist_features = np.empty((n, 16), dtype=float)

        for i, patch in enumerate(patches):
            hist, _ = np.histogram(
                patch,
                bins=16,
                range=(0.0, 1.0)
            )
            hist_features[i] = hist / hist.sum()

        parts.append(hist_features)

    if cfg.get("gradient", False):
        grad_y = np.gradient(patches, axis=1)
        grad_x = np.gradient(patches, axis=2)

        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        grad_mean = magnitude.mean(axis=(1, 2))
        grad_std = magnitude.std(axis=(1, 2))

        parts.append(
            np.column_stack([grad_mean, grad_std])
        )


    if cfg.get("position"):
        parts.append(np.column_stack([cy / h, cx / w]))

    if cfg.get("stats") or cfg.get("context"):
        flat = segments.ravel()
        mean_y = np.bincount(flat, weights=y.ravel(), minlength=n) / counts
    if cfg.get("stats"):
        var = np.bincount(flat, weights=(y ** 2).ravel(), minlength=n) / counts - mean_y ** 2
        parts.append(np.column_stack([mean_y, np.sqrt(np.maximum(var, 0)), counts / (h * w)]))
    if cfg.get("context"):
        parts.append(context_windows(y, cy, cx, cfg["context_size"], cfg["context_down"]))
        parts.append(neighbor_mean(mean_y, touching_pairs(segments), n)[:, None])

    return np.hstack(parts)


def superpixel_targets(uv, segments):
    n = segments.max() + 1
    flat = segments.ravel()
    counts = np.bincount(flat, minlength=n).astype(float)
    u = np.bincount(flat, weights=uv[..., 0].ravel(), minlength=n) / counts
    v = np.bincount(flat, weights=uv[..., 1].ravel(), minlength=n) / counts
    return u, v

def process_image(path, cfg, cache_dir, use_cache=True):
    key = "|".join([os.path.abspath(path), str(os.path.getmtime(path)),
                    json.dumps({k: cfg[k] for k in FEATURE_KEYS}, sort_keys=True)])
    cache_file = os.path.join(cache_dir, hashlib.md5(key.encode()).hexdigest() + ".npz")
    if use_cache and os.path.isfile(cache_file):
        d = np.load(cache_file)
        return d["X"], d["U"], d["V"]

    img = load_rgb(path)
    if img is None:
        print("  skip (grayscale file):", path)
        return None
    if min(img.shape[:2]) < cfg["square_size"]:
        print("  skip (image smaller than patch):", path)
        return None

    lab = rgb_to_lab(img)
    y = lab[..., 0] / 100.0
    segments = segment_gray(y, cfg["n_segments"], cfg["compactness"], cfg["slic_sigma"])
    X = superpixel_features(y, segments, cfg)
    A, B = superpixel_targets(lab[..., 1:], segments)


    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache_file, X=X, U=A, V=B)
    return X, A, B


def make_regressor(cfg, c, eps):
    if cfg["model"] == "hgb":
        return HistGradientBoostingRegressor(max_iter=HGB_MAX_ITER,
                                             learning_rate=HGB_LEARNING_RATE,
                                             random_state=0)
    svr = SVR(kernel="rbf", C=c, epsilon=eps)
    return make_pipeline(StandardScaler(), svr) if cfg["standardize"] else svr


def fit_and_save(cfg, X, U, V, c, eps, u_path, v_path):
    desc = "gradient boosting" if cfg["model"] == "hgb" else f"SVR C={c}, epsilon={eps}"
    print(f"Fitting {desc} on {len(X)} superpixels x {X.shape[1]} features...")
    for name, target, path in (("U", U, u_path), ("V", V, v_path)):
        t0 = time.time()
        model = make_regressor(cfg, c, eps)
        model.fit(X, target)
        joblib.dump(model, path)
        print(f"  {name} model: {time.time() - t0:.0f}s -> {path}")


def describe_features(cfg):
    parts = [f"|fft2| of {cfg['square_size']}x{cfg['square_size']} Y patch at centroid"]
    if cfg["position"]:
        parts.append("normalized position")
    if cfg["stats"]:
        parts.append("mean/std Y + relative area")
    if cfg["context"]:
        parts.append(f"{cfg['context_size']}px context window ({cfg['context_down']}x{cfg['context_down']}) + neighbor mean Y")
    return " + ".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Train the U and V regressors.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("-t", "--train-dir", default="data/train/",
                    help="folder of training .jpg images (searched recursively)")
    ap.add_argument("--models-dir", default="models", help="where to save models")
    ap.add_argument("--cache-dir", default="cache", help="where to cache extracted features")
    ap.add_argument("--no-cache", action="store_true", help="ignore/skip the feature cache")
    ap.add_argument("--sweep", action="store_true",
                    help="SVR only: train a model pair for every (C, epsilon) in C_LIST x EPSILON_LIST")
    ap.add_argument("--c", type=float, default=C)
    ap.add_argument("--epsilon", type=float, default=SVR_EPSILON)
    ap.add_argument("--n-segments", type=int, default=N_SEGMENTS)
    ap.add_argument("--compactness", type=float, default=COMPACTNESS)
    ap.add_argument("--slic-sigma", type=float, default=SLIC_SIGMA)
    ap.add_argument("--square-size", type=int, default=SQUARE_SIZE)
    ap.add_argument("--model", choices=["svr", "hgb"], default=MODEL,
                    help="svr = the paper's RBF SVR, hgb = gradient boosting")
    ap.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=STANDARDIZE,
                    help="standardize features before the SVR")
    ap.add_argument("--position", action=argparse.BooleanOptionalAction, default=USE_POSITION,
                    help="add normalized superpixel position")
    ap.add_argument("--context", action=argparse.BooleanOptionalAction, default=USE_CONTEXT,
                    help="add wide brightness window + neighbors' brightness")
    ap.add_argument("--stats", action=argparse.BooleanOptionalAction, default=USE_STATS,
                    help="add mean/std brightness and relative size")
    ap.add_argument("--histogram", action=argparse.BooleanOptionalAction, default=True, help="add a 16-bin luminance histogram")
    ap.add_argument("--gradient", action=argparse.BooleanOptionalAction, default=True, help="add gradient magnitude features")
    ap.add_argument("--context-size", type=int, default=CONTEXT_SIZE)
    ap.add_argument("--context-down", type=int, default=CONTEXT_DOWN)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="randomly keep at most this many superpixels (SVR gets slow with many)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.context_size % args.context_down:
        raise SystemExit("--context-size must be divisible by --context-down")
    if args.sweep and args.model != "svr":
        raise SystemExit("--sweep tunes C and epsilon, so it only applies to --model svr")

    cfg = {"n_segments": args.n_segments, "compactness": args.compactness,"histogram": args.histogram,
           "slic_sigma": args.slic_sigma, "square_size": args.square_size,"gradient": args.gradient,
           "position": args.position, "context": args.context, "stats": args.stats,
           "context_size": args.context_size, "context_down": args.context_down,
           "model": args.model, "standardize": args.standardize}
    print("Features:", describe_features(cfg))

    paths = []
    for root, _, files in os.walk(args.train_dir):
        paths += [os.path.join(root, f) for f in files if f.lower().endswith((".jpg", ".jpeg"))]
    paths.sort()
    if not paths:
        raise SystemExit(f"No .jpg files found under {args.train_dir}")
    print(f"Found {len(paths)} training images in {args.train_dir}")

    Xs, Us, Vs = [], [], []
    for i, path in enumerate(paths, 1):
        out = process_image(path, cfg, args.cache_dir, use_cache=not args.no_cache)
        if out is None:
            continue
        Xs.append(out[0]); Us.append(out[1]); Vs.append(out[2])
        if i % 10 == 0 or i == len(paths):
            print(f"  processed {i}/{len(paths)} images")

    X, U, V = np.concatenate(Xs), np.concatenate(Us), np.concatenate(Vs)
    print(f"Total training samples (superpixels): {len(X)}")

    if args.max_samples and len(X) > args.max_samples:
        idx = np.random.default_rng(args.seed).choice(len(X), args.max_samples, replace=False)
        X, U, V = X[idx], U[idx], V[idx]
        print(f"Subsampled to {len(X)} superpixels")

    os.makedirs(args.models_dir, exist_ok=True)
    config = {**cfg, "feature": describe_features(cfg),
              "segmentation_on": "grayscale (Y) image"}

    if args.sweep:
        for c in C_LIST:
            for eps in EPSILON_LIST:
                tag = f"C{c}_eps{eps}"
                u_path = os.path.join(args.models_dir, f"u_svr_{tag}.joblib")
                v_path = os.path.join(args.models_dir, f"v_svr_{tag}.joblib")
                if os.path.isfile(u_path) and os.path.isfile(v_path):
                    print(f"Skipping {tag} (already trained)")
                    continue
                fit_and_save(cfg, X, U, V, c, eps, u_path, v_path)
    else:
        fit_and_save(cfg, X, U, V, args.c, args.epsilon,
                     os.path.join(args.models_dir, "u_svr.joblib"),
                     os.path.join(args.models_dir, "v_svr.joblib"))

    with open(os.path.join(args.models_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()