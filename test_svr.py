#!/usr/bin/env python3

import argparse
import glob
import json
import os

import joblib
import numpy as np
from skimage.io import imread, imsave
from skimage.util import img_as_float, img_as_ubyte

from train_svr import (rgb_to_yuv, segment_gray, superpixel_features,
                   superpixel_patches, touching_pairs)

from constants import (U_MAX, V_MAX, RGB_FROM_YUV, WEIGHT_DIFF, COVAR, THRESHOLD,
                       ICM_ITERATIONS, ITER_EPSILON, BOOST,
                       WEIGHT_DIFF_LIST, THRESHOLD_LIST)

def yuv_to_rgb(yuv):
    return np.clip(yuv @ RGB_FROM_YUV, 0, 1)


def load_for_test(path):
    raw = imread(path)
    if raw.ndim == 2:
        return img_as_float(raw), None
    rgb = img_as_float(raw[..., :3])
    return rgb_to_yuv(rgb)[..., 0], rgb


def adjacent_pairs(segments, vecs):
    pairs = touching_pairs(segments)
    dists = np.linalg.norm(vecs[pairs[:, 0]] - vecs[pairs[:, 1]], axis=1)
    return pairs, dists


def prepare(path, cfg):
    y, rgb = load_for_test(path)
    segments = segment_gray(y, cfg["n_segments"], cfg["compactness"], cfg["slic_sigma"])
    feats = superpixel_features(y, segments, cfg)
    patches = superpixel_patches(y, segments, cfg["square_size"])
    fft = np.fft.fft2(patches).reshape(len(patches), -1)
    pairs, dists = adjacent_pairs(segments, fft)
    return {"path": path, "y": y, "rgb": rgb, "segments": segments,
            "feats": feats, "pairs": pairs, "dists": dists}

def predict_uv(p, u_svr, v_svr, boost):
    mu_u = np.clip(u_svr.predict(p["feats"]) * boost, -U_MAX, U_MAX)
    mu_v = np.clip(v_svr.predict(p["feats"]) * boost, -V_MAX, V_MAX)
    return mu_u, mu_v


def icm(mu, pairs, dists, sigma2, gamma, threshold, lo, hi, iters=ICM_ITERATIONS, tol=ITER_EPSILON):
    n = len(mu)
    pairs = pairs[dists < threshold]
    nbrs = [[] for _ in range(n)]
    for i, j in pairs:
        nbrs[i].append(j)
        nbrs[j].append(i)

    c = mu.copy()
    w_data = 1.0 / sigma2
    for _ in range(iters):
        change_sq = 0.0
        for i in range(n):
            if not nbrs[i]:
                continue
            new = (mu[i] * w_data + 2 * gamma * c[nbrs[i]].sum()) / (w_data + 2 * gamma * len(nbrs[i]))
            change_sq += (new - c[i]) ** 2
            c[i] = new
        if np.sqrt(change_sq) < tol:
            break
    return np.clip(c, lo, hi)


def colorize(p, u, v):
    yuv = np.stack([p["y"], u[p["segments"]], v[p["segments"]]], axis=-1)
    return yuv_to_rgb(yuv)


def gray_rgb(p):
    zeros = np.zeros(p["y"].shape)
    return yuv_to_rgb(np.stack([p["y"], zeros, zeros], axis=-1))


def error(pred_rgb, true_rgb):
    return np.linalg.norm(pred_rgb - true_rgb) / (true_rgb.shape[0] * true_rgb.shape[1])


def run_pipeline(p, u_svr, v_svr, args, gamma=None, threshold=None):
    gamma = args.gamma if gamma is None else gamma
    threshold = args.threshold if threshold is None else threshold
    mu_u, mu_v = predict_uv(p, u_svr, v_svr, args.boost)
    svr_rgb = colorize(p, mu_u, mu_v)
    u = icm(mu_u, p["pairs"], p["dists"], args.sigma2, gamma, threshold, -U_MAX, U_MAX)
    v = icm(mu_v, p["pairs"], p["dists"], args.sigma2, gamma, threshold, -V_MAX, V_MAX)
    return svr_rgb, colorize(p, u, v)

def load_config(models_dir):
    with open(os.path.join(models_dir, "config.json")) as f:
        return json.load(f)


def load_models(models_dir, tag=None):
    suffix = f"_{tag}" if tag else ""
    return (joblib.load(os.path.join(models_dir, f"u_svr{suffix}.joblib")),
            joblib.load(os.path.join(models_dir, f"v_svr{suffix}.joblib")))


def find_tags(models_dir):
    tags = []
    for u_path in sorted(glob.glob(os.path.join(models_dir, "u_svr_*.joblib"))):
        tag = os.path.basename(u_path)[len("u_svr_"):-len(".joblib")]
        if os.path.isfile(os.path.join(models_dir, f"v_svr_{tag}.joblib")):
            tags.append(tag)
    return tags


def list_images(path):
    found = []
    for root, _, files in os.walk(path):
        found += [os.path.join(root, f) for f in files if f.lower().endswith((".jpg", ".jpeg"))]
    return sorted(found)


def prepare_folder(folder, cfg):
    prepared = []
    paths = list_images(folder)
    for i, path in enumerate(paths, 1):
        p = prepare(path, cfg)
        if p["rgb"] is None:
            print("  skip (grayscale file, no ground truth):", path)
            continue
        prepared.append(p)
        if i % 10 == 0 or i == len(paths):
            print(f"  prepared {i}/{len(paths)} images")
    if not prepared:
        raise SystemExit(f"No color .jpg test images found under {folder}")
    return prepared


def mean_errors(prepared, u_svr, v_svr, args, gamma=None, threshold=None):
    gray, svr, full = [], [], []
    for p in prepared:
        svr_rgb, icm_rgb = run_pipeline(p, u_svr, v_svr, args, gamma, threshold)
        gray.append(error(gray_rgb(p), p["rgb"]))
        svr.append(error(svr_rgb, p["rgb"]))
        full.append(error(icm_rgb, p["rgb"]))
    return np.mean(gray), np.mean(svr), np.mean(full)

def single_image(args, cfg):
    u_svr, v_svr = load_models(args.models_dir, args.tag)
    p = prepare(args.path, cfg)
    svr_rgb, icm_rgb = run_pipeline(p, u_svr, v_svr, args)

    imsave(args.save, img_as_ubyte(icm_rgb), check_contrast=False)
    gray_img = gray_rgb(p)
    panels = [gray_img, icm_rgb] if p["rgb"] is None else [p["rgb"], gray_img, icm_rgb]
    stem, ext = os.path.splitext(args.save)
    imsave(f"{stem}_compare{ext or '.png'}", img_as_ubyte(np.concatenate(panels, axis=1)),
           check_contrast=False)
    print(f"Saved {args.save} and {stem}_compare{ext or '.png'}")

    if p["rgb"] is not None:
        print(f"Error  grayscale: {error(gray_img, p['rgb']):.3e}"
              f"   regressor only: {error(svr_rgb, p['rgb']):.3e}   + ICM: {error(icm_rgb, p['rgb']):.3e}")
    if args.show:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
        for ax, img in zip(np.atleast_1d(axes), panels):
            ax.imshow(img)
            ax.axis("off")
        plt.show()


def report(label, gray, svr, full):
    print(f"{label}")
    print(f"  Before processing (grayscale): {gray:.3e}")
    print(f"  With regressor only (no ICM):  {svr:.3e}   ({100 * (1 - svr / gray):.1f}% lower than grayscale)")
    print(f"  With regressor and ICM:        {full:.3e}   ({100 * (1 - full / gray):.1f}% lower than grayscale)")


def folder_eval(args, cfg):
    print(f"Preparing test images in {args.path} ...")
    prepared = prepare_folder(args.path, cfg)
    print(f"Evaluating on {len(prepared)} images\n")

    if args.sweep:
        tags = find_tags(args.models_dir)
        if not tags:
            raise SystemExit(f"No sweep models (u_svr_*.joblib) found in {args.models_dir}. "
                             f"Train them with: python train.py -t data/train/ --sweep")
        results = []
        for tag in tags:
            u_svr, v_svr = load_models(args.models_dir, tag)
            _, svr, full = mean_errors(prepared, u_svr, v_svr, args)
            results.append((full, svr, tag))
            print(f"  {tag:>22}   SVR only {svr:.3e}   SVR+ICM {full:.3e}")
        best = min(results)
        print(f"\nBest model by SVR+ICM error: {best[2]}  ({best[0]:.3e})")
        return

    u_svr, v_svr = load_models(args.models_dir, args.tag)

    if args.tune_icm:
        print(f"Tuning gamma (threshold fixed at {args.threshold}):")
        for g in WEIGHT_DIFF_LIST:
            _, _, full = mean_errors(prepared, u_svr, v_svr, args, gamma=g)
            print(f"  gamma {g:>4}: {full:.3e}")
        print(f"\nTuning neighbor threshold (gamma fixed at {args.gamma}); 0 = no smoothing:")
        for th in THRESHOLD_LIST:
            _, _, full = mean_errors(prepared, u_svr, v_svr, args, threshold=th)
            print(f"  threshold {th:>3}: {full:.3e}")
        return

    gray, svr, full = mean_errors(prepared, u_svr, v_svr, args)
    report(f"Average error over {len(prepared)} test images "
           f"(gamma={args.gamma}, sigma^2={args.sigma2}, threshold={args.threshold}, boost={args.boost}):",
           gray, svr, full)


def main():
    ap = argparse.ArgumentParser(description="Colorize images / evaluate the trained model.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("path", help="an image file, or a folder of color test images to evaluate")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--tag", default=None, help="use sweep models, e.g. C0.125_eps0.0625")
    ap.add_argument("--sweep", action="store_true", help="evaluate every sweep model on a folder")
    ap.add_argument("--tune-icm", action="store_true", help="sweep ICM gamma and threshold on a folder")
    ap.add_argument("-s", "--save", default="out.png", help="where to save a single colorized image")
    ap.add_argument("--show", action="store_true", help="show the result with matplotlib")
    ap.add_argument("--gamma", type=float, default=WEIGHT_DIFF, help="weight of neighbor smoothing")
    ap.add_argument("--sigma2", type=float, default=COVAR, help="variance of the SVR local potential")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help="max feature distance for touching superpixels to be neighbors (0 = no smoothing)")
    ap.add_argument("--boost", type=float, default=BOOST,
                    help="multiply SVR outputs before smoothing (original repo: 2; use 1 for none)")
    args = ap.parse_args()

    cfg = load_config(args.models_dir)
    print(f"Models: {args.models_dir}/  ({cfg.get('model', 'svr')}; features: "
          f"{cfg.get('feature', '|fft2| of Y patch')})")
    if os.path.isdir(args.path):
        folder_eval(args, cfg)
    else:
        single_image(args, cfg)


if __name__ == "__main__":
    main()