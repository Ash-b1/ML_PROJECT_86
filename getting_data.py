#!/usr/bin/env python3

import argparse
import csv
import io
import os
import random
import re
import shutil
import time
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"

# Scenic categories (names taken from the real crawl output). Crawled to --depth 1.
DEFAULT_CATEGORIES = [
    "Category:Landscapes of Yellowstone National Park",
    "Category:Quality images of Yellowstone National Park",
    "Category:Featured pictures of Yellowstone National Park",
    "Category:Views of Yellowstone National Park",
    "Category:Sunrises of Yellowstone National Park",
    "Category:Sunsets of Yellowstone National Park",
    "Category:Aerial photographs of Yellowstone National Park",
    "Category:Bodies of water in Yellowstone National Park",
    "Category:Hayden Valley",
    "Category:Mammoth Hot Springs",
    "Category:Norris Geyser Basin",
    "Category:Roaring Mountain",
    "Category:Obsidian Cliff",
    "Category:Bechler Meadows",
    "Category:Bechler River",
    "Category:Views from Calcite Springs Overlook",
    "Category:Yellowstone National Park in winter",
    "Category:Lower Falls of the Yellowstone River in winter",
    "Category:Upper Falls of the Yellowstone River in winter",
]

# Subcategories containing any of these words are not crawled
# (the paper's data had no animals, humans, or buildings).
EXCLUDE_WORDS = [
    "wildlife", "animal", "fauna", "bison", "elk", "bear", "wolf", "wolves",
    "bird", "fish", "deer", "moose", "woodpecker", "salamander", "coyote",
    "pelican", "pronghorn", "people", "visitor", "tourist", "portrait",
    "building", "lodge", "hotel", "inn", "museum", "center", "centre", "cabin",
    "fort", "road", "bridge", "railroad", "railway", "vehicle", "car ", "bus",
    "stagecoach", "history", "historic", "map", "diagram", "drawing", "painting",
    "engraving", "postcard", "stamp", "svg", "logo", "sign", "wildfire",
    "prescribed burn", "interior", "campground", "ranger", "by year", "by decade",
    "by century", "flickr stream", "video", "book", "publication", "flora",
    "fungi", "sound", "rescue", "maintenance", "snow removal", "snowmobile",
    "snowcoach", "skiing", "litter", "traffic", "youth", "research", "angling",
    "ceremon", "event", "fossil", "document", "boardwalk", "trail", "cycling",
    "horse", "canoe", "kayak", "teepe", "housing", "nursery", "helitack",
    "transport", "(18", "(19", " in 18", " in 19", " in 20", " in the 1",
    "yellowstone national park in art", "bacteria", "biofilm", "extremophile",
]

session = requests.Session()


def api_get(params):
    params = {**params, "format": "json", "formatversion": 2}
    r = None
    for attempt in range(5):
        # POST, because batches of long file titles overflow the URL limit of a GET
        r = session.post(API, data=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(2 ** attempt)
    r.raise_for_status()


def category_members(cat, cmtype):
    cont = {}
    while True:
        data = api_get({
            "action": "query", "list": "categorymembers",
            "cmtitle": cat, "cmtype": cmtype, "cmlimit": "max", **cont,
        })
        for m in data.get("query", {}).get("categorymembers", []):
            yield m["title"]
        if "continue" in data:
            cont = data["continue"]
        else:
            break


def crawl(root, depth):
    files, seen = set(), set()
    queue = [(root, 0)]
    while queue:
        cat, d = queue.pop(0)
        if cat in seen:
            continue
        seen.add(cat)
        found = list(category_members(cat, "file"))
        files.update(found)
        print(f"  [{d}] {cat}: {len(found)} files")
        if d < depth:
            for sub in category_members(cat, "subcat"):
                if not any(w in sub.lower() for w in EXCLUDE_WORDS):
                    queue.append((sub, d + 1))
    return files


def image_infos(titles, width):
    titles = list(titles)
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        data = api_get({
            "action": "query", "titles": "|".join(batch),
            "prop": "imageinfo", "iiprop": "url|mime|size",
            "iiurlwidth": width,
        })
        for page in data.get("query", {}).get("pages", []):
            info = page.get("imageinfo")
            if info:
                yield page["title"], info[0]


def download(url):
    for attempt in range(5):
        r = session.get(url, timeout=60)
        if r.status_code == 200:
            return r.content
        if r.status_code == 429:  # rate limited, back off
            time.sleep(5 * (attempt + 1))
        else:
            break
    return None


def is_grayscale(img, tol=6.0):
    """True if the R, G, B channels are nearly identical."""
    a = np.asarray(img.convert("RGB").resize((100, 100)), dtype=np.int16)
    spread = np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean()
    return spread < tol


def safe_name(title):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", title.replace("File:", ""))
    name = os.path.splitext(name)[0][:80]
    return name + ".jpg"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", help="output folder (default: data)")
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES,
                    help='Commons categories, e.g. "Category:Lakes of Yellowstone National Park"')
    ap.add_argument("--depth", type=int, default=1, help="subcategory crawl depth (default 1)")
    ap.add_argument("--max-images", type=int, default=300, help="stop after this many accepted images")
    ap.add_argument("--train-frac", type=float, default=0.8,
                    help="fraction for training (paper used 98/216, about 0.45)")
    ap.add_argument("--width", type=int, default=500, help="output width in pixels")
    ap.add_argument("--min-width", type=int, default=500, help="skip originals narrower than this")
    ap.add_argument("--min-aspect", type=float, default=1.0, help="min width/height (skip portraits)")
    ap.add_argument("--max-aspect", type=float, default=2.0, help="max width/height (skip extreme panoramas)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--contact", default="your-email@example.com",
                    help="your email, sent in the User-Agent as Wikimedia requests")
    args = ap.parse_args()

    session.headers["User-Agent"] = (
        f"recolorizer-dataset-builder/1.0 (student ML project; {args.contact})"
    )

    # 1. Collect candidate file titles
    print("Crawling categories...")
    titles = set()
    for cat in args.categories:
        titles |= crawl(cat, args.depth)
    titles = sorted(
        t for t in titles
        if t.lower().endswith((".jpg", ".jpeg"))
        and not any(w in t.lower() for w in EXCLUDE_WORDS)
    )
    print(f"{len(titles)} candidate JPEG files found.")

    # 2. Filter by metadata
    print("Fetching image info...")
    candidates = []
    for title, info in image_infos(titles, args.width):
        w, h = info.get("width", 0), info.get("height", 0)
        if info.get("mime") != "image/jpeg" or w < args.min_width or h == 0:
            continue
        if not (args.min_aspect <= w / h <= args.max_aspect):
            continue
        url = info.get("thumburl") or info["url"]
        candidates.append((title, url))
    print(f"{len(candidates)} candidates passed the size/shape filters.")

    random.Random(args.seed).shuffle(candidates)

    # 3-4. Download, resize, drop grayscale
    staging = os.path.join(args.out, "_all")
    os.makedirs(staging, exist_ok=True)
    accepted = []  # (filename, title)
    for title, url in candidates:
        if len(accepted) >= args.max_images:
            break
        data = download(url)
        if data is None:
            print("  skip (download failed):", title)
            continue
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            print("  skip (unreadable):", title)
            continue
        if is_grayscale(img):
            print("  skip (grayscale):", title)
            continue
        new_h = max(1, round(img.height * args.width / img.width))
        img = img.resize((args.width, new_h), Image.LANCZOS)
        fname = safe_name(title)
        if any(fname == a[0] for a in accepted):  # name collision after sanitizing
            fname = f"{len(accepted)}_{fname}"
        img.save(os.path.join(staging, fname), quality=95)
        accepted.append((fname, title))
        print(f"  [{len(accepted)}/{args.max_images}] {fname}")
        time.sleep(0.5)  # be polite to Wikimedia

    if not accepted:
        print("No images were saved. Check your categories and network connection.")
        return

    # 5. Split into train/test and write attribution
    random.Random(args.seed).shuffle(accepted)
    n_train = round(len(accepted) * args.train_frac)
    splits = {"train": accepted[:n_train], "test": accepted[n_train:]}
    for split, items in splits.items():
        os.makedirs(os.path.join(args.out, split), exist_ok=True)
        for fname, _ in items:
            shutil.move(os.path.join(staging, fname), os.path.join(args.out, split, fname))
    shutil.rmtree(staging, ignore_errors=True)

    with open(os.path.join(args.out, "attribution.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "file", "commons_page"])
        for split, items in splits.items():
            for fname, title in items:
                wr.writerow([split, fname, "https://commons.wikimedia.org/wiki/" + quote(title.replace(" ", "_"))])

    print(f"\nDone: {len(splits['train'])} train, {len(splits['test'])} test images in '{args.out}/'.")
    print(f"Next: python train_svr.py -t {os.path.join(args.out, 'train')}/")


if __name__ == "__main__":
    main()