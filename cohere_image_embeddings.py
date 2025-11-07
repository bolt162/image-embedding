
# -*- coding: utf-8 -*-
"""
Multimodal image embeddings with Cohere Embed v4.0
- Downloads the two SJSU sample images (if not already present)
- Optionally uses any local images placed next to this script
- Builds an image embedding index (L2-normalized)
- Runs two natural-language queries against the image index
- Prints cosine similarity scores and top matches

Usage:
  1) pip install cohere pillow numpy requests
  2) Set your key: export COHERE_API_KEY=...  (or edit API_KEY below)
  3) python cohere_image_embeddings.py
"""

import os
import base64
from io import BytesIO
from typing import List, Tuple
import requests
import numpy as np
from PIL import Image
import cohere

# ---------- Config ----------
API_KEY = os.getenv("COHERE_API_KEY", "API_KEY_HERE")  # or replace with your key
MODEL = "embed-v4.0"

SJSU_IMAGES = {
    "ADV_college-of-science_2.jpg": "https://www.sjsu.edu/_images/people/ADV_college-of-science_2.jpg",
    "ADV_college-of-social-sciences_2.jpg": "https://www.sjsu.edu/_images/people/ADV_college-of-social-sciences_2.jpg",
}

# You can also drop files named like these in the same folder to include them:
OPTIONAL_LOCAL_IMAGES = [
    "image.png",  # if you have a local image named image.png
]

QUERIES = [
    "person with tape and cap",
    "cart with single tire",
]

# ---------- Helpers ----------
def ensure_downloads(img_map: dict, dest_dir: str = ".") -> List[str]:
    """Download images from URLs if missing; return list of local paths."""
    paths = []
    for fname, url in img_map.items():
        fpath = os.path.join(dest_dir, fname)
        if not os.path.exists(fpath):
            try:
                print(f"Downloading {url} -> {fpath}")
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                with open(fpath, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                print(f"WARNING: could not download {url}: {e}")
                continue
        paths.append(fpath)
    return paths

def include_optional_locals(names: List[str]) -> List[str]:
    """Return only those optional local image paths that exist."""
    present = []
    for n in names:
        if os.path.exists(n):
            present.append(n)
    return present

def image_to_base64_data_url(image_path: str) -> str:
    """Convert an image file to a base64 data URL acceptable by Cohere."""
    with Image.open(image_path) as img:
        # Preserve original format if possible; default to PNG otherwise
        fmt = (img.format or "PNG").upper()
        if fmt not in {"PNG", "JPEG", "JPG", "WEBP", "GIF"}:
            fmt = "PNG"
        buf = BytesIO()
        img.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = "jpeg" if fmt in {"JPEG", "JPG"} else fmt.lower()
        return f"data:image/{mime};base64,{b64}"

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / n

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # expects L2-normalized vectors
    return float(np.dot(a, b))

# ---------- Cohere client ----------
co = cohere.ClientV2(api_key=API_KEY)

# ---------- Build image embedding store ----------
def embed_images(image_paths: List[str]) -> List[Tuple[str, np.ndarray]]:
    """Embed each image (one-per-call for maximum compatibility) and return normalized vectors."""
    out: List[Tuple[str, np.ndarray]] = []
    for p in image_paths:
        data_url = image_to_base64_data_url(p)
        res = co.embed(
            images=[data_url],           # one image per call (SDK/API safe)
            model=MODEL,
            embedding_types=["float"],
            input_type="image",
        )
        vec = np.array(res.embeddings.float[0], dtype=np.float32)
        vec = l2_normalize(vec)
        out.append((p, vec))
    return out

# ---------- Natural-language search over images ----------
def search_images(query: str, image_index: List[Tuple[str, np.ndarray]], top_k: int = 5) -> List[Tuple[str, float]]:
    qres = co.embed(
        texts=[query],
        model=MODEL,
        embedding_types=["float"],
        input_type="search_query",  # cross-modal retrieval (text->image)
    )
    qvec = np.array(qres.embeddings.float[0], dtype=np.float32)
    qvec = l2_normalize(qvec)

    scored = []
    for path, ivec in image_index:
        scored.append((path, cosine_sim(qvec, ivec)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

# ---------- Pairwise image similarity (optional) ----------
def pairwise_image_sims(image_index: List[Tuple[str, np.ndarray]]) -> None:
    names = [p for p, _ in image_index]
    vecs = np.stack([v for _, v in image_index], axis=0)
    sims = vecs @ vecs.T  # cosine since normalized
    print("\nPairwise image cosine similarities:")
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                print(f"  {a}  vs  {b}  ->  {sims[i, j]:.4f}")

def main():
    # 1) Collect images (download SJSU samples + include any local optionals)
    paths = ensure_downloads(SJSU_IMAGES)
    paths += include_optional_locals(OPTIONAL_LOCAL_IMAGES)
    if not paths:
        print("No images found. Add local images or check URLs.")
        return

    print("\nImages to index:")
    for p in paths:
        print(" -", p)

    # 2) Embed images
    image_index = embed_images(paths)
    print(f"\nIndexed {len(image_index)} images.")

    # 3) Run the two assignment queries
    for q in QUERIES:
        print(f"\nQuery: {q}")
        results = search_images(q, image_index, top_k=min(5, len(image_index)))
        for path, score in results:
            print(f"  {path}  |  cosine={score:.4f}")

    # 4) (Optional) directly compare images
    pairwise_image_sims(image_index)

if __name__ == "__main__":
    main()
