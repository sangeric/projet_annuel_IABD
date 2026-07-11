"""
build_dataset.py — multi-source "clear & centered" image collector.

Pulls candidate images for one or more categories (cheetah, lion, cat) from
several independent sources, then runs every candidate through a real
quality gate before counting it:

    1. minimum resolution
    2. blur detection (Laplacian variance)
    3. open-vocabulary object detection (YOLO-World) using the ACTUAL
       species name as the prompt — not a fixed COCO class list — so
       lion/cheetah are correctly detected instead of guessed as
       bear/cow/dog
    4. crop tightly to the detected animal with a small padding margin
    5. perceptual-hash dedup so near-identical shots don't inflate the count

It keeps pulling from each source, and moves on to the next source, until
either the target count is hit or every source is exhausted — and reports
the truth if the target can't be reached, rather than silently stopping.

Usage:
    pip install -r requirements.txt
    python build_dataset.py --categories cheetah lion cat --target 10000 --out dataset

Optional API keys (unlocks more sources / higher rate limits):
    --flickr-api-key YOUR_KEY
    --pixabay-api-key YOUR_KEY

Without those keys, the script still runs on iNaturalist + GBIF + Wikimedia
Commons, which require no key.
"""

import argparse
import csv
import io
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import requests

try:
    import cv2
    import numpy as np
    from PIL import Image
    import imagehash
except ImportError as e:
    raise SystemExit(
        f"Missing dependency: {e}. Run: pip install -r requirements.txt --break-system-packages"
    )


# --------------------------------------------------------------------------
# Category definitions — the actual names used to query each source and to
# prompt the open-vocabulary detector.
# --------------------------------------------------------------------------

CATEGORY_CONFIG = {
    "cheetah": {
        "scientific_name": "Acinonyx jubatus",
        "common_name": "cheetah",
        "detector_prompts": ["cheetah"],
        "wikimedia_categories": ["Category:Acinonyx jubatus"],
        "gbif_taxon_key": None,  # resolved at runtime
    },
    "lion": {
        "scientific_name": "Panthera leo",
        "common_name": "lion",
        "detector_prompts": ["lion"],
        "wikimedia_categories": ["Category:Panthera leo"],
        "gbif_taxon_key": None,
    },
    "cat": {
        "scientific_name": "Felis catus",
        "common_name": "cat",
        "detector_prompts": ["cat", "domestic cat"],
        "wikimedia_categories": ["Category:Felis catus", "Category:Cats"],
        "gbif_taxon_key": None,
    },
}

MIN_DIM = 400              # reject images smaller than this on either side
MIN_BLUR_VARIANCE = 60.0   # Laplacian variance below this = too blurry
MIN_AREA_RATIO = 0.08      # detected animal must fill at least this much of frame
MIN_CONFIDENCE = 0.25
CROP_PADDING_FRAC = 0.08   # padding around the detected box, as a fraction of box size
HASH_DISTANCE_THRESHOLD = 6  # perceptual-hash near-duplicate cutoff

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "MultiSourceDatasetBuilder/1.0 (research use)"})


def http_get_json(url: str, params: Optional[dict] = None, retries: int = 4, timeout: int = 30) -> Optional[dict]:
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                time.sleep(min(30, 2 ** attempt + random.uniform(0.5, 1.5)))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(min(15, 2 ** attempt))
    return None


def http_get_bytes(url: str, retries: int = 3, timeout: int = 30) -> Optional[bytes]:
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 429:
                time.sleep(min(30, 2 ** attempt + random.uniform(0.5, 1.5)))
                continue
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            time.sleep(min(15, 2 ** attempt))
    return None


# --------------------------------------------------------------------------
# Sources. Each is a generator that yields dicts:
#   {"url": str, "source": str, "license": str, "attribution": str}
# Generators should keep paging until they run out of results; the caller
# decides when to stop early (quota met).
# --------------------------------------------------------------------------

def source_inaturalist(cfg: dict, per_page: int = 200) -> Iterator[dict]:
    base = "https://api.inaturalist.org/v1"
    taxa = http_get_json(f"{base}/taxa", params={"q": cfg["scientific_name"], "rank": "species", "per_page": 5})
    if not taxa or not taxa.get("results"):
        return
    taxon_id = taxa["results"][0]["id"]

    page = 1
    while True:
        data = http_get_json(
            f"{base}/observations",
            params={
                "taxon_id": taxon_id,
                "photos": "true",
                "verifiable": "true",
                "quality_grade": "research",
                "per_page": per_page,
                "page": page,
                "order_by": "created_at",
                "order": "desc",
            },
        )
        if not data or not data.get("results"):
            return
        for obs in data["results"]:
            photos = obs.get("photos") or []
            if not photos:
                continue
            photo = photos[0]  # lead photo only — see prior discussion
            url = (photo.get("url") or "").replace("/square.", "/original.").replace("/medium.", "/original.")
            if not url:
                continue
            yield {
                "url": url,
                "source": "inaturalist",
                "license": photo.get("license_code", ""),
                "attribution": photo.get("attribution", ""),
            }
        page += 1
        time.sleep(0.4)


def source_gbif(cfg: dict, page_size: int = 100) -> Iterator[dict]:
    base = "https://api.gbif.org/v1"
    offset = 0
    while True:
        data = http_get_json(
            f"{base}/occurrence/search",
            params={
                "scientificName": cfg["scientific_name"],
                "mediaType": "StillImage",
                "limit": page_size,
                "offset": offset,
            },
        )
        if not data or not data.get("results"):
            return
        for occ in data["results"]:
            for media in occ.get("media", []):
                if media.get("type") != "StillImage":
                    continue
                url = media.get("identifier")
                if not url:
                    continue
                yield {
                    "url": url,
                    "source": "gbif",
                    "license": media.get("license", occ.get("license", "")),
                    "attribution": media.get("rightsHolder", occ.get("recordedBy", "")),
                }
        if offset + page_size >= data.get("count", 0):
            return
        offset += page_size
        time.sleep(0.4)


def source_wikimedia(cfg: dict) -> Iterator[dict]:
    base = "https://commons.wikimedia.org/w/api.php"
    for category in cfg["wikimedia_categories"]:
        gcmcontinue = None
        while True:
            params = {
                "action": "query",
                "generator": "categorymembers",
                "gcmtitle": category,
                "gcmtype": "file",
                "gcmlimit": "50",
                "prop": "imageinfo",
                "iiprop": "url|size|extmetadata",
                "format": "json",
            }
            if gcmcontinue:
                params["gcmcontinue"] = gcmcontinue

            data = http_get_json(base, params=params)
            if not data:
                return

            pages = (data.get("query") or {}).get("pages") or {}
            for page in pages.values():
                infos = page.get("imageinfo") or []
                if not infos:
                    continue
                info = infos[0]
                url = info.get("url")
                if not url or not url.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                meta = info.get("extmetadata", {}) or {}
                yield {
                    "url": url,
                    "source": "wikimedia",
                    "license": meta.get("LicenseShortName", {}).get("value", ""),
                    "attribution": meta.get("Artist", {}).get("value", ""),
                }

            cont = data.get("continue")
            if not cont:
                break
            gcmcontinue = cont.get("gcmcontinue")
            time.sleep(0.3)


def source_flickr(cfg: dict, api_key: str, per_page: int = 250) -> Iterator[dict]:
    if not api_key:
        return
    base = "https://api.flickr.com/services/rest/"
    page = 1
    while page <= 40:  # flickr search caps around 4000 results for unauthenticated search sanity
        data = http_get_json(
            base,
            params={
                "method": "flickr.photos.search",
                "api_key": api_key,
                "text": cfg["common_name"],
                "license": "1,2,3,4,5,6,9,10",  # CC-licensed variants
                "content_type": 1,
                "media": "photos",
                "sort": "relevance",
                "per_page": per_page,
                "page": page,
                "extras": "url_o,url_l,license,owner_name",
                "format": "json",
                "nojsoncallback": 1,
            },
        )
        if not data or data.get("stat") != "ok":
            return
        photos = (data.get("photos") or {}).get("photo", [])
        if not photos:
            return
        for p in photos:
            url = p.get("url_o") or p.get("url_l")
            if not url:
                continue
            yield {
                "url": url,
                "source": "flickr",
                "license": str(p.get("license", "")),
                "attribution": p.get("ownername", ""),
            }
        page += 1
        time.sleep(0.3)


def source_pixabay(cfg: dict, api_key: str = "56603180-e468c29edd0d42f296fb68d5c", per_page: int = 200) -> Iterator[dict]:
    if not api_key:
        return
    base = "https://pixabay.com/api/"
    page = 1
    while True:
        data = http_get_json(
            base,
            params={
                "key": api_key,
                "q": cfg["common_name"],
                "image_type": "photo",
                "per_page": per_page,
                "page": page,
                "safesearch": "true",
            },
        )
        if not data or not data.get("hits"):
            return
        for hit in data["hits"]:
            url = hit.get("largeImageURL") or hit.get("webformatURL")
            if not url:
                continue
            yield {
                "url": url,
                "source": "pixabay",
                "license": "Pixabay License",
                "attribution": hit.get("user", ""),
            }
        if page * per_page >= data.get("totalHits", 0):
            return
        page += 1
        time.sleep(0.3)


# --------------------------------------------------------------------------
# Quality gate: resolution -> blur -> open-vocabulary detection+crop -> dedupe
# --------------------------------------------------------------------------

class QualityGate:
    def __init__(self, prompts: List[str]):
        from ultralytics import YOLOWorld
        self.model = YOLOWorld("yolov8s-worldv2.pt")
        self.model.set_classes(prompts)
        self.seen_hashes: List[Any] = []

    def process(self, raw_bytes: bytes) -> Tuple[Optional[Image.Image], str]:
        """Returns (cropped_image_or_None, rejection_reason_or_'')."""
        try:
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception:
            return None, "unreadable"

        w, h = img.size
        if w < MIN_DIM or h < MIN_DIM:
            return None, "too_small"

        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_var < MIN_BLUR_VARIANCE:
            return None, "blurry"

        results = self.model.predict(cv_img, verbose=False)[0]
        img_area = w * h

        best_box = None
        best_conf = 0.0
        for box in results.boxes:
            conf = float(box.conf)
            if conf < MIN_CONFIDENCE:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area_ratio = ((x2 - x1) * (y2 - y1)) / img_area
            if area_ratio < MIN_AREA_RATIO:
                continue
            if conf > best_conf:
                best_conf = conf
                best_box = (x1, y1, x2, y2)

        if best_box is None:
            return None, "no_confident_detection"

        x1, y1, x2, y2 = best_box
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(bw * CROP_PADDING_FRAC), int(bh * CROP_PADDING_FRAC)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        cropped = img.crop((x1, y1, x2, y2))

        if cropped.width < MIN_DIM // 2 or cropped.height < MIN_DIM // 2:
            return None, "crop_too_small"

        phash = imagehash.phash(cropped)
        for existing in self.seen_hashes:
            if phash - existing <= HASH_DISTANCE_THRESHOLD:
                return None, "duplicate"
        self.seen_hashes.append(phash)

        return cropped, ""


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def build_category(
    category: str,
    cfg: dict,
    target: int,
    out_dir: Path,
    # flickr_key: str,
    pixabay_key: str,
) -> Dict[str, Any]:
    cat_dir = out_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{category}_log.csv"

    gate = QualityGate(cfg["detector_prompts"])

    sources = [
        ("inaturalist", source_inaturalist(cfg)),
        ("gbif", source_gbif(cfg)),
        ("wikimedia", source_wikimedia(cfg)),
        # ("flickr", source_flickr(cfg, flickr_key)),
        ("pixabay", source_pixabay(cfg, pixabay_key)),
    ]

    kept = 0
    per_source_kept = {name: 0 for name, _ in sources}
    per_source_seen = {name: 0 for name, _ in sources}
    rejection_counts: Dict[str, int] = {}
    seen_urls: Set[str] = set()

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "url", "status", "local_path", "license", "attribution"])

        for source_name, gen in sources:
            if kept >= target:
                break
            print(f"\n[{category}] pulling from {source_name} (kept so far: {kept}/{target})")

            for item in gen:
                if kept >= target:
                    break
                url = item["url"]
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                per_source_seen[source_name] += 1

                raw = http_get_bytes(url)
                if raw is None:
                    writer.writerow([source_name, url, "download_failed", "", "", ""])
                    continue

                cropped, reason = gate.process(raw)
                if cropped is None:
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    writer.writerow([source_name, url, f"rejected_{reason}", "", "", ""])
                    continue

                filename = f"{category}_{kept:06d}.jpg"
                dest = cat_dir / filename
                cropped.save(dest, quality=92)

                kept += 1
                per_source_kept[source_name] += 1
                writer.writerow([source_name, url, "kept", str(dest), item.get("license", ""), item.get("attribution", "")])

                if kept % 100 == 0:
                    print(f"  [{category}] {kept}/{target} kept")

            print(f"[{category}] {source_name} exhausted or quota reached "
                  f"({per_source_kept[source_name]} kept from {per_source_seen[source_name]} seen)")

    return {
        "category": category,
        "target": target,
        "kept": kept,
        "shortfall": max(0, target - kept),
        "per_source_kept": per_source_kept,
        "per_source_seen": per_source_seen,
        "rejection_counts": rejection_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-source clear/centered image dataset builder")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORY_CONFIG.keys()),
                         choices=list(CATEGORY_CONFIG.keys()))
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--out", type=str, default="dataset")
    # parser.add_argument("--flickr-api-key", type=str, default="")
    parser.add_argument("--pixabay-api-key", type=str, default="")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for category in args.categories:
        cfg = CATEGORY_CONFIG[category]
        result = build_category(category, cfg, args.target, out_dir, args.flickr_api_key, args.pixabay_api_key)
        summary.append(result)

    print("\n\n===== FINAL SUMMARY =====")
    for r in summary:
        status = "MET TARGET" if r["shortfall"] == 0 else f"SHORT BY {r['shortfall']}"
        print(f"\n{r['category'].upper()}: {r['kept']}/{r['target']} kept — {status}")
        print("  per-source kept:", r["per_source_kept"])
        print("  rejection reasons:", r["rejection_counts"])

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary written to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
