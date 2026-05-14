import csv
import time
import random
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Set

import requests


BASE_URL = "https://api.inaturalist.org/v1"
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "iNaturalistDatasetDownloader/1.0"
})


def request_json(url: str, params: Optional[dict] = None, retries: int = 5, timeout: int = 30) -> dict:
    last_err = None

    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)

            if r.status_code == 429:
                sleep_s = min(60, 2 ** attempt + random.uniform(0.5, 1.5))
                print(f"[429] Rate limited. Sleeping {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
            return r.json()

        except requests.RequestException as e:
            last_err = e
            sleep_s = min(30, 2 ** attempt + random.uniform(0.5, 1.5))
            print(f"[retry {attempt + 1}/{retries}] {e} -> sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed request after {retries} retries: {last_err}")


def sanitize_filename(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def resolve_taxon_id(taxon_name: str, rank: str = "species") -> int:
    data = request_json(
        f"{BASE_URL}/taxa",
        params={
            "q": taxon_name,
            "rank": rank,
            "per_page": 10
        }
    )

    results = data.get("results", [])
    if not results:
        raise RuntimeError(f"No taxon found for: {taxon_name}")

    exact = None
    lowered = taxon_name.strip().lower()

    for item in results:
        name = (item.get("name") or "").strip().lower()
        pref = (item.get("preferred_common_name") or "").strip().lower()
        if name == lowered or pref == lowered:
            exact = item
            break

    chosen = exact if exact else results[0]
    print(f"[taxon] Using taxon_id={chosen['id']} for {chosen.get('name')} ({chosen.get('preferred_common_name', 'no common name')})")
    return chosen["id"]


def choose_best_photo_url(photo: Dict[str, Any], size: str = "medium") -> Optional[str]:
    if size == "square":
        return photo.get("square_url") or photo.get("url")
    if size == "medium":
        return photo.get("medium_url") or photo.get("url")
    if size == "original":
        url = photo.get("url") or photo.get("medium_url") or photo.get("square_url")
        if not url:
            return None
        return url.replace("/square.", "/original.").replace("/medium.", "/original.")
    return photo.get("medium_url") or photo.get("url") or photo.get("square_url")


def stream_download(url: str, dest: Path, retries: int = 5, timeout: int = 60) -> bool:
    last_err = None

    for attempt in range(retries):
        try:
            with SESSION.get(url, stream=True, timeout=timeout) as r:
                if r.status_code == 429:
                    sleep_s = min(60, 2 ** attempt + random.uniform(0.5, 1.5))
                    print(f"[429 image] Sleeping {sleep_s:.1f}s...")
                    time.sleep(sleep_s)
                    continue

                r.raise_for_status()

                content_type = r.headers.get("Content-Type", "").lower()
                if "image" not in content_type and dest.suffix == "":
                    if "jpeg" in content_type or "jpg" in content_type:
                        dest = dest.with_suffix(".jpg")
                    elif "png" in content_type:
                        dest = dest.with_suffix(".png")
                    elif "webp" in content_type:
                        dest = dest.with_suffix(".webp")
                    else:
                        dest = dest.with_suffix(".img")

                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)

                return True

        except requests.RequestException as e:
            last_err = e
            sleep_s = min(30, 2 ** attempt + random.uniform(0.5, 1.5))
            print(f"[download retry {attempt + 1}/{retries}] {e} -> sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)

    print(f"[download failed] {url} -> {last_err}")
    return False


def fetch_observations(
    taxon_id: int,
    per_page: int = 200,
    quality_grade: str = "research",
    photo_license: Optional[str] = None,
    max_observations: Optional[int] = None,
    order_by: str = "created_at",
    order: str = "desc"
):
    page = 1
    yielded = 0

    while True:
        params = {
            "taxon_id": taxon_id,
            "photos": "true",
            "verifiable": "true",
            "per_page": per_page,
            "page": page,
            "order_by": order_by,
            "order": order,
        }

        if quality_grade and quality_grade != "any":
            params["quality_grade"] = quality_grade

        if photo_license:
            params["photo_license"] = photo_license

        data = request_json(f"{BASE_URL}/observations", params=params)
        results = data.get("results", [])

        if not results:
            break

        for obs in results:
            yield obs
            yielded += 1
            if max_observations is not None and yielded >= max_observations:
                return

        print(f"[page {page}] fetched {len(results)} observations")
        page += 1

        time.sleep(random.uniform(0.3, 0.6))


def main():
    parser = argparse.ArgumentParser(description="Bulk-download iNaturalist images for a species")
    parser.add_argument("--taxon-name", type=str, default="Panthera leo", help="Scientific/common name to resolve")
    parser.add_argument("--taxon-id", type=int, default=None, help="Use this taxon_id directly")
    parser.add_argument("--out", type=str, default="inat_lions", help="Output folder")
    parser.add_argument("--size", type=str, choices=["square", "medium", "original"], default="medium")
    parser.add_argument("--quality-grade", type=str, choices=["research", "needs_id", "casual", "any"], default="research")
    parser.add_argument(
        "--photo-license",
        type=str,
        default=None,
        help='Comma-separated, e.g. "cc0,cc-by,cc-by-sa"'
    )
    parser.add_argument("--max-observations", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = out_dir / "metadata.csv"

    taxon_id = args.taxon_id if args.taxon_id is not None else resolve_taxon_id(args.taxon_name)

    seen_photo_ids: Set[int] = set()
    total_downloaded = 0
    total_seen = 0

    existing_files = {p.name for p in images_dir.iterdir() if p.is_file()} if args.skip_existing else set()

    with open(metadata_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "observation_id",
            "photo_id",
            "taxon_id",
            "taxon_name",
            "preferred_common_name",
            "observed_on",
            "user_login",
            "photo_license_code",
            "photo_attribution",
            "photo_url",
            "local_path"
        ])

        for obs in fetch_observations(
            taxon_id=taxon_id,
            quality_grade=args.quality_grade,
            photo_license=args.photo_license,
            max_observations=args.max_observations
        ):
            obs_id = obs.get("id")
            taxon = obs.get("taxon") or {}
            taxon_name = taxon.get("name", "")
            common_name = taxon.get("preferred_common_name", "")
            observed_on = obs.get("observed_on_string") or obs.get("observed_on") or ""
            user_login = (obs.get("user") or {}).get("login", "")

            photos: List[Dict[str, Any]] = obs.get("photos") or []
            if not photos:
                continue

            for idx, photo in enumerate(photos):
                total_seen += 1

                photo_id = photo.get("id")
                if photo_id is None:
                    continue

                if photo_id in seen_photo_ids:
                    continue
                seen_photo_ids.add(photo_id)

                photo_url = choose_best_photo_url(photo, size=args.size)
                if not photo_url:
                    continue

                ext = ".jpg"
                lowered = photo_url.lower()
                if ".png" in lowered:
                    ext = ".png"
                elif ".webp" in lowered:
                    ext = ".webp"
                elif ".jpeg" in lowered:
                    ext = ".jpeg"
                elif ".jpg" in lowered:
                    ext = ".jpg"

                filename = sanitize_filename(f"obs_{obs_id}_photo_{photo_id}_{idx}{ext}")
                dest = images_dir / filename

                if args.skip_existing and filename in existing_files:
                    ok = True
                else:
                    ok = stream_download(photo_url, dest)

                if ok:
                    total_downloaded += 1
                    writer.writerow([
                        obs_id,
                        photo_id,
                        taxon_id,
                        taxon_name,
                        common_name,
                        observed_on,
                        user_login,
                        photo.get("license_code", ""),
                        photo.get("attribution", ""),
                        photo_url,
                        str(dest)
                    ])
                    csvfile.flush()

                print(f"[{total_downloaded}] {filename}")

                if args.max_images is not None and total_downloaded >= args.max_images:
                    print(f"Done. Downloaded {total_downloaded} images.")
                    return

    print(f"Done. Downloaded {total_downloaded} images from {len(seen_photo_ids)} unique photos seen.")


if __name__ == "__main__":
    main()