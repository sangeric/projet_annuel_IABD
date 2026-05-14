import os, time, csv, re, hashlib, random
import requests
from dotenv import load_dotenv
from pathlib import Path
from collections import Counter

load_dotenv()

BASE = "https://api.openverse.org/v1"
TOKEN_URL = f"{BASE}/auth_tokens/token/"
IMAGES_URL = f"{BASE}/images/"

CLIENT_ID = (os.environ.get("OPENVERSE_CLIENT_ID") or "").strip()
CLIENT_SECRET = (os.environ.get("OPENVERSE_CLIENT_SECRET") or "").strip()
if not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError("Missing OPENVERSE_CLIENT_ID / OPENVERSE_CLIENT_SECRET")

OUT_DIR = Path("data/downloads")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = Path("data/manifest_images.csv")
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

FAIL_LOG = Path("data/download_failures.csv")
FAIL_LOG.parent.mkdir(parents=True, exist_ok=True)

MAX_DEPTH = 240
PAGE_SIZE = 50
SAFE_MAX_PAGES = 4

USE_THUMBNAIL_FALLBACK = True

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "*/*",
})

REJECT_WORDS = {
    "cat", "kitten", "kitty", "housecat", "domestic cat", "pet cat",
    "lion", "lioness", "tiger", "panther", "jaguar", "leopard", "lynx",
    "cougar", "puma", "ocelot", "bobcat", "caracal", "serval", "wildcat",
    "snow leopard", "clouded leopard", "mountain lion",
    "dog", "dogs", "puppy", "wolf", "fox", "hyena", "horse", "zebra",
    "giraffe", "elephant", "monkey", "ape", "gorilla", "bird", "fish",
    "shark", "insect", "bug", "snake", "lizard",

    "cheater", "cheetos", "cheeto", "cheetah print", "animal print",
    "leopard print", "spotted pattern", "spots pattern",

    "cartoon", "drawing", "illustration", "painting", "sketch", "watercolor",
    "acrylic", "oil painting", "digital art", "concept art", "line art",
    "pixel art", "anime", "manga", "graffiti", "mural", "poster", "print",
    "wall art", "tattoo", "tattoos", "sticker", "decal", "clipart", "icon",
    "symbol", "emblem", "badge", "crest", "coat of arms", "heraldry", "seal",
    "logo", "brand", "mascot",

    "statue", "sculpture", "figurine", "toy", "plush", "stuffed animal",
    "costume", "mask", "puppet", "decoration", "ornament", "gargoyle",
    "carving", "engraving", "relief", "stone", "bronze", "wooden", "ceramic",
    "door", "doorknob", "knob", "furniture", "jewelry", "necklace", "ring",

    "shirt", "tshirt", "hoodie", "jacket", "dress", "shoe", "sneaker",
    "fashion", "fabric", "textile", "merch", "merchandise",

    "movie", "film", "theatre", "theater", "song", "album", "band", "book",
    "cover", "title", "word", "text", "quote", "poem", "character",
    "sports team", "team", "club", "party", "organization",

    "building", "city", "town", "restaurant", "hotel", "museum", "temple",
    "monument", "sea", "ocean", "air", "plane", "balloon", "country", "flag",

    "human", "person", "people", "man", "woman", "child", "crowd", "gathering",
    "selfie", "portrait person",
}

DOMAIN_WORDS = {
    "cheetah", "cheetah cub", "adult cheetah", "wild cheetah",
    "african cheetah", "cheetah face", "cheetah portrait",
}

CHEETAH_HINT_WORDS = {
    "cheetah portrait", "cheetah face", "cheetah head", "running cheetah",
    "sprinting cheetah", "walking cheetah", "resting cheetah",
    "sleeping cheetah", "cheetah cub", "wild cheetah",
    "african cheetah", "cheetah in savannah", "cheetah looking at camera",
    "cheetah full body", "close up cheetah",
}

BASE_QUERIES = [
    "cheetah",
    "wild cheetah",
    "african cheetah",
    "adult cheetah",
    "cheetah cub",
    "cheetah portrait",
    "cheetah face",
    "cheetah head",
    "running cheetah",
    "sprinting cheetah",
    "walking cheetah",
    "resting cheetah",
    "sleeping cheetah",
]

MODIFIERS = [
    "",
    "close up",
    "closeup",
    "portrait",
    "full body",
    "standing",
    "walking",
    "running",
    "sprinting",
    "resting",
    "sleeping",
    "in savannah",
    "looking at camera",
]

SOURCES = [None]


def build_queries():
    seen = set()
    out = []
    for b in BASE_QUERIES:
        for m in MODIFIERS:
            q = (b if not m else f"{b} {m}").strip()
            if q not in seen:
                out.append(q)
                seen.add(q)
    return out


QUERY_TERMS = build_queries()


def get_token():
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        timeout=30,
    )
    resp.raise_for_status()
    j = resp.json()
    if "access_token" not in j:
        raise RuntimeError(f"No access_token in response: {j}")
    return j["access_token"]


def search_images(token, q, page=1, page_size=50, **kwargs):
    params = {"q": q, "page": page, "page_size": page_size}
    for k, v in kwargs.items():
        if v is not None:
            params[k] = v

    resp = requests.get(
        IMAGES_URL,
        params=params,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=30,
    )

    if resp.status_code != 200:
        if resp.status_code == 401 and "pagination depth may not exceed" in resp.text:
            return None

        print("IMAGES STATUS:", resp.status_code)
        print("IMAGES URL:", resp.url)
        print("IMAGES BODY:", resp.text[:500])

    resp.raise_for_status()
    return resp.json()


def normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def text_blob(item) -> str:
    title = normalize(item.get("title") or "")
    creator = normalize(item.get("creator") or "")
    tags = item.get("tags") or []
    tag_names = []
    for t in tags:
        if isinstance(t, dict):
            tag_names.append(normalize(t.get("name", "")))
        else:
            tag_names.append(normalize(str(t)))
    return " ".join([title, " ".join(tag_names), creator]).strip()


def score_item(item) -> int:
    blob = text_blob(item)
    score = 0

    if any(w in blob for w in DOMAIN_WORDS):
        score += 10
    if any(w in blob for w in CHEETAH_HINT_WORDS):
        score += 5
    if any(w in blob for w in REJECT_WORDS):
        score -= 100

    return score


def should_keep(item):
    blob = text_blob(item)

    if any(w in blob for w in REJECT_WORDS):
        return False, "rejected_term"

    if any(w in blob for w in DOMAIN_WORDS) or any(w in blob for w in CHEETAH_HINT_WORDS):
        return True, ""

    return False, "not_cheetah"


def load_seen_ids(path: Path) -> set[str]:
    seen = set()
    if path.exists() and path.stat().st_size > 0:
        try:
            with path.open("r", encoding="utf-8") as rf:
                for row in csv.DictReader(rf):
                    if row.get("openverse_id"):
                        seen.add(row["openverse_id"])
        except Exception:
            pass
    return seen


def log_failure(row: dict):
    exists = FAIL_LOG.exists() and FAIL_LOG.stat().st_size > 0
    with FAIL_LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists or f.tell() == 0:
            w.writeheader()
        w.writerow(row)


def download_with_retries(urls, dest: Path, timeout=60, max_tries=4):
    last_exc = None

    for url in urls:
        if not url:
            continue

        for attempt in range(1, max_tries + 1):
            try:
                with SESSION.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
                    if r.status_code >= 400:
                        raise requests.HTTPError(f"HTTP {r.status_code}", response=r)

                    h = hashlib.sha256()
                    n = 0
                    dest.parent.mkdir(parents=True, exist_ok=True)

                    with dest.open("wb") as out:
                        for chunk in r.iter_content(chunk_size=1024 * 256):
                            if not chunk:
                                continue
                            out.write(chunk)
                            h.update(chunk)
                            n += len(chunk)

                    if n == 0:
                        raise RuntimeError("Downloaded 0 bytes")

                    return n, h.hexdigest(), url

            except Exception as e:
                last_exc = e
                sleep_s = min(2.5, 0.4 * (2 ** (attempt - 1))) + random.random() * 0.2
                time.sleep(sleep_s)

    raise last_exc if last_exc else RuntimeError("No URLs provided")


def main():
    token = get_token()
    seen_ids = load_seen_ids(MANIFEST_PATH)
    reject_stats = Counter()

    with MANIFEST_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "bucket", "query", "source_filter", "openverse_id", "source", "title",
            "license", "creator", "creator_url", "detail_url",
            "image_url", "thumbnail_url",
            "local_path", "bytes", "sha256",
            "score", "label"
        ])
        if f.tell() == 0:
            w.writeheader()

        bucket = "CHEETAH_CANDIDATES"

        for source_filter in SOURCES:
            for q in QUERY_TERMS:
                page = 1
                while page <= SAFE_MAX_PAGES:
                    payload = search_images(
                        token,
                        q=q,
                        page=page,
                        page_size=PAGE_SIZE,
                        source=source_filter,
                        mature=False,
                    )

                    if payload is None:
                        print(f"Depth cap hit. Stopping pagination for query='{q}' source='{source_filter}'.")
                        break

                    results = payload.get("results", [])
                    if not results:
                        break

                    print(f"\nsource_filter={source_filter} query={q} page={page} total={payload.get('result_count')} page_results={len(results)}")

                    downloaded = 0
                    for item in results:
                        ov_id = item.get("id")
                        if not ov_id or ov_id in seen_ids:
                            continue

                        ok, reason = should_keep(item)
                        if not ok:
                            reject_stats[reason] += 1
                            continue

                        candidate_urls = [
                            item.get("url"),
                            item.get("original_url"),
                        ]
                        if USE_THUMBNAIL_FALLBACK:
                            candidate_urls.append(item.get("thumbnail"))
                        candidate_urls = [u for u in candidate_urls if u]

                        if not candidate_urls:
                            reject_stats["no_url"] += 1
                            continue

                        ext = (item.get("extension") or "jpg").lower()
                        if ext not in {"jpg", "jpeg", "png"}:
                            ext = "jpg"

                        local = OUT_DIR / f"{ov_id}.{ext}"
                        if local.exists():
                            seen_ids.add(ov_id)
                            continue

                        try:
                            nbytes, digest, final_url = download_with_retries(candidate_urls, local)
                        except Exception as e:
                            reject_stats["download_fail"] += 1
                            log_failure({
                                "openverse_id": ov_id,
                                "query": q,
                                "source": item.get("source") or "",
                                "title": (item.get("title") or "")[:200],
                                "tried_urls": " | ".join(candidate_urls[:4]),
                                "error": repr(e),
                            })
                            continue

                        sc = score_item(item)
                        w.writerow({
                            "bucket": bucket,
                            "query": q,
                            "source_filter": source_filter or "",
                            "openverse_id": ov_id,
                            "source": item.get("source"),
                            "title": item.get("title"),
                            "license": item.get("license"),
                            "creator": item.get("creator"),
                            "creator_url": item.get("creator_url"),
                            "detail_url": item.get("foreign_landing_url"),
                            "image_url": final_url,
                            "thumbnail_url": item.get("thumbnail"),
                            "local_path": str(local),
                            "bytes": nbytes,
                            "sha256": digest,
                            "score": sc,
                            "label": "",
                        })
                        seen_ids.add(ov_id)
                        downloaded += 1

                    print("Downloaded this page:", downloaded)

                    time.sleep(0.35)
                    page += 1

        print("\nReject stats:", dict(reject_stats))


if __name__ == "__main__":
    main()