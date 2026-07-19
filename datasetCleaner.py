from ultralytics import YOLO
from pathlib import Path
from PIL import Image
from tqdm import tqdm

model = YOLO("yolov8n.pt")

ANIMAL_CLASSES = {15, 16, 17, 18, 19, 20, 21, 22, 23}
MIN_CONFIDENCE = 0.3
MIN_AREA_RATIO = 0.05

input_root = Path("felidae_classifier/dataset/")
output_root = Path("felidae_classifier/dataset_clean/")

img_paths = list(input_root.rglob("*.jpg")) + list(input_root.rglob("*.png"))

stats = {
    "total": len(img_paths),
    "kept": 0,
    "discarded_no_animal": 0,
    "discarded_too_small": 0,
    "by_class": {}
}

for img_path in tqdm(img_paths, desc="Cleaning dataset"):
    results = model(str(img_path), verbose=False)[0]
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    img_area = w * h

    best_box = None
    best_conf = 0.0
    found_animal = False

    for box in results.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        if cls in ANIMAL_CLASSES:
            found_animal = True
            if conf > MIN_CONFIDENCE:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area_ratio = ((x2 - x1) * (y2 - y1)) / img_area
                if area_ratio >= MIN_AREA_RATIO and conf > best_conf:
                    best_box = (x1, y1, x2, y2)
                    best_conf = conf

    if best_box is None:
        if found_animal:
            stats["discarded_too_small"] += 1
        else:
            stats["discarded_no_animal"] += 1
        continue

    x1, y1, x2, y2 = best_box
    pad = 20
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    cropped = img.crop((x1, y1, x2, y2))
    if cropped.mode != "RGB":
        cropped = cropped.convert("RGB")
    out_path = output_root / img_path.relative_to(input_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(out_path)

    class_name = img_path.parent.name
    stats["by_class"][class_name] = stats["by_class"].get(class_name, {"kept": 0, "discarded": 0})
    stats["by_class"][class_name]["kept"] += 1
    stats["kept"] += 1


for img_path in img_paths:
    class_name = img_path.parent.name
    if class_name not in stats["by_class"]:
        stats["by_class"][class_name] = {"kept": 0, "discarded": 0}

total_discarded = stats["discarded_no_animal"] + stats["discarded_too_small"]
for img_path in img_paths:
    class_name = img_path.parent.name
    total_in_class = sum(1 for p in img_paths if p.parent.name == class_name)
    kept_in_class = stats["by_class"].get(class_name, {}).get("kept", 0)
    stats["by_class"][class_name]["discarded"] = total_in_class - kept_in_class

print("\n===== Dataset Cleaning Summary =====")
print(f"Total images processed : {stats['total']}")
print(f"Kept                   : {stats['kept']} ({stats['kept']/stats['total']*100:.1f}%)")
print(f"Discarded (no animal)  : {stats['discarded_no_animal']}")
print(f"Discarded (too small)  : {stats['discarded_too_small']}")
print(f"Total discarded        : {total_discarded} ({total_discarded/stats['total']*100:.1f}%)")
print("\n--- Per class ---")
for class_name, counts in sorted(stats["by_class"].items()):
    total = counts["kept"] + counts["discarded"]
    print(f"  {class_name:<10} kept: {counts['kept']:<6} discarded: {counts['discarded']:<6} ({counts['kept']/total*100:.1f}% kept)")
print("====================================")
