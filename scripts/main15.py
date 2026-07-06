# ============================================================
# MAIN15 - BALANCIERTES TRACKING-PSEUDO-DATASET AUFBAUEN
# ============================================================
# Eingabe:
#   C:\thesis\runs\tracking_pseudo\videoXX\images + labels
#
# Ausgabe:
#   C:\thesis\datasets\tracking_pseudo_dataset\train\images
#   C:\thesis\datasets\tracking_pseudo_dataset\train\labels
#
# Zweck:
# Wählt die tracking-basierten Pseudo-Labels (main14) klassen-
# balanciert aus — gleiches Quota-Prinzip wie in main10, aber
# auf den Tracking-Pseudo-Labels statt den Detection-Pseudo-Labels.
# ============================================================

import os
import json
import shutil
import random
from collections import Counter


# =========================
# EINGABE / AUSGABE
# =========================
TRACKING_PSEUDO_ROOT = r"C:\thesis\runs\tracking_pseudo"
OUTPUT_ROOT = r"C:\thesis\datasets\tracking_pseudo_dataset"

CONFIG_PATH = r"C:\thesis\runs\pseudo\target_config.json"

TRAIN_IMG = os.path.join(OUTPUT_ROOT, "train", "images")
TRAIN_LBL = os.path.join(OUTPUT_ROOT, "train", "labels")

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


# =========================
# EINSTELLUNGEN
# =========================
random.seed(42)

# Zielwert pro Klasse: manual + pseudo ≈ dieser Wert.
# Zu hohe Werte lassen dominante Klassen (pool/storm_drain) überwiegen.
TARGET_TOTAL_PER_CLASS = 800

# Maximale Anzahl an Pseudo-Boxen pro Klasse.
MAX_PSEUDO_BBOX_PER_CLASS = 1000

# Obergrenze für die Gesamtzahl der ausgewählten Pseudo-Frames.
MAX_TOTAL_FRAMES = 3000


def load_config():
    # Lädt Klassen, Zielklassen und manuelle Verteilung aus target_config.json
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"Config bulunamadı: {CONFIG_PATH}. Önce main6.py ile target_config oluşmuş olmalı."
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    class_names = {int(k): v for k, v in config["class_names"].items()}
    target_classes = [int(x) for x in config["target_classes"]]

    manual_distribution = {
        int(k): int(v)
        for k, v in config.get("manual_distribution", {}).items()
    }

    return class_names, target_classes, manual_distribution


def clean_output():
    # Alte Ausgabe löschen und leere Ordnerstruktur anlegen
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)

    os.makedirs(TRAIN_IMG, exist_ok=True)
    os.makedirs(TRAIN_LBL, exist_ok=True)


def read_label_lines(label_path):
    """
    YOLO-Labelformat:
    class x_center y_center width height
    """
    lines = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            cls = int(parts[0])
            clean_line = " ".join(parts[:5]) + "\n"
            lines.append((cls, clean_line))

    return lines


def find_all_tracking_pairs():
    """
    Findet in tracking_pseudo die Paare:
    video01_regionA/images/frame_xxx.jpg
    video01_regionA/labels/frame_xxx.txt
    """
    pairs = []

    video_dirs = [
        d for d in sorted(os.listdir(TRACKING_PSEUDO_ROOT))
        if os.path.isdir(os.path.join(TRACKING_PSEUDO_ROOT, d))
    ]

    for video_name in video_dirs:
        img_dir = os.path.join(TRACKING_PSEUDO_ROOT, video_name, "images")
        lbl_dir = os.path.join(TRACKING_PSEUDO_ROOT, video_name, "labels")

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue

        images = [
            f for f in sorted(os.listdir(img_dir))
            if f.lower().endswith(IMAGE_EXTS)
        ]

        video_count = 0

        for img_file in images:
            base, ext = os.path.splitext(img_file)
            lbl_file = base + ".txt"

            src_img = os.path.join(img_dir, img_file)
            src_lbl = os.path.join(lbl_dir, lbl_file)

            if not os.path.exists(src_lbl):
                continue

            pairs.append({
                "video_name": video_name,
                "src_img": src_img,
                "src_lbl": src_lbl,
                "img_file": img_file,
                "base": base,
                "ext": ext,
            })

            video_count += 1

        print(f"📹 {video_name}: {video_count} tracking pseudo image-label bulundu")

    return pairs


def calculate_quota(class_names, target_classes, manual_distribution):
    # Pseudo-Quota pro Klasse: Zielwert minus manuelle Boxen, gedeckelt
    quota = {}

    print("\n📊 Manual dağılıma göre tracking pseudo quota:")

    for cls in sorted(target_classes):
        manual_count = manual_distribution.get(cls, 0)

        missing = TARGET_TOTAL_PER_CLASS - manual_count
        pseudo_quota = max(0, min(missing, MAX_PSEUDO_BBOX_PER_CLASS))

        quota[cls] = pseudo_quota

        print(
            f"{cls:2d} | {class_names.get(cls, cls):20s} "
            f"| manual: {manual_count:5d} "
            f"| pseudo quota: {pseudo_quota:5d}"
        )

    return quota


def main():
    print("🚀 MAIN15 - Balanced tracking pseudo dataset oluşturuluyor")

    class_names, target_classes, manual_distribution = load_config()
    target_classes = set(target_classes)

    clean_output()

    print("\n🎯 Target classes:")
    for cls in sorted(target_classes):
        print(f"{cls:2d} | {class_names.get(cls, cls)}")

    quota = calculate_quota(class_names, target_classes, manual_distribution)

    if sum(quota.values()) == 0:
        print("\n⚠️ Hiç pseudo quota yok. Output oluşturulmadı.")
        return

    all_pairs = find_all_tracking_pairs()
    random.shuffle(all_pairs)

    print(f"\n📦 Toplam aday tracking pseudo frame: {len(all_pairs)}")

    # Frames mit vielen Boxen aus unterrepräsentierten Klassen priorisieren
    def priority_score(item):
        label_lines = read_label_lines(item["src_lbl"])
        score = 0

        for cls, _line in label_lines:
            if cls in quota and quota[cls] > 0:
                score += quota[cls]

        return score

    all_pairs.sort(key=priority_score, reverse=True)

    used_bbox_counter = Counter()
    selected_frames = 0
    skipped_frames = 0
    total_seen_bbox = Counter()

    for item in all_pairs:
        if selected_frames >= MAX_TOTAL_FRAMES:
            break

        label_lines = read_label_lines(item["src_lbl"])

        kept_lines = []

        for cls, line in label_lines:
            total_seen_bbox[cls] += 1

            if cls not in target_classes:
                continue

            # Klassen mit erfüllter Quota nicht weiter aufnehmen
            if used_bbox_counter[cls] >= quota.get(cls, 0):
                continue

            kept_lines.append(line)
            used_bbox_counter[cls] += 1

        if not kept_lines:
            skipped_frames += 1
            continue

        video_name = item["video_name"]
        base = item["base"]
        ext = item["ext"]

        # Videoname als Präfix gegen Namenskollisionen
        new_img_name = f"{video_name}_{base}{ext}"
        new_lbl_name = f"{video_name}_{base}.txt"

        shutil.copy2(item["src_img"], os.path.join(TRAIN_IMG, new_img_name))

        with open(os.path.join(TRAIN_LBL, new_lbl_name), "w", encoding="utf-8") as f:
            f.writelines(kept_lines)

        selected_frames += 1

        # Abbruch, sobald alle Quotas erfüllt sind
        all_full = all(
            used_bbox_counter[cls] >= quota.get(cls, 0)
            for cls in target_classes
        )

        if all_full:
            break

    print("\n✅ Balanced tracking pseudo train dataset oluşturuldu")
    print(f"📄 Seçilen frame: {selected_frames}")
    print(f"⏭️ Boş/işe yaramayan frame: {skipped_frames}")

    print("\n📊 Seçilen tracking pseudo bbox dağılımı:")
    for cls in sorted(target_classes):
        print(
            f"{cls:2d} | {class_names.get(cls, cls):20s}: "
            f"{used_bbox_counter[cls]:5d} / quota {quota[cls]:5d}"
        )

    print("\n📊 Adaylarda görülen toplam bbox:")
    for cls in sorted(total_seen_bbox.keys()):
        print(
            f"{cls:2d} | {class_names.get(cls, cls):20s}: "
            f"{total_seen_bbox[cls]}"
        )

    # Auswahlbericht als JSON speichern
    report_path = os.path.join(OUTPUT_ROOT, "tracking_pseudo_dataset_report.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "settings": {
                    "target_total_per_class": TARGET_TOTAL_PER_CLASS,
                    "max_pseudo_bbox_per_class": MAX_PSEUDO_BBOX_PER_CLASS,
                    "max_total_frames": MAX_TOTAL_FRAMES,
                },
                "selected_frames": selected_frames,
                "skipped_frames": skipped_frames,
                "quota": {str(k): v for k, v in quota.items()},
                "selected_bbox": {str(k): v for k, v in used_bbox_counter.items()},
                "seen_bbox": {str(k): v for k, v in total_seen_bbox.items()},
                "class_names": {str(k): v for k, v in class_names.items()},
            },
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"\n🔥 Output: {OUTPUT_ROOT}")
    print(f"📄 Report: {report_path}")


if __name__ == "__main__":
    main()
