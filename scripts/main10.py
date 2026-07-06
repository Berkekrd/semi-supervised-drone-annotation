# ============================================================
# MAIN10 - KLASSENBALANCIERTE AUSWAHL DER PSEUDO-LABELS (QUOTA)
# ============================================================
# Zweck:
# Wählt aus dem Pseudo-Dataset (main9) eine klassenbalancierte
# Teilmenge für das Training aus. Pro Klasse wird eine Quota
# berechnet (Zielwert minus vorhandene manuelle Boxen), damit
# häufige Klassen das Training nicht dominieren.
# ============================================================

import os
import json
import shutil
import random
from collections import Counter, defaultdict

# =========================
# EINGABE / AUSGABE
# =========================
PSEUDO_DATASET_ROOT = r"C:\thesis\datasets\pseudo_dataset"
OUTPUT_ROOT = r"C:\thesis\datasets\pseudo_dataset_split"

CONFIG_PATH = r"C:\thesis\runs\pseudo\target_config.json"

TRAIN_IMG = os.path.join(OUTPUT_ROOT, "train", "images")
TRAIN_LBL = os.path.join(OUTPUT_ROOT, "train", "labels")

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# =========================
# EINSTELLUNGEN
# =========================
random.seed(42)

# Zielwert pro Klasse: manual count + pseudo count ≈ TARGET_TOTAL_PER_CLASS
TARGET_TOTAL_PER_CLASS = 800

# Maximale Anzahl an Pseudo-Boxen pro Klasse
MAX_PSEUDO_BBOX_PER_CLASS = 1000

# Obergrenze für die Gesamtzahl der Pseudo-Frames
MAX_TOTAL_FRAMES = 3000


def load_config():
    # Lädt Klassen, Zielklassen und manuelle Verteilung aus target_config.json
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config bulunamadı: {CONFIG_PATH}. Önce main6.py çalışmalı.")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    class_names = {int(k): v for k, v in config["class_names"].items()}
    target_classes = [int(x) for x in config["target_classes"]]
    manual_distribution = {
        int(k): int(v)
        for k, v in config["manual_distribution"].items()
    }

    return class_names, target_classes, manual_distribution


def read_label_lines(label_path):
    # Liest YOLO-Labelzeilen und gibt (Klasse, bereinigte Zeile) zurück
    lines = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            cls = int(parts[0])
            lines.append((cls, " ".join(parts[:5]) + "\n"))

    return lines


def find_all_pairs():
    # Sammelt alle Bild/Label-Paare aus allen Video-Unterordnern
    pairs = []

    video_names = [
        d for d in sorted(os.listdir(PSEUDO_DATASET_ROOT))
        if os.path.isdir(os.path.join(PSEUDO_DATASET_ROOT, d))
    ]

    for video_name in video_names:
        img_dir = os.path.join(PSEUDO_DATASET_ROOT, video_name, "images")
        lbl_dir = os.path.join(PSEUDO_DATASET_ROOT, video_name, "labels")

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"⏭️ Eksik klasör, geçiliyor: {video_name}")
            continue

        images = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(IMAGE_EXTS)
        ]

        count = 0

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

            count += 1

        print(f"📹 {video_name}: {count} pseudo image/label bulundu")

    return pairs


def main():
    class_names, target_classes, manual_distribution = load_config()
    target_classes = set(target_classes)

    # Alte Ausgabe wird gelöscht (sauberer Neuaufbau)
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)

    os.makedirs(TRAIN_IMG, exist_ok=True)
    os.makedirs(TRAIN_LBL, exist_ok=True)

    print("\n🎯 Target classes:")
    for cls in sorted(target_classes):
        print(f"{cls:2d} | {class_names.get(cls, cls)}")

    # =========================
    # 1) Pseudo-Quota pro Klasse berechnen
    # =========================
    quota = {}

    print("\n📊 Manual dağılıma göre pseudo quota:")

    for cls in sorted(target_classes):
        manual_count = manual_distribution.get(cls, 0)

        # Fehlende Boxen bis zum Zielwert, gedeckelt durch die Obergrenze
        missing = TARGET_TOTAL_PER_CLASS - manual_count
        pseudo_quota = max(0, min(missing, MAX_PSEUDO_BBOX_PER_CLASS))

        quota[cls] = pseudo_quota

        print(
            f"{cls:2d} | {class_names.get(cls, cls):20s} "
            f"| manual: {manual_count:5d} "
            f"| pseudo quota: {pseudo_quota:5d}"
        )

    if sum(quota.values()) == 0:
        print("\n⚠️ Hiç pseudo quota yok. Çıkılıyor.")
        return

    # =========================
    # 2) Kandidaten-Frames einlesen
    # =========================
    all_pairs = find_all_pairs()
    random.shuffle(all_pairs)

    print(f"\n📦 Toplam aday frame: {len(all_pairs)}")

    # Priorität: Frames mit vielen Boxen aus unterrepräsentierten Klassen zuerst
    def priority_score(item):
        label_lines = read_label_lines(item["src_lbl"])
        score = 0

        for cls, _line in label_lines:
            if cls in quota and quota[cls] > 0:
                score += quota[cls]

        return score

    all_pairs.sort(key=priority_score, reverse=True)

    # =========================
    # 3) Auswahl anhand der Bbox-Quota
    # =========================
    used_bbox_counter = Counter()
    selected_frames = 0
    skipped_frames = 0

    for item in all_pairs:
        if selected_frames >= MAX_TOTAL_FRAMES:
            break

        label_lines = read_label_lines(item["src_lbl"])

        kept_lines = []

        for cls, line in label_lines:
            if cls not in target_classes:
                continue

            # Klassen mit voller Quota nicht weiter aufnehmen
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

        # Videoname als Präfix, um Namenskollisionen zu vermeiden
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

    # =========================
    # 4) Bericht
    # =========================
    print("\n✅ Targeted balanced pseudo train oluşturuldu")
    print(f"📄 Seçilen frame: {selected_frames}")
    print(f"⏭️ Boş/işe yaramayan frame: {skipped_frames}")

    print("\n📊 Seçilen pseudo bbox dağılımı:")
    for cls in sorted(target_classes):
        print(
            f"{cls:2d} | {class_names.get(cls, cls):20s}: "
            f"{used_bbox_counter[cls]:5d} / quota {quota[cls]:5d}"
        )

    print(f"\n🔥 Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
