# ============================================================
# MAIN8 - FILTERUNG DER PSEUDO-LABELS NACH CONFIDENCE
# ============================================================
# Zweck:
# Filtert die rohen Pseudo-Labels aus main6 anhand der in main7
# vorgeschlagenen klassenweisen Schwellenwerte. Boxen unterhalb
# des Schwellenwerts oder außerhalb der Zielklassen werden
# verworfen; die Confidence-Spalte wird für das YOLO-Trainings-
# format entfernt.
# ============================================================

import os
import json
from collections import Counter

PSEUDO_ROOT = r"C:\thesis\runs\pseudo"
INPUT_ROOT = r"C:\thesis\runs\pseudo"
OUTPUT_ROOT = r"C:\thesis\runs\pseudo_filtered"

CONFIG_PATH = os.path.join(PSEUDO_ROOT, "target_config.json")
THRESHOLD_PATH = os.path.join(PSEUDO_ROOT, "threshold_suggestions.json")

def load_json(path):
    if not os.path.exists(path):
        print(f"❌ Dosya bulunamadı: {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    config = load_json(CONFIG_PATH)
    threshold_json = load_json(THRESHOLD_PATH)

    if config is None:
        print("Önce main6.py çalışmalı.")
        return

    if threshold_json is None:
        print("Önce main7.py çalışmalı.")
        return

    class_names = {int(k): v for k, v in config["class_names"].items()}
    target_classes = set(int(x) for x in config["target_classes"])

    thresholds = {
        int(k): float(v)
        for k, v in threshold_json.items()
    }

    # Sicherheits-Mindestschwelle:
    # main7 schlägt teils 0.55 vor — unter 0.60 lassen wir nicht zu.
    MIN_THRESHOLD = 0.60

    for cls in thresholds:
        thresholds[cls] = max(thresholds[cls], MIN_THRESHOLD)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("🎯 Kullanılacak target class ve threshold değerleri:")
    for cls in sorted(target_classes):
        print(f"{cls:2d} | {class_names.get(cls, cls):20s} | th={thresholds.get(cls, MIN_THRESHOLD):.2f}")

    total_kept = Counter()
    total_removed = Counter()
    total_files_written = 0

    video_names = [
        d for d in sorted(os.listdir(INPUT_ROOT))
        if os.path.isdir(os.path.join(INPUT_ROOT, d))
    ]

    for video_name in video_names:
        input_label_dir = os.path.join(INPUT_ROOT, video_name, "labels")
        output_label_dir = os.path.join(OUTPUT_ROOT, video_name, "labels")

        if not os.path.exists(input_label_dir):
            continue

        os.makedirs(output_label_dir, exist_ok=True)

        kept = Counter()
        removed = Counter()
        files_written = 0
        empty_after_filter = 0

        for file in sorted(os.listdir(input_label_dir)):
            if not file.endswith(".txt"):
                continue

            in_path = os.path.join(input_label_dir, file)
            out_path = os.path.join(output_label_dir, file)

            new_lines = []

            with open(in_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()

                    # Predict-Format:
                    # class x_center y_center width height conf
                    if len(parts) < 6:
                        continue

                    cls = int(parts[0])
                    conf = float(parts[5])

                    # Nicht-Zielklassen verwerfen
                    if cls not in target_classes:
                        removed[cls] += 1
                        total_removed[cls] += 1
                        continue

                    threshold = thresholds.get(cls, MIN_THRESHOLD)

                    if conf >= threshold:
                        # YOLO-Trainingsformat (ohne Confidence):
                        # class x_center y_center width height
                        new_lines.append(" ".join(parts[:5]) + "\n")
                        kept[cls] += 1
                        total_kept[cls] += 1
                    else:
                        removed[cls] += 1
                        total_removed[cls] += 1

            # Nur Dateien mit mindestens einer verbleibenden Box schreiben
            if new_lines:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)

                files_written += 1
                total_files_written += 1
            else:
                empty_after_filter += 1

        print("\n" + "=" * 60)
        print(f"📹 {video_name}")
        print("=" * 60)
        print(f"📄 Label kalan frame: {files_written}")
        print(f"🗑️ Boşa düşen frame: {empty_after_filter}")

        print("\n✅ Tutulan:")
        for cls in sorted(kept.keys()):
            print(f"{cls:2d} | {class_names.get(cls, cls):20s}: {kept[cls]}")

        print("\n❌ Silinen:")
        for cls in sorted(removed.keys()):
            print(f"{cls:2d} | {class_names.get(cls, cls):20s}: {removed[cls]}")

    print("\n" + "=" * 70)
    print("📊 TOPLAM FILTER SONUCU")
    print("=" * 70)

    print(f"📄 Toplam label kalan frame: {total_files_written}")

    print("\n✅ Toplam tutulan:")
    for cls in sorted(total_kept.keys()):
        print(f"{cls:2d} | {class_names.get(cls, cls):20s}: {total_kept[cls]}")

    print("\n❌ Toplam silinen:")
    for cls in sorted(total_removed.keys()):
        print(f"{cls:2d} | {class_names.get(cls, cls):20s}: {total_removed[cls]}")

    print("\n🔥 Filtering tamamlandı.")
    print(f"Output: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
