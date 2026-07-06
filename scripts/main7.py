# ============================================================
# MAIN7 - ANALYSE DER PSEUDO-LABELS UND THRESHOLD-EMPFEHLUNG
# ============================================================
# Zweck:
# Analysiert die in main6 erzeugten Pseudo-Labels pro Video und
# Klasse (Anzahl, mittlere/min/max Confidence) und leitet daraus
# klassenweise Confidence-Schwellenwerte für die Filterung ab.
# Die Empfehlungen werden als JSON für main8 gespeichert.
# ============================================================

import os
import json
from collections import Counter, defaultdict

PSEUDO_ROOT = r"C:\thesis\runs\pseudo"
CONFIG_PATH = os.path.join(PSEUDO_ROOT, "target_config.json")

DEFAULT_CLASS_NAMES = {
    0: "pool",
    1: "bottle",
    2: "bucket",
    3: "puddle",
    4: "tire",
    5: "watertank",
    6: "dumpster",
    7: "large trash bin",
    8: "plastic bag",
    9: "small trash bin",
    10: "storm drain",
}


def load_config():
    # Lädt die in main6 erzeugte target_config.json
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Config bulunamadı: {CONFIG_PATH}")
        print("Önce main6.py çalışmalı.")
        return None

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    class_names = {
        int(k): v for k, v in config.get("class_names", {}).items()
    }

    if not class_names:
        class_names = DEFAULT_CLASS_NAMES

    target_classes = [int(x) for x in config.get("target_classes", [])]
    excluded_classes = [int(x) for x in config.get("excluded_classes", [])]
    manual_review_classes = [int(x) for x in config.get("manual_review_classes", [])]

    return {
        "class_names": class_names,
        "target_classes": target_classes,
        "excluded_classes": excluded_classes,
        "manual_review_classes": manual_review_classes,
    }


def suggest_threshold(avg_conf):
    # Heuristik: je höher die mittlere Confidence einer Klasse,
    # desto höher darf der Filter-Schwellenwert sein
    if avg_conf >= 0.80:
        return 0.75
    elif avg_conf >= 0.65:
        return 0.60
    elif avg_conf >= 0.50:
        return 0.55
    else:
        return 0.65


def main():
    config = load_config()
    if config is None:
        return

    CLASS_NAMES = config["class_names"]
    TARGET_CLASSES = set(config["target_classes"])
    EXCLUDED_CLASSES = set(config["excluded_classes"])
    MANUAL_REVIEW_CLASSES = set(config["manual_review_classes"])

    print("🔍 Pseudo-label kontrolü başlıyor...\n")

    print("🎯 Target classes:")
    for cls in sorted(TARGET_CLASSES):
        print(f"{cls:2d} | {CLASS_NAMES.get(cls, cls)}")

    print("\n🚫 Excluded classes:")
    for cls in sorted(EXCLUDED_CLASSES):
        print(f"{cls:2d} | {CLASS_NAMES.get(cls, cls)}")

    print("\n👀 Manual review classes:")
    for cls in sorted(MANUAL_REVIEW_CLASSES):
        print(f"{cls:2d} | {CLASS_NAMES.get(cls, cls)}")

    total_counter = Counter()
    total_conf_values = defaultdict(list)
    unexpected_counter = Counter()

    video_names = [
        d for d in sorted(os.listdir(PSEUDO_ROOT))
        if os.path.isdir(os.path.join(PSEUDO_ROOT, d))
    ]

    for video_name in video_names:
        label_dir = os.path.join(PSEUDO_ROOT, video_name, "labels")

        if not os.path.exists(label_dir):
            continue

        txt_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

        video_counter = Counter()
        video_conf_values = defaultdict(list)
        video_unexpected = Counter()

        frames_with_labels = 0
        total_boxes = 0

        for file in txt_files:
            file_has_label = False

            with open(os.path.join(label_dir, file), "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()

                    # YOLO-predict mit save_conf=True liefert:
                    # class x y w h conf
                    if len(parts) < 6:
                        continue

                    cls = int(parts[0])
                    conf = float(parts[5])

                    if cls in TARGET_CLASSES:
                        video_counter[cls] += 1
                        total_counter[cls] += 1

                        video_conf_values[cls].append(conf)
                        total_conf_values[cls].append(conf)

                        total_boxes += 1
                        file_has_label = True
                    else:
                        # Klassen außerhalb der Zielmenge separat zählen
                        video_unexpected[cls] += 1
                        unexpected_counter[cls] += 1

            if file_has_label:
                frames_with_labels += 1

        print("\n" + "=" * 60)
        print(f"📹 {video_name}")
        print("=" * 60)
        print(f"📄 Target label içeren frame: {frames_with_labels}")
        print(f"📦 Target bbox toplam: {total_boxes}")

        if total_boxes == 0:
            print("⚠️ Bu videoda target pseudo-label yok.")
            continue

        print("\n📊 Target class dağılımı:")

        for cls in sorted(video_counter.keys()):
            values = video_conf_values[cls]
            avg_conf = sum(values) / len(values)
            min_conf = min(values)
            max_conf = max(values)
            suggested = suggest_threshold(avg_conf)

            print(
                f"{cls:2d} | {CLASS_NAMES.get(cls, cls):20s} "
                f"| count: {video_counter[cls]:6d} "
                f"| avg: {avg_conf:.3f} "
                f"| min: {min_conf:.3f} "
                f"| max: {max_conf:.3f} "
                f"| önerilen th: {suggested:.2f}"
            )

        if video_unexpected:
            print("\n⚠️ Target dışı gelen class var:")
            for cls in sorted(video_unexpected.keys()):
                print(f"{cls:2d} | {CLASS_NAMES.get(cls, cls):20s}: {video_unexpected[cls]}")

    print("\n" + "=" * 70)
    print("📊 TOPLAM TARGET PSEUDO-LABEL DAĞILIMI")
    print("=" * 70)

    total_boxes_all = sum(total_counter.values())
    print(f"📦 Toplam target bbox: {total_boxes_all}\n")

    threshold_suggestions = {}

    for cls in sorted(total_counter.keys()):
        values = total_conf_values[cls]
        avg_conf = sum(values) / len(values)
        min_conf = min(values)
        max_conf = max(values)
        suggested = suggest_threshold(avg_conf)
        threshold_suggestions[cls] = suggested

        print(
            f"{cls:2d} | {CLASS_NAMES.get(cls, cls):20s} "
            f"| count: {total_counter[cls]:6d} "
            f"| avg: {avg_conf:.3f} "
            f"| min: {min_conf:.3f} "
            f"| max: {max_conf:.3f} "
            f"| önerilen th: {suggested:.2f}"
        )

    if unexpected_counter:
        print("\n⚠️ TOPLAM target dışı class:")
        for cls in sorted(unexpected_counter.keys()):
            print(f"{cls:2d} | {CLASS_NAMES.get(cls, cls):20s}: {unexpected_counter[cls]}")

    # Empfohlene Schwellenwerte für main8 speichern
    out_path = os.path.join(PSEUDO_ROOT, "threshold_suggestions.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {str(k): v for k, v in threshold_suggestions.items()},
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"\n✅ Threshold önerileri kaydedildi: {out_path}")
    print("✅ Kontrol tamamlandı.")


if __name__ == "__main__":
    main()
