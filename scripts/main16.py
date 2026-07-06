# ============================================================
# MAIN16 - TRACKING-SEMI-SUPERVISED-DATASET AUFBAUEN
# ============================================================
# Eingabe:
#   Manuelles Dataset:
#       C:\thesis\datasets\merged_dataset
#
#   Tracking-Pseudo-Dataset:
#       C:\thesis\datasets\tracking_pseudo_dataset
#
# Ausgabe:
#   C:\thesis\datasets\tracking_semi_supervised_dataset
#
# Train:
#   manuelles Training + Tracking-Pseudo-Training
#
# Val:
#   ausschließlich manuelle Validierung
#
# Zusätzlich werden die Image/Label-Paare geprüft, die Klassen-
# verteilung ausgegeben und die data_tracking_semi.yaml erzeugt.
# ============================================================

import os
import shutil
import yaml
from collections import Counter


# =========================
# PFADE
# =========================
MANUAL_ROOT = r"C:\thesis\datasets\merged_dataset"
TRACKING_PSEUDO_ROOT = r"C:\thesis\datasets\tracking_pseudo_dataset"

OUTPUT_ROOT = r"C:\thesis\datasets\tracking_semi_supervised_dataset"

DATA_YAML_PATH = r"C:\thesis\data_tracking_semi.yaml"

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


CLASS_NAMES = {
    0: "pool",
    1: "bottle",
    2: "bucket",
    3: "puddle",
    4: "tire",
    5: "watertank",
    6: "dumpster",
    7: "large_trash_bin",
    8: "plastic_bag",
    9: "small_trash_bin",
    10: "storm_drain",
}


def clean_output():
    # Alte Ausgabe löschen und leere Ordnerstruktur anlegen
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, split, "labels"), exist_ok=True)


def copy_split(src_root, split, dst_split, prefix):
    # Kopiert einen Split mit Herkunfts-Präfix ("manual"/"trackingpseudo")
    src_img_dir = os.path.join(src_root, split, "images")
    src_lbl_dir = os.path.join(src_root, split, "labels")

    dst_img_dir = os.path.join(OUTPUT_ROOT, dst_split, "images")
    dst_lbl_dir = os.path.join(OUTPUT_ROOT, dst_split, "labels")

    if not os.path.exists(src_img_dir):
        print(f"❌ Image klasörü yok: {src_img_dir}")
        return 0

    if not os.path.exists(src_lbl_dir):
        print(f"❌ Label klasörü yok: {src_lbl_dir}")
        return 0

    count = 0

    for img_file in sorted(os.listdir(src_img_dir)):
        if not img_file.lower().endswith(IMAGE_EXTS):
            continue

        base, ext = os.path.splitext(img_file)
        lbl_file = base + ".txt"

        src_img = os.path.join(src_img_dir, img_file)
        src_lbl = os.path.join(src_lbl_dir, lbl_file)

        # Nur vollständige Bild/Label-Paare übernehmen
        if not os.path.exists(src_lbl):
            continue

        new_img_name = f"{prefix}_{base}{ext}"
        new_lbl_name = f"{prefix}_{base}.txt"

        shutil.copy2(src_img, os.path.join(dst_img_dir, new_img_name))
        shutil.copy2(src_lbl, os.path.join(dst_lbl_dir, new_lbl_name))

        count += 1

    print(f"✅ {prefix}: {split} → {dst_split}: {count} image-label kopyalandı")
    return count


def count_labels(label_dir):
    # Zählt Labeldateien und Boxen pro Klasse
    counter = Counter()
    image_label_files = 0

    if not os.path.exists(label_dir):
        return counter, image_label_files

    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue

        image_label_files += 1

        with open(os.path.join(label_dir, file), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                cls = int(parts[0])
                counter[cls] += 1

    return counter, image_label_files


def check_image_label_pairs(split):
    # Prüft, ob jedes Bild ein Label hat und umgekehrt
    img_dir = os.path.join(OUTPUT_ROOT, split, "images")
    lbl_dir = os.path.join(OUTPUT_ROOT, split, "labels")

    images = {
        os.path.splitext(f)[0]
        for f in os.listdir(img_dir)
        if f.lower().endswith(IMAGE_EXTS)
    }

    labels = {
        os.path.splitext(f)[0]
        for f in os.listdir(lbl_dir)
        if f.endswith(".txt")
    }

    missing_labels = images - labels
    extra_labels = labels - images

    print(f"\n🔍 Pair kontrolü - {split}")
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")
    print(f"Label olmayan image: {len(missing_labels)}")
    print(f"Image olmayan label: {len(extra_labels)}")

    return len(missing_labels), len(extra_labels)


def create_data_yaml():
    # Erzeugt die YOLO-Datenkonfiguration für das Training (main17)
    data = {
        "path": OUTPUT_ROOT.replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "names": CLASS_NAMES,
    }

    with open(DATA_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"\n✅ data_tracking_semi.yaml oluşturuldu:")
    print(DATA_YAML_PATH)


def print_distribution(split):
    # Gibt die Klassenverteilung eines Splits aus
    label_dir = os.path.join(OUTPUT_ROOT, split, "labels")
    counter, file_count = count_labels(label_dir)

    print(f"\n📊 {split.upper()} label dağılımı")
    print(f"Label dosyası: {file_count}")

    total = sum(counter.values())
    print(f"Toplam bbox: {total}")

    for cls in sorted(CLASS_NAMES.keys()):
        print(f"{cls:2d} | {CLASS_NAMES[cls]:20s}: {counter.get(cls, 0)}")


def main():
    print("🚀 MAIN16 - Tracking semi-supervised dataset oluşturuluyor")

    clean_output()

    # 1) Manuelles Training -> train
    manual_train = copy_split(
        src_root=MANUAL_ROOT,
        split="train",
        dst_split="train",
        prefix="manual"
    )

    # 2) Tracking-Pseudo-Training -> train
    tracking_pseudo_train = copy_split(
        src_root=TRACKING_PSEUDO_ROOT,
        split="train",
        dst_split="train",
        prefix="trackingpseudo"
    )

    # 3) Manuelle Validierung -> val (keine Pseudo-Labels in der Validierung!)
    manual_val = copy_split(
        src_root=MANUAL_ROOT,
        split="val",
        dst_split="val",
        prefix="manual"
    )

    print("\n🔥 Tracking semi-supervised dataset hazır:")
    print(OUTPUT_ROOT)

    print("\n📊 Özet:")
    print(f"Train toplam: {manual_train + tracking_pseudo_train}")
    print(f"  - Manual train: {manual_train}")
    print(f"  - Tracking pseudo train: {tracking_pseudo_train}")
    print(f"Val toplam: {manual_val}")
    print(f"  - Manual val only: {manual_val}")

    check_image_label_pairs("train")
    check_image_label_pairs("val")

    print_distribution("train")
    print_distribution("val")

    create_data_yaml()

    print("\n✅ MAIN16 tamamlandı.")


if __name__ == "__main__":
    main()
