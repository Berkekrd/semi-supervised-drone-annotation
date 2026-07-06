# ============================================================
# MAIN11 - ZUSAMMENFÜHRUNG ZUM SEMI-SUPERVISED-DATASET
# ============================================================
# Zweck:
# Kombiniert manuelle Trainingsdaten mit den balancierten
# Pseudo-Labels (main10) zu einem gemeinsamen Trainingsset.
# Wichtig: Die Validierung besteht ausschließlich aus manuellen
# Daten, damit die Evaluation nicht durch Pseudo-Labels
# verfälscht wird.
# ============================================================

import os
import shutil

MANUAL_ROOT = r"C:\thesis\datasets\merged_dataset"
PSEUDO_ROOT = r"C:\thesis\datasets\pseudo_dataset_split"
OUTPUT_ROOT = r"C:\thesis\datasets\semi_supervised_dataset"

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def clean_output():
    # Alte Ausgabe löschen und leere Ordnerstruktur anlegen
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, split, "labels"), exist_ok=True)


def copy_split(src_root, split, dst_split, prefix):
    # Kopiert einen Split und stellt jedem Dateinamen ein Präfix voran
    # ("manual" bzw. "pseudo"), um die Herkunft nachvollziehbar zu halten
    src_img = os.path.join(src_root, split, "images")
    src_lbl = os.path.join(src_root, split, "labels")

    dst_img = os.path.join(OUTPUT_ROOT, dst_split, "images")
    dst_lbl = os.path.join(OUTPUT_ROOT, dst_split, "labels")

    if not os.path.exists(src_img) or not os.path.exists(src_lbl):
        print(f"⏭️ Eksik klasör, geçiliyor: {src_root} / {split}")
        return 0

    count = 0

    for img_file in os.listdir(src_img):
        if not img_file.lower().endswith(IMAGE_EXTS):
            continue

        base, ext = os.path.splitext(img_file)
        lbl_file = base + ".txt"

        src_img_path = os.path.join(src_img, img_file)
        src_lbl_path = os.path.join(src_lbl, lbl_file)

        # Nur vollständige Bild/Label-Paare übernehmen
        if not os.path.exists(src_lbl_path):
            continue

        new_img_name = f"{prefix}_{base}{ext}"
        new_lbl_name = f"{prefix}_{base}.txt"

        shutil.copy2(src_img_path, os.path.join(dst_img, new_img_name))
        shutil.copy2(src_lbl_path, os.path.join(dst_lbl, new_lbl_name))

        count += 1

    print(f"✅ {prefix}: {split} → {dst_split}: {count} image-label kopyalandı")
    return count


def main():
    clean_output()

    # 1) Manuelles Training -> train
    manual_train = copy_split(
        src_root=MANUAL_ROOT,
        split="train",
        dst_split="train",
        prefix="manual"
    )

    # 2) Pseudo-Training -> train
    pseudo_train = copy_split(
        src_root=PSEUDO_ROOT,
        split="train",
        dst_split="train",
        prefix="pseudo"
    )

    # 3) Manuelle Validierung -> val (KEINE Pseudo-Labels in der Validierung!)
    manual_val = copy_split(
        src_root=MANUAL_ROOT,
        split="val",
        dst_split="val",
        prefix="manual"
    )

    print("\n🔥 Semi-supervised dataset hazır:")
    print(OUTPUT_ROOT)

    print("\n📊 Özet:")
    print(f"Train toplam: {manual_train + pseudo_train}")
    print(f"  - Manual train: {manual_train}")
    print(f"  - Pseudo train: {pseudo_train}")
    print(f"Val toplam: {manual_val}")
    print(f"  - Manual val only: {manual_val}")


if __name__ == "__main__":
    main()
