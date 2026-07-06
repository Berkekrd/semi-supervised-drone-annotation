# ============================================================
# MAIN9 - AUFBAU DES PSEUDO-DATASETS (BILDER + LABELS KOPIEREN)
# ============================================================
# Zweck:
# Kopiert zu jedem gefilterten Pseudo-Label (aus main8) das
# zugehörige Frame-Bild aus all_frames und baut so ein
# vollständiges Pseudo-Dataset (images + labels) pro Video auf.
# ============================================================

import os
import shutil

# =========================
# EINGABE
# =========================
FILTERED_ROOT = r"C:\thesis\runs\pseudo_filtered"
ALL_FRAMES_ROOT = r"C:\thesis\all_frames"

# =========================
# AUSGABE
# =========================
PSEUDO_DATASET_ROOT = r"C:\thesis\datasets\pseudo_dataset"

IMAGE_EXTENSIONS = [".jpg", ".png", ".jpeg"]


def find_image(frame_dir, base_name):
    # Sucht das zugehörige Bild — probiert alle unterstützten Endungen durch
    for ext in IMAGE_EXTENSIONS:
        img_path = os.path.join(frame_dir, base_name + ext)
        if os.path.exists(img_path):
            return img_path, base_name + ext
    return None, None


def main():
    # Altes pseudo_dataset ggf. löschen (sauberer Neuaufbau)
    if os.path.exists(PSEUDO_DATASET_ROOT):
        shutil.rmtree(PSEUDO_DATASET_ROOT)

    video_names = [
        d for d in sorted(os.listdir(FILTERED_ROOT))
        if os.path.isdir(os.path.join(FILTERED_ROOT, d))
    ]

    total_images = 0
    total_labels = 0
    total_missing = 0

    for video_name in video_names:
        label_dir = os.path.join(FILTERED_ROOT, video_name, "labels")
        frame_dir = os.path.join(ALL_FRAMES_ROOT, video_name)

        out_img_dir = os.path.join(PSEUDO_DATASET_ROOT, video_name, "images")
        out_lbl_dir = os.path.join(PSEUDO_DATASET_ROOT, video_name, "labels")

        if not os.path.exists(label_dir):
            print(f"⏭️ Label klasörü yok, geçiliyor: {label_dir}")
            continue

        if not os.path.exists(frame_dir):
            print(f"⏭️ Frame klasörü yok, geçiliyor: {frame_dir}")
            continue

        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        copied_images = 0
        copied_labels = 0
        missing_images = 0

        for label_file in sorted(os.listdir(label_dir)):
            if not label_file.endswith(".txt"):
                continue

            base_name = os.path.splitext(label_file)[0]

            src_img, image_file = find_image(frame_dir, base_name)
            src_lbl = os.path.join(label_dir, label_file)

            # Label ohne zugehöriges Bild wird übersprungen und gezählt
            if src_img is None:
                print(f"⚠️ Görsel bulunamadı: {os.path.join(frame_dir, base_name)}(.jpg/.png/.jpeg)")
                missing_images += 1
                continue

            dst_img = os.path.join(out_img_dir, image_file)
            dst_lbl = os.path.join(out_lbl_dir, label_file)

            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)

            copied_images += 1
            copied_labels += 1

        total_images += copied_images
        total_labels += copied_labels
        total_missing += missing_images

        print(f"\n📹 {video_name}")
        print(f"✅ Kopyalanan image: {copied_images}")
        print(f"✅ Kopyalanan label: {copied_labels}")
        print(f"⚠️ Eksik image: {missing_images}")

    print("\n🔥 Pseudo dataset image+label kopyalama tamamlandı.")
    print(f"📦 Toplam image: {total_images}")
    print(f"📦 Toplam label: {total_labels}")
    print(f"⚠️ Toplam eksik image: {total_missing}")


if __name__ == "__main__":
    main()
