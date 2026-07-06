# ============================================================
# MAIN4 - TRAIN/VAL-SPLIT (80/20)
# ============================================================
# Zweck:
# Teilt das konvertierte Dataset zufällig, aber reproduzierbar
# (fester Seed) in 80 % Trainings- und 20 % Validierungsdaten auf.
# Bilder ohne zugehöriges Label werden übersprungen.
# ============================================================

import os
import shutil
import random

DATASET_NAME = "video04_regionB"

BASE = f"datasets/{DATASET_NAME}"
IMG_DIR = f"{BASE}/images"
LBL_DIR = f"{BASE}/labels"

TRAIN_IMG = f"{BASE}/train/images"
TRAIN_LBL = f"{BASE}/train/labels"

VAL_IMG = f"{BASE}/val/images"
VAL_LBL = f"{BASE}/val/labels"

# Zielordner anlegen
for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
    os.makedirs(d, exist_ok=True)

# Fester Seed für Reproduzierbarkeit
random.seed(42)

# Sowohl PNG als auch JPG werden unterstützt
images = [f for f in os.listdir(IMG_DIR) if f.endswith((".png", ".jpg", ".jpeg"))]

print(f"Toplam image: {len(images)}")

random.shuffle(images)

# 80/20-Aufteilung
split = int(len(images) * 0.8)

train_files = images[:split]
val_files = images[split:]

def copy_files(files, img_out, lbl_out):
    count = 0

    for f in files:
        img_src = os.path.join(IMG_DIR, f)
        lbl_name = os.path.splitext(f)[0] + ".txt"
        lbl_src = os.path.join(LBL_DIR, lbl_name)

        # Bilder ohne Label werden übersprungen
        if not os.path.exists(lbl_src):
            print(f"⚠️ Label yok: {lbl_name}")
            continue

        shutil.copy(img_src, os.path.join(img_out, f))
        shutil.copy(lbl_src, os.path.join(lbl_out, lbl_name))

        count += 1

    return count

train_count = copy_files(train_files, TRAIN_IMG, TRAIN_LBL)
val_count = copy_files(val_files, VAL_IMG, VAL_LBL)

print(f"\n✅ Train: {train_count}")
print(f"✅ Val: {val_count}")
print("🔥 Split tamam!")
