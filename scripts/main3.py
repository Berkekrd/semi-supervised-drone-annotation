# ============================================================
# MAIN3 - KONSISTENZPRÜFUNG DES DATASETS (IMAGE/LABEL-PAARE)
# ============================================================
# Zweck:
# Prüft für die Train- und Val-Splits, ob jedes Bild ein
# zugehöriges Label besitzt und umgekehrt. Fehlende Paare
# würden das YOLO-Training verfälschen.
# ============================================================

import os

dataset = "datasets/video01_regionA"

for split in ["train", "val"]:
    img_dir = os.path.join(dataset, split, "images")
    lbl_dir = os.path.join(dataset, split, "labels")

    # Basisnamen (ohne Endung) von Bildern und Labels sammeln
    imgs = sorted([os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".png")])
    lbls = sorted([os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith(".txt")])

    print(f"\n--- {split.upper()} ---")
    print("image:", len(imgs))
    print("label:", len(lbls))

    # Mengendifferenz: Bilder ohne Label bzw. Labels ohne Bild
    missing_lbl = sorted(set(imgs) - set(lbls))
    extra_lbl = sorted(set(lbls) - set(imgs))

    print("labeli olmayan image:", len(missing_lbl))
    print("image'i olmayan label:", len(extra_lbl))
