# ============================================================
# MAIN18 - VERGLEICH: BASELINE / DETECTION-PSEUDO / TRACKING-PSEUDO
# ============================================================
# Zweck:
# Validiert alle drei Modelle (Baseline, Detection-Pseudo,
# Tracking-Pseudo) auf demselben manuellen Validierungsset und
# schreibt die Metriken (Precision, Recall, mAP50, mAP50-95)
# in eine Vergleichs-CSV.
#
# Wichtig:
# Es wird nur die Validierung genutzt. Da val in
# data_tracking_semi.yaml = manuelle Validierung ist, werden
# alle Modelle auf identischen Daten fair verglichen.
# ============================================================

import os
import csv
from ultralytics import YOLO

DATA_YAML = r"C:\thesis\data_tracking_semi.yaml"

OUTPUT_CSV = r"C:\thesis\runs\detect\model_comparison_same_val.csv"

# Die drei zu vergleichenden Modelle mit ihren Trainingsdaten
MODELS = [
    {
        "name": "baseline_manual_only",
        "training_data": "manual train only",
        "path": r"C:\thesis\runs\detect\train\weights\best.pt",
    },
    {
        "name": "detection_pseudo_iter1",
        "training_data": "manual train + detection pseudo-labels",
        "path": r"C:\thesis\runs\detect\semi_iter1_targeted\weights\best.pt",
    },
    {
        "name": "tracking_label_propagation_iter2_patience20",
        "training_data": "manual train + ByteTrack tracking pseudo-labels",
        "path": r"C:\thesis\runs\detect\tracking_label_propagation_iter2_patience20\weights\best.pt",
    },
]


def main():
    rows = []

    for item in MODELS:
        model_path = item["path"]

        # Fehlende Modelle überspringen statt abzubrechen
        if not os.path.exists(model_path):
            print(f"⚠️ Model bulunamadı, atlanıyor: {item['name']}")
            print(model_path)
            continue

        print("\n" + "=" * 100)
        print(f"🔍 Validating: {item['name']}")
        print("=" * 100)

        model = YOLO(model_path)

        # Validierung auf dem gemeinsamen manuellen Val-Set
        results = model.val(
            data=DATA_YAML,
            imgsz=1024,
            device=0,
            batch=16,
            split="val",
            project=r"C:\thesis\runs\detect\comparison_val",
            name=item["name"],
            verbose=True,
            plots=True
        )

        row = {
            "model": item["name"],
            "training_data": item["training_data"],
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
            "map50": float(results.box.map50),
            "map50_95": float(results.box.map),
        }

        rows.append(row)

        print("\n📊 Result:")
        print(row)

    if not rows:
        print("❌ Hiç model validate edilemedi.")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Ergebnisse als CSV speichern
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model",
            "training_data",
            "precision",
            "recall",
            "map50",
            "map50_95"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n🔥 Comparison CSV kaydedildi:")
    print(OUTPUT_CSV)

    # Abschließende Vergleichstabelle in der Konsole ausgeben
    print("\n📊 FINAL COMPARISON")
    print("=" * 110)
    print(
        f"{'Model':45s} "
        f"{'Precision':>10s} "
        f"{'Recall':>10s} "
        f"{'mAP50':>10s} "
        f"{'mAP50-95':>10s}"
    )
    print("-" * 110)

    for r in rows:
        print(
            f"{r['model']:45s} "
            f"{r['precision']:10.3f} "
            f"{r['recall']:10.3f} "
            f"{r['map50']:10.3f} "
            f"{r['map50_95']:10.3f}"
        )


if __name__ == "__main__":
    main()
