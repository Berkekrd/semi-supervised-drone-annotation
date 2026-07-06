# ============================================================
# MAIN17 - TRAINING MIT TRACKING-BASIERTER LABEL-PROPAGATION
# ============================================================
# Zweck:
# Fine-Tuning ausgehend vom Baseline-Modell mit dem Dataset aus
# manuellem Training + Tracking-Pseudo-Labels (main16).
#
# Validierung:
# Es wird ausschließlich die manuelle Validierung verwendet.
# ============================================================

from ultralytics import YOLO


MODEL_PATH = r"C:\thesis\runs\detect\train\weights\best.pt"
DATA_YAML = r"C:\thesis\data_tracking_semi.yaml"


def main():
    # Startpunkt: die besten Gewichte des Baseline-Trainings
    model = YOLO(MODEL_PATH)

    model.train(
        data=DATA_YAML,
        epochs=60,
        imgsz=1024,
        device=0,
        patience=20,
        batch=16,
        workers=4,
        optimizer="AdamW",
        lr0=0.00015,        # sehr kleine Lernrate für vorsichtiges Fine-Tuning
        cos_lr=True,
        augment=True,
        project=r"C:\thesis\runs\detect",
        name="tracking_label_propagation_iter2_patience20"
    )


if __name__ == "__main__":
    main()
