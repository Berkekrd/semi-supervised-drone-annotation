# ============================================================
# MAIN12 - FINE-TUNING MIT DEM SEMI-SUPERVISED-DATASET
# ============================================================
# Zweck:
# Feintuning des Baseline-Modells auf dem kombinierten Dataset
# (manuelle + Detection-Pseudo-Labels). Es wird bewusst eine
# kleinere Lernrate und weniger Epochen als beim Baseline-
# Training verwendet, da das Modell nur angepasst wird.
# ============================================================

from ultralytics import YOLO

def main():
    # Startpunkt: die besten Gewichte des Baseline-Trainings
    model = YOLO(r"C:\thesis\runs\detect\train\weights\best.pt")

    model.train(
        data=r"C:\thesis\data_semi.yaml",
        epochs=30,
        imgsz=1024,
        device=0,
        patience=7,
        batch=16,
        workers=4,
        optimizer="AdamW",
        lr0=0.0003,          # kleinere Lernrate fürs Fine-Tuning
        cos_lr=True,
        augment=True,
        name="semi_iter1_targeted"
    )

if __name__ == "__main__":
    main()
