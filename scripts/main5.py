# ============================================================
# MAIN5 - TRAINING DES BASELINE-MODELLS (YOLOv8n)
# ============================================================
# Zweck:
# Trainiert das Baseline-Modell ausschließlich auf den manuell
# annotierten Daten. Die hohe Bildauflösung (imgsz=1024) ist
# wichtig für die Erkennung kleiner Objekte aus Drohnenperspektive.
# ============================================================

from ultralytics import YOLO

def main():
    # Vortrainiertes YOLOv8n-Modell als Startpunkt
    model = YOLO("yolov8n.pt")

    # Frühere, einfachere Trainingskonfiguration (zum Vergleich behalten):
    # model.train(
    #     data="data_main.yaml",
    #     epochs=100,
    #     imgsz=640,
    #     device=0,
    #     patience=20,
    #     batch=16
    # )
    model.train(
    data="data_main.yaml",
    epochs=150,          # mehr Epochen für besseres Lernen
    imgsz=1024,          # hohe Auflösung — wichtig für kleine Objekte
    device=0,
    patience=30,         # Early Stopping nach 30 Epochen ohne Verbesserung
    batch=16,
    workers=4,
    optimizer="AdamW",   # stabilere Optimierung
    lr0=0.001,
    cos_lr=True,         # Cosine-Learning-Rate-Schedule
    augment=True
    )

if __name__ == "__main__":
    main()
