# Ergebnisse

Dieser Ordner enthält die Ergebnisse der Pipeline. Quantitative Ergebnisse (Trainingskurven, Konfusionsmatrizen, Metriken) der Baseline- und Pseudo-Labeling-Experimente liegen bei den zugehörigen Abbildungen der Arbeit unter [`thesis/figures/baseline/`](../thesis/figures/baseline/) und [`thesis/figures/pseudo/`](../thesis/figures/pseudo/) und werden dort direkt in die Thesis eingebunden.

## Struktur

```text
results/
└── tracking/                  # Qualitative Tracking-Ergebnisse (YOLO + ByteTrack + GPS)
    └── video04_regionB_01/    # Ein Ordner pro ausgewertetem Video
```

## Tracking-Ergebnisse

Für jedes ausgewertete Drohnenvideo existiert ein eigener Ordner mit dem frame-genauen GPS-Tracking-Report (`*.csv`), einer Zusammenfassung (`*.txt`) und einer Beschreibung (`README.md`). Die annotierten Videos selbst (`*.mp4`, mehrere GB, 4K) sind **nicht** im Repository enthalten (siehe [`.gitignore`](../.gitignore)) und werden extern gehostet.

| Video | Region | Beschreibung | Details | Video-Link |
|-------|--------|--------------|---------|------------|
| video04_regionB_01 | Region B | Multi-Object-Tracking mit GPS auf einem 4K-Drohnenvideo | [README](tracking/video04_regionB_01/README.md) | [▶ Video ansehen](https://drive.google.com/file/d/1wDNBesEX62_mUq_ulWIKpAWM8FOAn_20/view?usp=sharing) |

<!-- Neue Videos als weitere Zeile nach demselben Schema ergänzen. -->
