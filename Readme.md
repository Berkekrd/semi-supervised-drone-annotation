# Masterarbeit – Semi-Supervised Annotation von Drohnenvideos

Dieses Repository enthält den finalen Quellcode und die ergänzenden Materialien der Masterarbeit im Bereich Computer Vision.

Ziel der Arbeit ist es, den manuellen Annotierungsaufwand bei Drohnenvideos zu reduzieren. Dafür werden nur spärlich manuell annotierte Frames als Ausgangspunkt verwendet. Die nicht annotierten Zwischenframes werden anschließend automatisch mithilfe von Object Detection, Pseudo-Labeling, heuristischer Filterung und Tracking-basierter Konsistenzprüfung annotiert.

## Repository-Struktur

```text
repo/
├── scripts/      # Pipeline als 18 nummerierte Skripte (main1.py ... main18.py)
├── notebooks/    # pipeline.ipynb – dieselbe Pipeline mit deutschen Erläuterungen
├── thesis/       # LaTeX-Quellen der Masterarbeit (Kapitel, Abbildungen, Literatur)
├── results/      # Ergebnisse der Pipeline, v. a. qualitative Tracking-Ergebnisse
├── Readme.md     # Zentrale Projektbeschreibung
└── .gitignore    # Ausschluss großer oder temporärer Dateien (z. B. Videos, Modellgewichte)
```

- **[`scripts/`](scripts/README.md)** – vollständige Verarbeitungspipeline (Datenaufbereitung, Baseline-Training, Detection-Pseudo-Labeling, Tracking-basierte Label-Propagation, Evaluation).
- **[`notebooks/`](notebooks/pipeline.ipynb)** – dieselbe Pipeline als Notebook mit deutschen Erläuterungen.
- **[`thesis/`](thesis/)** – vollständige LaTeX-Struktur der Arbeit inkl. aller Kapitel und der Abbildungen der Experimente (Baseline- und Pseudo-Labeling-Ergebnisse unter `thesis/figures/`).
- **[`results/`](results/README.md)** – Tracking-Ergebnisse (YOLO + ByteTrack + GPS) pro Video mit Report, Zusammenfassung und Link zum extern gehosteten Video.

## Thema der Arbeit

In videobasierten Computer-Vision-Anwendungen ist die manuelle Annotation jedes einzelnen Frames sehr zeitaufwendig. Besonders bei Drohnenvideos entstehen große Datenmengen, bei denen eine vollständige manuelle Annotation nur mit hohem Aufwand möglich ist.

In dieser Arbeit wird untersucht, wie automatisch Labels für nicht annotierte Zwischenframes erzeugt werden können, wenn nur ausgewählte Frames manuell gelabelt sind. Im konkreten Fall dienen spärliche manuelle Annotationen, zum Beispiel jeder 30. Frame, als Grundlage für die automatische Erweiterung des Datensatzes.

Der Fokus liegt auf der Kombination von Object Detection, Pseudo-Labeling, heuristischer Filterung, Object Tracking und iterativem Retraining.

## Problemstellung

Die verfügbaren Drohnenvideos enthalten nur spärliche manuelle Annotationen. Die zentrale Fragestellung lautet:

> Wie können nicht annotierte Zwischenframes automatisch annotiert werden, ohne jeden Frame manuell labeln zu müssen?

Dabei müssen insbesondere folgende Probleme berücksichtigt werden:

- fehlerhafte oder unsichere Pseudo-Labels
- False Positives im Hintergrund
- instabile Detektionen kleiner Objekte
- Klassenungleichgewicht
- zeitliche Inkonsistenzen zwischen aufeinanderfolgenden Frames
- mögliche Fehlerfortpflanzung beim iterativen Retraining

## Methodik

Die Pipeline besteht aus mehreren Schritten:

```text
Sparse manuelle Annotationen
        ↓
Training eines Baseline-YOLO-Modells
        ↓
Pseudo-Label-Generierung für Zwischenframes
        ↓
Heuristische Filterung fehlerhafter Labels
        ↓
Tracking-basierte Konsistenzprüfung
        ↓
Iteratives Retraining / Self-Training
        ↓
Evaluation und Vergleich der Ergebnisse
```

## Voraussetzungen

- Python 3.10 oder höher
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Pandas
- PyYAML

## Ausführung

Die einzelnen Verarbeitungsschritte befinden sich im Ordner `scripts/`
und sind entsprechend der Ausführungsreihenfolge als `main1.py` bis
`main18.py` nummeriert. Weitere Hinweise befinden sich unter
[`scripts/README.md`](scripts/README.md).

## Hinweis zu großen Dateien

Videos, Datensätze und Modellgewichte sind aufgrund ihrer Dateigröße
nicht direkt im Repository enthalten. Links zu den extern bereitgestellten
Tracking-Videos befinden sich im Ordner [`results/`](results/README.md).
