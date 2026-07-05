# Masterarbeit – Semi-Supervised Annotation von Drohnenvideos

Dieses Repository enthält den aktuellen Entwicklungsstand meiner Masterarbeit im Bereich Computer Vision.

Ziel der Arbeit ist es, den manuellen Annotierungsaufwand bei Drohnenvideos zu reduzieren. Dafür werden nur spärlich manuell annotierte Frames als Ausgangspunkt verwendet. Die nicht annotierten Zwischenframes werden anschließend automatisch mithilfe von Object Detection, Pseudo-Labeling, heuristischer Filterung und Tracking-basierter Konsistenzprüfung annotiert.

> **Hinweis:** Dieses Repository stellt noch nicht die finale Abgabeversion dar. Es handelt sich um den Entwicklungsstand etwa einen Monat vor der geplanten Fertigstellung. Der Python-Code der Pipeline (Training, Pseudo-Labeling, Tracking, Evaluation) wird bis zur Abgabe ergänzt.

## Repository-Struktur

```text
repo/
├── thesis/       # LaTeX-Quellen der Masterarbeit (Kapitel, Abbildungen, Literatur)
├── results/      # Ergebnisse der Pipeline, v. a. qualitative Tracking-Ergebnisse
├── Readme.md     # Zentrale Projektbeschreibung
└── .gitignore    # Ausschluss großer oder temporärer Dateien (z. B. Videos, Modellgewichte)
```

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
