# Masterarbeit – Semi-Supervised Annotation von Drohnenvideos

Dieses Repository enthält den aktuellen Entwicklungsstand meiner Masterarbeit im Bereich Computer Vision.

Ziel der Arbeit ist es, den manuellen Annotierungsaufwand bei Drohnenvideos zu reduzieren. Dafür werden nur spärlich manuell annotierte Frames als Ausgangspunkt verwendet. Die nicht annotierten Zwischenframes sollen anschließend automatisch mithilfe von Object Detection, Pseudo-Labeling, heuristischer Filterung und Tracking-basierter Konsistenzprüfung annotiert werden.

> **Hinweis:** Dieses Repository stellt noch nicht die finale Abgabeversion dar. Es handelt sich um eine nahezu finale Entwicklungsstruktur etwa einen Monat vor der geplanten Fertigstellung.

## Repository-Struktur

repo/
├── thesis/ # LaTeX-Dateien der Masterarbeit
├── src/ # Wiederverwendbarer Python-Code für Training, Pseudo-Labeling, Tracking, Filterung und Evaluation
├── scripts/ # Direkt ausführbare Skripte für zentrale Arbeitsschritte
├── notebooks/ # Jupyter Notebooks für Experimente, Analysen und Visualisierungen
├── configs/ # Konfigurationsdateien für Dataset, Training, Pseudo-Labeling, Tracking und Evaluation
├── data/ # Beschreibung der Dataset-Struktur; keine vollständigen Rohdaten
├── models/ # Modellbeschreibungen; keine großen Modellgewichte
├── results/ # Metriken, Plots, Beispielvorhersagen und qualitative Ergebnisse
├── experiments/ # Experimentprotokolle, Versuchsdokumentation und Zwischenergebnisse
├── docs/ # Ergänzende Dokumentation, Methodik, Status und Meeting Notes
├── README.md # Zentrale Projektbeschreibung
├── .gitignore # Ausschluss großer oder temporärer Dateien aus Git
└── requirements.txt # Python-Abhängigkeiten des Projekts

## Thema der Arbeit

In videobasierten Computer-Vision-Anwendungen ist die manuelle Annotation jedes einzelnen Frames sehr zeitaufwendig. Besonders bei Drohnenvideos entstehen große Datenmengen, bei denen eine vollständige manuelle Annotation nur mit hohem Aufwand möglich ist.

In dieser Arbeit wird untersucht, wie automatisch Labels für nicht annotierte Zwischenframes erzeugt werden können, wenn nur ausgewählte Frames manuell gelabelt sind. Im konkreten Fall dienen spärliche manuelle Annotationen, zum Beispiel jeder 30. Frame, als Grundlage für die automatische Erweiterung des Datensatzes.

Der Fokus liegt auf der Kombination von Object Detection, Pseudo-Labeling, heuristischer Filterung, Object Tracking und iterativem Retraining.

## Problemstellung

Die verfügbaren Drohnenvideos enthalten nur spärliche manuelle Annotationen. Dadurch entsteht die Herausforderung, die nicht annotierten Zwischenframes möglichst zuverlässig automatisch zu labeln.

Die zentrale Fragestellung lautet:

> Wie können nicht annotierte Zwischenframes automatisch annotiert werden, ohne jeden Frame manuell labeln zu müssen?

Dabei müssen insbesondere folgende Probleme berücksichtigt werden:

- fehlerhafte oder unsichere Pseudo-Labels
- False Positives im Hintergrund
- instabile Detektionen kleiner Objekte
- Klassenungleichgewicht
- zeitliche Inkonsistenzen zwischen aufeinanderfolgenden Frames
- mögliche Fehlerfortpflanzung beim iterativen Retraining

## Methodik

Die geplante Pipeline besteht aus mehreren Schritten:

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
