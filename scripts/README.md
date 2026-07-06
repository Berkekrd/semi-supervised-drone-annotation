# Skripte der Pipeline (main1–main18)

Die vollständige Verarbeitungspipeline der Arbeit, aufgeteilt in 18 nummerierte Skripte.
Dieselben Skripte sind — mit deutschen Erläuterungen — auch als Notebook integriert:
[`notebooks/pipeline.ipynb`](../notebooks/pipeline.ipynb).

> **Hinweis:** Die Pfade verweisen auf die Original-Arbeitsumgebung (`C:\thesis\...` bzw.
> relative Pfade) und müssen zum Ausführen angepasst werden.

| Skript | Phase | Beschreibung |
|---|---|---|
| `main1.py` | Datenaufbereitung | Validierung der CLASS_MAP gegen die CVAT-XML-Labels |
| `main2.py` | Datenaufbereitung | Konvertierung CVAT-XML → YOLO-Format (mit automatischem Frame-Offset) |
| `main3.py` | Datenaufbereitung | Konsistenzprüfung der Image/Label-Paare |
| `main4.py` | Datenaufbereitung | Reproduzierbarer Train/Val-Split (80/20, Seed 42) |
| `main5.py` | Baseline | Training des Baseline-Modells (YOLOv8n, imgsz=1024) |
| `main6.py` | Detection-Pseudo | Gezielte Pseudo-Label-Generierung mit automatischer Klassenauswahl |
| `main7.py` | Detection-Pseudo | Analyse der Pseudo-Labels und Threshold-Empfehlung pro Klasse |
| `main8.py` | Detection-Pseudo | Confidence-Filterung der Pseudo-Labels |
| `main9.py` | Detection-Pseudo | Aufbau des Pseudo-Datasets (Bilder + Labels) |
| `main10.py` | Detection-Pseudo | Klassenbalancierte Auswahl per Quota-Verfahren |
| `main11.py` | Detection-Pseudo | Merge zu Semi-Supervised-Dataset (Val = nur manuell) |
| `main12.py` | Detection-Pseudo | Fine-Tuning mit dem Semi-Supervised-Dataset |
| `main13.py` | Tracking-Demo | ByteTrack-Tracking mit GPS-Zuordnung (First-Seen-Report) |
| `main14.py` | Tracking-Pseudo | Tracking-basierte Label-Propagation (Track-Qualitätsfilter) |
| `main15.py` | Tracking-Pseudo | Balanciertes Tracking-Pseudo-Dataset (Quota-Verfahren) |
| `main16.py` | Tracking-Pseudo | Merge zu Tracking-Semi-Supervised-Dataset + data.yaml |
| `main17.py` | Tracking-Pseudo | Fine-Tuning mit Tracking-Label-Propagation |
| `main18.py` | Evaluation | Vergleich aller drei Modelle auf identischem Validierungsset |
