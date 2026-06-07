# video04_regionB_01 – Tracking-Ergebnis

Multi-Object-Tracking auf dem Drohnenvideo **video04_regionB** (Region B). Das annotierte Video zeigt die mit dem YOLO-Modell erkannten Objekte sowie die über ByteTrack vergebenen Track-IDs als Overlay. Ergänzend werden die GPS-Koordinaten des Flugs aus dem zugehörigen Log eingebunden.

## 🎬 Video

> Die Videodatei (`tracked_video.mp4`, ca. 3,7 GB, 4K) ist aufgrund ihrer Größe **nicht** im Repository enthalten (siehe [`.gitignore`](../../../.gitignore)). Das vollständige Video ist hier abrufbar:

**▶ [Video ansehen (Google Drive)](https://drive.google.com/file/d/1wDNBesEX62_mUq_ulWIKpAWM8FOAn_20/view?usp=sharing)**

## Technische Details

| Eigenschaft | Wert |
|-------------|------|
| Quellvideo | `video04_regionB.avi` |
| Modell | YOLO (`best.pt`, eigenes Training) |
| Tracker | ByteTrack (`bytetrack.yaml`) |
| Auflösung | 3840 × 2160 (4K UHD) |
| Framerate | 29,97 FPS |
| Verarbeitete Frames | 6.285 (≈ 3,5 min) |
| Detektionen gesamt | 51.867 |
| Eindeutige Tracks | 716 |

## Erkannte Klassen

| Klasse | Detektionen | Eindeutige Tracks |
|--------|-------------|-------------------|
| watertank | 42.135 | 498 |
| pool | 4.147 | 47 |
| storm drain | 2.682 | 68 |
| tire | 1.284 | 33 |
| bucket | 1.011 | 75 |
| plastic bag | 608 | 3 |

## Dateien in diesem Ordner

| Datei | Beschreibung | Im Repo |
|-------|--------------|---------|
| `tracking_report_with_gps.csv` | Frame-genauer Tracking-Report inkl. GPS-Koordinaten | ✅ |
| `tracking_summary.txt` | Zusammenfassung der Tracking-Statistiken | ✅ |
| `tracked_video.mp4` | Annotiertes Video (Overlay) | ❌ extern, siehe Link oben |
