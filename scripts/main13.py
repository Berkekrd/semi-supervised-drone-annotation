# ============================================================
# MAIN13 - OBJEKT-TRACKING MIT GPS-ZUORDNUNG (FIRST-SEEN-REPORT)
# ============================================================
# Zweck:
# Führt ByteTrack-Tracking mit dem Baseline-Modell auf einem
# Drohnenvideo aus. Für jede Track-ID wird nur der Zeitpunkt
# des ERSTEN Auftretens in die CSV geschrieben, zusammen mit
# der zu diesem Frame passenden GPS-Position aus dem Fluglog
# (inkl. Google-Maps-Link). Zusätzlich werden ein annotiertes
# Video und eine Zusammenfassung erzeugt.
#
# Hinweis: Eine frühere Variante dieses Skripts schrieb jede
# Detektion pro Frame in die CSV (vollständiger Frame-Report);
# sie wurde durch diese First-Seen-Variante ersetzt.
# ============================================================

import os
import cv2
import csv
import pandas as pd
from collections import defaultdict
from ultralytics import YOLO

# =========================
# PFADE
# =========================
MODEL_PATH = r"C:\thesis\runs\detect\train\weights\best.pt"

VIDEO_PATH = r"C:\thesis\flight-mbg-v2-02-regionA\avi\video02_regionA.avi"
LOG_PATH = r"C:\thesis\flight-mbg-v2-02-regionA\log\video02_regionA.csv"

OUTPUT_ROOT = r"C:\thesis\runs\tracking_demo3"
RUN_NAME = "video02_regionA_tracking_first_seen_gps"

# =========================
# TRACKER-KONFIGURATION
# =========================
TRACKER_CFG = "bytetrack.yaml"
CONF_TH = 0.25
IMGSZ = 1024

# =========================
# KLASSENNAMEN
# =========================
CLASS_NAMES = {
    0: "pool",
    1: "bottle",
    2: "bucket",
    3: "puddle",
    4: "tire",
    5: "watertank",
    6: "dumpster",
    7: "large trash bin",
    8: "plastic bag",
    9: "small trash bin",
    10: "storm drain",
}

# =========================
# AUSGABEDATEIEN
# =========================
SAVE_DIR = os.path.join(OUTPUT_ROOT, RUN_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

OUTPUT_VIDEO = os.path.join(SAVE_DIR, "tracked_video.mp4")
OUTPUT_CSV = os.path.join(SAVE_DIR, "first_seen_tracking_report_with_gps.csv")
OUTPUT_SUMMARY = os.path.join(SAVE_DIR, "tracking_summary.txt")


def load_gps_log(log_path):
    # Lädt das GPS-Fluglog (CSV) und bereinigt es:
    # nur gültige Zeilen mit Zeitstempel, Breiten- und Längengrad
    if not os.path.exists(log_path):
        print(f"⚠️ GPS log bulunamadı: {log_path}")
        return None

    df = pd.read_csv(log_path)

    required_cols = ["latitude", "longitude", "time(millisecond)"]
    for col in required_cols:
        if col not in df.columns:
            print(f"⚠️ GPS log içinde kolon yok: {col}")
            return None

    df = df.dropna(subset=["latitude", "longitude", "time(millisecond)"]).copy()

    df["time(millisecond)"] = pd.to_numeric(df["time(millisecond)"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude", "time(millisecond)"])
    df = df.sort_values("time(millisecond)").reset_index(drop=True)

    print(f"✅ GPS log yüklendi: {len(df)} satır")
    return df


def get_nearest_gps(log_df, frame_idx, fps):
    """
    Ordnet die Video-Frame-Zeit der GPS-Log-Zeit zu.
    Der Zeitstempel der ersten Logzeile gilt als Videostart.
    """
    if log_df is None or len(log_df) == 0:
        return {
            "gps_time_ms": "",
            "datetime_utc": "",
            "datetime_local": "",
            "latitude": "",
            "longitude": "",
            "altitude": "",
            "google_maps_url": "",
        }

    log_start_ms = float(log_df["time(millisecond)"].iloc[0])
    frame_time_ms = log_start_ms + (frame_idx / fps) * 1000.0

    # Zeile mit der geringsten Zeitdifferenz zum Frame-Zeitpunkt suchen
    idx = (log_df["time(millisecond)"] - frame_time_ms).abs().idxmin()
    row = log_df.loc[idx]

    lat = float(row["latitude"])
    lon = float(row["longitude"])

    altitude = row["altitude(m)"] if "altitude(m)" in log_df.columns else ""
    datetime_utc = row["datetime(utc)"] if "datetime(utc)" in log_df.columns else ""
    datetime_local = row["datetime(local)"] if "datetime(local)" in log_df.columns else ""
    gps_time_ms = row["time(millisecond)"]

    google_maps_url = f"https://www.google.com/maps?q={lat},{lon}"

    return {
        "gps_time_ms": gps_time_ms,
        "datetime_utc": datetime_utc,
        "datetime_local": datetime_local,
        "latitude": lat,
        "longitude": lon,
        "altitude": altitude,
        "google_maps_url": google_maps_url,
    }


def main():
    model = YOLO(MODEL_PATH)
    gps_log = load_gps_log(LOG_PATH)

    # Video-Metadaten (FPS, Auflösung, Frame-Anzahl) auslesen
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Video açılamadı: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"🎥 Video FPS: {fps}")
    print(f"🎥 Video size: {width}x{height}")
    print(f"🎥 Toplam video frame: {total_video_frames}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    csv_file = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    # In die CSV wird pro Track-ID nur das erste Auftreten geschrieben
    csv_writer.writerow([
        "track_id",
        "first_seen_frame_idx",
        "class_id",
        "class_name",
        "first_seen_confidence",
        "x1", "y1", "x2", "y2",
        "center_x", "center_y",
        "bbox_width", "bbox_height",
        "gps_time_ms",
        "datetime_utc",
        "datetime_local",
        "drone_latitude",
        "drone_longitude",
        "altitude_m",
        "google_maps_url"
    ])

    total_detections = 0
    unique_tracks_all = set()
    unique_tracks_by_class = defaultdict(set)
    detections_per_class = defaultdict(int)

    # Dieses Set stellt sicher, dass jede track_id nur EINMAL in die CSV kommt
    written_track_ids = set()

    print("\n🚀 Tracking başlıyor...")

    results = model.track(
        source=VIDEO_PATH,
        tracker=TRACKER_CFG,
        conf=CONF_TH,
        imgsz=IMGSZ,
        persist=True,
        stream=True,
        save=False
    )

    frame_idx = 0

    for result in results:
        # Annotierten Frame ins Ausgabevideo schreiben
        plotted_frame = result.plot()
        writer.write(plotted_frame)

        boxes = result.boxes

        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy()

            cls_list = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
            conf_list = boxes.conf.cpu().numpy() if boxes.conf is not None else []

            # Ohne Track-ID liefert der Tracker -1; solche Boxen kommen nicht in den Report
            id_list = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(xyxy)

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                cls_id = int(cls_list[i]) if i < len(cls_list) else -1
                conf = float(conf_list[i]) if i < len(conf_list) else 0.0
                track_id = int(id_list[i]) if i < len(id_list) else -1

                class_name = CLASS_NAMES.get(cls_id, str(cls_id))

                total_detections += 1
                detections_per_class[class_name] += 1

                if track_id == -1:
                    continue

                unique_tracks_all.add(track_id)
                unique_tracks_by_class[class_name].add(track_id)

                # Bereits geschriebene Track-IDs überspringen (First-Seen-Prinzip)
                if track_id in written_track_ids:
                    continue

                written_track_ids.add(track_id)

                gps_info = get_nearest_gps(gps_log, frame_idx, fps)

                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2
                cy = y1 + h / 2

                csv_writer.writerow([
                    track_id,
                    frame_idx,
                    cls_id,
                    class_name,
                    round(conf, 4),
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(x2), 2),
                    round(float(y2), 2),
                    round(float(cx), 2),
                    round(float(cy), 2),
                    round(float(w), 2),
                    round(float(h), 2),
                    gps_info["gps_time_ms"],
                    gps_info["datetime_utc"],
                    gps_info["datetime_local"],
                    gps_info["latitude"],
                    gps_info["longitude"],
                    gps_info["altitude"],
                    gps_info["google_maps_url"],
                ])

        if frame_idx % 100 == 0:
            print(f"🎞️ Frame işleniyor: {frame_idx}")

        frame_idx += 1

    writer.release()
    csv_file.close()

    # Zusammenfassung als Textdatei schreiben
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("TRACKING SUMMARY WITH FIRST-SEEN GPS\n")
        f.write("============================\n")
        f.write(f"Video: {VIDEO_PATH}\n")
        f.write(f"GPS log: {LOG_PATH}\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Tracker: {TRACKER_CFG}\n")
        f.write(f"FPS: {fps}\n")
        f.write(f"Video size: {width}x{height}\n")
        f.write(f"Processed frames: {frame_idx}\n")
        f.write(f"Total detections across frames: {total_detections}\n")
        f.write(f"Total unique tracked objects: {len(unique_tracks_all)}\n")
        f.write(f"Tracks written to CSV: {len(written_track_ids)}\n\n")

        f.write("Class-based summary:\n")
        for class_name in sorted(detections_per_class.keys()):
            f.write(
                f"- {class_name}: "
                f"detections={detections_per_class[class_name]}, "
                f"unique_tracks={len(unique_tracks_by_class[class_name])}\n"
            )

    print("\n✅ Tracking tamamlandı.")
    print(f"🎥 Video: {OUTPUT_VIDEO}")
    print(f"📄 First-seen CSV rapor: {OUTPUT_CSV}")
    print(f"📊 Özet rapor: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
