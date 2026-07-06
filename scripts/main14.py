# ============================================================
# MAIN14 - TRACKING-BASIERTE LABEL-PROPAGATION
# ============================================================
# Zweck:
# Führt das Baseline-YOLO-Modell mit ByteTrack auf allen Videos
# aus und erzeugt Pseudo-Labels anhand der Track-Kontinuität:
# Nur Tracks, die lang genug sind, eine ausreichende mittlere
# Confidence haben und klassenkonsistent sind, werden behalten.
#
# Ausgabe:
# C:\thesis\runs\tracking_pseudo\
#   video01_regionA\
#       images\
#       labels\
#   video02_regionA\
#       images\
#       labels\
#   tracking_pseudo_summary.csv
#   tracking_pseudo_detections.csv
# ============================================================

import os
import re
import cv2
import csv
import glob
import json
from collections import defaultdict, Counter
from ultralytics import YOLO


# =========================
# PFADE
# =========================
THESIS_ROOT = r"C:\thesis"

MODEL_PATH = r"C:\thesis\runs\detect\train\weights\best.pt"

# Falls main6.py bereits eine target_config.json erzeugt hat, wird sie genutzt.
# Andernfalls werden alle Klassen verwendet.
TARGET_CONFIG_PATH = r"C:\thesis\runs\pseudo\target_config.json"

OUTPUT_ROOT = r"C:\thesis\runs\tracking_pseudo"

# Bilder aus dem manuellen Dataset werden ausgeschlossen,
# um Val-/Test-Leakage zu verhindern.
MANUAL_DATASET_ROOT = r"C:\thesis\datasets\merged_dataset"


# =========================
# TRACKING-EINSTELLUNGEN
# =========================
TRACKER_CFG = "bytetrack.yaml"

# Detection-Schwellenwert für ByteTrack.
# Bewusst niedrig, damit der Tracker auch schwache Detektionen assoziieren kann.
TRACK_CONF_TH = 0.25

IMGSZ = 1024

# Mindest-Confidence, damit eine Box als Pseudo-Label akzeptiert wird.
KEEP_MIN_CONF = 0.60

# Mindestanzahl an Frames, in denen ein Track sichtbar sein muss.
MIN_TRACK_LENGTH = 3

# Schwellenwert für die mittlere Track-Confidence.
MIN_AVG_TRACK_CONF = 0.60

# Anteil der dominanten Klasse innerhalb eines Tracks.
# Beispiel: mindestens 70 % der Detektionen eines Tracks müssen dieselbe Klasse haben.
MIN_CLASS_CONSISTENCY = 0.70

# Minimale normierte Fläche, um winzige Boxen auszusortieren.
# 0.000001 = sehr klein, filtert nur völlig unsinnige Boxen heraus.
MIN_BOX_AREA_NORM = 0.000001

# Ausgabeformat der Bilder
OUTPUT_IMAGE_EXT = ".jpg"
JPEG_QUALITY = 90


# =========================
# KLASSENNAMEN
# =========================
DEFAULT_CLASS_NAMES = {
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


def load_target_config():
    """
    Nutzt — falls vorhanden — die von main6.py erzeugte target_config.json:
    - target_classes
    - class_names

    Ohne Config werden alle Klassen als Ziel betrachtet.
    """
    if not os.path.exists(TARGET_CONFIG_PATH):
        print("⚠️ target_config.json bulunamadı. Tüm class'lar kullanılacak.")
        return DEFAULT_CLASS_NAMES, set(DEFAULT_CLASS_NAMES.keys())

    with open(TARGET_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    class_names = {
        int(k): v
        for k, v in config.get("class_names", {}).items()
    }

    if not class_names:
        class_names = DEFAULT_CLASS_NAMES

    target_classes = set(int(x) for x in config.get("target_classes", []))

    if not target_classes:
        print("⚠️ target_config içinde target_classes boş. Tüm class'lar kullanılacak.")
        target_classes = set(class_names.keys())

    print("✅ target_config yüklendi.")
    print("🎯 Tracking pseudo target classes:")
    for cls in sorted(target_classes):
        print(f"{cls:2d} | {class_names.get(cls, cls)}")

    return class_names, target_classes


def find_videos():
    """
    Sucht die Videos unter C:\\thesis\\flight-mbg-v2-*\\avi\\*.avi.
    """
    pattern = os.path.join(THESIS_ROOT, "flight-mbg-v2-*", "avi", "*.avi")
    video_paths = sorted(glob.glob(pattern))

    videos = {}

    for path in video_paths:
        video_name = os.path.splitext(os.path.basename(path))[0]
        videos[video_name] = path

    return videos


def extract_frame_token(filename_or_base):
    """
    Extrahiert den Frame-Token (z. B. frame_000123) aus einem Dateinamen.
    Funktioniert auch, wenn der Dateiname im manuellen Dataset ein Präfix trägt.
    """
    base = os.path.splitext(os.path.basename(filename_or_base))[0]
    match = re.search(r"(frame_\d+)", base)
    if match:
        return match.group(1)
    return base


def load_forbidden_frame_tokens():
    """
    Sammelt die Frames, die bereits im manuellen Train/Val/Test enthalten sind,
    damit sie nicht ins Pseudo-Dataset gelangen. So wird verhindert:
    - Duplikate im manuellen Training
    - Leakage in die manuelle Validierung
    """
    forbidden = set()

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(MANUAL_DATASET_ROOT, split, "images")

        if not os.path.exists(img_dir):
            continue

        for file in os.listdir(img_dir):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            token = extract_frame_token(file)
            forbidden.add(token)

    print(f"✅ Manual dataset içinden exclude edilecek frame token sayısı: {len(forbidden)}")
    return forbidden


def get_video_info(video_path):
    # Liest FPS, Auflösung und Frame-Anzahl eines Videos aus
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return fps, width, height, frame_count


def clamp(value, min_value, max_value):
    return max(min_value, min(float(value), max_value))


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """
    Wandelt Pixelkoordinaten (xyxy) ins normierte YOLO-Format um.
    Ungültige oder zu kleine Boxen liefern None.
    """
    x1 = clamp(x1, 0, img_w - 1)
    y1 = clamp(y1, 0, img_h - 1)
    x2 = clamp(x2, 0, img_w - 1)
    y2 = clamp(y2, 0, img_h - 1)

    if x2 <= x1 or y2 <= y1:
        return None

    box_w = x2 - x1
    box_h = y2 - y1

    x_center = (x1 + box_w / 2.0) / img_w
    y_center = (y1 + box_h / 2.0) / img_h
    width = box_w / img_w
    height = box_h / img_h

    area = width * height

    if area < MIN_BOX_AREA_NORM:
        return None

    return x_center, y_center, width, height


def analyze_tracks(track_records):
    """
    Berechnet pro track_id:
    - Länge (Anzahl Frames)
    - mittlere Confidence
    - dominante Klasse
    - Klassenkonsistenz
    und entscheidet, ob der Track behalten wird.
    """
    track_info = {}

    for track_id, records in track_records.items():
        if not records:
            continue

        cls_counter = Counter(r["class_id"] for r in records)
        dominant_class, dominant_count = cls_counter.most_common(1)[0]

        length = len(records)
        avg_conf = sum(r["confidence"] for r in records) / length
        class_consistency = dominant_count / length

        keep = (
            length >= MIN_TRACK_LENGTH
            and avg_conf >= MIN_AVG_TRACK_CONF
            and class_consistency >= MIN_CLASS_CONSISTENCY
        )

        # Ablehnungsgrund für den Bericht festhalten
        reason = "kept"

        if length < MIN_TRACK_LENGTH:
            reason = "too_short"
        elif avg_conf < MIN_AVG_TRACK_CONF:
            reason = "low_avg_conf"
        elif class_consistency < MIN_CLASS_CONSISTENCY:
            reason = "class_inconsistent"

        track_info[track_id] = {
            "length": length,
            "avg_conf": avg_conf,
            "dominant_class": dominant_class,
            "dominant_count": dominant_count,
            "class_consistency": class_consistency,
            "keep": keep,
            "reason": reason,
            "class_counter": dict(cls_counter),
        }

    return track_info


def export_frames_and_labels(video_path, video_name, labels_by_frame, forbidden_tokens):
    """
    Zweiter Durchlauf:
    Öffnet das Video erneut und speichert nur die Frames als Bild,
    für die auch Labels geschrieben werden.
    """
    out_img_dir = os.path.join(OUTPUT_ROOT, video_name, "images")
    out_lbl_dir = os.path.join(OUTPUT_ROOT, video_name, "labels")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Export için video açılamadı: {video_path}")
        return 0, 0, 0

    target_frames = set(labels_by_frame.keys())

    frame_idx = 0
    saved_images = 0
    saved_labels = 0
    skipped_forbidden = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx not in target_frames:
            frame_idx += 1
            continue

        frame_token = f"frame_{frame_idx:06d}"

        # Frames, die im manuellen Train/Val vorkommen, nicht übernehmen
        if frame_token in forbidden_tokens:
            skipped_forbidden += 1
            frame_idx += 1
            continue

        labels = labels_by_frame[frame_idx]

        if not labels:
            frame_idx += 1
            continue

        image_name = f"{frame_token}{OUTPUT_IMAGE_EXT}"
        label_name = f"{frame_token}.txt"

        image_path = os.path.join(out_img_dir, image_name)
        label_path = os.path.join(out_lbl_dir, label_name)

        cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        with open(label_path, "w", encoding="utf-8") as f:
            for line in labels:
                f.write(line)

        saved_images += 1
        saved_labels += 1

        frame_idx += 1

    cap.release()

    return saved_images, saved_labels, skipped_forbidden


def process_video(model, video_name, video_path, class_names, target_classes, forbidden_tokens):
    print("\n" + "=" * 80)
    print(f"🎬 İşleniyor: {video_name}")
    print("=" * 80)

    fps, img_w, img_h, frame_count = get_video_info(video_path)

    print(f"🎥 FPS: {fps}")
    print(f"🎥 Size: {img_w}x{img_h}")
    print(f"🎥 Frame count: {frame_count}")

    # track_id -> Liste der Detektionen dieses Tracks
    track_records = defaultdict(list)

    total_raw_detections = 0
    total_with_track_id = 0
    total_without_track_id = 0
    target_class_detections = 0

    print("🚀 ByteTrack başlıyor...")

    results = model.track(
        source=video_path,
        tracker=TRACKER_CFG,
        conf=TRACK_CONF_TH,
        imgsz=IMGSZ,
        persist=True,
        stream=True,
        save=False,
        verbose=False,
    )

    frame_idx = 0

    # Erster Durchlauf: alle Detektionen mit Track-ID einsammeln
    for result in results:
        boxes = result.boxes

        if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls_list = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
            conf_list = boxes.conf.cpu().numpy() if boxes.conf is not None else []

            if boxes.id is not None:
                id_list = boxes.id.cpu().numpy().astype(int)
            else:
                id_list = [-1] * len(xyxy)

            for i in range(len(xyxy)):
                total_raw_detections += 1

                cls_id = int(cls_list[i]) if i < len(cls_list) else -1
                conf = float(conf_list[i]) if i < len(conf_list) else 0.0
                track_id = int(id_list[i]) if i < len(id_list) else -1

                if cls_id not in target_classes:
                    continue

                target_class_detections += 1

                if track_id == -1:
                    total_without_track_id += 1
                    continue

                total_with_track_id += 1

                x1, y1, x2, y2 = xyxy[i]

                track_records[track_id].append({
                    "video_name": video_name,
                    "frame_idx": frame_idx,
                    "track_id": track_id,
                    "class_id": cls_id,
                    "class_name": class_names.get(cls_id, str(cls_id)),
                    "confidence": conf,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                })

        if frame_idx % 250 == 0:
            print(f"🎞️ Frame: {frame_idx}")

        frame_idx += 1

    print("✅ ByteTrack tamamlandı.")
    print(f"📦 Raw detections: {total_raw_detections}")
    print(f"🎯 Target class detections: {target_class_detections}")
    print(f"🆔 Track ID bulunan: {total_with_track_id}")
    print(f"⚠️ Track ID olmayan: {total_without_track_id}")
    print(f"🧵 Unique track: {len(track_records)}")

    # Qualitätsanalyse der Tracks
    track_info = analyze_tracks(track_records)

    kept_tracks = {
        tid: info
        for tid, info in track_info.items()
        if info["keep"]
    }

    removed_tracks = {
        tid: info
        for tid, info in track_info.items()
        if not info["keep"]
    }

    print(f"✅ Kept tracks: {len(kept_tracks)}")
    print(f"❌ Removed tracks: {len(removed_tracks)}")

    removed_reason_counter = Counter(info["reason"] for info in removed_tracks.values())

    if removed_reason_counter:
        print("❌ Removed reasons:")
        for reason, count in removed_reason_counter.items():
            print(f"   - {reason}: {count}")

    # frame_idx -> YOLO-Labelzeilen
    labels_by_frame = defaultdict(list)

    kept_bbox_counter = Counter()
    removed_bbox_counter = Counter()

    detection_rows = []

    for track_id, records in track_records.items():
        info = track_info.get(track_id)

        if info is None or not info["keep"]:
            removed_bbox_counter["track_filtered"] += len(records)
            continue

        dominant_class = info["dominant_class"]

        for r in records:
            # Detektionen außerhalb der dominanten Track-Klasse verwerfen
            if r["class_id"] != dominant_class:
                removed_bbox_counter["non_dominant_class"] += 1
                continue

            # Boxen mit zu niedriger Confidence verwerfen
            if r["confidence"] < KEEP_MIN_CONF:
                removed_bbox_counter["low_conf_bbox"] += 1
                continue

            yolo_box = xyxy_to_yolo(
                r["x1"], r["y1"], r["x2"], r["y2"],
                img_w, img_h
            )

            if yolo_box is None:
                removed_bbox_counter["invalid_box"] += 1
                continue

            x_center, y_center, width, height = yolo_box

            label_line = (
                f"{dominant_class} "
                f"{x_center:.6f} {y_center:.6f} "
                f"{width:.6f} {height:.6f}\n"
            )

            labels_by_frame[r["frame_idx"]].append(label_line)
            kept_bbox_counter[dominant_class] += 1

            detection_rows.append({
                **r,
                "dominant_class": dominant_class,
                "track_length": info["length"],
                "track_avg_conf": info["avg_conf"],
                "class_consistency": info["class_consistency"],
            })

    saved_images, saved_labels, skipped_forbidden = export_frames_and_labels(
        video_path=video_path,
        video_name=video_name,
        labels_by_frame=labels_by_frame,
        forbidden_tokens=forbidden_tokens
    )

    print(f"💾 Saved images: {saved_images}")
    print(f"💾 Saved labels: {saved_labels}")
    print(f"🚫 Skipped manual frames: {skipped_forbidden}")

    print("📊 Kept bbox by class:")
    for cls in sorted(kept_bbox_counter.keys()):
        print(f"{cls:2d} | {class_names.get(cls, cls):20s}: {kept_bbox_counter[cls]}")

    summary = {
        "video_name": video_name,
        "video_path": video_path,
        "fps": fps,
        "width": img_w,
        "height": img_h,
        "frame_count": frame_count,
        "processed_frames": frame_idx,
        "raw_detections": total_raw_detections,
        "target_class_detections": target_class_detections,
        "detections_with_track_id": total_with_track_id,
        "detections_without_track_id": total_without_track_id,
        "unique_tracks": len(track_records),
        "kept_tracks": len(kept_tracks),
        "removed_tracks": len(removed_tracks),
        "saved_images": saved_images,
        "saved_labels": saved_labels,
        "skipped_manual_frames": skipped_forbidden,
        "kept_bbox_total": sum(kept_bbox_counter.values()),
        "removed_bbox_total": sum(removed_bbox_counter.values()),
        "kept_bbox_by_class": dict(kept_bbox_counter),
        "removed_bbox_by_reason": dict(removed_bbox_counter),
        "removed_tracks_by_reason": dict(removed_reason_counter),
    }

    return summary, detection_rows


def save_global_reports(all_summaries, all_detection_rows, class_names):
    # Speichert die globalen Berichte (CSV + JSON) über alle Videos
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    summary_csv = os.path.join(OUTPUT_ROOT, "tracking_pseudo_summary.csv")
    detections_csv = os.path.join(OUTPUT_ROOT, "tracking_pseudo_detections.csv")
    summary_json = os.path.join(OUTPUT_ROOT, "tracking_pseudo_summary.json")

    # Zusammenfassung als CSV
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "video_name",
            "processed_frames",
            "raw_detections",
            "target_class_detections",
            "detections_with_track_id",
            "detections_without_track_id",
            "unique_tracks",
            "kept_tracks",
            "removed_tracks",
            "saved_images",
            "saved_labels",
            "skipped_manual_frames",
            "kept_bbox_total",
            "removed_bbox_total",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for s in all_summaries:
            writer.writerow({k: s.get(k, "") for k in fieldnames})

    # Detektionen als CSV
    with open(detections_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "video_name",
            "frame_idx",
            "track_id",
            "class_id",
            "class_name",
            "confidence",
            "x1", "y1", "x2", "y2",
            "dominant_class",
            "track_length",
            "track_avg_conf",
            "class_consistency",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in all_detection_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # Einstellungen + Zusammenfassungen als JSON
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "settings": {
                    "model_path": MODEL_PATH,
                    "tracker": TRACKER_CFG,
                    "track_conf_th": TRACK_CONF_TH,
                    "imgsz": IMGSZ,
                    "keep_min_conf": KEEP_MIN_CONF,
                    "min_track_length": MIN_TRACK_LENGTH,
                    "min_avg_track_conf": MIN_AVG_TRACK_CONF,
                    "min_class_consistency": MIN_CLASS_CONSISTENCY,
                    "min_box_area_norm": MIN_BOX_AREA_NORM,
                },
                "class_names": {str(k): v for k, v in class_names.items()},
                "summaries": all_summaries,
            },
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n🔥 Global raporlar kaydedildi:")
    print(f"📄 {summary_csv}")
    print(f"📄 {detections_csv}")
    print(f"📄 {summary_json}")


def main():
    print("🚀 MAIN14 - Tracking-based label propagation başlıyor")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Baseline model bulunamadı: {MODEL_PATH}")
        print("Önce main5.py ile baseline modelin oluşmuş olması gerekiyor.")
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    class_names, target_classes = load_target_config()
    forbidden_tokens = load_forbidden_frame_tokens()

    videos = find_videos()

    if not videos:
        print("❌ Video bulunamadı.")
        print(r"Beklenen pattern: C:\thesis\flight-mbg-v2-*\avi\*.avi")
        return

    print("\n🎬 Bulunan videolar:")
    for name, path in videos.items():
        print(f"- {name}: {path}")

    model = YOLO(MODEL_PATH)

    all_summaries = []
    all_detection_rows = []

    for video_name, video_path in videos.items():
        try:
            summary, detection_rows = process_video(
                model=model,
                video_name=video_name,
                video_path=video_path,
                class_names=class_names,
                target_classes=target_classes,
                forbidden_tokens=forbidden_tokens
            )

            all_summaries.append(summary)
            all_detection_rows.extend(detection_rows)

        except Exception as e:
            print(f"❌ Hata oluştu: {video_name}")
            print(str(e))

    save_global_reports(all_summaries, all_detection_rows, class_names)

    print("\n✅ MAIN14 tamamlandı.")
    print(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
