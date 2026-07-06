# ============================================================
# MAIN6 - GEZIELTE PSEUDO-LABEL-GENERIERUNG (DETECTION-BASIERT)
# ============================================================
# Zweck:
# Wendet das Baseline-Modell auf alle Video-Frames an, um
# Pseudo-Labels zu erzeugen. Zielklassen werden automatisch
# anhand der manuellen Klassenverteilung ausgewählt:
# unterrepräsentierte Klassen werden bevorzugt, sehr seltene
# oder bereits ausreichend vertretene Klassen ausgeschlossen.
# ============================================================

import os
import cv2
import glob
import json
from collections import Counter
from ultralytics import YOLO

# =========================
# PFADE
# =========================
MODEL_PATH = r"C:\thesis\runs\detect\train\weights\best.pt"
MANUAL_DATASET_ROOT = r"C:\thesis\datasets\merged_dataset"

THESIS_ROOT = r"C:\thesis"
ALL_FRAMES_ROOT = r"C:\thesis\all_frames"
PSEUDO_OUTPUT_ROOT = r"C:\thesis\runs\pseudo"

model = YOLO(MODEL_PATH)

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
# EINSTELLUNGEN FÜR DIE AUTOMATISCHE KLASSENAUSWAHL
# =========================
# Klassen mit sehr wenigen manuellen Beispielen sind für Pseudo-Labeling riskant.
# Diese werden statt automatischem Pseudo-Labeling zur manuellen Prüfung markiert.
MIN_MANUAL_FOR_AUTO_PSEUDO = 20

# Oberhalb dieser Anzahl gilt eine Klasse als ausreichend vertreten.
# So wird z. B. "watertank" automatisch ausgeschlossen.
MAX_MANUAL_FOR_AUTO_PSEUDO = 1000

# Für Klassen ohne manuelle Beispiele keine Pseudo-Labels erzeugen.
ALLOW_ZERO_SHOT_PSEUDO = False


def read_manual_distribution():
    # Zählt die Bounding-Boxen pro Klasse im manuellen Dataset (train + val)
    counter = Counter()

    for split in ["train", "val"]:
        label_dir = os.path.join(MANUAL_DATASET_ROOT, split, "labels")

        if not os.path.exists(label_dir):
            print(f"⚠️ Manual label klasörü yok: {label_dir}")
            continue

        for file in os.listdir(label_dir):
            if not file.endswith(".txt"):
                continue

            path = os.path.join(label_dir, file)

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    cls = int(parts[0])
                    counter[cls] += 1

    return counter


def decide_target_classes(manual_counter):
    # Teilt die Klassen anhand der manuellen Verteilung in drei Gruppen auf:
    # target (Pseudo-Labeling), excluded (ausgeschlossen), manual_review (manuell prüfen)
    target_classes = []
    excluded_classes = []
    manual_review_classes = []

    for cls, name in CLASS_NAMES.items():
        count = manual_counter.get(cls, 0)

        if count == 0 and not ALLOW_ZERO_SHOT_PSEUDO:
            excluded_classes.append(cls)
            continue

        if count < MIN_MANUAL_FOR_AUTO_PSEUDO:
            manual_review_classes.append(cls)
            continue

        if count > MAX_MANUAL_FOR_AUTO_PSEUDO:
            excluded_classes.append(cls)
            continue

        target_classes.append(cls)

    return target_classes, excluded_classes, manual_review_classes


def find_videos():
    # Sucht alle AVI-Videos nach dem Muster flight-mbg-v2-*/avi/*.avi
    pattern = os.path.join(THESIS_ROOT, "flight-mbg-v2-*", "avi", "*.avi")
    video_paths = sorted(glob.glob(pattern))

    videos = {}

    for path in video_paths:
        # Ordnername: flight-mbg-v2-01-regionA
        parent = os.path.basename(os.path.dirname(os.path.dirname(path)))

        # Videoname: video01_regionA
        video_name = os.path.splitext(os.path.basename(path))[0]

        videos[video_name] = path

    return videos


def extract_all_frames(video_name, video_path):
    # Extrahiert alle Frames eines Videos als JPEG.
    # Bereits extrahierte Videos werden übersprungen.
    output_dir = os.path.join(ALL_FRAMES_ROOT, video_name)
    os.makedirs(output_dir, exist_ok=True)

    existing_frames = [
        f for f in os.listdir(output_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if len(existing_frames) > 0:
        print(f"✅ Frame zaten var, geçiliyor: {output_dir}")
        return output_dir, True

    if not os.path.exists(video_path):
        print(f"❌ Video dosyası bulunamadı: {video_path}")
        return output_dir, False

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Video açılamadı: {video_path}")
        return output_dir, False

    frame_id = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_name = f"frame_{frame_id:06d}.jpg"
        frame_path = os.path.join(output_dir, frame_name)

        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame_id += 1

    cap.release()

    if frame_id == 0:
        print(f"❌ Frame çıkarılamadı: {video_path}")
        return output_dir, False

    print(f"✅ {frame_id} frame çıkarıldı: {output_dir}")
    return output_dir, True


def save_target_config(target_classes, excluded_classes, manual_review_classes, manual_counter):
    # Speichert die Klassenauswahl als JSON — wird von main7/main8/main10 weiterverwendet
    os.makedirs(PSEUDO_OUTPUT_ROOT, exist_ok=True)

    config = {
        "target_classes": target_classes,
        "excluded_classes": excluded_classes,
        "manual_review_classes": manual_review_classes,
        "manual_distribution": {str(k): v for k, v in manual_counter.items()},
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
    }

    out_path = os.path.join(PSEUDO_OUTPUT_ROOT, "target_config.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Target config kaydedildi: {out_path}")


def run_pseudo_prediction(video_name, frames_dir, target_classes):
    # Führt die YOLO-Prädiktion auf allen Frames aus und speichert
    # die Ergebnisse als TXT-Labels inkl. Confidence-Werten
    print(f"\n🚀 Pseudo-label başlıyor: {video_name}")

    results = model.predict(
        source=frames_dir,
        project=PSEUDO_OUTPUT_ROOT,
        name=video_name,
        save=False,
        save_txt=True,
        save_conf=True,
        conf=0.25,
        imgsz=1024,
        classes=target_classes,
        exist_ok=True,
        stream=True
    )

    # stream=True liefert einen Generator — Iteration stößt die Verarbeitung an
    for _ in results:
        pass

    print(f"✅ Pseudo-label tamamlandı: {video_name}")


def main():
    manual_counter = read_manual_distribution()

    print("\n📊 Manual dataset class dağılımı:")
    for cls in sorted(CLASS_NAMES.keys()):
        print(f"{cls:2d} | {CLASS_NAMES[cls]:20s}: {manual_counter.get(cls, 0)}")

    target_classes, excluded_classes, manual_review_classes = decide_target_classes(manual_counter)

    print("\n🎯 Otomatik seçilen pseudo target class'lar:")
    for cls in target_classes:
        print(f"{cls:2d} | {CLASS_NAMES[cls]}")

    print("\n🚫 Pseudo'dan çıkarılan class'lar:")
    for cls in excluded_classes:
        print(f"{cls:2d} | {CLASS_NAMES[cls]}")

    print("\n👀 Manuel kontrol önerilen çok az class'lar:")
    for cls in manual_review_classes:
        print(f"{cls:2d} | {CLASS_NAMES[cls]}")

    if not target_classes:
        print("\n❌ Target class bulunamadı. Ayarları kontrol et.")
        return

    save_target_config(target_classes, excluded_classes, manual_review_classes, manual_counter)

    videos = find_videos()

    print("\n🎬 Bulunan videolar:")
    for name, path in videos.items():
        print(f"{name}: {path}")

    if not videos:
        print("\n❌ Video bulunamadı.")
        return

    for video_name, video_path in videos.items():
        print("\n==============================")
        print(f"🎬 İşleniyor: {video_name}")
        print("==============================")

        frames_dir, ok = extract_all_frames(video_name, video_path)

        if ok:
            run_pseudo_prediction(video_name, frames_dir, target_classes)
        else:
            print(f"⏭️ Atlandı: {video_name}")

    print("\n🔥 Targeted pseudo-label üretimi tamamlandı!")


if __name__ == "__main__":
    main()
