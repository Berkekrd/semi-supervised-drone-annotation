# ============================================================
# MAIN2 - KONVERTIERUNG CVAT-XML -> YOLO-FORMAT (MIT FRAME-OFFSET)
# ============================================================
# Zweck:
# Wandelt die CVAT-XML-Annotationen in YOLO-Trainingslabels um.
# Da die Frame-Nummern auf der Festplatte und in der XML nicht
# identisch sind, wird zunächst automatisch der beste Offset
# zwischen beiden Nummerierungen bestimmt.
# ============================================================

import os
import cv2
import xml.etree.ElementTree as ET

DATASET_NAME = "video04_regionB"

# Eingabe: extrahierte Frames und CVAT-XML-Annotationen
FRAMES_DIR = "flight-mbg-v2-04-regionB/frames"
XML_PATH = f"flight-mbg-v2-04-regionB/ann/{DATASET_NAME}.xml"

# Ausgabe: YOLO-Dataset (Bilder + Labels)
OUT_IMG_DIR = f"datasets/{DATASET_NAME}/images"
OUT_LABEL_DIR = f"datasets/{DATASET_NAME}/labels"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

# Zuordnung: Klassenname -> YOLO-Klassen-ID
CLASS_MAP = {
    "pool": 0,
    "bottle": 1,
    "bucket": 2,
    "puddle": 3,
    "tire": 4,
    "watertank": 5,
    "dumpster": 6,
    "large_trash_bin": 7,
    "plastic_bag": 8,
    "small_trash_bin": 9,
    "storm_drain": 10,
}

def parse_frame_id(filename: str) -> int:
    # Extrahiert die Frame-Nummer aus dem Dateinamen (z. B. frame_000123.png -> 123)
    return int(os.path.splitext(filename)[0].split("_")[1])

def build_frame_dict(xml_path: str, class_map: dict):
    # Liest alle Bounding-Boxen aus der XML ein und gruppiert sie pro Frame
    tree = ET.parse(xml_path)
    root = tree.getroot()

    frame_dict = {}
    unknown_labels = set()

    for track in root.findall("track"):
        label = track.get("label")
        if not label:
            continue

        label_clean = label.strip().lower().replace(" ", "_")

        # Unbekannte Labels werden gesammelt und übersprungen
        if label_clean not in class_map:
            unknown_labels.add(label_clean)
            continue

        class_id = class_map[label_clean]

        for box in track.findall("box"):
            frame = int(box.get("frame"))
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            frame_dict.setdefault(frame, []).append((class_id, xtl, ytl, xbr, ybr))

    return frame_dict, unknown_labels

def find_best_offset(disk_ids, xml_ids, step=24):
    # Bestimmt den besten Offset zwischen Disk-Frame-Nummern und XML-Frame-Nummern
    xml_set = set(xml_ids)

    # Kandidaten-Offsets nur als Differenz xml_frame - disk_frame erzeugen
    candidate_offsets = set()
    for d in disk_ids[:20]:
        for x in xml_ids[:500]:
            candidate_offsets.add(x - d)

    best_offset = None
    best_score = -1

    # Konsistenz des Musters nur über die ersten 20 Disk-Frames prüfen
    sample = disk_ids[:20]

    for offset in sorted(candidate_offsets):
        score = 0
        streak = 0

        for d in sample:
            if (d + offset) in xml_set:
                score += 1
                streak += 1
            else:
                streak = 0

        # Optional könnte man hier auch die Streak-Länge belohnen
        score2 = score

        if score2 > best_score:
            best_score = score2
            best_offset = offset

    return best_offset, best_score

frame_dict, unknown_labels = build_frame_dict(XML_PATH, CLASS_MAP)

print(f"✅ XML annotated unique frame sayısı: {len(frame_dict)}")
print(f"⚠️ Unknown labels: {unknown_labels}")

frame_files = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.png')])
disk_ids = [parse_frame_id(f) for f in frame_files]
xml_ids = sorted(frame_dict.keys())

print(f"✅ Diskteki toplam frame: {len(frame_files)}")

best_offset, best_score = find_best_offset(disk_ids, xml_ids, step=24)

print(f"✅ Seçilen offset: {best_offset}")
print(f"✅ İlk 20 frame üstünde eşleşme: {best_score}/20")

# Debug-Ausgabe: erste 10 Zuordnungen prüfen
print("\nİlk 10 eşleşme kontrolü:")
for d in disk_ids[:10]:
    print(f"local={d} -> xml={d + best_offset} -> {'OK' if (d + best_offset) in frame_dict else 'MISS'}")

written = 0
for filename in frame_files:
    local_frame_id = parse_frame_id(filename)
    xml_frame_id = local_frame_id + best_offset

    # Frames ohne Annotation überspringen
    if xml_frame_id not in frame_dict:
        continue

    img_path = os.path.join(FRAMES_DIR, filename)
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w, _ = img.shape

    cv2.imwrite(os.path.join(OUT_IMG_DIR, filename), img)

    # Bounding-Boxen ins normalisierte YOLO-Format umrechnen:
    # class x_center y_center width height (alle Werte relativ zur Bildgröße)
    label_path = os.path.join(OUT_LABEL_DIR, filename.replace(".png", ".txt"))
    with open(label_path, "w", encoding="utf-8") as f:
        for (class_id, xtl, ytl, xbr, ybr) in frame_dict[xml_frame_id]:
            x_center = ((xtl + xbr) / 2.0) / w
            y_center = ((ytl + ybr) / 2.0) / h
            width = (xbr - xtl) / w
            height = (ybr - ytl) / h
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    written += 1

print(f"\n✅ Dataset oluşturuldu! ({written} image)")
