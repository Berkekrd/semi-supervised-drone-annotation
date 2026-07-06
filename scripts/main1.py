# ============================================================
# MAIN1 - VALIDIERUNG DER KLASSEN-ZUORDNUNG (CLASS_MAP vs. CVAT-XML)
# ============================================================
# Zweck:
# Prüft, ob alle Labels aus der CVAT-XML-Annotationsdatei in der
# CLASS_MAP enthalten sind, bevor die Konvertierung ins YOLO-Format
# gestartet wird. So werden fehlende oder ungenutzte Klassen früh erkannt.
# ============================================================

import xml.etree.ElementTree as ET

# Pfad zur CVAT-XML-Annotationsdatei
XML_PATH = "flight-mbg-v2-01-regionA/ann/video01_regionA.xml"

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

tree = ET.parse(XML_PATH)
root = tree.getroot()

xml_labels = set()

# Alle Track-Labels aus der XML einsammeln und normalisieren
# (Kleinschreibung, Leerzeichen -> Unterstrich)
for track in root.findall("track"):
    label = track.get("label")
    if label:
        label_clean = label.strip().lower().replace(" ", "_")
        xml_labels.add(label_clean)

map_labels = set(CLASS_MAP.keys())

missing_in_map = sorted(xml_labels - map_labels)   # In XML vorhanden, aber nicht in CLASS_MAP
unused_in_map = sorted(map_labels - xml_labels)    # In CLASS_MAP vorhanden, aber nicht in XML

print("XML içindeki label'lar:")
for lbl in sorted(xml_labels):
    print("-", lbl)

print("\nCLASS_MAP içindeki label'lar:")
for lbl in sorted(map_labels):
    print("-", lbl)

print("\nXML'de var ama CLASS_MAP'te yok:")
print(missing_in_map if missing_in_map else "YOK")

print("\nCLASS_MAP'te var ama XML'de yok:")
print(unused_in_map if unused_in_map else "YOK")

# Ergebnis: CLASS_MAP ist nur dann ausreichend, wenn kein Label fehlt
if not missing_in_map:
    print("\n✅ CLASS_MAP bu XML için yeterli.")
else:
    print("\n⚠️ Yeni label bulundu, işlemden önce CLASS_MAP güncellenmeli.")
