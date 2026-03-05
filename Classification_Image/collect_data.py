"""
Étape 1 — Collecte landmarks flexibles (visage / main(s))
==========================================================
Features par sample :
  face   : 468 pts × 2 (x,y)         = 936
  hand1  : 21 pts  × 3 (x,y,z)       =  63
  hand2  : 21 pts  × 3 (x,y,z)       =  63
  flags  : [face_ok, hand1_ok, hand2_ok] =   3
  TOTAL  = 1065  (zéros si modality absente)

Labels par défaut (à adapter à tes memes) :
  1 = blowing -> visage
  2 = dontknow -> visage + 2 mains
  3 = dumb -> visage
  4 = happy -> visage
  5 = headscraching -> visage + 1 main
  6 = heart_hands -> visage + 2 mains
  7 = sad -> visage
  8 = silly -> visage + 2 mains
  9 = smiling -> visage
  10 = stare -> visage
  11 = thinking -> visage
  12 = zamn -> visage
  13 = hand -> visage + 1 main
  14 = malicious -> visage + 2 mains

ESPACE = sauvegarder le sample courant
ESC    = quitter
"""

import cv2, csv, os, time, urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODELS_DIR = "models"
HAND_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
FACE_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
os.makedirs(MODELS_DIR, exist_ok=True)


def dl(url, path):
    if not os.path.exists(path):
        print(f"[INFO] Téléchargement {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)


dl(
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    HAND_PATH,
)
dl(
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    FACE_PATH,
)

KEY_MAP = {
    ord("a"): "blowing",
    ord("z"): "dontknow",
    ord("e"): "dumb",
    ord("r"): "happy",
    ord("t"): "headscraching",
    ord("y"): "heart_hands",
    ord("u"): "sad",
    ord("i"): "silly",
    ord("o"): "smiling",
    ord("p"): "stare",
    ord("q"): "thinking",
    ord("s"): "zamn",
    ord("d"): "hand",
    ord("f"): "malicious",
}

os.makedirs("data", exist_ok=True)
CSV_PATH = "data/combined_data.csv"

FACE_COLS = [f"f_{a}{i}" for i in range(468) for a in ("x", "y")]
HAND1_COLS = [f"h1_{a}{i}" for i in range(21) for a in ("x", "y", "z")]
HAND2_COLS = [f"h2_{a}{i}" for i in range(21) for a in ("x", "y", "z")]
FLAG_COLS = ["flag_face", "flag_hand1", "flag_hand2"]
ALL_COLS = ["label"] + FACE_COLS + HAND1_COLS + HAND2_COLS + FLAG_COLS  # 1 + 1065

if os.path.exists(CSV_PATH):
    with open(CSV_PATH) as f:
        existing_header = f.readline().strip().split(",")
    if len(existing_header) != len(ALL_COLS):
        print(
            f"[WARN] CSV existant a {len(existing_header)} colonnes, attendu {len(ALL_COLS)}."
        )
        print(
            f"       → Ancien fichier renommé en combined_data_old.csv, nouveau créé."
        )
        os.rename(CSV_PATH, CSV_PATH.replace(".csv", "_old.csv"))

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(ALL_COLS)


def append_row(row):
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow(row)


ZEROS_FACE = [0.0] * 936
ZEROS_HAND = [0.0] * 63


FACE_OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
    10,
]
LEFT_EYE = [
    33,
    246,
    161,
    160,
    159,
    158,
    157,
    173,
    133,
    155,
    154,
    153,
    145,
    144,
    163,
    7,
    33,
]
RIGHT_EYE = [
    362,
    398,
    384,
    385,
    386,
    387,
    388,
    466,
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
    362,
]
LIPS_OUTER = [
    61,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    291,
    375,
    321,
    405,
    314,
    17,
    84,
    181,
    91,
    146,
    61,
]
LIPS_INNER = [
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
    78,
]


def draw_face(frame, lm, w, h):
    pts = [(int(l.x * w), int(l.y * h)) for l in lm]
    n = len(pts)

    def poly(indices, color, t=1):
        for i in range(len(indices) - 1):
            a, b = indices[i], indices[i + 1]
            if a < n and b < n:
                cv2.line(frame, pts[a], pts[b], color, t)

    poly(FACE_OVAL, (70, 140, 255), 1)
    poly(LEFT_EYE, (0, 220, 220), 1)
    poly(RIGHT_EYE, (0, 220, 220), 1)
    poly(LIPS_OUTER, (80, 100, 255), 2)
    poly(LIPS_INNER, (60, 80, 200), 1)


hand_det = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
)
face_det = mp_vision.FaceLandmarker.create_from_options(
    mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
    )
)

cap = cv2.VideoCapture(0)
current_label = None
count = 0
flash_ok = False
flash_err = False
flash_until = 0
label_counts = {v: 0 for v in KEY_MAP.values()}

print("ESPACE = sauvegarder  |  ESC = quitter")
print("Au moins le visage doit être détecté pour enregistrer.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    hand_res = hand_det.detect(mp_img)
    face_res = face_det.detect(mp_img)

    n_hands = len(hand_res.hand_landmarks) if hand_res.hand_landmarks else 0
    face_ok = bool(face_res.face_landmarks)

    # Dessin
    if face_ok:
        draw_face(frame, face_res.face_landmarks[0], w, h)
    for hi in range(n_hands):
        color = (0, 220, 100) if hi == 0 else (0, 150, 255)
        for lm in hand_res.hand_landmarks[hi]:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, color, -1)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key in KEY_MAP:
        current_label = KEY_MAP[key]

    if key == ord(" ") and current_label:
        if not face_ok:
            flash_ok = False
            flash_err = True
            flash_until = time.time() + 0.2
        else:
            # Visage
            face_feats = [
                v for p in face_res.face_landmarks[0][:478] for v in (p.x, p.y)
            ]
            # Main 1
            hand1_feats = (
                [v for p in hand_res.hand_landmarks[0] for v in (p.x, p.y, p.z)]
                if n_hands >= 1
                else ZEROS_HAND
            )
            # Main 2
            hand2_feats = (
                [v for p in hand_res.hand_landmarks[1] for v in (p.x, p.y, p.z)]
                if n_hands >= 2
                else ZEROS_HAND
            )
            flags = [1.0, 1.0 if n_hands >= 1 else 0.0, 1.0 if n_hands >= 2 else 0.0]

            append_row([current_label] + face_feats + hand1_feats + hand2_feats + flags)
            count += 1
            label_counts[current_label] += 1
            flash_ok = True
            flash_err = False
            flash_until = time.time() + 0.12

    # Flash bordure
    now = time.time()
    if now < flash_until:
        color = (0, 220, 80) if flash_ok else (0, 60, 220)
        cv2.rectangle(frame, (0, 0), (w, h), color, 6)

    # Statut
    def c(ok):
        return (0, 210, 80) if ok else (60, 60, 200)

    cv2.putText(
        frame,
        f"Visage: {'OK' if face_ok else 'non detecte'}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        c(face_ok),
        2,
    )
    cv2.putText(
        frame,
        f"Mains:  {n_hands} detectee(s)",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        c(n_hands > 0),
        2,
    )
    cv2.putText(
        frame,
        f"Label:  {current_label or 'aucun (touches 1-6)'}",
        (10, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 60),
        2,
    )
    cv2.putText(
        frame,
        f"Total:  {count} samples",
        (10, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (160, 160, 160),
        1,
    )

    for i, (lbl, cnt) in enumerate(label_counts.items()):
        bw = min(cnt * 5, 220)
        y = 148 + i * 22
        cv2.rectangle(frame, (10, y), (10 + bw, y + 16), (50, 110, 70), -1)
        cv2.putText(
            frame,
            f"{lbl}: {cnt}",
            (12, y + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (210, 210, 210),
            1,
        )

    cv2.putText(
        frame,
        "ESPACE=sauver  ESC=quitter  (visage obligatoire)",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (90, 90, 90),
        1,
    )

    cv2.imshow("Collecte — labels a-f puis ESPACE", frame)

cap.release()
cv2.destroyAllWindows()
hand_det.close()
face_det.close()
print(f"\nTotal: {count} samples → {CSV_PATH}")
print("Compteurs:", label_counts)
