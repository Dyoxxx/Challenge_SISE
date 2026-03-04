"""
Webcam Tracker - Détection d'expressions faciales + signes de mains
Utilise la nouvelle API MediaPipe Tasks (compatible 0.10+)

Dataset attendu :
    dataset/
        happy/ sad/ surprised/ neutral/ angry/ fear/ disgust/
        thumbs_up/ thumbs_down/ peace/ fist/ open_hand/ ok/ pointing/

Si absent → placeholders colorés créés automatiquement.
"""

import cv2
import numpy as np
import os
import random
import time
import urllib.request
from pathlib import Path

# ─── MediaPipe nouvelle API ────────────────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─── DeepFace pour les émotions ───────────────────────────────────────────────
try:
    from deepface import DeepFace

    USE_DEEPFACE = True
    print("[OK] DeepFace chargé")
except ImportError:
    USE_DEEPFACE = False
    print("[WARN] DeepFace non disponible (pip install deepface)")

# ─── Téléchargement des modèles MediaPipe (.task) ─────────────────────────────
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

HAND_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def download_model(url, path):
    if not os.path.exists(path):
        print(f"[INFO] Téléchargement {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)
        print(f"[OK]  {os.path.basename(path)} prêt")


download_model(HAND_MODEL_URL, HAND_MODEL_PATH)
download_model(FACE_MODEL_URL, FACE_MODEL_PATH)

# ─── Labels & couleurs ────────────────────────────────────────────────────────
EXPRESSION_LABELS = ["happy", "sad", "surprised", "neutral", "angry", "fear", "disgust"]
HAND_LABELS = [
    "thumbs_up",
    "thumbs_down",
    "peace",
    "fist",
    "open_hand",
    "ok",
    "pointing",
]

PLACEHOLDER_COLORS = {
    "happy": (0, 220, 220),
    "sad": (200, 100, 0),
    "surprised": (0, 180, 255),
    "neutral": (150, 150, 150),
    "angry": (0, 50, 220),
    "fear": (200, 0, 200),
    "disgust": (40, 180, 40),
    "thumbs_up": (0, 220, 0),
    "thumbs_down": (0, 80, 200),
    "peace": (230, 220, 0),
    "fist": (180, 60, 0),
    "open_hand": (0, 200, 200),
    "ok": (200, 200, 0),
    "pointing": (180, 0, 180),
}

DATASET_PATH = "dataset"
PANEL_W = 280
FPS_ALPHA = 0.1


# ─── Placeholder dataset ──────────────────────────────────────────────────────
def create_placeholder_dataset():
    for label in EXPRESSION_LABELS + HAND_LABELS:
        d = os.path.join(DATASET_PATH, label)
        os.makedirs(d, exist_ok=True)
        for i in range(3):
            p = os.path.join(d, f"placeholder_{i}.jpg")
            if not os.path.exists(p):
                color = PLACEHOLDER_COLORS.get(label, (100, 100, 100))
                img = np.zeros((300, 300, 3), dtype=np.uint8)
                for y in range(300):
                    t = y / 300
                    img[y] = (np.array(color) * (0.5 + 0.5 * t)).astype(np.uint8)
                cv2.putText(
                    img,
                    label,
                    (15, 155),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    img,
                    f"#{i+1}",
                    (230, 280),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    1,
                )
                cv2.imwrite(p, img)
    print(f"[OK] Dataset placeholder dans ./{DATASET_PATH}/")


def load_dataset(path):
    ds = {}
    if not os.path.exists(path):
        return ds
    for d in Path(path).iterdir():
        if d.is_dir():
            files = (
                list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))
            )
            if files:
                ds[d.name] = [str(f) for f in files]
    return ds


def get_random_image(dataset, label, size):
    if label not in dataset:
        return None
    img = cv2.imread(random.choice(dataset[label]))
    return cv2.resize(img, (size, size)) if img is not None else None


# ─── Dessin landmarks mains ───────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


def draw_hand_landmarks(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 220, 100), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)
        cv2.circle(frame, pt, 4, (0, 180, 80), 1)


# ─── Contours du visage (indices MediaPipe FaceMesh) ─────────────────────────
# Ovale visage, yeux, sourcils, nez, bouche — ~100 points au lieu de 468
FACE_CONTOURS = {
    "oval": [
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
    ],
    "left_eye": [
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        173,
        157,
        158,
        159,
        160,
        161,
        246,
        33,
    ],
    "right_eye": [
        362,
        382,
        381,
        380,
        374,
        373,
        390,
        249,
        263,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
        362,
    ],
    "left_brow": [46, 53, 52, 65, 55, 70, 63, 105, 66, 107, 46],
    "right_brow": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336, 276],
    "nose": [
        168,
        6,
        197,
        195,
        5,
        4,
        45,
        220,
        115,
        48,
        64,
        98,
        97,
        2,
        326,
        327,
        294,
        278,
        344,
        440,
        275,
        4,
        168,
    ],
    "lips_out": [
        61,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        291,
        409,
        270,
        269,
        267,
        0,
        37,
        39,
        40,
        185,
        61,
    ],
    "lips_in": [
        78,
        95,
        88,
        178,
        87,
        14,
        317,
        402,
        318,
        324,
        308,
        415,
        310,
        311,
        312,
        13,
        82,
        81,
        80,
        191,
        78,
    ],
}

CONTOUR_COLORS = {
    "oval": (80, 160, 255),
    "left_eye": (0, 230, 230),
    "right_eye": (0, 230, 230),
    "left_brow": (100, 200, 255),
    "right_brow": (100, 200, 255),
    "nose": (80, 160, 255),
    "lips_out": (80, 120, 255),
    "lips_in": (60, 100, 220),
}


def draw_face_landmarks(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for name, indices in FACE_CONTOURS.items():
        color = CONTOUR_COLORS.get(name, (100, 180, 255))
        for i in range(len(indices) - 1):
            a, b = indices[i], indices[i + 1]
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], color, 1)


# ─── Classification signe de main ─────────────────────────────────────────────
def classify_hand_sign(lm):
    thumb = lm[4].x < lm[3].x
    index = lm[8].y < lm[6].y
    middle = lm[12].y < lm[10].y
    ring = lm[16].y < lm[14].y
    pinky = lm[20].y < lm[18].y

    if thumb and not index and not middle and not ring and not pinky:
        return "thumbs_up"
    if not thumb and not index and not middle and not ring and not pinky:
        return "fist"
    if thumb and index and middle and ring and pinky:
        return "open_hand"
    if not thumb and index and middle and not ring and not pinky:
        return "peace"
    if not thumb and index and not middle and not ring and not pinky:
        return "pointing"

    dx = lm[4].x - lm[8].x
    dy = lm[4].y - lm[8].y
    if (dx**2 + dy**2) ** 0.5 < 0.06 and middle and ring and pinky:
        return "ok"

    return None


# ─── Panneau latéral ──────────────────────────────────────────────────────────
def make_side_panel(h, emotion_label, emotion_img, hand_label, hand_img):
    panel = np.full((h, PANEL_W, 3), 28, dtype=np.uint8)
    img_h = min(PANEL_W - 10, (h - 80) // 2)
    margin = (PANEL_W - img_h) // 2

    def draw_slot(y0, prefix, label, img):
        color = (220, 220, 80) if label else (160, 160, 160)
        cv2.putText(
            panel,
            f"{prefix}: {label or '---'}",
            (8, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
        )
        slot = (
            img if img is not None else np.full((img_h, img_h, 3), 55, dtype=np.uint8)
        )
        if img is None:
            cv2.putText(
                slot,
                "no image",
                (10, img_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (100, 100, 100),
                1,
            )
        slot = cv2.resize(slot, (img_h, img_h))
        panel[y0 + 28 : y0 + 28 + img_h, margin : margin + img_h] = slot

    draw_slot(4, "Expression", emotion_label, emotion_img)
    draw_slot(h // 2 + 4, "Signe main", hand_label, hand_img)
    return panel


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(DATASET_PATH):
        print("[INFO] Pas de dataset → création de placeholders")
        create_placeholder_dataset()

    dataset = load_dataset(DATASET_PATH)
    print(f"[INFO] Dataset: {list(dataset.keys())}")

    # Détecteurs MediaPipe Tasks
    hand_detector = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
    )
    face_detector = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
        )
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la webcam")
        return

    cached_emotion_img = None
    cached_hand_img = None
    last_emotion_label = None
    last_hand_label = None
    last_update = 0
    UPDATE_INTERVAL = 0.7

    DEEPFACE_EVERY = 15  # appeler DeepFace toutes les N frames
    frame_count = 0
    emotion_label = None

    fps = 0.0
    prev_t = time.time()

    print("[INFO] Appuyez sur Q pour quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_count += 1

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        # ── Mains ────────────────────────────────────────────────────────────
        hand_res = hand_detector.detect(mp_img)
        hand_label = None
        if hand_res.hand_landmarks:
            lm = hand_res.hand_landmarks[0]
            draw_hand_landmarks(frame, lm, w, h)
            hand_label = classify_hand_sign(lm)

        # ── Visage ────────────────────────────────────────────────────────────
        face_res = face_detector.detect(mp_img)
        if face_res.face_landmarks:
            draw_face_landmarks(frame, face_res.face_landmarks[0], w, h)

        # ── Émotion DeepFace (toutes les N frames) ────────────────────────────
        if USE_DEEPFACE and frame_count % DEEPFACE_EVERY == 0:
            try:
                res = DeepFace.analyze(
                    frame, actions=["emotion"], enforce_detection=False, silent=True
                )
                if res:
                    emotion_label = res[0]["dominant_emotion"]
            except Exception:
                pass

        # ── Refresh images ────────────────────────────────────────────────────
        now = time.time()
        img_size = max(50, min(PANEL_W - 10, (h - 80) // 2))
        if now - last_update > UPDATE_INTERVAL:
            last_update = now
            if emotion_label != last_emotion_label:
                last_emotion_label = emotion_label
                cached_emotion_img = get_random_image(dataset, emotion_label, img_size)
            if hand_label != last_hand_label:
                last_hand_label = hand_label
                cached_hand_img = get_random_image(dataset, hand_label, img_size)
            if not hand_label:
                last_hand_label = None
                cached_hand_img = None

        """# ── FPS ───────────────────────────────────────────────────────────────
        cur_t  = time.time()
        fps    = fps * (1 - FPS_ALPHA) + FPS_ALPHA / max(cur_t - prev_t, 1e-5)
        prev_t = cur_t"""

        # ── Texte overlay ─────────────────────────────────────────────────────
        """cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 255, 80),
            2,
        )"""
        if emotion_label:
            cv2.putText(
                frame,
                f"Expression: {emotion_label}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 230, 200),
                2,
            )
        if hand_label:
            cv2.putText(
                frame,
                f"Signe: {hand_label}",
                (10, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (230, 200, 0),
                2,
            )

        # ── Affichage ─────────────────────────────────────────────────────────
        side = make_side_panel(
            h, last_emotion_label, cached_emotion_img, last_hand_label, cached_hand_img
        )
        display = np.hstack([frame, side])
        cv2.imshow("Webcam Tracker  —  Q pour quitter", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    hand_detector.close()
    face_detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
