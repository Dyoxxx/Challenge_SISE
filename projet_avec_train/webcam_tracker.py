"""
Webcam Tracker v5 — MLP flexible visage / 1 main / 2 mains
============================================================
Input MLP : 1065 features avec zero-padding sur modalities absentes
  [0   :936 ] visage  (468 pts × x,y)
  [936 :999 ] main 1  (21  pts × x,y,z)  ← zéros si absente
  [999 :1062] main 2  (21  pts × x,y,z)  ← zéros si absente
  [1062:1065] flags   [face_ok, hand1_ok, hand2_ok]

Workflow :
  1. uv run python collect_data.py
  2. uv run python train_mlp.py
  3. uv run python webcam_tracker.py
"""

import cv2, json, os, random, time, urllib.request
import numpy as np
import torch, torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path

ZEROS_HAND = np.zeros(63, dtype=np.float32)

# ─── MLP ──────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1085, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,  256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,  128), nn.ReLU(),
            nn.Linear(128,  num_classes),
        )
    def forward(self, x): return self.net(x)

def load_mlp():
    mp_, lp_ = "models/meme_mlp.pt", "models/meme_labels.json"
    if not (os.path.exists(mp_) and os.path.exists(lp_)):
        return None, None
    labels = json.load(open(lp_))
    m = MLP(len(labels))
    m.load_state_dict(torch.load(mp_, map_location="cpu"))
    m.eval()
    print(f"[OK] MLP chargé — classes: {labels}")
    return m, labels

mlp, meme_labels = load_mlp()
if mlp is None:
    print("[INFO] Pas de modèle — lance collect_data.py puis train_mlp.py")

# ─── MediaPipe ────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
HAND_PATH  = os.path.join(MODELS_DIR, "hand_landmarker.task")
FACE_PATH  = os.path.join(MODELS_DIR, "face_landmarker.task")
os.makedirs(MODELS_DIR, exist_ok=True)

def dl(url, path):
    if not os.path.exists(path):
        print(f"[INFO] Téléchargement {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)

dl("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", HAND_PATH)
dl("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", FACE_PATH)

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_PATH = "dataset"
PANEL_W      = 300

def load_dataset():
    ds = {}
    if not os.path.exists(DATASET_PATH): return ds
    for d in Path(DATASET_PATH).iterdir():
        if not d.is_dir(): continue
        files = list(d.glob("*.jpg"))+list(d.glob("*.jpeg"))+list(d.glob("*.png"))
        if files: ds[d.name] = [str(f) for f in files]
    return ds

def get_image(ds, label, size):
    if not label or label not in ds: return None
    img = cv2.imread(random.choice(ds[label]))
    return cv2.resize(img, (size, size)) if img is not None else None

def create_placeholders(labels):
    colors = [(0,200,200),(0,180,255),(200,100,0),(230,220,0),(0,80,220),(150,150,150)]
    for i, lbl in enumerate(labels):
        d = os.path.join(DATASET_PATH, lbl)
        os.makedirs(d, exist_ok=True)
        for j in range(3):
            p = os.path.join(d, f"placeholder_{j}.jpg")
            if os.path.exists(p): continue
            img = np.zeros((300,300,3), dtype=np.uint8)
            c   = colors[i % len(colors)]
            for y in range(300):
                img[y] = (np.array(c)*(0.4+0.6*y/300)).astype(np.uint8)
            cv2.putText(img, lbl[:16], (10,145), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2)
            cv2.imwrite(p, img)

# ─── Dessin main ──────────────────────────────────────────────────────────────
HAND_CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),
             (11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]

HAND_COLORS = [
    {"line": (0,210,90),  "dot": (180,255,180)},   # main 1 — vert
    {"line": (0,140,255), "dot": (140,200,255)},   # main 2 — bleu
]

def draw_hand(frame, lm, w, h, hand_idx=0):
    col = HAND_COLORS[hand_idx % 2]
    pts = [(int(l.x*w), int(l.y*h)) for l in lm]
    for a, b in HAND_CONN:
        cv2.line(frame, pts[a], pts[b], col["line"], 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255,255,255), -1)
        cv2.circle(frame, pt, 4, col["dot"], 1)

# ─── Dessin visage ────────────────────────────────────────────────────────────
FACE_OVAL  = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
              152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]
LEFT_EYE   = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,33]
RIGHT_EYE  = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382,362]
LIPS_OUTER = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,61]
LIPS_INNER = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95,78]

def draw_face(frame, lm, w, h):
    pts = [(int(l.x*w), int(l.y*h)) for l in lm]
    n   = len(pts)
    def poly(indices, color, t=1):
        for i in range(len(indices)-1):
            a, b = indices[i], indices[i+1]
            if a < n and b < n:
                cv2.line(frame, pts[a], pts[b], color, t)
    poly(FACE_OVAL,  (70,140,255), 1)
    poly(LEFT_EYE,   (0, 220,220), 1)
    poly(RIGHT_EYE,  (0, 220,220), 1)
    poly(LIPS_OUTER, (80,100,255), 2)
    poly(LIPS_INNER, (60, 80,200), 1)

# ─── Inférence ────────────────────────────────────────────────────────────────
def predict(face_lm, hand_lms):
    """
    face_lm  : liste de landmarks (toujours présent)
    hand_lms : liste de 0, 1 ou 2 listes de landmarks
    """
    if mlp is None: return None, None

    face_f = np.array([v for p in face_lm[:478] for v in (p.x, p.y)], dtype=np.float32)

    h1_f = (np.array([v for p in hand_lms[0] for v in (p.x,p.y,p.z)], dtype=np.float32)
            if len(hand_lms) >= 1 else ZEROS_HAND)
    h2_f = (np.array([v for p in hand_lms[1] for v in (p.x,p.y,p.z)], dtype=np.float32)
            if len(hand_lms) >= 2 else ZEROS_HAND)

    flags = np.array([1.0,
                      1.0 if len(hand_lms) >= 1 else 0.0,
                      1.0 if len(hand_lms) >= 2 else 0.0], dtype=np.float32)

    feat = torch.tensor(np.concatenate([face_f, h1_f, h2_f, flags])).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(mlp(feat)[0], dim=0)
        idx   = probs.argmax().item()
    return meme_labels[idx], probs[idx].item()

# ─── Panneau droit ────────────────────────────────────────────────────────────
def make_panel(h, label, img):
    panel    = np.full((h, PANEL_W, 3), 25, dtype=np.uint8)
    img_size = min(PANEL_W - 10, h - 50)
    mx       = (PANEL_W - img_size) // 2
    my       = (h - img_size) // 2

    if img is not None:
        panel[my:my+img_size, mx:mx+img_size] = cv2.resize(img, (img_size, img_size))
    else:
        cv2.putText(panel, label or "---", (10, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70,70,70), 1)

    if label:
        cv2.putText(panel, label, (mx, my-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220,220,80), 1)
    return panel

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    labels = meme_labels or ["heart_hands","thumbs_up","surprised","peace_smile","fist_angry","neutral"]
    if not os.path.exists(DATASET_PATH):
        create_placeholders(labels)

    dataset = load_dataset()
    print(f"[INFO] Dataset: {list(dataset.keys())}")

    hand_det = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2, min_hand_detection_confidence=0.6))
    face_det = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_PATH),
            running_mode=mp_vision.RunningMode.IMAGE, num_faces=1))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la webcam"); return

    label       = None
    conf        = None
    cached_img  = None
    last_label  = None
    last_update = 0
    UPDATE_INT  = 0.5

    fps    = 0.0
    prev_t = time.time()
    print("[INFO] Q pour quitter")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hand_res = hand_det.detect(mp_img)
        face_res = face_det.detect(mp_img)

        face_ok  = bool(face_res.face_landmarks)
        n_hands  = len(hand_res.hand_landmarks) if hand_res.hand_landmarks else 0

        # Dessin
        if face_ok:
            draw_face(frame, face_res.face_landmarks[0], w, h)
        for hi in range(n_hands):
            draw_hand(frame, hand_res.hand_landmarks[hi], w, h, hi)

        # Inférence — visage obligatoire, mains optionnelles
        if face_ok:
            hands = hand_res.hand_landmarks[:n_hands]
            label, conf = predict(face_res.face_landmarks[0], hands)
        else:
            label, conf = None, None

        # Refresh image dataset
        now = time.time()
        if now - last_update > UPDATE_INT:
            last_update = now
            if label != last_label:
                last_label = label
                img_size   = min(PANEL_W-10, h-50)
                cached_img = get_image(dataset, label, img_size)
            if not label:
                last_label = None; cached_img = None

        # FPS
        cur_t = time.time()
        fps   = fps*0.9 + 0.1/max(cur_t-prev_t, 1e-5)
        prev_t = cur_t

        # Overlay texte
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80,255,80), 2)

        # Indicateurs de détection (petits points colorés en haut à droite)
        dot_y = 18
        for i, (ok, txt) in enumerate([(face_ok,"F"),(n_hands>=1,"H1"),(n_hands>=2,"H2")]):
            col = (0,220,80) if ok else (60,60,180)
            cx  = w - 80 + i*28
            cv2.circle(frame, (cx, dot_y), 9, col, -1)
            cv2.putText(frame, txt, (cx-8, dot_y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

        if label:
            conf_str = f"  {conf*100:.0f}%" if conf else ""
            cv2.putText(frame, f"{label}{conf_str}", (10,58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (220,220,60), 2)
        elif not face_ok:
            cv2.putText(frame, "visage non detecte", (10,58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70,70,180), 2)

        panel   = make_panel(h, last_label, cached_img)
        display = np.hstack([frame, panel])
        cv2.imshow("Webcam Tracker  —  Q pour quitter", display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    hand_det.close(); face_det.close()
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
