"""
hand_tracker.py — Abstraction MediaPipe compatible toutes versions

Stratégie :
  1. Essaie l'ancienne API  mp.solutions.hands  (mediapipe < 0.10.14)
  2. Sinon utilise la nouvelle API  mp.tasks  (mediapipe >= 0.10.14)
     → télécharge automatiquement hand_landmarker.task si absent

Retourne toujours des landmarks normalisés (x, y, z) dans [0,1].
"""

import os
import sys
import cv2
import numpy as np

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")


# ════════════════════════════════════════════════════════════
#  Wrapper landmark (uniforme entre les deux APIs)
# ════════════════════════════════════════════════════════════
class Landmark:
    __slots__ = ("x", "y", "z")
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


# ════════════════════════════════════════════════════════════
#  Tracker — ancienne API (solutions)
# ════════════════════════════════════════════════════════════
class _TrackerLegacy:
    def __init__(self):
        import mediapipe as mp
        self._mp_hands   = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles  = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,        # 0 = léger et rapide, 1 = lourd (inutile ici)
            min_detection_confidence=0.65,
            min_tracking_confidence=0.55,
        )

    def process(self, frame_bgr: np.ndarray):
        """
        frame_bgr : image BGR de la webcam (déjà miroir)
        Retourne (landmarks_list | None, annotated_frame)
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)
        annotated = frame_bgr.copy()

        if res.multi_hand_landmarks:
            lm_raw = res.multi_hand_landmarks[0]
            self._mp_drawing.draw_landmarks(
                annotated, lm_raw,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )
            lm = [Landmark(p.x, p.y, p.z) for p in lm_raw.landmark]
            return lm, annotated

        return None, annotated

    def close(self):
        self._hands.close()


# ════════════════════════════════════════════════════════════
#  Tracker — nouvelle API Tasks (0.10.14+)
# ════════════════════════════════════════════════════════════
class _TrackerTasks:

    # Connexions pour dessiner le squelette manuellement
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17),
    ]

    def __init__(self):
        self._ensure_model()
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker, HandLandmarkerOptions, RunningMode
        )
        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.60,
            min_tracking_confidence=0.55,
        )
        self._detector = HandLandmarker.create_from_options(opts)

    def _ensure_model(self):
        if os.path.exists(MODEL_PATH):
            return
        print(f"[HandTracker] Modèle introuvable : {MODEL_PATH}")
        print("[HandTracker] Téléchargement en cours...")
        try:
            import urllib.request
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[HandTracker] ✓ Modèle téléchargé.")
        except Exception as e:
            print(f"[HandTracker] ✗ Échec du téléchargement : {e}")
            print(f"\nTéléchargez manuellement :\n  {MODEL_URL}")
            print(f"et placez-le ici : {os.path.abspath(MODEL_PATH)}\n")
            sys.exit(1)

    def process(self, frame_bgr: np.ndarray):
        import mediapipe as mp
        rgb_arr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_arr,
        )
        result    = self._detector.detect(mp_img)
        annotated = frame_bgr.copy()

        if result.hand_landmarks:
            raw_lm = result.hand_landmarks[0]
            h, w   = frame_bgr.shape[:2]

            # Dessine le squelette manuellement
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in raw_lm]
            for a, b in self.CONNECTIONS:
                cv2.line(annotated, pts[a], pts[b], (80, 200, 120), 1, cv2.LINE_AA)
            for px, py in pts:
                cv2.circle(annotated, (px, py), 4, (0, 229, 255), -1, cv2.LINE_AA)

            lm = [Landmark(p.x, p.y, p.z) for p in raw_lm]
            return lm, annotated

        return None, annotated

    def close(self):
        self._detector.close()


# ════════════════════════════════════════════════════════════
#  FACTORY — choisit automatiquement la bonne implémentation
# ════════════════════════════════════════════════════════════
def create_tracker():
    """Retourne le bon tracker selon la version de mediapipe."""
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        print("[HandTracker] Utilisation de l'API legacy (solutions)")
        return _TrackerLegacy()
    else:
        print("[HandTracker] Utilisation de la nouvelle API (Tasks)")
        return _TrackerTasks()
