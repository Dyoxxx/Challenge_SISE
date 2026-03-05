"""
gesture.py — Détection des gestes
Compatible mediapipe 0.10+ (nouvelle API Tasks) ET ancienne API solutions.
"""


class GestureDetector:
    """
    Prend en entrée une liste de 21 landmarks.
    Chaque landmark a des attributs .x, .y, .z
    """
    TIPS = [8, 12, 16, 20]   # index, majeur, annulaire, auriculaire
    PIPS = [6, 10, 14, 18]

    def detect(self, landmarks) -> str:
        fingers_up = tuple(
            landmarks[t].y < landmarks[p].y
            for t, p in zip(self.TIPS, self.PIPS)
        )
        # Pouce : compare x (main miroir)
        thumb_up = landmarks[4].x > landmarks[3].x

        idx, mid, rng, pnk = fingers_up

        if thumb_up and idx and mid and rng and pnk:
            return "open_hand"
        if thumb_up and not idx and not mid and not rng and pnk:
            return "undo"
        if not idx and not mid and not rng and not pnk:
            return "erase"
        if idx and mid and not rng and not pnk:
            return "pause"
        if idx and not mid:
            return "draw"
        return "pause"
