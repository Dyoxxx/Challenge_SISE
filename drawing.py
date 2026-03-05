"""
drawing.py — Moteur de dessin OpenCV
"""

import cv2
import numpy as np
import os
from datetime import datetime


class DrawingEngine:
    MAX_UNDO = 30
    SAVE_DIR = "saves"

    def __init__(self, cfg):
        self.cfg = cfg
        self._canvas = np.zeros((cfg.CANVAS_H, cfg.CANVAS_W, 3), dtype=np.uint8)
        self._undo_stack = []
        self.stroke_count = 0
        os.makedirs(self.SAVE_DIR, exist_ok=True)

    def get_canvas(self) -> np.ndarray:
        """Vue lecture-seule du canvas (pas de copie inutile)."""
        return self._canvas

    # ── Dessin ───────────────────────────────────────────────
    def draw_line(self, x1, y1, x2, y2):
        col = self.cfg.color_bgr
        size = self.cfg.brush_size
        alpha = self.cfg.opacity
        if alpha >= 0.99:
            cv2.line(self._canvas, (x1, y1), (x2, y2), col, size, lineType=cv2.LINE_AA)
        else:
            ov = self._canvas.copy()
            cv2.line(ov, (x1, y1), (x2, y2), col, size, lineType=cv2.LINE_AA)
            cv2.addWeighted(ov, alpha, self._canvas, 1 - alpha, 0, self._canvas)
        self.stroke_count += 1

    def erase(self, x, y):
        r = self.cfg.brush_size * 4
        cv2.circle(self._canvas, (x, y), r, (0, 0, 0), -1)

    def clear(self):
        self._canvas[:] = 0
        self.stroke_count = 0

    def draw_perfect_shape(self, shape: dict):
        """Dessine une forme parfaite en effaçant l'ancien trait détecté."""
        c = self.cfg.color_bgr
        t = self.cfg.brush_size
        kind = shape["type"]

        # ── Calcul de la zone à nettoyer ─────────────────────
        if kind == "circle":
            cx, cy, r = int(shape["cx"]), int(shape["cy"]), int(shape["radius"])
            # efface légèrement plus grand pour couvrir le trait original
            cv2.circle(self._canvas, (cx, cy), r + t, (0, 0, 0), -1)
            cv2.circle(self._canvas, (cx, cy), r, c, t, cv2.LINE_AA)

        elif kind == "rect":
            x1, y1 = int(shape["x"]), int(shape["y"])
            w, h = int(shape["w"]), int(shape["h"])
            if shape.get("is_square"):
                s = min(w, h)
                w = h = s
            x2, y2 = x1 + w, y1 + h
            # efface un peu plus large
            cv2.rectangle(
                self._canvas, (x1 - t, y1 - t), (x2 + t, y2 + t), (0, 0, 0), -1
            )
            cv2.rectangle(self._canvas, (x1, y1), (x2, y2), c, t, cv2.LINE_AA)

        elif kind == "triangle":
            pts = np.array([(int(p[0]), int(p[1])) for p in shape["points"]], np.int32)
            # zone de bounding box
            x, y, w, h = cv2.boundingRect(pts)
            cv2.rectangle(
                self._canvas, (x - t, y - t), (x + w + t, y + h + t), (0, 0, 0), -1
            )
            cv2.polylines(self._canvas, [pts], True, c, t, cv2.LINE_AA)

    # ── Enhancement IA ───────────────────────────────────────
    def apply_enhanced(self, enhanced_canvas: np.ndarray):
        """
        Remplace le canvas par la version améliorée par l'IA.
        L'appelant doit avoir sauvegardé l'undo AVANT d'appeler cette méthode.
        """
        h, w = self._canvas.shape[:2]
        eh, ew = enhanced_canvas.shape[:2]
        if (eh, ew) != (h, w):
            enhanced_canvas = cv2.resize(enhanced_canvas, (w, h))
        self._canvas = enhanced_canvas.copy()

    # ── Undo ─────────────────────────────────────────────────
    def save_undo(self):
        self._undo_stack.append(self._canvas.copy())
        if len(self._undo_stack) > self.MAX_UNDO:
            self._undo_stack.pop(0)

    def undo(self):
        if self._undo_stack:
            self._canvas = self._undo_stack.pop()

    # ── Save ─────────────────────────────────────────────────
    def save_png(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.SAVE_DIR, f"gesture_draw_{ts}.png")
        bg = np.full_like(self._canvas, 13)
        mask = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        bg[mask > 0] = self._canvas[mask > 0]
        cv2.imwrite(path, bg)
        print(f"[GestureDraw] Sauvegardé → {path}")
        return path
