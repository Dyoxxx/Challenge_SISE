"""
shape_detector.py — Reconnaissance de formes géométriques
"""

import cv2
import numpy as np
import math


class ShapeDetector:
    MIN_POINTS = 25
    MIN_SIZE   = 40

    def detect(self, points: list) -> dict | None:
        if len(points) < self.MIN_POINTS:
            return None
        pts = np.array(points, dtype=np.float32)
        xs, ys = pts[:, 0], pts[:, 1]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        w, h = x_max - x_min, y_max - y_min
        if w < self.MIN_SIZE or h < self.MIN_SIZE:
            return None

        # Fermeture
        close = math.hypot(pts[-1][0]-pts[0][0], pts[-1][1]-pts[0][1])
        peri  = sum(math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
                    for i in range(1, len(pts)))
        if close > peri * 0.35:
            return None

        scores = {
            "circle":   self._circle_score(pts, xs, ys, w, h),
            "rect":     self._rect_score(pts),
            "triangle": self._triangle_score(pts),
        }
        best = max(scores, key=scores.get)
        if scores[best] < 0.55:
            return None
        return self._build(best, pts, xs, ys, x_min, y_min, w, h)

    def _circle_score(self, pts, xs, ys, w, h):
        aspect = min(w, h) / max(w, h)
        cx, cy = xs.mean(), ys.mean()
        d  = np.sqrt((xs-cx)**2 + (ys-cy)**2)
        mr = d.mean()
        if mr == 0: return 0.0
        circ = max(0.0, 1.0 - (d.std()/mr)*2.5)
        return circ*0.65 + aspect*0.35

    def _rect_score(self, pts):
        approx = self._approx(pts, 0.06)
        n = len(approx)
        if n < 3 or n > 7: return 0.0
        cs = 1.0 if n==4 else (0.7 if n in(3,5) else 0.4)
        as_ = self._angle_score(approx) if n==4 else 0.5
        return cs*0.5 + as_*0.5

    def _triangle_score(self, pts):
        approx = self._approx(pts, 0.08)
        n = len(approx)
        if n==3: return 0.85
        if n==4: return 0.3
        return 0.0

    def _approx(self, pts, eps_ratio):
        pi = pts.astype(np.int32)
        hull = cv2.convexHull(pi)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, eps_ratio*peri, True)
        return approx.reshape(-1, 2).astype(np.float32)

    def _angle_score(self, corners):
        n = len(corners)
        s = 0.0
        for i in range(n):
            p0 = corners[(i-1)%n]
            p1 = corners[i]
            p2 = corners[(i+1)%n]
            v1, v2 = p0-p1, p2-p1
            nm = np.linalg.norm(v1)*np.linalg.norm(v2)
            if nm == 0: continue
            ca = np.clip(np.dot(v1,v2)/nm, -1, 1)
            ang = math.degrees(math.acos(ca))
            s += max(0.0, 1.0 - abs(ang-90)/90)
        return s/n

    def _build(self, kind, pts, xs, ys, x_min, y_min, w, h):
        if kind == "circle":
            cx, cy = float(xs.mean()), float(ys.mean())
            r = float(np.sqrt((xs-cx)**2+(ys-cy)**2).mean())
            return {"type":"circle","label":"⭕ Cercle","cx":cx,"cy":cy,"radius":r}
        elif kind == "rect":
            sq = abs(w-h)/max(w,h) < 0.15
            return {"type":"rect","label":"◻ Carré" if sq else "▬ Rectangle",
                    "x":float(x_min),"y":float(y_min),
                    "w":float(w),"h":float(h),"is_square":sq}
        elif kind == "triangle":
            approx = self._approx(pts, 0.08)
            tri = approx[:3].tolist() if len(approx)>=3 else [
                [float(x_min+w/2),float(y_min)],
                [float(x_min),float(y_min+h)],
                [float(x_min+w),float(y_min+h)],
            ]
            return {"type":"triangle","label":"△ Triangle","points":tri}
