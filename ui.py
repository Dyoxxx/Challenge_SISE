"""
ui.py — GestureDraw v3  (réécriture propre, RGB pur)

Règle d'or : ImageDraw travaille TOUJOURS sur une image RGB.
Toutes les couleurs sont des 3-tuples (r,g,b). Pas d'alpha dans les
appels Pillow → zéro TypeError.

Les effets semi-transparents sont simulés par mélange numpy
(cv2.addWeighted) sur le tableau BGR final, uniquement quand nécessaire.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math, time

from config import Config
from fonts import FONT_BOLD, FONT_REG, FONT_MONO, FONT_MONOR, get_font as F


# ── Conversion ───────────────────────────────────────────────
def _bgr2rgb(arr):
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _pil2bgr(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ════════════════════════════════════════════════════════════
#  Primitives Pillow  — RGB uniquement, couleurs = 3-tuples
# ════════════════════════════════════════════════════════════


def _r(d, x1, y1, x2, y2, fill=None, outline=None, ow=1, rad=0):
    """Rectangle (optionnel : coins arrondis)."""
    kw = {"fill": fill, "outline": outline, "width": ow}
    kw = {k: v for k, v in kw.items() if v is not None}
    if rad:
        d.rounded_rectangle([x1, y1, x2, y2], radius=rad, **kw)
    else:
        d.rectangle([x1, y1, x2, y2], **kw)


def _l(d, x1, y1, x2, y2, col, w=1):
    d.line([(x1, y1), (x2, y2)], fill=col, width=w)


def _c(d, cx, cy, r, fill=None, outline=None, ow=1):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=ow)


def _t(d, txt, x, y, col, font, anchor="lt"):
    d.text((x, y), txt, fill=col, font=font, anchor=anchor)


def _tw(font, txt):
    bb = font.getbbox(txt)
    return bb[2] - bb[0]


def _sep(d, y, x1, x2, col):
    d.line([(x1, y), (x2, y)], fill=col, width=1)


def _blend(col, bg, alpha):
    """Mélange linéaire d'une couleur avec un fond (simule la transparence)."""
    a = alpha / 255.0
    return tuple(int(col[i] * a + bg[i] * (1 - a)) for i in range(3))


# ── Glow sur crop local (RGB tout le long) ───────────────────
def _glow_circle(img, cx, cy, r, col, gr=10, s=2):
    """Cercle lumineux (halo) sur un crop local, en RGB."""
    pad = gr * 3
    x1, y1 = max(0, cx - r - pad), max(0, cy - r - pad)
    x2, y2 = min(img.width, cx + r + pad), min(img.height, cy + r + pad)
    cw, ch = x2 - x1, y2 - y1
    lx, ly = cx - x1, cy - y1

    # Calque RGBA pour le glow (seulement ici, isolé)
    L = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    ld = ImageDraw.Draw(L)
    ld.ellipse([lx - r, ly - r, lx + r, ly + r], fill=(*col, 255))

    solid = Image.new("RGBA", (cw, ch), (*col, 0))
    _, _, _, a = L.split()
    solid.putalpha(a)
    glow = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    for _ in range(s):
        glow = Image.alpha_composite(glow, solid.filter(ImageFilter.GaussianBlur(gr)))
    merged = Image.alpha_composite(glow, L).convert("RGB")

    # Blend le halo sur le crop RGB existant
    crop = img.crop((x1, y1, x2, y2))
    blended = Image.blend(crop, merged, 0.8)
    img.paste(blended, (x1, y1))


# ════════════════════════════════════════════════════════════
class UIManager:

    GESTURE_LABEL = {
        "draw": "● DRAW",
        "pause": "❙❙ PAUSE",
        "erase": "⊘ GOMME",
        "undo": "↩ ANNUL.",
        "open_hand": "✕ CLEAR",
        "none": " —  — ",
    }
    GESTURE_GUIDE = [
        ("Index seul", "DESSINER", "draw"),
        ("Index+Majeur", "PAUSE", "pause"),
        ("Poing fermé", "GOMME", "erase"),
        ("Pouce+Auricu.", "ANNULER", "undo"),
        ("Main ouverte", "EFFACER", "open_hand"),
    ]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.W = cfg.WIN_W
        self.H = cfg.WIN_H
        self.PL = cfg.PANEL_L
        self.PR = cfg.PANEL_R
        self.HH = cfg.HEADER_H
        self.FH = cfg.FOOTER_H
        self.CW = cfg.CANVAS_W
        self.CH = cfg.CANVAS_H
        self.CX = self.PL
        self.CY = self.HH

        # Couleurs (3-tuples uniquement)
        self.BG = cfg.C_BG
        self.SF = cfg.C_SURFACE
        self.PAN = cfg.C_PANEL
        self.BOR = cfg.C_BORDER
        self.TXT = cfg.C_TEXT
        self.MUT = cfg.C_MUTED
        self.ACC = cfg.C_ACCENT
        self.A2 = cfg.C_ACCENT2
        self.A3 = cfg.C_ACCENT3
        self.OR = cfg.C_ORANGE
        self.PUR = cfg.C_PURPLE
        self.GR = cfg.C_GREEN

        # Tailles de police
        self.FST = cfg.FS_TITLE
        self.FSH = cfg.FS_HEAD
        self.FS = cfg.FS_SEC
        self.FSB = cfg.FS_BODY
        self.FSS = cfg.FS_SMALL
        self.FSN = cfg.FS_NOTIF

        # Polices
        self.f_title = F(FONT_BOLD, self.FST)
        self.f_head = F(FONT_BOLD, self.FSH)
        self.f_sec = F(FONT_MONO, self.FS)
        self.f_body = F(FONT_MONOR, self.FSB)
        self.f_small = F(FONT_MONOR, self.FSS)
        self.f_key = F(FONT_MONO, self.FSB)
        self.f_notif = F(FONT_MONO, self.FSN)

        self._pulse_t = 0.0
        self._notif_text = ""
        self._notif_until = 0.0
        self._ai_loading = False
        self._ai_subject = ""
        self._ai_subj_end = 0.0

        # Couleurs de geste
        self._gcol = {
            "draw": self.A3,
            "pause": self.MUT,
            "erase": self.A2,
            "undo": self.OR,
            "open_hand": self.A2,
            "none": self.MUT,
        }

        # Caches
        self._bg = self._build_bg()  # Image RGB
        self._logo = self._build_logo()  # Image RGB
        self._left_key = None
        self._left_img = None  # Image RGB

    # ── Notifications ────────────────────────────────────────
    def notify(self, msg, dur=2.5):
        self._notif_text = msg
        self._notif_until = time.time() + dur

    def show_ai_subject(self, s, dur=5.0):
        self._ai_subject = s
        self._ai_subj_end = time.time() + dur

    # ════════════════════════════════════════════════════════
    #  FOND STATIQUE
    # ════════════════════════════════════════════════════════
    def _build_bg(self):
        img = Image.new("RGB", (self.W, self.H), self.BG)
        d = ImageDraw.Draw(img)

        # Header
        _r(d, 0, 0, self.W, self.HH, fill=self.SF)
        _sep(d, self.HH, 0, self.W, self.BOR)

        # Panneau gauche
        _r(d, 0, self.HH, self.PL, self.H, fill=self.SF)
        _l(d, self.PL, self.HH, self.PL, self.H, self.BOR)

        # Canvas
        cx1, cy1 = self.CX, self.CY
        cx2, cy2 = self.CX + self.CW, self.CY + self.CH
        _r(d, cx1, cy1, cx2, cy2, fill=(13, 13, 20))
        step = max(30, self.CW // 24)
        gc = (25, 25, 38)
        for x in range(cx1, cx2, step):
            _l(d, x, cy1, x, cy2, gc)
        for y in range(cy1, cy2, step):
            _l(d, cx1, y, cx2, y, gc)

        # Panneau droit
        rx = self.CX + self.CW
        _r(d, rx, self.HH, self.W, self.H, fill=self.SF)
        _l(d, rx, self.HH, rx, self.H, self.BOR)

        # Footer
        fy = self.H - self.FH
        _r(d, 0, fy, self.W, self.H, fill=self.SF)
        _sep(d, fy, 0, self.W, self.BOR)
        _t(
            d,
            "GestureDraw  ✦  I=Enhance  A=Artwork IA  C=Effacer  Z=Annuler  S=Sauver  Q=Quitter",
            14,
            fy + self.FH // 2,
            self.MUT,
            self.f_small,
            anchor="lm",
        )

        # Labels statiques panneau droit
        self._static_right(d, rx)
        return img

    def _static_right(self, d, rx):
        pw = self.PR
        y = self.HH + 10

        _t(d, "CAMÉRA LIVE", rx + 10, y, self.MUT, self.f_sec)
        _sep(d, y + self.FS + 4, rx, rx + pw, self.BOR)
        y += self.FS + 8 + self.cfg.THUMB_H + 14

        _t(d, "GESTES", rx + 10, y, self.MUT, self.f_sec)
        _sep(d, y + self.FS + 4, rx, rx + pw, self.BOR)
        # (les cartes gestes sont dynamiques — on ne les pré-rend pas)
        y += self.FS + 8 + len(self.GESTURE_GUIDE) * (self.FSH + self.FSS + 12) + 10

        _sep(d, y, rx, rx + pw, self.BOR)
        y += 10
        _t(d, "IA MISTRAL", rx + 10, y, self.PUR, self.f_sec)
        y += self.FS + 8
        for txt, col in [
            ("I  = Enhance  (remplace canvas)", self.ACC),
            ("A  = Artwork  (vraie image IA)", self.A3),
            ("Multi-provider : FLUX, DALL-E…", self.MUT),
        ]:
            _t(d, txt, rx + 10, y, col, self.f_small)
            y += self.FSS + 5

    # ════════════════════════════════════════════════════════
    #  LOGO (cache, pré-rendu une fois)
    # ════════════════════════════════════════════════════════
    def _build_logo(self):
        lw, lh = self.PL, self.HH
        img = Image.new("RGB", (lw, lh), self.SF)
        d = ImageDraw.Draw(img)

        # Hexagone
        hx, hy, hr = lh // 2, lh // 2, lh // 3
        pts = [
            (
                int(hx + hr * math.cos(math.radians(60 * i - 30))),
                int(hy + hr * math.sin(math.radians(60 * i - 30))),
            )
            for i in range(6)
        ]
        d.polygon(pts, fill=self.ACC)

        tx = lh + 8
        _t(d, "GESTURE", tx, 5, self.ACC, self.f_title)
        _t(d, "DRAW  v3", tx, self.FST + 7, self.MUT, self.f_small)
        return img

    # ════════════════════════════════════════════════════════
    #  COMPOSE  (appelé chaque frame)
    # ════════════════════════════════════════════════════════
    def compose(
        self,
        canvas,
        cam_frame,
        gesture,
        finger_pos,
        fps,
        shape_hint,
        brush_color,
        brush_size,
        brush_opacity,
        shape_mode,
        strokes,
        ai_loading=False,
    ) -> np.ndarray:

        self._pulse_t = time.time()
        self._ai_loading = ai_loading

        img = self._bg.copy()  # RGB
        d = ImageDraw.Draw(img)  # dessin RGB uniquement

        # ── Blocs ────────────────────────────────────────────
        self._blit_canvas(img, canvas)

        # Panneau gauche (cache)
        lk = (brush_color, brush_size, int(brush_opacity * 100), shape_mode, strokes)
        if lk != self._left_key:
            self._left_img = self._build_left(
                brush_color, brush_size, brush_opacity, shape_mode, strokes
            )
            self._left_key = lk
        img.paste(self._left_img, (0, self.HH))

        # Logo + header
        img.paste(self._logo, (0, 0))
        d = ImageDraw.Draw(img)
        self._draw_header(d, fps, gesture)

        # Panneau droit
        self._draw_right(img, d, cam_frame, gesture)

        # Curseur main
        fx, fy = finger_pos
        if fx is not None:
            self._draw_cursor(
                d, self.CX + fx, self.CY + fy, gesture, brush_color, brush_size
            )

        # Overlays
        if shape_hint:
            self._draw_shape_hint(d, shape_hint)
        if ai_loading:
            self._draw_ai_loading(img, d)
        if self._ai_subject and time.time() < self._ai_subj_end:
            self._draw_ai_subject(d)
        if time.time() < self._notif_until:
            self._draw_notif(img)

        return _pil2bgr(img)

    # ── Canvas ───────────────────────────────────────────────
    def _blit_canvas(self, img, canvas):
        rgb = _bgr2rgb(canvas)
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        src = Image.fromarray(rgb)
        img.paste(src, (self.CX, self.CY), Image.fromarray(mask))

    # ── Header ───────────────────────────────────────────────
    def _draw_header(self, d, fps, gesture):
        hh = self.HH
        cx = self.PL + self.CW // 2
        col = self._gcol.get(gesture, self.MUT)
        lbl = self.GESTURE_LABEL.get(gesture, " —  — ")

        tw = _tw(self.f_head, lbl)
        bx = cx - tw // 2 - 14
        # Fond du badge (simulé par mélange avec SF)
        bg = _blend(col, self.SF, 35)
        _r(d, bx, 8, bx + tw + 28, hh - 8, fill=bg, outline=col, ow=1, rad=6)
        _t(d, lbl, cx, hh // 2, col, self.f_head, anchor="mm")

        # FPS
        fc = self.GR if fps >= 20 else (self.OR if fps >= 12 else self.A2)
        _t(d, f"{fps} FPS", self.W - 14, hh // 2, fc, self.f_body, anchor="rm")

        # Point CAM
        pulse = abs(math.sin(self._pulse_t * 2.5)) > 0.5
        dot_c = self.A3 if pulse else (50, 130, 50)
        _c(d, self.W - 80, hh // 2, 5, fill=dot_c)
        _t(d, "CAM", self.W - 68, hh // 2, self.MUT, self.f_small, anchor="lm")

        # Indicateur IA
        if self._ai_loading:
            dots = "." * (int(time.time() * 3) % 4)
            _t(
                d,
                f"IA{dots}",
                cx + tw // 2 + 36,
                hh // 2,
                self.PUR,
                self.f_small,
                anchor="lm",
            )

    # ════════════════════════════════════════════════════════
    #  PANNEAU GAUCHE (cache)
    # ════════════════════════════════════════════════════════
    def _build_left(self, color_rgb, brush_size, brush_opacity, shape_mode, strokes):
        pw = self.PL
        ph = self.H - self.HH - self.FH
        img = Image.new("RGB", (pw, ph), self.SF)
        d = ImageDraw.Draw(img)
        PAD = 12
        bw = pw - PAD * 2
        y = 14

        def sec(title):
            nonlocal y
            _t(d, title, PAD, y, self.MUT, self.f_sec)
            y += self.FS + 4
            _sep(d, y, 0, pw, self.BOR)
            y += 8

        # ── Couleur active ────────────────────────────────────
        sec("COULEUR ACTIVE")
        rc = 20
        ccx, ccy = PAD + rc + 4, y + rc + 4
        _glow_circle(img, ccx, ccy, rc, color_rgb, gr=rc // 14, s=1)
        d = ImageDraw.Draw(img)
        _c(d, ccx, ccy, rc, outline=self.TXT, ow=2)
        _t(
            d,
            f"R{color_rgb[0]:>3}  G{color_rgb[1]:>3}  B{color_rgb[2]:>3}",
            ccx + rc + 10,
            ccy,
            self.MUT,
            self.f_small,
            anchor="lm",
        )
        y += rc * 2 + 12

        # ── Palette ───────────────────────────────────────────
        sec("PALETTE  [1-6]")
        cr = max(14, pw // 16)
        sp = (pw - PAD * 2) // 3
        for i, (c, _) in enumerate(self.cfg.COLORS_RGB):
            col_ = i % 3
            row_ = i // 3
            px = PAD + col_ * sp + sp // 2
            py = y + row_ * (cr * 2 + 18) + cr
            is_a = c == color_rgb
            if is_a:
                # _glow_circle(img, px, py, cr, c, gr=cr, s=2)
                d = ImageDraw.Draw(img)
            _c(d, px, py, cr, fill=c)
            _c(
                d,
                px,
                py,
                cr,
                outline=self.TXT if is_a else self.BOR,
                ow=2 if is_a else 1,
            )
            _t(d, str(i + 1), px, py + cr + 3, self.MUT, self.f_small, anchor="mt")
        y += (cr * 2 + 18) * 2 + 8

        # ── Taille pinceau ────────────────────────────────────
        sec(f"PINCEAU  {brush_size}px")
        filled = int(bw * min(1.0, brush_size / 60))
        _r(d, PAD, y, PAD + bw, y + 8, fill=self.BOR, rad=4)
        if filled:
            _r(d, PAD, y, PAD + filled, y + 8, fill=color_rgb, rad=4)
        y += 16
        pr = min(brush_size // 2 + 2, pw // 5)
        _glow_circle(img, pw // 2, y + pr + 4, pr, color_rgb, gr=max(6, pr) // 14, s=1)
        d = ImageDraw.Draw(img)
        y += pr * 2 + 12

        # ── Opacité ───────────────────────────────────────────
        sec(f"OPACITÉ  {int(brush_opacity * 100)}%")
        op = int(bw * min(1.0, brush_opacity))
        _r(d, PAD, y, PAD + bw, y + 8, fill=self.BOR, rad=4)
        if op:
            _r(d, PAD, y, PAD + op, y + 8, fill=self.TXT, rad=4)
        y += 18

        # ── Mode formes ───────────────────────────────────────
        sec("FORMES AUTO")
        mc = self.A3 if shape_mode else (55, 55, 75)
        txt = "ON  ✓  [M]" if shape_mode else "OFF      [M]"
        bg2 = _blend(mc, self.SF, 35) if shape_mode else self.PAN
        _r(d, PAD, y, pw - PAD, y + self.FSB + 10, fill=bg2, outline=mc, ow=1, rad=5)
        _t(d, txt, pw // 2, y + self.FSB // 2 + 5, mc, self.f_body, anchor="mm")
        y += self.FSB + 18

        # ── Stats ─────────────────────────────────────────────
        sec("STATS")
        _t(d, f"Traits : {strokes}", PAD, y, self.TXT, self.f_body)
        y += self.FSB + 16

        # ── Raccourcis ────────────────────────────────────────
        sec("RACCOURCIS")
        for k, desc in [
            ("C", "Effacer"),
            ("Z", "Annuler"),
            ("S", "Sauver"),
            ("M", "Formes"),
            ("I", "IA Enhance"),
            ("A", "Artwork IA"),
            ("Q", "Quitter"),
        ]:
            if y + self.FSB > ph - 4:
                break
            kw = _tw(self.f_key, k)
            kcol = self.PUR if k in ("I", "A") else self.ACC
            _r(d, PAD, y - 2, PAD + kw + 10, y + self.FSB + 2, fill=self.BOR, rad=3)
            _t(d, k, PAD + 5, y, kcol, self.f_key)
            _t(d, desc, PAD + kw + 16, y, self.TXT, self.f_body)
            y += self.FSB + 5

        return img

    # ════════════════════════════════════════════════════════
    #  PANNEAU DROIT
    # ════════════════════════════════════════════════════════
    def _draw_right(self, img, d, cam_frame, gesture):
        rx = self.CX + self.CW
        pw = self.PR
        tw, th = self.cfg.THUMB_W, self.cfg.THUMB_H
        tx = rx + (pw - tw) // 2
        ty = self.HH + self.FS + 14

        # Miniature webcam
        if cam_frame is not None:
            thumb = cv2.resize(cam_frame, (tw, th))
            img.paste(Image.fromarray(_bgr2rgb(thumb)), (tx, ty))
        d.rectangle([tx, ty, tx + tw, ty + th], outline=self.ACC)

        pulse = abs(math.sin(self._pulse_t * 2.5)) > 0.5
        rc = (200, 30, 30) if pulse else (100, 20, 20)
        _c(d, tx + tw - 12, ty + 12, 5, fill=rc)
        _t(d, "REC", tx + tw - 28, ty + 12, rc, self.f_small, anchor="lm")

        # Cartes gestes
        gcols = {
            "draw": self.A3,
            "pause": self.MUT,
            "erase": self.A2,
            "undo": self.OR,
            "open_hand": self.A2,
        }
        gy = ty + th + 14 + self.FS + 8
        for desc, action, gkey in self.GESTURE_GUIDE:
            is_a = gesture == gkey
            col = gcols.get(gkey, self.MUT)
            bh = self.FSH + self.FSS + 10
            bg = _blend(col, self.SF, 35) if is_a else self.PAN
            bdr = col if is_a else self.BOR
            _r(d, rx + 8, gy, rx + pw - 8, gy + bh, fill=bg, outline=bdr, ow=1, rad=5)
            tc = col if is_a else self.MUT
            _t(d, action, rx + 14, gy + 3, tc, self.f_head if is_a else self.f_body)
            _t(
                d,
                desc,
                rx + 14,
                gy + self.FSH + 5,
                self.TXT if is_a else self.MUT,
                self.f_small,
            )
            gy += bh + 5

    # ── Curseur ──────────────────────────────────────────────
    def _draw_cursor(self, d, x, y, gesture, color, size):
        """Toutes les couleurs sont des 3-tuples — aucun alpha."""
        if gesture == "draw":
            r = max(10, size + 6)
            halo = _blend(color, self.BG, 50)
            _c(d, x, y, r + 4, outline=halo, ow=1)
            _c(d, x, y, r, outline=color, ow=2)
            _c(d, x, y, 3, fill=color)
        elif gesture == "erase":
            r = size * 4
            _c(d, x, y, r, outline=self.A2, ow=2)
            _l(d, x - r, y, x + r, y, self.A2)
            _l(d, x, y - r, x, y + r, self.A2)
        elif gesture == "pause":
            _c(d, x, y, 10, outline=self.MUT, ow=1)
        elif gesture in ("undo", "open_hand"):
            _c(d, x, y, 13, outline=self.OR, ow=2)

    # ── Shape hint ───────────────────────────────────────────
    def _draw_shape_hint(self, d, label):
        tw = _tw(self.f_body, label)
        cx = self.CX + self.CW // 2
        bx = cx - tw // 2 - 14
        by = self.CY + 12
        _r(
            d,
            bx,
            by,
            bx + tw + 28,
            by + self.FSB + 12,
            fill=self.PAN,
            outline=self.ACC,
            ow=1,
            rad=6,
        )
        _t(d, label, cx, by + self.FSB // 2 + 6, self.ACC, self.f_body, anchor="mm")

    # ── Overlay IA chargement ────────────────────────────────
    def _draw_ai_loading(self, img, d):
        cx = self.CX + self.CW // 2
        cy = self.CY + self.CH // 2
        bw = min(340, self.CW - 40)
        bh = 66
        bx, by = cx - bw // 2, cy - bh // 2

        _r(d, bx, by, bx + bw, by + bh, fill=self.SF, outline=self.PUR, ow=2, rad=10)

        # Barre animée
        t = time.time()
        pw2 = bw - 40
        px = bx + 20
        _r(d, px, by + bh - 18, px + pw2, by + bh - 8, fill=self.BOR, rad=4)
        seg = int((t % 1.5) / 1.5 * pw2)
        end = min(px + seg + 50, px + pw2)
        _r(d, px, by + bh - 18, end, by + bh - 8, fill=self.PUR, rad=4)

        dots = "." * (int(t * 3) % 4)
        _t(
            d,
            f"IA en cours{dots}",
            cx,
            by + bh // 2 - 6,
            self.TXT,
            self.f_body,
            anchor="mm",
        )

    def _draw_ai_subject(self, d):
        s = self._ai_subject
        if not isinstance(s, str):
            s = str(s)
        tw = _tw(self.f_body, s)
        cx = self.CX + self.CW // 2
        bx = cx - tw // 2 - 20
        by = self.CY + 14
        _r(
            d,
            bx,
            by,
            bx + tw + 40,
            by + self.FSB + 14,
            fill=self.PAN,
            outline=self.PUR,
            ow=2,
            rad=8,
        )
        _t(d, f"✨ {s}", cx, by + self.FSB // 2 + 7, self.TXT, self.f_body, anchor="mm")

    # ── Notification toast ───────────────────────────────────
    def _draw_notif(self, img):
        txt = self._notif_text
        tw = _tw(self.f_notif, txt)
        cx = self.W // 2
        bw = tw + 40
        bh = self.FSN + 18
        bx = cx - bw // 2
        by = self.HH + 12

        # Halo flou (crop local en RGBA, isolé)
        pad = 10
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(self.W, bx + bw + pad)
        y2 = min(self.H, by + bh + pad)
        crop = img.crop((x1, y1, x2, y2))
        halo = Image.new("RGBA", crop.size, (0, 0, 0, 0))
        hd = ImageDraw.Draw(halo)
        hd.rounded_rectangle(
            [bx - x1, by - y1, bx - x1 + bw, by - y1 + bh],
            radius=8,
            fill=(*self.ACC, 30),
            outline=(*self.ACC, 180),
            width=2,
        )
        blurred = halo.filter(ImageFilter.GaussianBlur(6))
        merged = Image.alpha_composite(crop.convert("RGBA"), blurred).convert("RGB")
        img.paste(merged, (x1, y1))

        d = ImageDraw.Draw(img)
        _r(d, bx, by, bx + bw, by + bh, fill=self.PAN, outline=self.ACC, ow=1, rad=8)
        _t(d, txt, cx, by + bh // 2, self.TXT, self.f_notif, anchor="mm")
