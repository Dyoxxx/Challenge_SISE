"""
renderer.py — Primitives de rendu néon avec Pillow + conversion OpenCV
Fournit : texte anti-aliasé, glow, rounded-rect, badges, barres, etc.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
from fonts import (FONT_BOLD as FONT_SANS_BOLD, FONT_REG as FONT_SANS,
                   FONT_MONO, FONT_MONOR as FONT_MONO_REG,
                   FONT_BOLD as FONT_CONDENSED, get_font)


# ════════════════════════════════════════════════════════════
#  CONVERSIONS
# ════════════════════════════════════════════════════════════
def pil_to_cv(img: Image.Image) -> np.ndarray:
    """PIL RGB → OpenCV BGR"""
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv_to_pil(arr: np.ndarray) -> Image.Image:
    """OpenCV BGR → PIL RGB"""
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def cv_to_pil_rgba(arr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).convert("RGBA")


# ════════════════════════════════════════════════════════════
#  GLOW
# ════════════════════════════════════════════════════════════
def add_glow(draw_img: Image.Image, color_rgb: tuple,
             radius: int = 12, strength: int = 3) -> Image.Image:
    """
    Ajoute un halo lumineux autour des pixels colorés.
    draw_img : image RGBA avec les éléments à faire briller.
    """
    glow_layer = Image.new("RGBA", draw_img.size, (0, 0, 0, 0))
    # Crée un calque uni de la couleur avec l'alpha de l'original
    r, g, b = color_rgb
    solid = Image.new("RGBA", draw_img.size, (r, g, b, 0))
    _, _, _, alpha = draw_img.split()
    solid.putalpha(alpha)

    for _ in range(strength):
        blurred = solid.filter(ImageFilter.GaussianBlur(radius=radius))
        glow_layer = Image.alpha_composite(glow_layer, blurred)

    result = Image.alpha_composite(glow_layer, draw_img)
    return result


# ════════════════════════════════════════════════════════════
#  CLASSE PRINCIPALE
# ════════════════════════════════════════════════════════════
class Renderer:
    """
    Rendu haute qualité sur une image Pillow.
    Toutes les coordonnées sont en pixels absolus.
    """

    def __init__(self, width: int, height: int, bg: tuple = (10, 10, 15)):
        self.w = width
        self.h = height
        self.bg = bg
        self._img  = Image.new("RGB", (width, height), bg)
        self._draw = ImageDraw.Draw(self._img, "RGBA")

    def reset(self, bg: tuple | None = None):
        color = bg or self.bg
        self._img  = Image.new("RGB", (self.w, self.h), color)
        self._draw = ImageDraw.Draw(self._img, "RGBA")

    def get_image(self) -> Image.Image:
        return self._img

    def to_cv(self) -> np.ndarray:
        return pil_to_cv(self._img)

    # ────────────────────────────────────────────────────────
    #  PASTE (sous-image)
    # ────────────────────────────────────────────────────────
    def paste(self, sub_img: Image.Image, x: int, y: int,
              mask: Image.Image | None = None):
        self._img.paste(sub_img, (x, y), mask)
        self._draw = ImageDraw.Draw(self._img, "RGBA")

    def paste_cv(self, cv_arr: np.ndarray, x: int, y: int):
        pil = cv_to_pil(cv_arr)
        self._img.paste(pil, (x, y))
        self._draw = ImageDraw.Draw(self._img, "RGBA")

    # ────────────────────────────────────────────────────────
    #  RECTANGLE
    # ────────────────────────────────────────────────────────
    def rect(self, x1, y1, x2, y2,
             fill=None, outline=None, width=1, radius=0):
        fill_c    = (*fill,   255) if fill    else None
        outline_c = (*outline, 255) if outline else None
        self._draw.rounded_rectangle(
            [x1, y1, x2, y2],
            radius=radius,
            fill=fill_c,
            outline=outline_c,
            width=width,
        )

    def rect_alpha(self, x1, y1, x2, y2,
                   fill_rgba=None, outline=None, width=1, radius=0):
        self._draw.rounded_rectangle(
            [x1, y1, x2, y2],
            radius=radius,
            fill=fill_rgba,
            outline=(*outline, 255) if outline else None,
            width=width,
        )

    def line(self, x1, y1, x2, y2, color, width=1):
        self._draw.line([(x1, y1), (x2, y2)],
                        fill=(*color, 255), width=width)

    def circle(self, cx, cy, r, fill=None, outline=None, width=1):
        self._draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=(*fill, 255) if fill else None,
            outline=(*outline, 255) if outline else None,
            width=width,
        )

    # ────────────────────────────────────────────────────────
    #  TEXTE
    # ────────────────────────────────────────────────────────
    def text(self, txt: str, x: int, y: int,
             color: tuple, font_path: str, size: int,
             anchor="lt"):
        font = get_font(font_path, size)
        self._draw.text((x, y), txt,
                        fill=(*color, 255),
                        font=font, anchor=anchor)

    def text_size(self, txt: str, font_path: str, size: int) -> tuple:
        font = get_font(font_path, size)
        bb = font.getbbox(txt)
        return (bb[2] - bb[0], bb[3] - bb[1])

    # ────────────────────────────────────────────────────────
    #  TEXTE GLOW (néon)
    # ────────────────────────────────────────────────────────
    def text_glow(self, txt: str, x: int, y: int,
                  color: tuple, font_path: str, size: int,
                  glow_radius: int = 10, anchor="lt"):
        font = get_font(font_path, size)
        # Calque transparent pour le glow
        layer = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        ld.text((x, y), txt, fill=(*color, 220), font=font, anchor=anchor)
        glowed = add_glow(layer, color, radius=glow_radius, strength=2)
        self._img = Image.alpha_composite(
            self._img.convert("RGBA"), glowed
        ).convert("RGB")
        # Re-draw le texte net par-dessus
        self._draw = ImageDraw.Draw(self._img, "RGBA")
        self._draw.text((x, y), txt,
                        fill=(*color, 255), font=font, anchor=anchor)

    # ────────────────────────────────────────────────────────
    #  BADGE (pill)
    # ────────────────────────────────────────────────────────
    def badge(self, txt: str, cx: int, cy: int,
              color: tuple, bg: tuple | None = None,
              font_path=FONT_MONO, size=12, padding_x=14, padding_y=6):
        font = get_font(font_path, size)
        bb = font.getbbox(txt)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        x1 = cx - tw // 2 - padding_x
        y1 = cy - th // 2 - padding_y
        x2 = cx + tw // 2 + padding_x
        y2 = cy + th // 2 + padding_y
        fill_c = (*bg, 255) if bg else (*color, 30)
        self._draw.rounded_rectangle(
            [x1, y1, x2, y2], radius=20,
            fill=fill_c,
            outline=(*color, 200), width=1,
        )
        self._draw.text((cx, cy), txt,
                        fill=(*color, 255), font=font, anchor="mm")

    # ────────────────────────────────────────────────────────
    #  BARRE DE PROGRESSION
    # ────────────────────────────────────────────────────────
    def progress_bar(self, x, y, w, h, value, max_val,
                     color, bg=(42, 42, 58), radius=4):
        # Fond
        self.rect(x, y, x + w, y + h, fill=bg, radius=radius)
        # Rempli
        filled = int(w * min(1.0, value / max_val))
        if filled > 0:
            self.rect(x, y, x + filled, y + h, fill=color, radius=radius)

    # ────────────────────────────────────────────────────────
    #  DOT animé (pulsé simulé)
    # ────────────────────────────────────────────────────────
    def status_dot(self, cx, cy, active: bool, color):
        if active:
            # halo extérieur
            self.circle(cx, cy, 8, fill=(*color[:3],) if len(color)==3 else color,
                        outline=None)
            # point blanc central
            self.circle(cx, cy, 4, fill=(255,255,255))
        else:
            self.circle(cx, cy, 5, fill=(70, 70, 90))

    # ────────────────────────────────────────────────────────
    #  SEPARATOR LINE
    # ────────────────────────────────────────────────────────
    def separator(self, y, x1=0, x2=None, color=(42, 42, 58)):
        x2 = x2 or self.w
        self._draw.line([(x1, y), (x2, y)], fill=(*color, 255), width=1)
