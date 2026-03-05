"""
config.py — GestureDraw v3
Multiplateforme : Windows / Linux / macOS.
Rendu natif à la résolution d'affichage.
"""

import os, re, subprocess, platform

from fonts import FONT_BOLD, FONT_REG, FONT_MONO, FONT_MONOR, get_font, report

# Alias pour compatibilité avec renderer.py d'origine
FONT_SANS_BOLD = FONT_BOLD
FONT_SANS = FONT_REG
FONT_MONO_REG = FONT_MONOR
FONT_CONDENSED = FONT_BOLD


# ════════════════════════════════════════════════════════════
#  Détection de la résolution écran
# ════════════════════════════════════════════════════════════
def _screen_size():
    """Retourne (w, h) ou (None, None)."""
    _OS = platform.system()

    # Variable d'env (override manuel toujours prioritaire)
    ew = os.environ.get("GESTURE_W")
    eh = os.environ.get("GESTURE_H")
    if ew and eh:
        try:
            return int(ew), int(eh)
        except ValueError:
            pass

    if _OS == "Windows":
        try:
            import ctypes

            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        except Exception:
            pass

    else:  # Linux
        for cmd, pat in [
            (["xdpyinfo"], r"dimensions:\s+(\d+)x(\d+)"),
            (["xrandr", "--current"], r"current (\d+) x (\d+)"),
        ]:
            try:
                out = subprocess.check_output(
                    cmd, text=True, stderr=subprocess.DEVNULL, timeout=2
                )
                m = re.search(pat, out)
                if m:
                    return int(m.group(1)), int(m.group(2))
            except Exception:
                pass

    return None, None


# ════════════════════════════════════════════════════════════
class Config:
    _sw, _sh = _screen_size()

    if _sw and _sh:
        WIN_W = max(800, min(1600, _sw - 40))
        WIN_H = max(500, min(900, _sh - 60))
    else:
        WIN_W, WIN_H = 1280, 720

    # ── Mise en page ─────────────────────────────────────────
    HEADER_H = max(48, WIN_H // 15)
    FOOTER_H = max(26, WIN_H // 28)
    PANEL_L = max(180, WIN_W // 7)
    PANEL_R = max(200, WIN_W // 6)

    # Zone disponible pour le canvas
    _avail_w = WIN_W - PANEL_L - PANEL_R
    _avail_h = WIN_H - HEADER_H - FOOTER_H

    # Canvas contraint au ratio 4:3 de la caméra (640×480)
    # → chaque pixel de la caméra correspond exactement à une zone du canvas
    CAM_RATIO = 640 / 480  # 4/3
    if _avail_w / _avail_h > CAM_RATIO:
        # Trop large : on limite la largeur
        CANVAS_H = _avail_h
        CANVAS_W = int(CANVAS_H * CAM_RATIO)
    else:
        # Trop haut : on limite la hauteur
        CANVAS_W = _avail_w
        CANVAS_H = int(CANVAS_W / CAM_RATIO)

    # Recalcule les panneaux pour centrer le canvas
    _total_panels = WIN_W - CANVAS_W
    PANEL_L = max(180, _total_panels // 2 - (WIN_W // 60))
    PANEL_R = _total_panels - PANEL_L

    PANEL_LEFT_W = PANEL_L  # alias pour compatibilité
    PANEL_RIGHT_W = PANEL_R

    # ── Caméra ───────────────────────────────────────────────
    CAM_W = 640
    CAM_H = 480
    THUMB_W = max(160, PANEL_R - 28)
    THUMB_H = int(THUMB_W * 0.75)

    # ── Tailles de police (proportionnelles à WIN_H) ─────────
    _s = max(0.75, WIN_H / 720)
    FS_TITLE = int(18 * _s)
    FS_HEAD = int(15 * _s)
    FS_SEC = int(13 * _s)
    FS_BODY = int(12 * _s)
    FS_SMALL = int(11 * _s)
    FS_NOTIF = int(14 * _s)

    # ── Palette ──────────────────────────────────────────────
    COLORS_RGB = [
        ((0, 229, 255), "Cyan"),
        ((184, 255, 87), "Vert"),
        ((255, 51, 102), "Rose"),
        ((255, 170, 0), "Orange"),
        ((180, 80, 255), "Violet"),
        ((255, 255, 255), "Blanc"),
    ]

    # ── Couleurs UI ──────────────────────────────────────────
    C_BG = (10, 10, 15)
    C_SURFACE = (17, 17, 24)
    C_PANEL = (22, 22, 31)
    C_BORDER = (42, 42, 58)
    C_TEXT = (232, 232, 240)
    C_MUTED = (90, 90, 120)
    C_ACCENT = (0, 229, 255)
    C_ACCENT2 = (255, 51, 102)
    C_ACCENT3 = (184, 255, 87)
    C_ORANGE = (255, 170, 0)
    C_PURPLE = (180, 80, 255)
    C_GREEN = (80, 255, 100)

    # ── Mistral ──────────────────────────────────────────────
    MISTRAL_API_KEY = ""

    def __init__(self):
        self._color_idx = 0
        self.brush_size = 4
        self.opacity = 1.0
        self.shape_mode = False
        self.eraser_mode = False
        if not self.MISTRAL_API_KEY:
            self.MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")

    @property
    def color_rgb(self):
        return self.COLORS_RGB[self._color_idx][0]

    @property
    def color_bgr(self):
        r, g, b = self.color_rgb
        return (b, g, r)

    @property
    def color_name(self):
        return self.COLORS_RGB[self._color_idx][1]

    def set_color_by_index(self, idx):
        self._color_idx = idx % len(self.COLORS_RGB)
