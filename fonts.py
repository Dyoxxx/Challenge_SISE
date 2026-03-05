"""
fonts.py — Détection multiplateforme des polices + cache Pillow.
Supporte Windows, Linux.
"""

import os, sys, platform
from PIL import ImageFont

# ════════════════════════════════════════════════════════════
#  Détection OS
# ════════════════════════════════════════════════════════════
_OS = platform.system()  # "Windows" | "Linux" | "Darwin"


def _candidates(*names):
    """
    Retourne une liste de chemins possibles pour les polices demandées,
    selon le système d'exploitation.
    """
    paths = []
    if _OS == "Windows":
        win = os.environ.get("WINDIR", r"C:\Windows")
        for name in names:
            paths.append(os.path.join(win, "Fonts", name))
    elif _OS == "Darwin":
        for name in names:
            for base in [
                "/Library/Fonts",
                "/System/Library/Fonts",
                os.path.expanduser("~/Library/Fonts"),
            ]:
                paths.append(os.path.join(base, name))
    else:  # Linux
        for name in names:
            for base in [
                "/usr/share/fonts/truetype/dejavu",
                "/usr/share/fonts/truetype/liberation",
                "/usr/share/fonts/truetype/freefont",
                "/usr/share/fonts/truetype",
            ]:
                paths.append(os.path.join(base, name))
    return paths


def _first_existing(*groups):
    """Retourne le premier fichier existant parmi les groupes."""
    for group in groups:
        for p in _candidates(*group) if isinstance(group, (list, tuple)) else [group]:
            if os.path.isfile(p):
                return p
    return None  # fallback Pillow default


# ════════════════════════════════════════════════════════════
#  Résolution des chemins selon la plateforme
# ════════════════════════════════════════════════════════════

FONT_BOLD = _first_existing(
    # Windows
    ["arialbd.ttf", "Arial Bold.ttf"],
    # macOS
    ["Arial Bold.ttf", "Helvetica Bold.ttf"],
    # Linux
    ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"],
)

FONT_REG = _first_existing(
    ["arial.ttf", "Arial.ttf"],
    ["Arial.ttf", "Helvetica.ttf"],
    ["DejaVuSans.ttf", "LiberationSans-Regular.ttf"],
)

FONT_MONO = _first_existing(
    ["cour.ttf", "CourierNew Bold.ttf", "consola.ttf"],
    ["Courier New Bold.ttf", "Menlo Bold.ttf"],
    ["DejaVuSansMono-Bold.ttf", "LiberationMono-Bold.ttf"],
)

FONT_MONOR = _first_existing(
    ["cour.ttf", "CourierNew.ttf", "consola.ttf"],
    ["Courier New.ttf", "Menlo.ttf"],
    ["DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"],
)

# Fallback ultime : police Pillow intégrée (toujours disponible)
_BUILTIN = None

# ════════════════════════════════════════════════════════════
#  Cache et chargement
# ════════════════════════════════════════════════════════════
_CACHE: dict = {}


def get_font(path, size: int):
    """
    Charge et met en cache une police TrueType.
    Si le fichier est introuvable, utilise la police intégrée Pillow
    redimensionnée (approximation).
    """
    global _BUILTIN
    key = (path, size)
    if key in _CACHE:
        return _CACHE[key]

    font = None
    if path and os.path.isfile(path):
        try:
            font = ImageFont.truetype(path, size)
        except Exception:
            font = None

    if font is None:
        # Police intégrée Pillow — disponible partout, non redimensionnable
        # On stocke quand même pour éviter les crashs
        if _BUILTIN is None:
            try:
                _BUILTIN = ImageFont.load_default(size=max(10, size))
            except TypeError:
                # Pillow < 10.0 : load_default ne prend pas size
                _BUILTIN = ImageFont.load_default()
        try:
            font = ImageFont.load_default(size=max(10, size))
        except TypeError:
            font = ImageFont.load_default()

        if path:
            print(f"[fonts] ⚠ Police introuvable : {path!r}  → fallback Pillow")

    _CACHE[key] = font
    return font


def report():
    """Affiche un résumé des polices détectées (utile au démarrage)."""
    print(f"  OS     : {_OS}")
    for name, val in [
        ("BOLD", FONT_BOLD),
        ("REG", FONT_REG),
        ("MONO", FONT_MONO),
        ("MONOR", FONT_MONOR),
    ]:
        status = "✓" if val and os.path.isfile(val) else "✗ fallback"
        print(f"  {name:<6}: {val or '(builtin)'}  [{status}]")
