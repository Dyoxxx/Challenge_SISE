"""
mistral_enhance.py — Mistral Pixtral : enhance (replace canvas) + artwork (new PNG)
"""

import cv2, numpy as np, base64, json, re, requests, os, subprocess
from datetime import datetime


# ════════════════════════════════════════════════════════════
#  ENCODE
# ════════════════════════════════════════════════════════════
def _encode(canvas: np.ndarray, max_w=768) -> str:
    h, w = canvas.shape[:2]
    if w > max_w:
        canvas = cv2.resize(canvas, (max_w, int(h * max_w / w)))
    bg = np.full_like(canvas, 13)
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    bg[mask > 0] = canvas[mask > 0]
    _, buf = cv2.imencode(".jpg", bg, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode()


# ════════════════════════════════════════════════════════════
#  PARSING JSON ROBUSTE
# ════════════════════════════════════════════════════════════
def _parse_json(raw: str) -> dict:
    text = raw.strip()

    # ─────────────────────────────
    # 1. Retire markdown ```json
    # ─────────────────────────────
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.M)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.M)

    # ─────────────────────────────
    # 2. Supprime commentaires // ...
    # ─────────────────────────────
    text = re.sub(r"//.*", "", text)

    # ─────────────────────────────
    # 3. Supprime commentaires /* ... */
    # ─────────────────────────────
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)

    # ─────────────────────────────
    # 4. Supprime virgules finales
    # ─────────────────────────────
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # ─────────────────────────────
    # 5. Tente parsing direct
    # ─────────────────────────────
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[Mistral] ⚠ JSON non parsable après nettoyage :\n{text[:400]}")
        raise e


# ════════════════════════════════════════════════════════════
#  RENDU DES FORMES
# ════════════════════════════════════════════════════════════
def _parse_color(raw):
    if isinstance(raw, dict):
        raw = [raw.get("r", 255), raw.get("g", 255), raw.get("b", 255)]

    # si couleurs normalisées 0-1 → convertir en 0-255
    if all(isinstance(x, float) and 0 <= x <= 1 for x in raw):
        raw = [int(x * 255) for x in raw]

    return (int(raw[2]), int(raw[1]), int(raw[0]))


def _glow(img):
    b = cv2.GaussianBlur(img.astype(np.float32), (7, 7), 0)
    return np.clip(img.astype(np.float32) + b * 0.35, 0, 255).astype(np.uint8)


def render_shapes(data: dict, W: int, H: int, base: np.ndarray = None) -> np.ndarray:
    """
    Rend les formes Mistral sur un canvas (noir ou 'base' si fourni).
    base : image BGR de référence (pour artwork = sketch original en transparence)
    """
    out = np.zeros((H, W, 3), dtype=np.uint8)
    if base is not None:
        # Fond = sketch original assombri (silhouette à 30%)
        b = cv2.resize(base, (W, H)) if base.shape[:2] != (H, W) else base.copy()
        out = (b.astype(np.float32) * 0.25).astype(np.uint8)

    for s in data.get("shapes", []):
        col = _parse_color(s.get("color", [255, 255, 255]))
        thick = max(1, min(8, int(s.get("thickness", 2))))
        fill = s.get("filled", False)
        t = -1 if fill else thick
        try:
            st = s["type"]
            if st == "circle":
                cx, cy = int(s["cx"] * W), int(s["cy"] * H)
                r = int(s["r"] * min(W, H))
                cv2.circle(out, (cx, cy), max(1, r), col, t, cv2.LINE_AA)
            elif st == "ellipse":
                cx, cy = int(s["cx"] * W), int(s["cy"] * H)
                rx, ry = int(s["rx"] * W), int(s["ry"] * H)
                cv2.ellipse(
                    out,
                    (cx, cy),
                    (max(1, rx), max(1, ry)),
                    float(s.get("angle", 0)),
                    0,
                    360,
                    col,
                    t,
                    cv2.LINE_AA,
                )
            elif st == "rect":
                x1, y1 = int(s["x"] * W), int(s["y"] * H)
                x2, y2 = int((s["x"] + s["w"]) * W), int((s["y"] + s["h"]) * H)
                cv2.rectangle(out, (x1, y1), (x2, y2), col, t, cv2.LINE_AA)
            elif st == "line":
                cv2.line(
                    out,
                    (int(s["x1"] * W), int(s["y1"] * H)),
                    (int(s["x2"] * W), int(s["y2"] * H)),
                    col,
                    thick,
                    cv2.LINE_AA,
                )
            elif st == "poly":
                pts_key = "pts" if "pts" in s else "points"
                pts = np.array(
                    [(int(p[0] * W), int(p[1] * H)) for p in s[pts_key]], np.int32
                )
                (
                    (
                        cv2.fillPoly
                        if fill
                        else lambda *a, **k: cv2.polylines(*a, True, **k)
                    )(out, [pts], col)
                    if fill
                    else cv2.polylines(out, [pts], True, col, thick, cv2.LINE_AA)
                )
        except (KeyError, ValueError, TypeError, IndexError):
            continue

    return _glow(out)


# ════════════════════════════════════════════════════════════
#  APPEL MISTRAL GÉNÉRIQUE
# ════════════════════════════════════════════════════════════
_SYS = (
    "Tu es un assistant artistique qui dois améliorer les dessins en ajoutant à la base. "
    "Tu réponds STRICTEMENT en JSON valide. "
    "AUCUN commentaire. "
    "AUCUN texte hors JSON. "
    "PAS de markdown. "
    "JSON valide RFC8259 uniquement."
)

_PROMPT_ENHANCE = """Analyse ce croquis gestuel. Retourne UNIQUEMENT ce JSON :
{
  "subject": "sujet français (3 mots max)",
  "palette": [[r,g,b]],
  "shapes": [
    {"type":"circle","cx":0.5,"cy":0.5,"r":0.15,"color":[r,g,b],"filled":false,"thickness":2}
  ]
}
Types : circle, ellipse, rect, line, poly.
Coords normalisées 0.0-1.0. Max 20 formes. thickness 1-6."""

_PROMPT_ARTWORK = """Analyse ce croquis gestuel et crée un dessin artistique complet basé dessus.
Retourne UNIQUEMENT ce JSON :
{
  "subject": "sujet français (3 mots max)",
  "style": "description du style (ex: cartoon, réaliste, minimaliste)",
  "palette": [[r,g,b]],
  "shapes": [
    {"type":"circle","cx":0.5,"cy":0.5,"r":0.15,"color":[r,g,b],"filled":true,"thickness":2}
  ]
}
IMPORTANT :
- Reprends la composition du croquis comme base
- Complète avec des détails artistiques (ombres, contours, décors)
- Mélange formes pleines (filled:true) et contours
- Utilise une palette cohérente et belle
- Max 35 formes. Types : circle, ellipse, rect, line, poly. Coords 0.0-1.0."""


def _call(b64: str, prompt: str, api_key: str) -> dict:
    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "pixtral-12b-2409",
            "messages": [
                {"role": "system", "content": _SYS},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "max_tokens": 3500,
            "temperature": 0.2,
        },
        timeout=50,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Mistral {resp.status_code}: {resp.text[:120]}")
    content = resp.json()["choices"][0]["message"]["content"]
    print("TYPE:", type(content))
    print("RAW:", content)
    return _parse_json(resp.json()["choices"][0]["message"]["content"])


# ════════════════════════════════════════════════════════════
#  API PUBLIQUE
# ════════════════════════════════════════════════════════════


def call_mistral(canvas: np.ndarray, api_key: str) -> dict:
    """Enhance : retourne données pour remplacer le canvas."""
    return _call(_encode(canvas), _PROMPT_ENHANCE, api_key)


def call_mistral_artwork(
    canvas: np.ndarray, api_key: str, save_dir: str = "saves"
) -> tuple[str, str]:
    """
    Artwork : génère un PNG indépendant.
    Retourne (chemin_png, sujet).
    """
    b64 = _encode(canvas)
    data = _call(b64, _PROMPT_ARTWORK, api_key)

    W, H = canvas.shape[1], canvas.shape[0]
    artwork = render_shapes(data, W, H, base=canvas)

    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"artwork_{ts}.png")
    cv2.imwrite(path, artwork)

    # Ouvre avec le visualiseur par défaut (non-bloquant)
    try:
        subprocess.Popen(
            ["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        pass  # silencieux si xdg-open absent

    subj = data.get("subject", "Artwork")
    return path, subj


# Alias pour ancienne API
render_enhanced = lambda data, W, H: render_shapes(data, W, H)
