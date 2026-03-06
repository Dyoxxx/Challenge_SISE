

import cv2
import numpy as np
import time


def rgb(r, g, b): return (b, g, r)


WIN_NAME = "GestureDraw — Image Générée (Q pour fermer)"


def show_generated(sketch: np.ndarray, gen_result: dict):
    """
    Affiche une fenêtre côte à côte : dessin original vs image IA.
    Non-bloquant : la fenêtre se rafraîchit en background.
    """
    gen_img  = gen_result.get("image")
    name_fr  = gen_result.get("name_fr", "dessin")
    prompt   = gen_result.get("prompt", "")
    style    = gen_result.get("style", "")
    dur      = gen_result.get("duration", 0)
    gen_name = gen_result.get("generator", "")
    emoji    = gen_result.get("emoji", "")

    if gen_img is None:
        return

    SIZE = 512  # taille de chaque image

    # ── Prépare le dessin (fond sombre) ──────────────────────
    sk = cv2.resize(sketch, (SIZE, SIZE))
    sk_bg = np.full((SIZE, SIZE, 3), 13, dtype=np.uint8)
    gray = cv2.cvtColor(sk, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    sk_bg[mask > 0] = sk[mask > 0]

    # ── Prépare l'image générée ───────────────────────────────
    gen = cv2.resize(gen_img, (SIZE, SIZE))

    # ── Header + footer ───────────────────────────────────────
    W     = SIZE * 2 + 40   # 1064px
    H     = SIZE + 120       # 632px
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:] = rgb(10, 10, 15)

    # Header
    frame[:50, :] = rgb(17, 17, 24)
    frame[50, :]  = rgb(0, 229, 255)

    title = f"  {emoji}  {name_fr.upper()}  |  {gen_name}  |  {dur:.1f}s"
    cv2.putText(frame, title, (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                rgb(0, 229, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, "Q / Echap = fermer", (W - 200, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                rgb(90, 90, 120), 1, cv2.LINE_AA)

    # Images
    y0, y1 = 58, 58 + SIZE
    x_sk  = 10
    x_gen = SIZE + 30

    frame[y0:y1, x_sk:x_sk+SIZE]   = sk_bg
    frame[y0:y1, x_gen:x_gen+SIZE] = gen

    # Labels sur les images
    cv2.rectangle(frame, (x_sk, y0), (x_sk+SIZE, y0+22), rgb(0,0,0), -1)
    cv2.putText(frame, "VOTRE DESSIN", (x_sk+8, y0+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb(90,90,120), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (x_gen, y0), (x_gen+SIZE, y0+22), rgb(0,0,0), -1)
    cv2.putText(frame, "VERSION IA", (x_gen+8, y0+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb(0,229,255), 1, cv2.LINE_AA)

    # Bordures
    cv2.rectangle(frame, (x_sk,  y0), (x_sk+SIZE,  y1), rgb(42,42,58), 1)
    cv2.rectangle(frame, (x_gen, y0), (x_gen+SIZE, y1), rgb(0,229,255), 1)

    # Footer — prompt
    frame[y1+4:, :] = rgb(17, 17, 24)
    frame[y1+4, :]  = rgb(42, 42, 58)

    prompt_short = prompt[:90] + ("..." if len(prompt) > 90 else "")
    cv2.putText(frame, f"Prompt: {prompt_short}", (12, y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                rgb(90, 90, 120), 1, cv2.LINE_AA)

    style_txt = f"Style: {style}  |  Appuyez sur S pour sauvegarder"
    cv2.putText(frame, style_txt, (12, y1 + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                rgb(60, 60, 90), 1, cv2.LINE_AA)

    # ── Affiche la fenêtre ────────────────────────────────────
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, W, H)
    cv2.imshow(WIN_NAME, frame)

    return frame


class ImageWindowManager:
    """
    Gère la fenêtre image en mode non-bloquant.
    À appeler dans la boucle principale.
    """

    def __init__(self):
        self._frame      = None
        self._visible    = False
        self._last_result_id = None

    def update(self, gen_result, sketch, gen_status):
        """
        Appeler à chaque frame dans la boucle principale.
        Ouvre/met à jour la fenêtre si nouveau résultat.
        """
        if gen_result is None or gen_result.get("image") is None:
            return

        # Nouveau résultat disponible → (ré)affiche
        rid = id(gen_result)
        if rid != self._last_result_id and gen_status == "done":
            self._last_result_id = rid
            self._frame   = show_generated(sketch, gen_result)
            self._visible = True
            return

        # Rafraîchit la fenêtre existante (pour qu'elle reste visible)
        if self._visible and self._frame is not None:
            cv2.imshow(WIN_NAME, self._frame)

    def handle_key(self, key):
        """Retourne True si la fenêtre doit se fermer."""
        if key in (ord('q'), ord('Q'), 27) and self._visible:
            cv2.destroyWindow(WIN_NAME)
            self._visible = False
            return True
        # Sauvegarde
        if key == ord('s') and self._visible and self._frame is not None:
            fname = f"generated_{int(time.time())}.png"
            cv2.imwrite(fname, self._frame)
            print(f"[ImageWindow] Sauvegardé : {fname}")
        return False

    def close(self):
        if self._visible:
            try:
                cv2.destroyWindow(WIN_NAME)
            except Exception:
                pass
        self._visible = False
