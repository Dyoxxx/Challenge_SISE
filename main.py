"""
main.py — GestureDraw v3  Néon + IA Mistral

Lancer :    python main.py
Résolution: GESTURE_W=1280 GESTURE_H=720 python main.py
Clé Mistral: MISTRAL_API_KEY=xxx python main.py
"""

import cv2, numpy as np, time, os, threading
from collections import deque

from config import Config
from hand_tracker import create_tracker
from gesture import GestureDetector
from drawing import DrawingEngine
from ui import UIManager
from shape_detector import ShapeDetector
from mistral_enhance import call_mistral, render_shapes, call_mistral_artwork
from fonts import report as fonts_report


# ════════════════════════════════════════════════════════════
#  Clavier QWERTY + AZERTY + majuscules
# ════════════════════════════════════════════════════════════
def _k(*chars):
    out = []
    for c in chars:
        out += [ord(c.lower()), ord(c.upper())]
    return tuple(out)


KEY_QUIT = _k("q") + (27,)
KEY_CLEAR = _k("c")
KEY_UNDO = _k("z")
KEY_SAVE = _k("s")
KEY_MODE = _k("m")
KEY_AI = _k("i")
KEY_ARTWORK = _k("a")
KEY_UP = (ord("+"), ord("="), 43)
KEY_DN = (ord(")"), ord("°"), 95)  # pour éviter '-' aussi AZERTY couleur 6
KEY_OP_DN = (ord("ù"), ord("%"))
KEY_OP_UP = (ord("*"), ord("µ"))

COLOR_KEYS = {
    ord("1"): 0,
    38: 0,  # 1  &
    ord("2"): 1,
    233: 1,  # 2  é
    ord("3"): 2,
    34: 2,  # 3  "
    ord("4"): 3,
    39: 3,  # 4  '
    ord("5"): 4,
    40: 4,  # 5  (
    ord("6"): 5,
    45: 5,  # 6  -   (45 aussi KEY_DN si pas dans COLOR_KEYS)
}


# ════════════════════════════════════════════════════════════
#  Thread IA
# ════════════════════════════════════════════════════════════
class AIState:
    def __init__(self):
        self.loading = False
        self.result = None
        self.error = None


def _run_enhance(ai, snap, key, W, H):
    try:
        data = call_mistral(snap, key)
        # si on veut l'ancien canva en plus
        # enhanced = render_shapes(data, W, H, base=snap)
        enhanced = render_shapes(data, W, H)

        ai.result = ("enhance", enhanced, data.get("subject", ""))
    except Exception as e:
        ai.error = str(e)[:80]
    finally:
        ai.loading = False


def _run_artwork(ai, snap, key, save_dir):
    try:
        path, subj = call_mistral_artwork(snap, key, save_dir)
        ai.result = ("artwork", path, subj)
    except Exception as e:
        ai.error = str(e)[:80]
    finally:
        ai.loading = False


# ════════════════════════════════════════════════════════════
def main():
    cfg = Config()
    engine = DrawingEngine(cfg)
    gdet = GestureDetector()
    ui = UIManager(cfg)
    shapes = ShapeDetector()
    ai = AIState()

    print("=" * 62)
    print(f"  ✦ GestureDraw v3  —  rendu {cfg.WIN_W}×{cfg.WIN_H}")
    print(f"  (forcer: GESTURE_W=1280 GESTURE_H=720 python main.py)")
    print("=" * 62)
    if not cfg.MISTRAL_API_KEY:
        print("  ⚠ MISTRAL_API_KEY non définie — I et A désactivés")
    else:
        print("  ✓ Mistral API prête  (I=Enhance  A=Artwork)")
    print("=" * 62 + "\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERR] Webcam introuvable")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    tracker = create_tracker()

    prev_x = prev_y = None
    drawing = False
    shape_path = []
    fps_times = deque(maxlen=30)
    last_undo = 0.0
    shape_hint = ""
    shape_hint_t = 0.0
    cam_frame = None

    WIN = "✦ GestureDraw"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, cfg.WIN_W, cfg.WIN_H)
    ui.notify("Montrez votre main à la caméra !", 4.0)

    while True:
        ret, raw = cap.read()
        if not ret:
            break
        frame = cv2.flip(raw, 1)

        fps_times.append(time.time())
        fps = (
            int((len(fps_times) - 1) / (fps_times[-1] - fps_times[0]))
            if len(fps_times) >= 2
            else 0
        )

        landmarks, cam_frame = tracker.process(frame)
        gesture = "none"
        fx = fy = None

        if landmarks:
            ix = int(landmarks[8].x * cfg.CANVAS_W)
            iy = int(landmarks[8].y * cfg.CANVAS_H)
            fx, fy = ix, iy
            gesture = gdet.detect(landmarks)

            if gesture == "draw":
                if prev_x is not None:
                    if not drawing:
                        engine.save_undo()
                        drawing = True
                    engine.draw_line(prev_x, prev_y, ix, iy)
                    if cfg.shape_mode:
                        shape_path.append((ix, iy))
                prev_x, prev_y = ix, iy
            elif gesture == "pause":
                if drawing and cfg.shape_mode and len(shape_path) > 25:
                    det = shapes.detect(shape_path)
                    if det:
                        engine.save_undo()
                        engine.draw_perfect_shape(det)
                        shape_hint = det["label"]
                        shape_hint_t = time.time() + 2.5
                        ui.notify(f"✓ {det['label']}")
                drawing = False
                shape_path = []
                prev_x = prev_y = None
            elif gesture == "erase":
                engine.erase(ix, iy)
                drawing = False
                shape_path = []
                prev_x = prev_y = None
            elif gesture == "undo":
                now = time.time()
                if now - last_undo > 1.0:
                    engine.undo()
                    last_undo = now
                    ui.notify("↩ Annulé")
                drawing = False
                prev_x = prev_y = None
            elif gesture == "open_hand":
                now = time.time()
                if now - last_undo > 2.0:
                    engine.save_undo()
                    engine.clear()
                    last_undo = now
                    ui.notify("✕ Effacé")
                drawing = False
                prev_x = prev_y = None
            else:
                drawing = False
                prev_x = prev_y = None
        else:
            if drawing and cfg.shape_mode and len(shape_path) > 25:
                det = shapes.detect(shape_path)
                if det:
                    engine.save_undo()
                    engine.draw_perfect_shape(det)
                    shape_hint = det["label"]
                    shape_hint_t = time.time() + 2.5
            drawing = False
            shape_path = []
            prev_x = prev_y = None

        # ── Résultat IA ──────────────────────────────────────
        if ai.result:
            res = ai.result
            ai.result = None
            if res[0] == "enhance":
                _, ecanvas, subj = res
                if isinstance(subj, dict):
                    subj = subj.get("fr") or subj.get("en") or ""

                subj = str(subj)
                engine.save_undo()
                engine.apply_enhanced(ecanvas)
                ui.show_ai_subject(subj, 5.0)
                ui.notify(f"✨ {subj}")
            elif res[0] == "artwork":
                _, path, subj = res
                if isinstance(subj, dict):
                    subj = subj.get("fr") or subj.get("en") or ""

                subj = str(subj)
                ui.notify(f"🎨 {subj}  →  {os.path.basename(path)}", 6.0)
        if ai.error:
            ui.notify(f"❌ IA : {ai.error}", 4.0)
            ai.error = None

        # ── Rendu ────────────────────────────────────────────
        sh = shape_hint if time.time() < shape_hint_t else ""
        display = ui.compose(
            canvas=engine.get_canvas(),
            cam_frame=cam_frame,
            gesture=gesture,
            finger_pos=(fx, fy),
            fps=fps,
            shape_hint=sh,
            brush_color=cfg.color_rgb,
            brush_size=cfg.brush_size,
            brush_opacity=cfg.opacity,
            shape_mode=cfg.shape_mode,
            strokes=engine.stroke_count,
            ai_loading=ai.loading,
        )
        cv2.imshow(WIN, display)

        # ── Touches ──────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in KEY_QUIT:
            break
        elif key in KEY_CLEAR:
            engine.save_undo()
            engine.clear()
            ui.notify("✕ Canvas effacé")
        elif key in KEY_UNDO:
            engine.undo()
            ui.notify("↩ Annulé")
        elif key in KEY_SAVE:
            p = engine.save_png()
            ui.notify(f"💾 {os.path.basename(p)}")
        elif key in KEY_MODE:
            cfg.shape_mode = not cfg.shape_mode
            ui.notify(f"Formes : {'ON ✓' if cfg.shape_mode else 'OFF'}")
        elif key in KEY_UP:
            cfg.brush_size = min(60, cfg.brush_size + 2)
        elif key in KEY_DN and key not in COLOR_KEYS:
            cfg.brush_size = max(1, cfg.brush_size - 2)
        elif key in KEY_OP_DN:
            cfg.opacity = max(cfg.opacity - 0.05, 0.0)
        elif key in KEY_OP_UP:
            cfg.opacity = min(cfg.opacity + 0.05, 1.0)

        elif key in COLOR_KEYS:
            cfg.set_color_by_index(COLOR_KEYS[key])
            ui.notify(f"Couleur : {cfg.color_name}")

        elif key in KEY_AI:
            if not cfg.MISTRAL_API_KEY:
                ui.notify("⚠ Clé Mistral manquante")
            elif ai.loading:
                ui.notify("⏳ IA déjà en cours…")
            elif (
                cv2.countNonZero(cv2.cvtColor(engine.get_canvas(), cv2.COLOR_BGR2GRAY))
                < 100
            ):
                ui.notify("⚠ Canvas vide !")
            else:
                ai.loading = True
                snap = engine.get_canvas().copy()
                threading.Thread(
                    target=_run_enhance,
                    args=(ai, snap, cfg.MISTRAL_API_KEY, cfg.CANVAS_W, cfg.CANVAS_H),
                    daemon=True,
                ).start()
                ui.notify("🤖 Enhance en cours…", 3.0)

        elif key in KEY_ARTWORK:
            if not cfg.MISTRAL_API_KEY:
                ui.notify("⚠ Clé Mistral manquante")
            elif ai.loading:
                ui.notify("⏳ IA déjà en cours…")
            elif (
                cv2.countNonZero(cv2.cvtColor(engine.get_canvas(), cv2.COLOR_BGR2GRAY))
                < 100
            ):
                ui.notify("⚠ Canvas vide !")
            else:
                ai.loading = True
                snap = engine.get_canvas().copy()
                threading.Thread(
                    target=_run_artwork,
                    args=(ai, snap, cfg.MISTRAL_API_KEY, "saves"),
                    daemon=True,
                ).start()
                ui.notify("🎨 Artwork en cours…", 3.0)

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    print("\n[GestureDraw v3] Au revoir !")


if __name__ == "__main__":
    main()
