"""
main.py — GestureDraw v3 + Génération IA

Touches :
  I  → Enhance Mistral (remplace le canvas, annulable avec Z)
  A  → Artwork IA      (génère une vraie image dans une fenêtre dédiée)
  C / Z / S / M / Q    → commandes habituelles
  1-6 / & é " ' ( -   → couleurs QWERTY + AZERTY
"""

import cv2, numpy as np, time, os, threading
from collections import deque

from config import Config
from hand_tracker import create_tracker
from gesture import GestureDetector
from drawing import DrawingEngine
from ui import UIManager
from shape_detector import ShapeDetector
from mistral_enhance import call_mistral, render_shapes
from fonts import report as fonts_report

# Génération IA (pipeline v6)
from llm_providers import create_provider
from image_generator import create_smart_generator
from image_completer import ImageCompleter
from image_window import ImageWindowManager


# ════════════════════════════════════════════════════════════
#  Clavier QWERTY + AZERTY + majuscules
# ════════════════════════════════════════════════════════════
def _k(*chars):
    out = []
    for c in chars: out += [ord(c.lower()), ord(c.upper())]
    return tuple(out)

KEY_QUIT    = _k("q") + (27,)
KEY_CLEAR   = _k("c")
KEY_UNDO    = _k("z")
KEY_SAVE    = _k("s")
KEY_MODE    = _k("m")
KEY_ENHANCE = _k("i")       # I → Mistral Enhance (remplace canvas)
KEY_ARTWORK = _k("a")       # A → Artwork IA (fenêtre dédiée)
KEY_UP      = (ord("+"), ord("="), 43)
KEY_DN      = (ord(")"), ord("°"), 95)
KEY_OP_DN   = (ord("ù"), ord("%"))
KEY_OP_UP   = (ord("*"), ord("µ"))

COLOR_KEYS = {
    ord("1"): 0,  38: 0,
    ord("2"): 1, 233: 1,
    ord("3"): 2,  34: 2,
    ord("4"): 3,  39: 3,
    ord("5"): 4,  40: 4,
    ord("6"): 5,  45: 5,
}


# ════════════════════════════════════════════════════════════
#  Thread Enhance (Mistral → formes géométriques)
# ════════════════════════════════════════════════════════════
class EnhanceState:
    def __init__(self): self.loading = False; self.result = None; self.error = None

def _run_enhance(state, snap, key, W, H):
    try:
        data = call_mistral(snap, key)
        enhanced = render_shapes(data, W, H)
        state.result = (enhanced, data.get("subject", ""))
    except Exception as e:
        state.error = str(e)[:100]
    finally:
        state.loading = False


# ════════════════════════════════════════════════════════════
def main():
    cfg    = Config()
    engine = DrawingEngine(cfg)
    gdet   = GestureDetector()
    ui     = UIManager(cfg)
    shapes = ShapeDetector()
    enh    = EnhanceState()

    # ── Affichage startup ────────────────────────────────────
    print("=" * 62)
    print(f"  ✦ GestureDraw  —  rendu {cfg.WIN_W}×{cfg.WIN_H}")
    print(f"  (forcer: GESTURE_W=1280 GESTURE_H=720 python main.py)")
    print("=" * 62)
    fonts_report()
    print()

    # ── Provider LLM pour Enhance (Mistral Pixtral) ──────────
    if cfg.MISTRAL_API_KEY:
        print(f"  ✓ Mistral API  →  I=Enhance activé")
    else:
        print(f"  ⚠ MISTRAL_API_KEY manquante  →  I=Enhance désactivé")

    # ── Pipeline génération image (Artwork) ──────────────────
    completer = None
    img_win   = None
    _llm_for_gen = None

    # Choisit le meilleur LLM dispo pour analyser le dessin
    for pname, key in [
        ("mistral",   cfg.MISTRAL_API_KEY),
        ("anthropic", cfg.ANTHROPIC_API_KEY),
        ("openai",    cfg.OPENAI_API_KEY),
        ("gemini",    cfg.GEMINI_API_KEY),
    ]:
        if key:
            try:
                _llm_for_gen = create_provider(pname, key)
                print(f"  ✓ LLM génération : {_llm_for_gen.NAME}")
                break
            except Exception as e:
                print(f"  ⚠ {pname} : {e}")

    if _llm_for_gen:
        try:
            img_gen   = create_smart_generator(os.environ)
            completer = ImageCompleter(
                llm_provider    = _llm_for_gen,
                image_generator = img_gen,
            ).start()
            img_win = ImageWindowManager()
            print(f"  ✓ Artwork IA activé  →  A=Générer")
        except Exception as e:
            print(f"  ⚠ Artwork désactivé : {e}")
    else:
        print(f"  ⚠ Aucun LLM configuré  →  A=Artwork désactivé")
        print(f"     Ajoutez MISTRAL_API_KEY (ou autre) dans .env")

    print("=" * 62 + "\n")

    # ── Webcam ───────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("[ERR] Webcam introuvable"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    tracker = create_tracker()

    prev_x = prev_y = None
    drawing = False; shape_path = []
    fps_times = deque(maxlen=30); last_undo = 0.0
    shape_hint = ""; shape_hint_t = 0.0
    cam_frame = None
    gen_result = None; gen_status = "idle"
    last_gen_submit = 0.0

    WIN = "✦ GestureDraw"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, cfg.WIN_W, cfg.WIN_H)
    ui.notify("Montrez votre main à la caméra !", 4.0)

    # ════════════════════════════════════════════════════════
    while True:
        ret, raw = cap.read()
        if not ret: break
        frame = cv2.flip(raw, 1)

        fps_times.append(time.time())
        fps = (int((len(fps_times)-1) / (fps_times[-1]-fps_times[0]))
               if len(fps_times) >= 2 else 0)

        landmarks, cam_frame = tracker.process(frame)
        gesture = "none"; fx = fy = None

        if landmarks:
            ix = int(landmarks[8].x * cfg.CANVAS_W)
            iy = int(landmarks[8].y * cfg.CANVAS_H)
            fx, fy = ix, iy
            gesture = gdet.detect(landmarks)

            if gesture == "draw":
                if prev_x is not None:
                    if not drawing: engine.save_undo(); drawing = True
                    engine.draw_line(prev_x, prev_y, ix, iy)
                    if cfg.shape_mode: shape_path.append((ix, iy))
                prev_x, prev_y = ix, iy
            elif gesture == "pause":
                if drawing and cfg.shape_mode and len(shape_path) > 25:
                    det = shapes.detect(shape_path)
                    if det:
                        engine.save_undo(); engine.draw_perfect_shape(det)
                        shape_hint = det["label"]; shape_hint_t = time.time()+2.5
                        ui.notify(f"✓ {det['label']}")
                drawing = False; shape_path = []; prev_x = prev_y = None
            elif gesture == "erase":
                engine.erase(ix, iy); drawing=False; shape_path=[]; prev_x=prev_y=None
            elif gesture == "undo":
                now = time.time()
                if now-last_undo > 1.0: engine.undo(); last_undo=now; ui.notify("↩ Annulé")
                drawing=False; prev_x=prev_y=None
            elif gesture == "open_hand":
                now = time.time()
                if now-last_undo > 2.0:
                    engine.save_undo(); engine.clear(); last_undo=now; ui.notify("✕ Effacé")
                drawing=False; prev_x=prev_y=None
            else:
                drawing=False; prev_x=prev_y=None
        else:
            if drawing and cfg.shape_mode and len(shape_path) > 25:
                det = shapes.detect(shape_path)
                if det:
                    engine.save_undo(); engine.draw_perfect_shape(det)
                    shape_hint = det["label"]; shape_hint_t = time.time()+2.5
            drawing=False; shape_path=[]; prev_x=prev_y=None

        # ── Résultat Enhance ─────────────────────────────────
        if enh.result is not None:
            ecanvas, subj = enh.result; enh.result = None
            subj = str(subj) if not isinstance(subj, str) else subj
            engine.save_undo(); engine.apply_enhanced(ecanvas)
            ui.show_ai_subject(subj, 5.0); ui.notify(f"✨ {subj}")
        if enh.error:
            ui.notify(f"❌ Enhance : {enh.error}", 4.0); enh.error = None

        # ── Résultat Artwork (génération image) ──────────────
        if completer:
            new_result = completer.get_result()
            new_status = completer.get_status()
            if new_result and new_result is not gen_result:
                gen_result = new_result
                gen_status = new_status
                if not gen_result.get("error"):
                    name = gen_result.get("name_fr", "artwork")
                    ui.notify(f"🎨 {gen_result.get('emoji','')} {name}  — fenêtre ouverte", 5.0)
                    if img_win:
                        img_win.update(gen_result, engine.get_canvas(), "done")
                else:
                    ui.notify(f"❌ Artwork : {gen_result.get('error','?')[:60]}", 4.0)
            else:
                gen_status = new_status

        # Maintient la fenêtre artwork visible
        if img_win and gen_result:
            img_win.update(gen_result, engine.get_canvas(), gen_status)

        # ── Rendu principal ───────────────────────────────────
        ai_loading = enh.loading or (completer is not None and gen_status in ("analyzing","generating"))
        sh = shape_hint if time.time() < shape_hint_t else ""

        display = ui.compose(
            canvas=engine.get_canvas(), cam_frame=cam_frame,
            gesture=gesture, finger_pos=(fx, fy), fps=fps,
            shape_hint=sh,
            brush_color=cfg.color_rgb, brush_size=cfg.brush_size,
            brush_opacity=cfg.opacity, shape_mode=cfg.shape_mode,
            strokes=engine.stroke_count, ai_loading=ai_loading,
        )
        cv2.imshow(WIN, display)

        # ── Touches ──────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if img_win: img_win.handle_key(key)

        if   key in KEY_QUIT:  break
        elif key in KEY_CLEAR:
            engine.save_undo(); engine.clear(); ui.notify("✕ Canvas effacé")
        elif key in KEY_UNDO:  engine.undo(); ui.notify("↩ Annulé")
        elif key in KEY_SAVE:
            p = engine.save_png(); ui.notify(f"💾 {os.path.basename(p)}")
        elif key in KEY_MODE:
            cfg.shape_mode = not cfg.shape_mode
            ui.notify(f"Formes : {'ON ✓' if cfg.shape_mode else 'OFF'}")
        elif key in KEY_UP: cfg.brush_size = min(60, cfg.brush_size + 2)
        elif key in KEY_DN and key not in COLOR_KEYS:
            cfg.brush_size = max(1, cfg.brush_size - 2)
        elif key in KEY_OP_DN: cfg.opacity = max(0.0, cfg.opacity - 0.05)
        elif key in KEY_OP_UP: cfg.opacity = min(1.0, cfg.opacity + 0.05)
        elif key in COLOR_KEYS:
            cfg.set_color_by_index(COLOR_KEYS[key]); ui.notify(f"Couleur : {cfg.color_name}")

        # I → Mistral Enhance (remplace le canvas)
        elif key in KEY_ENHANCE:
            if not cfg.MISTRAL_API_KEY:
                ui.notify("⚠ MISTRAL_API_KEY manquante")
            elif enh.loading:
                ui.notify("⏳ Enhance déjà en cours…")
            elif cv2.countNonZero(cv2.cvtColor(engine.get_canvas(), cv2.COLOR_BGR2GRAY)) < 100:
                ui.notify("⚠ Canvas vide !")
            else:
                enh.loading = True
                snap = engine.get_canvas().copy()
                threading.Thread(
                    target=_run_enhance,
                    args=(enh, snap, cfg.MISTRAL_API_KEY, cfg.CANVAS_W, cfg.CANVAS_H),
                    daemon=True).start()
                ui.notify("🤖 Enhance en cours…", 3.0)

        # A → Artwork IA (génération vraie image, fenêtre dédiée)
        elif key in KEY_ARTWORK:
            if completer is None:
                ui.notify("⚠ Artwork IA non disponible — vérifiez .env")
            elif gen_status in ("analyzing", "generating"):
                ui.notify("⏳ Génération déjà en cours…")
            elif cv2.countNonZero(cv2.cvtColor(engine.get_canvas(), cv2.COLOR_BGR2GRAY)) < 100:
                ui.notify("⚠ Canvas vide !")
            else:
                ok = completer.submit(engine.get_canvas())
                if ok:
                    gen_result = None; gen_status = "analyzing"
                    last_gen_submit = time.time()
                    if img_win: img_win.close()
                    ui.notify("🎨 Analyse + génération en cours…", 3.0)
                else:
                    wait = 5.0 - (time.time() - last_gen_submit)
                    ui.notify(f"⏳ Attendez {wait:.0f}s avant la prochaine génération")

    # ── Nettoyage ─────────────────────────────────────────────
    if completer: completer.stop()
    if img_win:   img_win.close()
    cap.release(); tracker.close()
    cv2.destroyAllWindows()
    print("\n[GestureDraw] Au revoir !")


if __name__ == "__main__":
    main()
