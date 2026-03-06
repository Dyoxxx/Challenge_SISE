

import threading
import time
import base64
import cv2
import numpy as np


COOLDOWN = 5.0   # secondes entre deux générations (API payante)


class ImageCompleter:
    """
    Usage :
        completer = ImageCompleter(llm_provider, image_generator)
        completer.start()
        completer.submit(canvas_bgr)
        result = completer.get_result()
        # result = {
        #   'image':     np.ndarray BGR,
        #   'prompt':    str,
        #   'name_fr':   str,
        #   'style':     str,
        #   'emoji':     str,
        #   'duration':  float,
        #   'generator': str,
        # }
    """

    def __init__(self, llm_provider, image_generator):
        self._llm       = llm_provider
        self._gen       = image_generator
        self._lock      = threading.Lock()
        self._canvas_in = None
        self._result    = None
        self._status    = "idle"   # idle | analyzing | generating | done | error
        self._running    = True
        self._event      = threading.Event()
        self._last_gen   = 0.0
        self._ai_context = None
        self._thread    = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def submit(self, canvas_bgr: np.ndarray, ai_result: dict = None):
        """
        Soumet un dessin pour génération.
        Si ai_result est fourni, bypasse l'analyse LLM et utilise
        directement le nom/tags déjà détectés par AIAnalyzer.
        """
        now = time.time()
        if now - self._last_gen < COOLDOWN:
            remaining = COOLDOWN - (now - self._last_gen)
            print(f"[Completer] Attendre {remaining:.0f}s avant la prochaine génération")
            return False
        with self._lock:
            self._canvas_in  = canvas_bgr.copy()
            self._ai_context = ai_result   # résultat AIAnalyzer à réutiliser
            self._result     = None
        self._event.set()
        return True

    def get_result(self):
        with self._lock:
            return self._result

    def get_status(self):
        with self._lock:
            return self._status

    def clear(self):
        with self._lock:
            self._result = None
            self._status = "idle"

    def stop(self):
        self._running = False
        self._event.set()

    # ════════════════════════════════════════════════════════
    def _run(self):
        while self._running:
            self._event.wait(timeout=1.0)
            self._event.clear()
            with self._lock:
                canvas     = self._canvas_in
                ai_context = self._ai_context
                self._canvas_in  = None
                self._ai_context = None
            if canvas is None:
                continue

            self._last_gen = time.time()
            t0 = time.time()

            try:
                # ── Étape 1 : prépare l'image ─────────────────
                b64 = self._prepare(canvas)

                # ── Étape 2 : LLM analyse → prompt ───────────
                self._set_status("analyzing")
                print(f"[Completer] Analyse du dessin par {self._llm.NAME}...")
                _hint = ""
                if ai_context:
                    _rec = ai_context.get("tags", [])
                    _hint = _rec[0] if _rec else ai_context.get("name", "")

                # Fallback prompt si le LLM est indisponible (401, timeout…)
                prompt   = "a hand-drawn sketch, artistic, colorful, high quality"
                negative = "blurry, low quality, distorted, sketch lines"
                name_fr  = "dessin"
                style    = "illustration"
                emoji    = "🎨"

                try:
                    meta     = self._llm.analyze_for_generation(b64, subject_hint=_hint)
                    prompt   = meta.get("prompt",   prompt)
                    negative = meta.get("negative", negative)
                    name_fr  = meta.get("name_fr",  name_fr)
                    style    = meta.get("style",    style)
                    emoji    = meta.get("emoji",    emoji)
                except Exception as llm_err:
                    print(f"[Completer] ⚠ LLM analyse échouée ({llm_err}) — prompt générique utilisé")

              
                if ai_context:
                    rec_name = ai_context.get("name", "")
                    rec_conf = ai_context.get("confidence", 0)
                    rec_tags = ai_context.get("tags", [])  # mots anglais
                    if rec_name and rec_conf >= 50:
                        # Utilise le premier tag anglais comme sujet principal
                        subject_en = rec_tags[0] if rec_tags else rec_name
                        # Construit un nouveau prompt centré sur le sujet reconnu
                        prompt = f"{subject_en}, {prompt}"[:450]
                        name_fr = rec_name
                        print(f"[Completer] Sujet forcé : '{subject_en}' ({rec_name} à {rec_conf}%)")

                print(f"[Completer] Prompt: {prompt[:60]}...")

                # ── Étape 3 : génère l'image ──────────────────
                self._set_status("generating")
                print(f"[Completer] Génération via {self._gen.NAME}...")
                gen_image = self._gen.generate(prompt=prompt, negative=negative)

                if gen_image is None:
                    raise ValueError("Image générée est None")

                # Redimensionne pour l'affichage
                gen_image = cv2.resize(gen_image, (512, 512))

                duration = time.time() - t0
                print(f"[Completer] ✓ Généré en {duration:.1f}s — {name_fr}")

                with self._lock:
                    self._result = {
                        "image":     gen_image,
                        "prompt":    prompt,
                        "name_fr":   name_fr,
                        "style":     style,
                        "emoji":     emoji,
                        "duration":  duration,
                        "generator": self._gen.NAME,
                    }
                    self._status = "done"

            except Exception as e:
                print(f"[Completer] Erreur : {e}")
                with self._lock:
                    self._result = {
                        "image":     None,
                        "prompt":    "",
                        "name_fr":   "Erreur",
                        "style":     "",
                        "emoji":     "⚠",
                        "duration":  time.time() - t0,
                        "generator": getattr(self._gen, 'NAME', '?'),
                        "error":     str(e),
                    }
                    self._status = "error"

    def _set_status(self, s):
        with self._lock:
            self._status = s

    def _prepare(self, canvas: np.ndarray) -> str:
        """Fond sombre + redimension + base64 JPEG."""
        bg = np.full_like(canvas, 20)
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        result = bg.copy()
        result[mask > 0] = canvas[mask > 0]
        h, w = result.shape[:2]
        if w > 512:
            result = cv2.resize(result, (512, int(h * 512 / w)))
        _, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return base64.b64encode(buf).decode()
