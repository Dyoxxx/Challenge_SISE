

import json, base64, urllib.request, urllib.parse, urllib.error
import time, cv2, numpy as np


# ════════════════════════════════════════════════════════════
#  UTILITAIRES
# ════════════════════════════════════════════════════════════
def _post_json(url, payload, headers, timeout=60):
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _download_image(url: str, timeout=60) -> np.ndarray:
    req = urllib.request.Request(url, headers={"User-Agent": "GestureDraw/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError("Image invalide reçue")
    return img


def _placeholder(msg="Service indisponible", w=512, h=512) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        t = i / h
        img[i] = np.clip(
            np.array([15,10,25])*(1-t) + np.array([35,15,55])*t,
            0, 255).astype(np.uint8)
    cv2.rectangle(img, (30, h//2-60), (w-30, h//2+60), (70,40,120), 1)
    cv2.putText(img, "Generation impossible", (50, h//2-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,130,255), 1, cv2.LINE_AA)
    cv2.putText(img, msg[:30], (50, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120,90,180), 1, cv2.LINE_AA)
    cv2.putText(img, "Verifiez votre cle API", (50, h//2+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90,70,130), 1, cv2.LINE_AA)
    cv2.putText(img, "ou reessayez dans 1 min", (50, h//2+52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70,55,100), 1, cv2.LINE_AA)
    return img


# ════════════════════════════════════════════════════════════
#  TOGETHER AI  — FLUX.1-schnell (GRATUIT avec clé)
#  Clé gratuite sur : api.together.xyz
# ════════════════════════════════════════════════════════════
class TogetherGenerator:
    NAME  = "Together AI (FLUX)"
    MODEL = "black-forest-labs/FLUX.1-schnell-Free"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(self, prompt: str, **kw) -> np.ndarray:
        print(f"[Together] Génération FLUX.1-schnell...")
        payload = {
            "model":  self.MODEL,
            "prompt": prompt[:1000],
            "width":  512,
            "height": 512,
            "steps":  4,   # FLUX schnell = rapide en 4 steps
            "n":      1,
            "response_format": "b64_json",
        }
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        resp = _post_json(
            "https://api.together.xyz/v1/images/generations",
            payload, headers, timeout=60)
        b64 = resp["data"][0]["b64_json"]
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Image Together invalide")
        print(f"[Together] ✓ {img.shape[1]}x{img.shape[0]}")
        return img


# ════════════════════════════════════════════════════════════
#  HUGGING FACE  — FLUX.1-schnell (GRATUIT avec token)
#  Token gratuit sur : huggingface.co/settings/tokens
# ════════════════════════════════════════════════════════════
class HuggingFaceGenerator:
    """
    HuggingFace Inference API (nouvel endpoint router.huggingface.co)
    Gratuit avec token — token sur huggingface.co → Settings → Access Tokens
    """
    NAME = "HuggingFace"

    # Modèles sur le nouvel endpoint (router.huggingface.co)
    MODELS = [
        "black-forest-labs/FLUX.1-schnell",
        "stabilityai/stable-diffusion-3-medium-diffusers",
        "stabilityai/stable-diffusion-xl-base-1.0",
    ]

    BASE = "https://router.huggingface.co/hf-inference/models"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _try_model(self, model: str, prompt: str) -> np.ndarray:
        url     = f"{self.BASE}/{model}"
        payload = {"inputs": prompt[:400]}
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "x-wait-for-model": "true",
        }
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(url, data=data, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=90) as r:
            raw = r.read()

        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None and img.size > 0:
            return img
        raise ValueError(f"{model}: réponse non-image")

    def generate(self, prompt: str, **kw) -> np.ndarray:
        for model in self.MODELS:
            name = model.split("/")[1]
            try:
                print(f"[HuggingFace] Essai : {name}...")
                img = self._try_model(model, prompt)
                print(f"[HuggingFace] ✓ {name} — {img.shape[1]}x{img.shape[0]}")
                return img
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="ignore")[:100]
                print(f"[HuggingFace] {name} → HTTP {e.code}: {body}")
                if e.code in (401, 403):
                    raise ConnectionError(f"HuggingFace token invalide ({e.code})")
            except Exception as e:
                print(f"[HuggingFace] {name} → {e}")
        raise ConnectionError("HuggingFace: aucun modèle disponible")


# ════════════════════════════════════════════════════════════
#  POLLINATIONS AI  (100% gratuit, sans clé, sans limite)
#  Fallback principal quand Mistral est en rate limit
# ════════════════════════════════════════════════════════════
class PollinationsGenerator:
    NAME  = "Pollinations FLUX"
    MODEL = "flux"

    # Deux endpoints alternatifs pour le fallback
    ENDPOINTS = [
        "https://image.pollinations.ai/prompt/{encoded}?width=768&height=768&model=flux&seed={seed}&nologo=true&enhance=true",
        "https://image.pollinations.ai/prompt/{encoded}?width=512&height=512&model=flux-realism&seed={seed}&nologo=true",
    ]

    def generate(self, prompt: str, **kw) -> np.ndarray:
        seed    = int(time.time()) % 99999
        # Prompt court et efficace pour Pollinations
        clean   = prompt.strip()[:350]
        encoded = urllib.parse.quote(clean)

        for ep_idx, ep_template in enumerate(self.ENDPOINTS):
            url = ep_template.format(encoded=encoded, seed=seed)
            print(f"[Pollinations] Endpoint {ep_idx+1}/{len(self.ENDPOINTS)}...")
            try:
                img = _download_image(url, timeout=45)
                if img is not None and img.size > 0:
                    print(f"[Pollinations] OK {img.shape[1]}x{img.shape[0]}")
                    return img
            except urllib.error.HTTPError as e:
                if e.code == 530:
                    print(f"[Pollinations] Serveur surchargé (530) — essai endpoint suivant")
                    time.sleep(3)
                else:
                    print(f"[Pollinations] HTTP {e.code}")
            except Exception as e:
                print(f"[Pollinations] {type(e).__name__}: {e}")
                time.sleep(2)

        raise ConnectionError("Pollinations indisponible sur tous les endpoints")


# ════════════════════════════════════════════════════════════
#  MISTRAL AGENTS API — FLUX1.1 Pro Ultra (Black Forest Labs)
#  Même clé MISTRAL_API_KEY — rien à configurer en plus !
# ════════════════════════════════════════════════════════════
class MistralImageGenerator:
    NAME  = "Mistral FLUX1.1 Pro"
    MODEL = "mistral-medium-2505"

    def __init__(self, api_key: str):
        self.api_key   = api_key
        self._agent_id = None

    def _get_agent(self) -> str:
        if self._agent_id:
            return self._agent_id
        print("[Mistral] Creation agent image_generation...")
        payload = {
            "model":        self.MODEL,
            "name":         "GestureDrawImageAgent",
            "description":  "Genere des images depuis des prompts",
            "instructions": "Use the image generation tool to create the requested image.",
            "tools":        [{"type": "image_generation"}],
            "completion_args": {"temperature": 0.3}
        }
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        resp = _post_json("https://api.mistral.ai/v1/agents",
                          payload, headers, timeout=30)
        self._agent_id = resp["id"]
        print(f"[Mistral] Agent : {self._agent_id}")
        return self._agent_id

    def generate(self, prompt: str, **kw) -> np.ndarray:
        clean = prompt.strip()[:400]
        print(f"[Mistral FLUX1.1] Generation — {clean[:55]}...")
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            agent_id = self._get_agent()
            payload  = {
                "agent_id": agent_id,
                "inputs":   f"Generate this image: {clean}"
            }
            resp = _post_json("https://api.mistral.ai/v1/conversations",
                              payload, headers, timeout=90)

            # Cherche le file_id dans les outputs
            file_id = None
            for output in resp.get("outputs", []):
                for chunk in output.get("content", []):
                    if chunk.get("type") == "tool_file":
                        file_id = chunk.get("file_id")
                        break
                if file_id:
                    break

            if not file_id:
                raise ValueError("Mistral: aucun file_id dans la reponse")

            print(f"[Mistral] Telechargement file_id={file_id}...")
            req = urllib.request.Request(
                f"https://api.mistral.ai/v1/files/{file_id}/content",
                headers={"Authorization": f"Bearer {self.api_key}"})
            with urllib.request.urlopen(req, timeout=30) as r:
                img_bytes = r.read()

            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                raise ValueError("Mistral: image invalide")
            print(f"[Mistral FLUX1.1] OK {img.shape[1]}x{img.shape[0]}")
            return img

        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")
            print(f"[Mistral FLUX1.1] Erreur {e.code}: {body[:80]}")
            self._agent_id = None
            # Jamais de retry ici — le SmartGenerator gère le cooldown
            if e.code == 429:
                raise ConnectionError("Mistral rate limit 429")
            elif e.code in (401, 403):
                raise ConnectionError(f"Mistral cle invalide {e.code}")
            raise ValueError(f"Mistral HTTP {e.code}")


# ════════════════════════════════════════════════════════════
#  GEMINI IMAGEN  (Google) — génération via gemini-2.0-flash
#  Clé gratuite sur : aistudio.google.com
# ════════════════════════════════════════════════════════════
class GeminiImagenGenerator:
    NAME  = "Gemini Imagen (Google)"
    MODEL = "gemini-2.0-flash-preview-image-generation"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(self, prompt: str, **kw) -> np.ndarray:
        # Prompt court et simple — Gemini est strict sur le contenu
        clean = prompt.strip()[:400]
        clean = " ".join(clean.split())
        print(f"[Gemini] Génération (prompt: {clean[:60]}...)")

        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.MODEL}:generateContent?key={self.api_key}")
        payload = {
            "contents": [{"parts": [{"text": clean}]}],
            "generationConfig": {"responseModalities": ["IMAGE"]}
        }
        headers = {"Content-Type": "application/json"}
        try:
            resp = _post_json(url, payload, headers, timeout=60)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")
            print(f"[Gemini] Erreur {e.code}: {body[:300]}")
            # Si modèle preview indisponible, essaie imagen-3.0
            if e.code == 400:
                print("[Gemini] Essai avec imagen-3.0-generate-002...")
                url2 = ("https://generativelanguage.googleapis.com/v1beta/models/"
                        f"imagen-3.0-generate-002:predict?key={self.api_key}")
                payload2 = {
                    "instances": [{"prompt": clean}],
                    "parameters": {"sampleCount": 1, "aspectRatio": "1:1"}
                }
                try:
                    resp2 = _post_json(url2, payload2, headers, timeout=60)
                    preds = resp2.get("predictions", [])
                    if preds:
                        b64 = preds[0].get("bytesBase64Encoded", "")
                        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            print(f"[Gemini] ✓ Imagen-3 {img.shape[1]}x{img.shape[0]}")
                            return img
                except urllib.error.HTTPError as e2:
                    body2 = e2.read().decode(errors="ignore")
                    print(f"[Gemini] Imagen-3 erreur {e2.code}: {body2[:200]}")
                    raise ConnectionError(f"Gemini Imagen-3 {e2.code}: {body2[:100]}")
            raise

        # Extrait image de la réponse gemini flash
        parts = resp.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        for part in parts:
            if "inlineData" in part:
                b64 = part["inlineData"]["data"]
                b64 += "==" * ((-len(b64)) % 4)
                arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None and img.size > 0:
                    print(f"[Gemini] ✓ {img.shape[1]}x{img.shape[0]}")
                    return img

        raise ValueError("Gemini : aucune image dans la réponse")


# ════════════════════════════════════════════════════════════
#  DALL-E 3  (OpenAI)
# ════════════════════════════════════════════════════════════
class DALLEGenerator:
    NAME  = "DALL-E 3"
    MODEL = "dall-e-3"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(self, prompt: str, size="1024x1024", **kw) -> np.ndarray:
        # Nettoie et raccourcit le prompt pour éviter les erreurs 400
        clean = prompt.strip()[:900]
        # Supprime les termes potentiellement bloqués par le filtre DALL-E
        for term in ["realistic depiction", "highly detailed", "nude", "blood",
                     "gore", "violent", "majestic stag", "dead", "weapon"]:
            clean = clean.replace(term, "")
        clean = " ".join(clean.split())  # normalise les espaces

        print(f"[DALL-E 3] Génération (prompt: {clean[:60]}...)")
        payload = {
            "model":   self.MODEL,
            "prompt":  clean,
            "n": 1, "size": "1024x1024", "quality": "standard",
        }
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            resp = _post_json(
                "https://api.openai.com/v1/images/generations",
                payload, headers, timeout=60)
            url = resp["data"][0]["url"]
            img = _download_image(url)
            print(f"[DALL-E 3] ✓ {img.shape[1]}x{img.shape[0]}")
            return img
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")
            print(f"[DALL-E 3] Erreur {e.code}: {body[:200]}")
            raise


# ════════════════════════════════════════════════════════════
#  STABILITY AI
# ════════════════════════════════════════════════════════════
class StabilityGenerator:
    NAME  = "Stability AI"
    MODEL = "stable-diffusion-xl-1024-v1-0"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(self, prompt: str, negative="blurry, bad quality", **kw) -> np.ndarray:
        print(f"[Stability] Génération...")
        payload = {
            "text_prompts": [
                {"text": prompt,   "weight": 1.0},
                {"text": negative, "weight": -1.0},
            ],
            "cfg_scale": 7, "height": 512, "width": 512,
            "samples": 1, "steps": 30,
        }
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept":        "application/json",
        }
        resp = _post_json(
            f"https://api.stability.ai/v1/generation/{self.MODEL}/text-to-image",
            payload, headers, timeout=90)
        b64 = resp["artifacts"][0]["base64"]
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        print(f"[Stability] ✓")
        return img


# ════════════════════════════════════════════════════════════
#  STABLE DIFFUSION LOCAL
# ════════════════════════════════════════════════════════════
class LocalSDGenerator:
    NAME  = "Stable Diffusion (local)"
    MODEL = "local"

    def __init__(self, host="http://localhost:7860"):
        self.host = host

    def generate(self, prompt: str, negative="blurry, bad quality", steps=25, **kw) -> np.ndarray:
        payload = {
            "prompt": prompt, "negative_prompt": negative,
            "steps": steps, "width": 512, "height": 512,
            "cfg_scale": 7.5, "sampler_name": "DPM++ 2M Karras",
        }
        resp = _post_json(f"{self.host}/sdapi/v1/txt2img",
                          {"Content-Type": "application/json"}, {}, timeout=180)
        b64 = resp["images"][0]
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ════════════════════════════════════════════════════════════
#  CRAIYON  (GRATUIT — AUCUNE CLÉ, AUCUN COMPTE)
#  Anciennement DALL-E mini — stable et totalement libre
# ════════════════════════════════════════════════════════════
class CraiyonGenerator:
    NAME  = "Craiyon (gratuit, sans cle)"
    MODEL = "photo"  # art | drawing | photo | none

    BACKENDS = [
        "https://backend.craiyon.com/generate",
        "https://backend-j52LittGmA-uc.a.run.app/generate",
    ]

    def generate(self, prompt: str, **kw) -> np.ndarray:
        payload = json.dumps({
            "prompt":          prompt[:500],
            "negative_prompt": "blurry, bad quality, distorted, sketch, rough",
            "model":           self.MODEL,
            "token":           None,
            "version":         "35s5hfwn9n78gb06",
            "orientations":    "square",
        }).encode()

        headers = {
            "Content-Type": "application/json",
            "User-Agent":   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept":       "application/json",
            "Origin":       "https://www.craiyon.com",
            "Referer":      "https://www.craiyon.com/",
        }

        for backend in self.BACKENDS:
            for attempt in range(2):
                try:
                    print(f"[Craiyon] {backend.split('/')[2]} — tentative {attempt+1}/2...")
                    req = urllib.request.Request(
                        backend, data=payload, headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=120) as r:
                        resp = json.loads(r.read())

                    images = resp.get("images", [])
                    if not images:
                        raise ValueError("Aucune image retournée")

                    # Prend la première image (meilleure qualité généralement)
                    b64 = images[0]
                    # Craiyon retourne parfois avec ou sans padding
                    b64 += "==" * ((-len(b64)) % 4)
                    arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None and img.size > 0:
                        print(f"[Craiyon] ✓ Image {img.shape[1]}x{img.shape[0]}")
                        return img

                except urllib.error.HTTPError as e:
                    print(f"[Craiyon] HTTP {e.code} — attente 5s...")
                    time.sleep(5)
                except Exception as e:
                    print(f"[Craiyon] Erreur : {e}")
                    time.sleep(3)

        raise ConnectionError("Craiyon indisponible")


# ════════════════════════════════════════════════════════════
#  SMART GENERATOR — fallback automatique
# ════════════════════════════════════════════════════════════
class SmartGenerator:
    """
    Essaie les providers dans l'ordre jusqu'à succès.
    - Cooldown automatique après un 429 (rate limit)
    - Jamais de retry infini — 1 essai par provider max
    """
    def __init__(self, generators: list):
        self.generators = generators
        self._cooldowns = {}   # {name: timestamp_disponible}
        names = " → ".join(g.NAME for g in generators)
        print(f"[SmartGenerator] Chaîne : {names}")

    @property
    def NAME(self):
        return self.generators[0].NAME if self.generators else "?"

    @property
    def MODEL(self):
        return "auto-fallback"

    def _available(self, name: str) -> bool:
        until = self._cooldowns.get(name, 0)
        if time.time() < until:
            print(f"[SmartGenerator] {name} cooldown ({int(until-time.time())}s) — skip")
            return False
        return True

    def _cooldown(self, name: str, seconds: int):
        self._cooldowns[name] = time.time() + seconds
        print(f"[SmartGenerator] {name} mis en pause {seconds}s")

    def generate(self, prompt: str, **kw) -> np.ndarray:
        errors = []
        for gen in self.generators:
            if not self._available(gen.NAME):
                continue
            try:
                print(f"[SmartGenerator] Essai : {gen.NAME}")
                img = gen.generate(prompt, **kw)
                if img is not None and img.size > 0:
                    return img
            except Exception as e:
                msg = str(e)
                print(f"[SmartGenerator] {gen.NAME} échoué : {msg[:70]}")
                errors.append(msg)
                if "429" in msg or "rate limit" in msg.lower():
                    self._cooldown(gen.NAME, 70)
                elif any(k in msg.lower() for k in ["billing","expired","invalid","401","403"]):
                    self._cooldown(gen.NAME, 3600)

        print(f"[SmartGenerator] Tous providers échoués → placeholder")
        return _placeholder("Reessayez dans 1 minute")


# ════════════════════════════════════════════════════════════
#  FACTORY
# ════════════════════════════════════════════════════════════
def create_generator(name: str, api_key: str = ""):
    name = name.lower()
    if name in ("pollinations", ""):
        return PollinationsGenerator()
    elif name == "together":
        if not api_key: raise ValueError("Clé Together AI requise")
        return TogetherGenerator(api_key)
    elif name in ("huggingface", "hf"):
        if not api_key: raise ValueError("Token HuggingFace requis")
        return HuggingFaceGenerator(api_key)
    elif name in ("dalle", "dall-e", "openai"):
        if not api_key: raise ValueError("Clé OpenAI requise")
        return DALLEGenerator(api_key)
    elif name == "stability":
        if not api_key: raise ValueError("Clé Stability requise")
        return StabilityGenerator(api_key)
    elif name in ("local", "sd"):
        return LocalSDGenerator()
    else:
        raise ValueError(f"Générateur inconnu : {name}")


def create_smart_generator(env: dict) -> SmartGenerator:
    """
    Crée automatiquement la meilleure chaîne de fallback
    selon les clés disponibles dans env (os.environ).
    Ordre : DALL-E 3 → Gemini Imagen → Together → HuggingFace
            → Stability → Craiyon → Pollinations
    """
    chain = []

    # 1. HuggingFace — PRIORITÉ si token dispo
    #    Gratuit, sans limite quotidienne, token créé en 2 min sur huggingface.co
    k = env.get("HUGGINGFACE_TOKEN", "") or env.get("HF_TOKEN", "")
    if k:
        chain.append(HuggingFaceGenerator(k))
        print(f"[SmartGenerator] + HuggingFace (SD-XL, gratuit illimité)")

    # 2. Mistral FLUX1.1 — uniquement si MISTRAL_IMAGEN=1 (nécessite plan Agents payant)
    #    Par défaut désactivé car API /v1/agents donne 401 sur plan standard
    k = env.get("MISTRAL_API_KEY", "")
    if k and env.get("MISTRAL_IMAGEN", "") == "1":
        chain.append(MistralImageGenerator(k))
        print(f"[SmartGenerator] + Mistral FLUX1.1 Pro (Agents API)")

    # 3. DALL-E 3 si disponible
    k = env.get("OPENAI_API_KEY", "")
    if k:
        chain.append(DALLEGenerator(k))
        print(f"[SmartGenerator] + DALL-E 3 (OpenAI)")

    # 4. Pollinations — gratuit sans clé
    chain.append(PollinationsGenerator())
    print(f"[SmartGenerator] + Pollinations FLUX (sans clé)")

    # 5. Craiyon — dernier recours
    chain.append(CraiyonGenerator())
    print(f"[SmartGenerator] + Craiyon (dernier recours)")

    return SmartGenerator(chain)
