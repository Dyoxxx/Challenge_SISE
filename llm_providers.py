

import json
import urllib.request
import urllib.error
import base64


# ════════════════════════════════════════════════════════════
#  PROMPT COMMUN
# ════════════════════════════════════════════════════════════
SYSTEM_PROMPT = (
    "Tu es un expert en reconnaissance de dessins faits à la main. "
    "Réponds UNIQUEMENT en JSON valide, sans markdown, sans explication."
)

USER_PROMPT = """\
Analyse ce dessin fait à la main et réponds UNIQUEMENT en JSON avec ces champs :
{
  "name": "nom court en français (ex: un chat, une maison, un soleil)",
  "description": "description poétique en 1 phrase courte",
  "confidence": nombre entre 0 et 100,
  "tags": ["mot_anglais1", "mot_anglais2", "mot_anglais3"],
  "emoji": "un emoji représentatif",
  "style": "géométrique|abstrait|figuratif|symbole|texte"
}
Si le dessin est trop vague, mets confidence < 40. Réponds UNIQUEMENT avec le JSON, rien d'autre."""


def _parse_response(text: str) -> dict:
    """Nettoie et parse le JSON retourné par n'importe quel LLM."""
    text = text.strip()
    # Retire les balises markdown si présentes
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    # Cherche le premier { ... }
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    return json.loads(text)


def _post(url: str, payload: dict, headers: dict, timeout=20) -> dict:
    """POST JSON → JSON."""
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ════════════════════════════════════════════════════════════
#  MISTRAL  (pixtral-12b-2409)
# ════════════════════════════════════════════════════════════
class MistralProvider:
    NAME  = "Mistral"
    MODEL = "pixtral-12b-2409"
    URL   = "https://api.mistral.ai/v1/chat/completions"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def analyze(self, image_b64: str) -> dict:
        payload = {
            "model": self.MODEL,
            "max_tokens": 400,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT + "\n\n" + USER_PROMPT
                        }
                    ]
                }
            ]
        }
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        resp = _post(self.URL, payload, headers)
        text = resp["choices"][0]["message"]["content"]
        return _parse_response(text)


# ════════════════════════════════════════════════════════════
#  ANTHROPIC  (claude-sonnet)
# ════════════════════════════════════════════════════════════
class AnthropicProvider:
    NAME  = "Claude"
    MODEL = "claude-sonnet-4-20250514"
    URL   = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def analyze(self, image_b64: str) -> dict:
        payload = {
            "model":      self.MODEL,
            "max_tokens": 400,
            "system":     SYSTEM_PROMPT,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/jpeg",
                            "data":       image_b64,
                        }
                    },
                    {"type": "text", "text": USER_PROMPT}
                ]
            }]
        }
        headers = {
            "Content-Type":      "application/json",
            "x-api-key":         self.api_key,
            "anthropic-version": "2023-06-01",
        }
        resp = _post(self.URL, payload, headers)
        text = resp["content"][0]["text"]
        return _parse_response(text)


# ════════════════════════════════════════════════════════════
#  OPENAI  (gpt-4o-mini)
# ════════════════════════════════════════════════════════════
class OpenAIProvider:
    NAME  = "OpenAI"
    MODEL = "gpt-4o-mini"
    URL   = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def analyze(self, image_b64: str) -> dict:
        payload = {
            "model":      self.MODEL,
            "max_tokens": 400,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":    f"data:image/jpeg;base64,{image_b64}",
                                "detail": "low"   # économise des tokens
                            }
                        },
                        {"type": "text", "text": USER_PROMPT}
                    ]
                }
            ]
        }
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        resp = _post(self.URL, payload, headers)
        text = resp["choices"][0]["message"]["content"]
        return _parse_response(text)


# ════════════════════════════════════════════════════════════
#  GOOGLE GEMINI  (gemini-1.5-flash — tier gratuit)
# ════════════════════════════════════════════════════════════
class GeminiProvider:
    NAME  = "Gemini"
    MODEL = "gemini-1.5-flash"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.URL = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.MODEL}:generateContent?key={api_key}"
        )

    def analyze(self, image_b64: str) -> dict:
        payload = {
            "contents": [{
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data":      image_b64
                        }
                    },
                    {
                        "text": SYSTEM_PROMPT + "\n\n" + USER_PROMPT
                    }
                ]
            }],
            "generationConfig": {
                "maxOutputTokens": 400,
                "temperature":     0.3,
            }
        }
        headers = {"Content-Type": "application/json"}
        resp = _post(self.URL, payload, headers)
        text = resp["candidates"][0]["content"]["parts"][0]["text"]
        return _parse_response(text)


# ════════════════════════════════════════════════════════════
#  OLLAMA  (llava — 100% local, gratuit)
# ════════════════════════════════════════════════════════════
class OllamaProvider:
    NAME  = "Ollama (local)"
    MODEL = "llava"
    URL   = "http://localhost:11434/api/generate"

    def __init__(self, model: str = "llava"):
        self.MODEL = model

    def analyze(self, image_b64: str) -> dict:
        payload = {
            "model":  self.MODEL,
            "prompt": SYSTEM_PROMPT + "\n\n" + USER_PROMPT,
            "images": [image_b64],
            "stream": False,
            "options": {"temperature": 0.3}
        }
        headers = {"Content-Type": "application/json"}
        resp = _post(self.URL, payload, headers, timeout=60)
        text = resp.get("response", "")
        return _parse_response(text)


# ════════════════════════════════════════════════════════════
#  FACTORY
# ════════════════════════════════════════════════════════════
PROVIDERS = {
    "mistral":   MistralProvider,
    "anthropic": AnthropicProvider,
    "openai":    OpenAIProvider,
    "gemini":    GeminiProvider,
    "ollama":    OllamaProvider,
}

def create_provider(name: str, api_key: str = ""):
    """
    Crée un provider par son nom.
    name    : 'mistral' | 'anthropic' | 'openai' | 'gemini' | 'ollama'
    api_key : clé API (non nécessaire pour ollama)
    """
    name = name.lower()
    if name not in PROVIDERS:
        raise ValueError(f"Provider inconnu : {name}. Choix : {list(PROVIDERS)}")
    cls = PROVIDERS[name]
    if name == "ollama":
        return cls(api_key or "llava")   # api_key = nom du modèle pour Ollama
    if not api_key:
        raise ValueError(f"Clé API requise pour {name}")
    return cls(api_key)


# ════════════════════════════════════════════════════════════
#  MÉTHODE analyze_for_generation — ajoutée à chaque provider
#  Retourne un prompt optimisé pour la génération d'image
# ════════════════════════════════════════════════════════════

_GEN_SYSTEM = (
    "Tu es un expert en prompt engineering pour génération d'images IA. "
    "Réponds UNIQUEMENT en JSON valide, sans markdown ni explication."
)

_GEN_USER = """\
Analyse ce dessin fait à la main et génère un prompt court pour créer une version
réaliste de ce dessin.

Réponds en JSON avec exactement ces champs :
{
  "name_fr": "nom du dessin en français (3 mots max)",
  "prompt": "prompt en anglais, 15-25 mots MAX. Format: [sujet], [style], [qualité]. \
Exemple: a cat sitting, digital art, vibrant colors, high quality",
  "negative": "blurry, low quality, distorted, sketch, rough lines",
  "style": "realistic|illustration|cartoon|watercolor",
  "emoji": "emoji représentatif"
}

IMPORTANT: Le prompt doit être court (15-25 mots), simple et sans termes violents."""


def _build_gen_user(subject_hint: str = "") -> str:
    if subject_hint:
        hint_line = "\nIMPORTANT: Le sujet du dessin EST '" + subject_hint + "'. Ton prompt DOIT commencer par ce sujet."
    else:
        hint_line = ""
    return (
        "Analyse ce dessin et genere un prompt pour creer une version realiste." + hint_line + "\n\n"
        "Reponds en JSON avec ces champs:\n"
        "{\n"
        '  "name_fr": "nom du dessin en francais (3 mots max)",\n'
        '  "prompt": "prompt anglais 15-25 mots: [sujet precis], [style], [qualite]",\n'
        '  "negative": "blurry, low quality, distorted, sketch",\n'
        '  "style": "realistic|illustration|cartoon|watercolor",\n'
        '  "emoji": "emoji representatif"\n'
        "}\n\n"
        "IMPORTANT: 15-25 mots max dans le prompt."
    )


def _add_gen_method(cls, url_key, model_key, key_header, extra_headers=None):
    """Injecte analyze_for_generation dans un provider existant."""
    def analyze_for_generation(self, image_b64: str, subject_hint: str = "") -> dict:
        import json, urllib.request

        if cls == GeminiProvider:
            payload = {
                "contents": [{
                    "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                        {"text": _GEN_SYSTEM + "\n\n" + _build_gen_user(subject_hint)}
                    ]
                }],
                "generationConfig": {"maxOutputTokens": 400, "temperature": 0.3}
            }
            headers = {"Content-Type": "application/json"}
            url = self.URL
        elif cls == OllamaProvider:
            payload = {
                "model": self.MODEL,
                "prompt": _GEN_SYSTEM + "\n\n" + _build_gen_user(subject_hint),
                "images": [image_b64],
                "stream": False,
            }
            headers = {"Content-Type": "application/json"}
            url = self.URL
        elif cls == AnthropicProvider:
            payload = {
                "model": self.MODEL, "max_tokens": 400,
                "system": _GEN_SYSTEM,
                "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                    {"type": "text", "text": _build_gen_user(subject_hint)}
                ]}]
            }
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }
            url = self.URL
        else:  # Mistral / OpenAI
            payload = {
                "model": self.MODEL, "max_tokens": 400,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": _GEN_SYSTEM + "\n\n" + _build_gen_user(subject_hint)}
                ]}]
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            url = self.URL

        data = json.dumps(payload).encode()
        req  = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=20) as r:
            resp = json.loads(r.read())

        # Extrait le texte selon le provider
        if cls == GeminiProvider:
            text = resp["candidates"][0]["content"]["parts"][0]["text"]
        elif cls == OllamaProvider:
            text = resp.get("response", "{}")
        elif cls == AnthropicProvider:
            text = resp["content"][0]["text"]
        else:
            text = resp["choices"][0]["message"]["content"]

        return _parse_response(text)

    cls.analyze_for_generation = analyze_for_generation


# Injecte la méthode dans tous les providers
for _cls in [MistralProvider, AnthropicProvider, OpenAIProvider,
             GeminiProvider, OllamaProvider]:
    _add_gen_method(_cls, None, None, None)
