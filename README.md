# ✦ GestureDraw v3 — Design Néon

Application Python de dessin par gestes de la main.
**Compatible toutes versions de mediapipe** (ancienne et nouvelle API).

---

## 📁 Structure

```
gesture_draw_v3/
├── main.py             ← Point d'entrée
├── config.py           ← Paramètres + palette néon
├── hand_tracker.py     ← ★ Abstraction mediapipe (auto-détecte la version)
├── gesture.py          ← Détection des 5 gestes
├── drawing.py          ← Moteur dessin + undo + PNG
├── shape_detector.py   ← Reconnaissance formes géo.
├── renderer.py         ← Primitives Pillow (glow, texte antialiasé)
├── ui.py               ← Interface néon complète
├── requirements.txt
├── hand_landmarker.task  ← (téléchargé automatiquement au 1er lancement)
└── saves/              ← Dessins (créé automatiquement)
```

---

## 🚀 Installation & Lancement

```bash
# 1. Installer les dépendances
pip install opencv-python mediapipe Pillow numpy

# 2. Lancer
python main.py
```

> ✅ **Python 3.10+** · Webcam obligatoire  
> ✅ Compatible **Windows / Linux / macOS**  
> ✅ Compatible **mediapipe < 0.10.14** (ancienne API `solutions`)  
> ✅ Compatible **mediapipe >= 0.10.14** (nouvelle API `tasks`) — télécharge `hand_landmarker.task` automatiquement

---

## ⚠️ Première exécution (mediapipe 0.10+)

Au premier lancement, si `hand_landmarker.task` est absent, il sera **téléchargé automatiquement** (~5 Mo) depuis Google Storage.

Si le téléchargement échoue (réseau restreint) :
1. Téléchargez manuellement :  
   `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
2. Placez le fichier **dans le même dossier** que `main.py`

---

## ✋ Gestes reconnus

| Geste | Action |
|-------|--------|
| ☝ Index seul | **Dessiner** |
| ✌ Index + Majeur | **Pause** (lève le crayon) |
| ✊ Poing fermé | **Gomme** |
| 🤙 Pouce + Auriculaire | **Annuler** (Ctrl+Z) |
| 🖐 Main ouverte | **Effacer tout** |

---

## ⌨️ Raccourcis clavier

| Touche | Action |
|--------|--------|
| `C` | Effacer tout |
| `Z` | Annuler |
| `S` | Sauvegarder PNG |
| `M` | Activer/désactiver formes auto |
| `+` / `)` | Taille du pinceau |
| `ù` / `*` | Opacité du pinceau |
| `1` à `6` | Choisir une couleur |
| `Q` / `Échap` | Quitter |

---

## 🎨 Design

Interface identique à la version HTML :
- Fond sombre `#0a0a0f` + panneaux latéraux
- Polices antialiasées (Liberation / DejaVu)
- Effets de **glow néon** via Pillow `GaussianBlur`
- Curseur animé selon le geste détecté
- Toast notifications avec halo lumineux
- Palette de 6 couleurs néon + indicateur actif
- Barre de progression pour taille et opacité

---

*✦ GestureDraw v3*
