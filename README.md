# Webcam Tracker — Expressions & Signes de mains

Détection temps réel via webcam :
- **Expressions faciales** : happy, sad, surprised, neutral, angry, fear, disgust
- **Signes de mains** : thumbs_up, fist, peace, pointing, open_hand, ok

Affichage côte-à-côte : webcam (gauche) + image du dataset associée (droite).

---

## Installation

```bash
pip install opencv-python mediapipe fer torch torchvision numpy
```

> `fer` dépend de TensorFlow. Si tu veux éviter TF, commente le bloc FER
> dans le script et branche DeepFace ou un modèle PyTorch custom à la place.

---

## Lancer

```bash
python webcam_tracker.py
```

Appuie sur **Q** pour quitter.

---

## Ajouter tes propres images

Structure attendue :

```
dataset/
├── happy/
│   ├── image1.jpg
│   └── image2.png
├── sad/
├── peace/
├── thumbs_up/
└── ...
```

Si le dossier `dataset/` n'existe pas au lancement, des images **placeholder colorées**
sont créées automatiquement pour tester.

---

## Architecture

```
webcam frame
    ├─► MediaPipe Hands  → classify_hand_sign() → label (ex: "peace")
    │                                            → image aléatoire dans dataset/peace/
    └─► MediaPipe FaceMesh (landmarks)
        FER detector      → emotion dominante    → image aléatoire dans dataset/happy/
```

---

## Remplacer FER par un modèle PyTorch custom

Dans `main()`, remplace le bloc FER par :

```python
import torch
import torchvision.transforms as T

model = torch.load("mon_modele_emotion.pt").eval()
transform = T.Compose([T.ToPILImage(), T.Resize((48,48)), T.Grayscale(), T.ToTensor()])

# Dans la boucle :
tensor = transform(face_roi).unsqueeze(0)
with torch.no_grad():
    pred = model(tensor).argmax().item()
emotion_label = EXPRESSION_LABELS[pred]
```
