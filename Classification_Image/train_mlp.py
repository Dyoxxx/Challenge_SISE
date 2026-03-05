"""
Étape 2 — Entraînement MLP PyTorch from scratch
=================================================
Input  : 1065 features
           936  = visage (468 pts × x,y)
            63  = main 1 (21 pts  × x,y,z)
            63  = main 2 (21 pts  × x,y,z)
             3  = flags [face_ok, hand1_ok, hand2_ok]
Output : N classes meme

Usage : uv run python train_mlp.py
"""

import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS      = 100
BATCH_SIZE  = 32
LR          = 1e-3
JITTER_STD  = 0.004
CSV_PATH    = "data/combined_data.csv"
MODEL_PATH  = "models/meme_mlp.pt"
LABELS_PATH = "models/meme_labels.json"

N_FEATURES  = 1085
FACE_END    = 956
H1_START    = 956;  H1_END    = 1019
H2_START    = 1019; H2_END    = 1082
FLAG_START  = 1082

os.makedirs("models", exist_ok=True)
print(f"[INFO] Device: {DEVICE}")

# ─── Dataset ──────────────────────────────────────────────────────────────────
class LandmarkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ─── Augmentation ─────────────────────────────────────────────────────────────
def augment(X: np.ndarray, n_jitter: int = 4) -> np.ndarray:
    out = [X]

    # Mirror x — visage et mains
    Xm = X.copy()
    Xm[:, 0:FACE_END:2] = 1.0 - X[:, 0:FACE_END:2]          # visage x
    h1 = Xm[:, H1_START:H1_END].reshape(-1, 21, 3)
    h1[:, :, 0] = 1.0 - X[:, H1_START:H1_END].reshape(-1, 21, 3)[:, :, 0]
    Xm[:, H1_START:H1_END] = h1.reshape(-1, 63)
    h2 = Xm[:, H2_START:H2_END].reshape(-1, 21, 3)
    h2[:, :, 0] = 1.0 - X[:, H2_START:H2_END].reshape(-1, 21, 3)[:, :, 0]
    Xm[:, H2_START:H2_END] = h2.reshape(-1, 63)
    out.append(Xm)

    # Swap main1 / main2 si les deux présentes
    flags = X[:, FLAG_START:]
    if flags.shape[1] >= 3:
        both = (flags[:, 1] == 1.0) & (flags[:, 2] == 1.0)
        if both.any():
            Xs = X[both].copy()
            tmp = Xs[:, H1_START:H1_END].copy()
            Xs[:, H1_START:H1_END] = Xs[:, H2_START:H2_END]
            Xs[:, H2_START:H2_END] = tmp
            out.append(Xs)

    # Jitter
    base = np.vstack(out)
    for _ in range(n_jitter):
        noise = np.random.normal(0, JITTER_STD, base.shape).astype(np.float32)
        noise[:, FLAG_START:] = 0   # ne pas bruiter les flags
        out.append(np.clip(base + noise, 0, 1))

    return np.vstack(out)

# ─── MLP ──────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),        nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),        nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x): return self.net(x)

# ─── Chargement CSV ───────────────────────────────────────────────────────────
if not os.path.exists(CSV_PATH):
    print(f"[ERREUR] {CSV_PATH} introuvable — lance d'abord collect_data.py"); exit(1)

# Lecture brute pour détecter le vrai nombre de colonnes dans le header
with open(CSV_PATH) as f:
    header = f.readline().strip().split(",")
n_cols_in_file = len(header)

expected_cols = 1 + N_FEATURES   # 1086

if n_cols_in_file != expected_cols:
    print(f"[WARN] Header CSV : {n_cols_in_file} colonnes (attendu {expected_cols}) — ignoré, lecture ligne par ligne.")

# Lecture brute ligne par ligne — insensible au header et aux lignes corrompues
def is_float(s):
    try: float(s); return True
    except: return False

rows = []
with open(CSV_PATH, encoding="utf-8-sig") as f:
    for raw_line in f:
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(",")
        label_val = parts[0].strip()
        # Ignore header(s) et lignes avec label numérique
        if label_val == "label" or is_float(label_val):
            continue
        # Garde uniquement les lignes avec exactement N_FEATURES valeurs numériques
        feat_parts = parts[1:]
        if len(feat_parts) != N_FEATURES:
            print(f"[WARN] Ligne ignorée ({len(feat_parts)} features, attendu {N_FEATURES}) : {label_val!r}")
            continue
        try:
            feats = [float(v) for v in feat_parts]
        except ValueError:
            continue
        rows.append([label_val] + feats)

if not rows:
    print("[ERREUR] Aucune ligne valide trouvée dans le CSV.")
    print("         Vérifie que collect_data.py a bien tourné et appuyé sur ESPACE.")
    exit(1)

feat_cols = [f"f{i}" for i in range(N_FEATURES)]
df = pd.DataFrame(rows, columns=["label"] + feat_cols)
print(f"[INFO] {len(df)} samples | {df['label'].value_counts().to_dict()}")
if df["label"].nunique() < 2:
    print("[ERREUR] Minimum 2 classes différentes requises"); exit(1)

le = LabelEncoder()
y  = le.fit_transform(df["label"].values)
X  = df.drop(columns=["label"]).values.astype(np.float32)

X_aug = augment(X, n_jitter=4)
# Tile y pour correspondre exactement à X_aug (le swap peut ajouter des lignes variables)
y_aug = np.resize(y, len(X_aug))   # resize répète et tronque proprement

X_tr, X_val, y_tr, y_val = train_test_split(
    X_aug, y_aug, test_size=0.15, stratify=y_aug, random_state=42)
print(f"[INFO] Après augmentation : {len(X_tr)} train, {len(X_val)} val")

tr_loader  = DataLoader(LandmarkDataset(X_tr,  y_tr),  batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(LandmarkDataset(X_val, y_val), batch_size=BATCH_SIZE)

model     = MLP(len(le.classes_)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
criterion = nn.CrossEntropyLoss()

best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for Xb, yb in tr_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    if epoch % 10 == 0 or epoch == EPOCHS:
        model.eval(); correct = total = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                preds    = model(Xb.to(DEVICE)).argmax(1).cpu()
                correct += (preds == yb).sum().item()
                total   += len(yb)
        acc = correct / total
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={total_loss/len(tr_loader):.4f}  val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)

with open(LABELS_PATH, "w") as f:
    json.dump(list(le.classes_), f)

print(f"\n[OK] Modèle → {MODEL_PATH}  (best val_acc: {best_acc:.3f})")
print(f"[OK] Labels → {LABELS_PATH}: {list(le.classes_)}")
