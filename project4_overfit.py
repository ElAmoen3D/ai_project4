"""
Cirrhosis Patient Outcomes — Neural Network Assignment (Parts 1–5)
PyTorch | 128 neurons per hidden layer | Sigmoid activation throughout
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import time

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cpu"

# =============================================================================
# PART 1 — DATA LOADING & EXPLORATORY ANALYSIS
# =============================================================================
print("=" * 70)
print("PART 1: DATA LOADING & EXPLORATORY ANALYSIS")
print("=" * 70)

df = pd.read_csv("./cirrhosis.csv")

print(f"\nDataset shape: {df.shape}")
print(f"\nData types:\n{df.dtypes.to_string()}")
print(f"\nClass distribution (Status):\n{df['Status'].value_counts().to_string()}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum()>0].to_string()}")
print(f"\nSummary statistics:\n{df.describe().round(2).to_string()}")

# Missing-value imputation — pandas CoW-safe (no inplace)
# Numerical  → median  (robust to the heavy-tailed distributions visible here)
# Categorical→ mode    (most frequent category)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols  = [c for c in df.select_dtypes(include=["object","str"]).columns
             if c != "Status"]

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

assert df.isnull().sum().sum() == 0
print("\nAfter imputation — missing values: 0 ok")

# Encode target (C=0, CL=1, D=2) and binary/ordinal categoricals
le_status = LabelEncoder()
df["Status_enc"] = le_status.fit_transform(df["Status"])
print(f"\nStatus encoding: "
      f"{dict(zip(le_status.classes_, le_status.transform(le_status.classes_)))}")
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

df       = df.drop(columns=["ID", "Status"])
features = [c for c in df.columns if c != "Status_enc"]
X        = df[features].values.astype(np.float32)
y        = df["Status_enc"].values.astype(np.int64)
INPUT_DIM   = X.shape[1]
NUM_CLASSES = len(np.unique(y))
label_map   = {0: "C", 1: "CL", 2: "D"}

# --- Visualisations ---
pal      = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}
num_feat = [c for c in features if df[c].nunique() > 10]

# 1. Histograms coloured by class
fig, axes = plt.subplots(4, 4, figsize=(18, 14))
axes = axes.flatten()
for i, feat in enumerate(num_feat[:16]):
    ax = axes[i]
    for cls in sorted(np.unique(y)):
        ax.hist(df.loc[df["Status_enc"] == cls, feat], bins=20,
                alpha=0.55, color=pal[cls], label=label_map[cls])
    ax.set_title(feat, fontsize=9)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
axes[0].legend(title="Status")
fig.suptitle("Part 1 — Feature Distributions by Class", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("./results/part1_histograms.png", dpi=110, bbox_inches="tight")
plt.close()

# 2. Correlation heatmap
fig, ax = plt.subplots(figsize=(14, 11))
corr = df[features + ["Status_enc"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.4, ax=ax, annot_kws={"size": 7})
ax.set_title("Part 1 — Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("./results/part1_correlation_heatmap.png", dpi=110, bbox_inches="tight")
plt.close()

# 3. Class distribution bar chart
fig, ax = plt.subplots(figsize=(6, 4))
cnt  = pd.Series(y).map(label_map).value_counts()
bars = ax.bar(cnt.index, cnt.values, color=list(pal.values()), edgecolor="white")
for bar, v in zip(bars, cnt.values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 1, str(v),
            ha="center", fontsize=11, fontweight="bold")
ax.set_title("Part 1 — Class Distribution (Status)", fontsize=12, fontweight="bold")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("./results/part1_class_distribution.png", dpi=110, bbox_inches="tight")
plt.close()
print("Part 1 plots saved.")

# --- Train / Val / Test split (70 / 15 / 15) ---------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)
print(f"\nSplit — Train:{len(X_train)}  Val:{len(X_val)}  Test:{len(X_test)}")


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def build_model(dropout=0.0, n_layers=4, n_hidden=128):
    """4 hidden layers of 128 sigmoid neurons; optional per-layer dropout."""
    layers = []
    in_d = INPUT_DIM
    for _ in range(n_layers):
        layers.append(nn.Linear(in_d, n_hidden))
        layers.append(nn.Sigmoid())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_d = n_hidden
    layers.append(nn.Linear(n_hidden, NUM_CLASSES))
    m = nn.Sequential(*layers)
    for layer in m:                             # Xavier init for sigmoid nets
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return m


def get_acc(model, X_np, y_np):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_np)).argmax(1).numpy()
    return accuracy_score(y_np, preds)


def ema_smooth(values, alpha=0.05):
    """Exponential moving average used only for plotting readability."""
    out = []
    prev = None
    for v in values:
        prev = v if prev is None else alpha * v + (1 - alpha) * prev
        out.append(prev)
    return out


def train_model(model, opt, X_tr, y_tr, X_v, y_v,
                epochs=4000, bs=32, patience=None, verbose=0,
                target_train_acc=None, shuffle=True):
    crit  = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(X_tr); ytr_t = torch.tensor(y_tr)
    Xv_t  = torch.tensor(X_v);  yv_t  = torch.tensor(y_v)
    TL, VL, TA, VA = [], [], [], []
    best, pat_cnt, stopped = 1e9, 0, None

    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(len(Xtr_t)) if shuffle else torch.arange(len(Xtr_t))
        eloss = 0.0
        for i in range(0, len(Xtr_t), bs):
            xb = Xtr_t[idx[i:i+bs]]; yb = ytr_t[idx[i:i+bs]]
            opt.zero_grad()
            loss = crit(model(xb), yb)
            if torch.isnan(loss):
                break
            loss.backward()
            opt.step()
            eloss += loss.item() * len(xb)
        eloss /= len(Xtr_t)

        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv_t), yv_t).item()
        TL.append(eloss); VL.append(vl)
        TA.append(get_acc(model, X_tr, y_tr))
        VA.append(get_acc(model, X_v,  y_v))

        if verbose and ep % verbose == 0:
            print(f"    ep={ep:4d}  tr_loss={eloss:.4f}  vl={vl:.4f}  "
                  f"tr_acc={TA[-1]:.4f}  va={VA[-1]:.4f}")
        
        # Early stopping if target training accuracy reached
        if target_train_acc is not None and TA[-1] >= target_train_acc:
            stopped = ep
            if verbose:
                print(f"    Target training accuracy {target_train_acc:.4f} reached at epoch {ep}")
            break
        
        if patience is not None:
            if vl < best - 1e-4:
                best, pat_cnt = vl, 0
            else:
                pat_cnt += 1
                if pat_cnt >= patience:
                    stopped = ep; break

    return TL, VL, TA, VA, stopped


# =============================================================================
# PART 2 — OVERFITTING BASELINE
# =============================================================================
print("\n" + "=" * 70)
print("PART 2: OVERFITTING BASELINE — TRAINING TO NEAR-PERFECT ACCURACY")
print("=" * 70)

m_base  = build_model()
op_base = optim.Adam(m_base.parameters(), lr=1e-4)
print("\nTraining baseline to near-perfect training accuracy (70/15/15 fixed split)...")
print("(Full-batch training — no mini-batching)\n")
tl_b, vl_b, ta_b, va_b, _ = train_model(
    m_base, op_base, X_train, y_train, X_val, y_val,
    epochs=20000, bs=len(X_train), target_train_acc=0.99, verbose=500, shuffle=False)

print(f"\nBaseline  tr_acc={ta_b[-1]:.4f}  val_acc={va_b[-1]:.4f}  "
      f"gap={ta_b[-1]-va_b[-1]:.4f}")
test_acc = get_acc(m_base, X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
ep_r = range(1, len(tl_b)+1)
a1.plot(ep_r, tl_b, label="Train (raw)", alpha=0.25)
a1.plot(ep_r, vl_b, label="Val (raw)", alpha=0.25)
a1.plot(ep_r, ema_smooth(tl_b, alpha=0.03), label="Train (smoothed)")
a1.plot(ep_r, ema_smooth(vl_b, alpha=0.03), label="Val (smoothed)")
a1.set_title("Part 2 — Loss (to Perfect Training Accuracy)")
a1.set_xlabel("Epoch")
a1.legend()
a2.plot(ep_r, ta_b, label="Train (raw)", alpha=0.25)
a2.plot(ep_r, va_b, label="Val (raw)", alpha=0.25)
a2.plot(ep_r, ema_smooth(ta_b, alpha=0.05), label="Train (smoothed)")
a2.plot(ep_r, ema_smooth(va_b, alpha=0.05), label="Val (smoothed)")
a2.set_title("Part 2 — Accuracy (to Perfect Training Accuracy)")
a2.set_xlabel("Epoch")
a2.legend()
a2.axhline(0.99, color='red', linestyle='--', alpha=0.5, label='99% target')
a2.legend()
fig.suptitle("Part 2 — Overfitting Baseline to Near-Perfect Training Accuracy", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("./results/part2_baseline.png", dpi=110, bbox_inches="tight")
plt.close()
print("Part 2 plot saved.")

print("""
Why does the model overfit?
  1. Massive capacity mismatch: ~50k parameters vs ~292 training samples.
  2. No regularisation — weights are free to grow, memorising training noise.
  3. Deep sigmoid networks suffer vanishing gradients in lower layers,
     causing only upper layers to update strongly (unstable training).
  4. Class imbalance (C:232, D:161, CL:25) biases predictions to majority.
""")


# =============================================================================
# PART 3 — REGULARISATION COMPARISON
# =============================================================================
print("=" * 70)
print("PART 3: REGULARISATION COMPARISON")
print("=" * 70)

configs3 = [
    ("No Reg",            0.0, 0.0,  None),
    ("L2 lam=1e-3",       0.0, 1e-3, None),
    ("L2 lam=1e-2",       0.0, 1e-2, None),
    ("Dropout p=0.3",     0.3, 0.0,  None),
    ("Dropout p=0.5",     0.5, 0.0,  None),
    ("Early Stop p=20",   0.0, 0.0,  20),
]

res3 = {}
for label, dr, wd, pat in configs3:
    m  = build_model(dropout=dr)
    op = optim.Adam(m.parameters(), lr=5e-4, weight_decay=wd)
    tl, vl, ta, va, stopped = train_model(
        m, op, X_train, y_train, X_val, y_val, epochs=400, bs=32, patience=pat, target_train_acc=None)
    tr_a = ta[-1]; va_a = va[-1]; tst_a = get_acc(m, X_test, y_test)
    res3[label] = dict(tl=tl, vl=vl, tr_a=tr_a, va_a=va_a,
                       tst_a=tst_a, gap=tr_a-va_a, stopped=stopped)
    extra = f"  stopped@{stopped}" if stopped else ""
    print(f"  {label:<22}  tr={tr_a:.4f}  val={va_a:.4f}  "
          f"test={tst_a:.4f}  gap={tr_a-va_a:+.4f}{extra}")

print(f"\n{'Method':<22} {'Train':>7} {'Val':>7} {'Test':>7} {'Gap':>7}")
print("-" * 50)
for label, r in res3.items():
    print(f"{label:<22} {r['tr_a']:>7.4f} {r['va_a']:>7.4f} "
          f"{r['tst_a']:>7.4f} {r['gap']:>+7.4f}")

fig, ax = plt.subplots(figsize=(11, 5))
for label, r in res3.items():
    ax.plot(r["vl"], label=label)
ax.set_title("Part 3 — Validation Loss Overlay", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("./results/part3_val_loss_overlay.png", dpi=110, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, (label, r) in enumerate(res3.items()):
    a = axes[i]
    a.plot(r["tl"], label="Train"); a.plot(r["vl"], label="Val")
    a.set_title(label, fontsize=9); a.legend(fontsize=7)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Part 3 — Train vs Val Loss", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("./results/part3_individual_curves.png", dpi=110, bbox_inches="tight")
plt.close()
print("\nPart 3 plots saved.")

print("""
Observations:
  L2 (lam=1e-3/1e-2): penalises large weights; reduces gap gradually.
  Dropout (0.3/0.5): trains ensemble of sub-networks; most robust features.
  Early Stopping: cleanest — halts before val loss rises; no added bias.
  Trade-offs: strong L2/Dropout may underfit; early stopping may vary by run.
""")


# =============================================================================
# PART 4 — INPUT NORMALISATION COMPARISON
# =============================================================================
print("=" * 70)
print("PART 4: INPUT NORMALISATION COMPARISON")
print("=" * 70)
print("Config: L2 lam=1e-3  (best test accuracy vs regularisation cost in P3)\n")

TARGET = 1.05

def make_scaled(mode):
    if mode == "none":
        return X_train.copy(), X_val.copy(), X_test.copy()
    sc = StandardScaler() if mode == "standard" else MinMaxScaler()
    return (sc.fit_transform(X_train).astype(np.float32),
            sc.transform(X_val).astype(np.float32),
            sc.transform(X_test).astype(np.float32))

res4 = {}
for mode in ["none", "standard", "minmax"]:
    Xtr_n, Xv_n, Xt_n = make_scaled(mode)
    m  = build_model()
    op = optim.Adam(m.parameters(), lr=5e-4, weight_decay=1e-3)
    tl, vl, ta, va, _ = train_model(
        m, op, Xtr_n, y_train, Xv_n, y_val, epochs=300, bs=32)
    ep_t = next((i+1 for i, v in enumerate(vl) if v <= TARGET), None)
    va_a = va[-1]; tst_a = get_acc(m, Xt_n, y_test)
    res4[mode] = dict(tl=tl, vl=vl, va_a=va_a, tst_a=tst_a, ep_t=ep_t)
    print(f"  {mode:<10}  val={va_a:.4f}  test={tst_a:.4f}  "
          f"epochs_to_val<={TARGET}: {ep_t or 'never'}")

print(f"\n{'Mode':<10} {'Val':>8} {'Test':>8} {'Epochs to target':>18}")
print("-" * 46)
for mode, r in res4.items():
    print(f"{mode:<10} {r['va_a']:>8.4f} {r['tst_a']:>8.4f} "
          f"{str(r['ep_t'] or 'never'):>18}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
titles4 = {"none": "No Normalisation",
           "standard": "Standardisation (Z-score)", "minmax": "Min-Max [0,1]"}
for ax, (mode, r) in zip(axes, res4.items()):
    ax.plot(r["tl"], label="Train"); ax.plot(r["vl"], label="Val")
    ax.axhline(TARGET, color="red", ls="--", alpha=0.5, label=f"target={TARGET}")
    ax.set_title(titles4[mode], fontsize=10)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(fontsize=7)
fig.suptitle("Part 4 — Input Normalisation", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("./results/part4_normalisation.png", dpi=110, bbox_inches="tight")
plt.close()
print("\nPart 4 plot saved.")

print("""
Why normalisation helps:
  Raw features span wildly different scales (N_Days ~41-4795 vs Albumin ~1-4).
  Gradient magnitudes scale with input, so large features dominate updates and
  create an elongated loss landscape, slowing convergence.
  Normalisation equalises contributions, roundens the loss surface, and keeps
  sigmoid inputs away from saturation zones — reducing vanishing gradients.
""")


# =============================================================================
# PART 5 — MINI-BATCH GRADIENT DESCENT COMPARISON
# =============================================================================
print("=" * 70)
print("PART 5: MINI-BATCH GRADIENT DESCENT COMPARISON")
print("=" * 70)
print("Config: L2 lam=1e-3 + Standardisation\n")

Xtr_s, Xv_s, Xt_s = make_scaled("standard")

batch_cfgs = [
    ("Full-Batch",          len(X_train), 0.03),
    ("SGD (bs=1)",          1,            0.005),
    ("Mini-Batch (bs=32)",  32,           0.015),
    ("Mini-Batch (bs=128)", 128,          0.02),
]

res5 = {}
for label, bs, lr in batch_cfgs:
    m  = build_model()
    op = optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
    t0 = time.perf_counter()
    tl, vl, ta, va, _ = train_model(
        m, op, Xtr_s, y_train, Xv_s, y_val, epochs=200, bs=bs, target_train_acc=None)
    tpe = (time.perf_counter() - t0) / len(tl)
    va_a = va[-1]; tst_a = get_acc(m, Xt_s, y_test)
    res5[label] = dict(tl=tl, vl=vl, va_a=va_a, tst_a=tst_a, tpe=tpe, bs=bs, lr=lr)
    print(f"  {label:<26}  val={va_a:.4f}  test={tst_a:.4f}  "
          f"lr={lr:.4f}  ms/epoch={tpe*1000:.1f}")

print(f"\n{'Method':<26} {'Val':>8} {'Test':>8} {'ms/epoch':>10}")
print("-" * 54)
for label, r in res5.items():
    print(f"{label:<26} {r['va_a']:>8.4f} {r['tst_a']:>8.4f} {r['tpe']*1000:>10.1f}")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()
for ax, (label, r) in zip(axes, res5.items()):
    ax.plot(r["tl"], label="Train", alpha=0.8)
    ax.plot(r["vl"], label="Val",   alpha=0.8)
    ax.set_title(f"{label}  (bs={r['bs']}, lr={r['lr']})", fontsize=9)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(fontsize=7)
fig.suptitle("Part 5 — Batch Size Effect on Loss", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("./results/part5_batch_comparison.png", dpi=110, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
for label, r in res5.items():
    ax.plot(r["vl"], label=label, alpha=0.85)
ax.set_title("Part 5 — Validation Loss Overlay", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Val Loss"); ax.legend()
plt.tight_layout()
plt.savefig("./results/part5_val_loss_overlay.png", dpi=110, bbox_inches="tight")
plt.close()
print("\nPart 5 plots saved.")

print("""
Batch-size trade-offs:
  Full-Batch: exact gradient, smooth curve; no stochasticity to escape minima.
  SGD (bs=1): noisiest updates, may escape saddle points; slow per-sample ops.
  Mini-batch (bs=32): balanced noise + vectorisation; good generalisation.
  Mini-batch (bs=128): faster per epoch; may converge to sharper (worse) minima.
  On this small dataset (~292 samples) timing differences are modest; effects
  amplify on larger data.
""")

print("=" * 70)
print("All 5 parts complete.  Plots saved to ./results")
print("=" * 70)