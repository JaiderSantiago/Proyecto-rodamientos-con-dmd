"""Figuras finales para los slides 9 y 10 (letra grande, títulos incluidos)."""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dmd_cwru as m

OUT = "slides_figs"
os.makedirs(OUT, exist_ok=True)
NAVY = "#14213D"
plt.rcParams.update({"font.size": 18, "axes.grid": True, "grid.alpha": 0.25, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 20, "axes.titleweight": "bold", "text.color": NAVY,
                     "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY})
NAMES = list(m.CONDITIONS)
COL = m.COLORS
R, DELAY, WIN, NW = 30, 300, 4096, 25

res, wins = {}, {}
for n, f in m.CONDITIONS.items():
    x = m.load_signal(os.path.join("data", f))
    ws = [w - w.mean() for w in m.make_windows(x, WIN, NW)]
    wins[n] = ws
    res[n] = [m.dmd(m.hankel(w, DELAY), R) for w in ws]

# ---------- Slide 9: señal -> matriz de Hankel (apilado)
w = wins["Pista externa"][0]
ncol = 151
seg = w[:DELAY + ncol - 1]
Hs = m.hankel(seg, DELAY)
fig, ax = plt.subplots(2, 1, figsize=(9.0, 5.3), gridspec_kw={"height_ratios": [1, 1.55]})
t = np.arange(len(seg)) / m.FS * 1e3
ax[0].plot(t, seg, color=NAVY, lw=1.3)
ax[0].axvspan(0, DELAY / m.FS * 1e3, color="tab:orange", alpha=0.28, label="Columna 1")
ax[0].axvspan(ncol / m.FS * 1e3, len(seg) / m.FS * 1e3, color="tab:blue", alpha=0.18,
              label=f"Columna {ncol}")
ax[0].set_xlabel("Tiempo [ms]")
ax[0].set_ylabel("Aceleración [g]")
ax[0].set_title("Ventana de señal (pista externa)")
ax[0].set_ylim(-3.2, 3.2)
ax[0].text(0.1, 2.3, "Columna 1", fontsize=16, color="#9A4D00", fontweight="bold", va="bottom")
ax[0].text(25.6, 2.3, f"Columna {ncol}", fontsize=16, color="#1F4E86", fontweight="bold", va="bottom")
vmax = np.abs(Hs).max()
im = ax[1].imshow(Hs, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  extent=[0, ncol / m.FS * 1e3, DELAY / m.FS * 1e3, 0])
ax[1].grid(False)
ax[1].set_xlabel("Instante de la columna [ms]")
ax[1].set_ylabel("Retardo (fila) [ms]")
ax[1].set_title("Matriz de Hankel H (recorte de 300 × 151)")
cb = fig.colorbar(im, ax=ax[1], pad=0.02)
cb.set_label("[g]")
fig.tight_layout()
fig.savefig(f"{OUT}/s9_hankel_v2.png", dpi=200)
plt.close(fig)

# ---------- Slide 9: ecuaciones
fig = plt.figure(figsize=(9.0, 1.6))
fig.patch.set_alpha(0)
eqs = [
    r"$H_{ij}=x_{i+j-1},\quad H\in\mathbb{R}^{300\times 3797}$",
    r"$X=H_{:,\,1:m-1},\;\; Y=H_{:,\,2:m},\quad Y\approx AX$",
    r"$X=U\Sigma V^{*}\;\Rightarrow\;\tilde{A}=U_r^{*}\,Y\,V_r\,\Sigma_r^{-1},\quad f=\angle\lambda\,/\,(2\pi\,\Delta t)$",
]
for i, e in enumerate(eqs):
    fig.text(0.02, 0.84 - i * 0.34, e, fontsize=21, color=NAVY, va="center", ha="left")
fig.savefig(f"{OUT}/s9_ecuaciones.png", dpi=300, transparent=True)
plt.close(fig)

SZ = (6.8, 5.6)
# ---------- Slide 10 (A): eigenvalores
fig, ax = plt.subplots(figsize=SZ)
th = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(th), np.sin(th), "k--", lw=1.4)
for n in NAMES:
    lam = res[n][0]["lam"]
    ax.scatter(lam.real, lam.imag, s=70, color=COL[n], alpha=0.85, label=n, edgecolor="white", lw=0.6)
ax.set_aspect("equal")
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("Parte real de λ [-]")
ax.set_ylabel("Parte imag. de λ [-]")
ax.set_title("Eigenvalores (1.ª ventana)")
ax.legend(fontsize=14, loc="center", frameon=True)
fig.savefig(f"{OUT}/s10_a_eigenvalores.png", dpi=200)
plt.close(fig)

# ---------- Slide 10 (B): frecuencias de los modos
edges = np.linspace(0, 6000, 61)
ctr = 0.5 * (edges[1:] + edges[:-1])
fig, ax = plt.subplots(figsize=SZ)
for n in NAMES:
    acc = np.zeros(len(ctr))
    for r in res[n]:
        pos = (r["freq"] > 0) & (r["freq"] < m.FS / 2)
        h, _ = np.histogram(r["freq"][pos], bins=edges, weights=r["amp"][pos] ** 2)
        acc += h / h.sum()
    acc /= len(res[n])
    ax.plot(ctr, acc, color=COL[n], lw=3, label=n)
    ax.fill_between(ctr, acc, color=COL[n], alpha=0.12)
ax.set_xlabel("Frecuencia [Hz]")
ax.set_ylabel("Energía relativa [-]")
ax.set_title("Frecuencia de los modos DMD")
ax.legend(fontsize=14)
fig.savefig(f"{OUT}/s10_b_frecuencias.png", dpi=200)
plt.close(fig)

# ---------- Slide 10 (C): predicción a un paso
fig, ax = plt.subplots(figsize=SZ)
x = np.arange(len(NAMES))
for off, key, lab, col in [(-0.2, "pred_err", "DMD (r = 30)", "#E07B00"),
                           (0.2, "persist_err", "Persistencia", "#9AA5B1")]:
    mu = np.array([np.mean([r[key] for r in res[n]]) for n in NAMES])
    sd = np.array([np.std([r[key] for r in res[n]]) for n in NAMES])
    ax.bar(x + off, mu, 0.38, yerr=sd, capsize=4, color=col, label=lab)
ax.set_xticks(x)
ax.set_xticklabels(["Normal", "P. int.", "Bola", "P. ext."], fontsize=17)
ax.set_ylabel("Error relativo [-]")
ax.set_title("Predicción a un paso")
ax.legend(fontsize=14, loc="upper left")
ax.set_ylim(0, 2.0)
fig.savefig(f"{OUT}/s10_c_prediccion.png", dpi=200)
plt.close(fig)
print("ok")