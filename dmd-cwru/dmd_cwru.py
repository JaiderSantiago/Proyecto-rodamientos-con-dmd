"""
Prueba de concepto: DMD estándar sobre señales de vibración del CWRU Bearing Data Center.

Condiciones (0 HP, ~1797 RPM, 12 kHz, acelerómetro Drive End, falla de 0.007"):
    Normal -> 97.mat | Pista interna -> 105.mat | Bola -> 118.mat | Pista externa -> 130.mat

Flujo: señal cruda -> ventanas -> matriz de Hankel -> DMD -> métricas y figuras.
Línea base: FFT (espectro) y persistencia (x_{k+1} = x_k) para la predicción.

Uso:
    python dmd_cwru.py --data data/        # con los .mat reales
    python dmd_cwru.py --demo              # datos SINTÉTICOS, solo para probar el código
"""
import argparse
import os

import numpy as np
import scipy.io as sio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 12_000          # Hz, frecuencia de muestreo
RPM = 1797           # 0 HP
DT = 1.0 / FS

CONDITIONS = {       # nombre: archivo
    "Normal": "97.mat",
    "Pista interna": "105.mat",
    "Bola": "118.mat",
    "Pista externa": "130.mat",
}
COLORS = {"Normal": "tab:green", "Pista interna": "tab:orange",
          "Bola": "tab:red", "Pista externa": "tab:purple"}

# Frecuencias características del rodamiento SKF 6205-2RS (múltiplos de la frecuencia de giro)
FR = RPM / 60.0
FAULT_FREQ = {"BPFI": 5.4152 * FR, "BPFO": 3.5848 * FR, "BSF": 4.7135 * FR}

plt.rcParams.update({"font.size": 13, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 110, "savefig.bbox": "tight"})


# ----------------------------------------------------------------- datos
def load_signal(path):
    """Devuelve la señal DE_time de un .mat del CWRU."""
    m = sio.loadmat(path)
    key = [k for k in m if k.endswith("_DE_time")][0]
    return m[key].ravel().astype(float)


def make_demo(folder, n=120_000, seed=0):
    """Señales SINTÉTICAS (no son del CWRU) para verificar que el código corre."""
    rng = np.random.default_rng(seed)
    os.makedirs(folder, exist_ok=True)
    t = np.arange(n) * DT
    fres = 3200.0
    for name, fname in CONDITIONS.items():
        x = 0.05 * np.sin(2 * np.pi * FR * t) + 0.02 * rng.standard_normal(n)
        f0 = {"Pista interna": FAULT_FREQ["BPFI"], "Bola": FAULT_FREQ["BSF"],
              "Pista externa": FAULT_FREQ["BPFO"]}.get(name)
        if f0:
            imp = np.zeros(n)
            imp[(np.arange(0, n / FS, 1 / f0) * FS).astype(int)] = 1.0
            k = np.arange(0, 200)
            ring = np.exp(-k / 25) * np.sin(2 * np.pi * fres * k * DT)
            x += 1.5 * np.convolve(imp, ring)[:n]
        key = "X" + fname.split(".")[0].zfill(3) + "_DE_time"
        sio.savemat(os.path.join(folder, fname), {key: x[:, None]})


def make_windows(x, n, max_windows):
    """Ventanas consecutivas sin traslape de n muestras."""
    k = min(len(x) // n, max_windows)
    return [x[i * n:(i + 1) * n] for i in range(k)]


def hankel(x, d):
    """Embedding de retardos: fila i = señal desplazada i muestras; columna j = instante j."""
    m = len(x) - d + 1
    idx = np.arange(d)[:, None] + np.arange(m)[None, :]
    return x[idx]


# ----------------------------------------------------------------- DMD
def dmd(H, r, dt=DT, n_rec=500):
    """DMD exacta con truncamiento SVD de rango r sobre la matriz de snapshots H."""
    X, Y = H[:, :-1], H[:, 1:]
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    Ur, Sr, Vr = U[:, :r], S[:r], Vh[:r].conj().T
    YV = Y @ Vr / Sr                       # Y V S^-1
    Atilde = Ur.conj().T @ YV
    lam, W = np.linalg.eig(Atilde)
    Phi = YV @ W                           # modos DMD exactos
    b = np.linalg.lstsq(Phi, X[:, 0], rcond=None)[0]

    freq = np.angle(lam) / (2 * np.pi * dt)        # Hz
    growth = np.log(np.abs(lam)) / dt               # 1/s (<0 decae)
    amp = np.abs(b) * np.linalg.norm(Phi, axis=0)

    # métricas
    pred = Ur @ Atilde @ Ur.conj().T @ X                       # predicción un paso
    pred_err = np.linalg.norm(Y - pred) / np.linalg.norm(Y)
    persist_err = np.linalg.norm(Y - X) / np.linalg.norm(Y)    # línea base
    m = min(n_rec, H.shape[1])
    Hrec = (Phi @ (b[:, None] * lam[:, None] ** np.arange(m)[None, :])).real
    rec_err = np.linalg.norm(H[:, :m] - Hrec) / np.linalg.norm(H[:, :m])
    energy_r = (S[:r] ** 2).sum() / (S ** 2).sum()

    return dict(lam=lam, freq=freq, growth=growth, amp=amp, pred_err=pred_err,
                persist_err=persist_err, rec_err=rec_err, energy_r=energy_r,
                recon_x=Hrec[0], S=S)


def get_bands(tol):
    """Bandas de interés: alrededor de cada frecuencia de falla y la banda de alta frecuencia
    (donde suele aparecer la resonancia excitada por los impactos de la falla)."""
    b = {k: (f - tol, f + tol) for k, f in FAULT_FREQ.items()}
    b["HF"] = (1000.0, 6000.0)
    return b


def band_energy_dmd(res, band):
    lo, hi = band
    pos = (res["freq"] > 0) & (res["freq"] < FS / 2)
    a2 = res["amp"] ** 2
    sel = pos & (res["freq"] >= lo) & (res["freq"] <= hi)
    return a2[sel].sum() / a2[pos].sum()


def spectrum_fft(w):
    P = np.abs(np.fft.rfft(w * np.hanning(len(w)))) ** 2
    f = np.fft.rfftfreq(len(w), DT)
    return f, P


def band_energy_fft(w, band):
    lo, hi = band
    f, P = spectrum_fft(w)
    return P[(f >= lo) & (f <= hi)].sum() / P.sum()


# ----------------------------------------------------------------- pipeline
def run(args):
    os.makedirs(args.out, exist_ok=True)
    data, results, rows = {}, {}, []
    bands = get_bands(args.tol)
    for name, fname in CONDITIONS.items():
        x = load_signal(os.path.join(args.data, fname))
        wins = [w - w.mean() for w in make_windows(x, args.win, args.max_windows)]
        data[name] = wins
        per_win = []
        for w in wins:
            res = dmd(hankel(w, args.delay), args.rank)
            res["bands_dmd"] = {k: band_energy_dmd(res, b) for k, b in bands.items()}
            res["bands_fft"] = {k: band_energy_fft(w, b) for k, b in bands.items()}
            per_win.append(res)
        results[name] = per_win
        print(f"{name}: {len(wins)} ventanas de {args.win} muestras "
              f"({args.win / FS:.2f} s), Hankel {args.delay} x {args.win - args.delay + 1}")

    # ---- tabla de métricas (media ± desviación sobre ventanas)
    def ms(name, fn):
        v = np.array([fn(r) for r in results[name]])
        return v.mean(), v.std()

    header = ["condicion", "n_ventanas", "energia_rango_r", "err_reconstruccion",
              "err_prediccion_DMD", "err_prediccion_persistencia"]
    for k in bands:
        header += [f"{k}_DMD_media", f"{k}_DMD_std", f"{k}_FFT_media", f"{k}_FFT_std"]
    lines = [",".join(header)]
    for name in CONDITIONS:
        row = [name, str(len(results[name]))]
        for key in ["energy_r", "rec_err", "pred_err", "persist_err"]:
            m, s = ms(name, lambda r, key=key: r[key])
            row.append(f"{m:.4f}±{s:.4f}")
        for k in bands:
            for src in ["bands_dmd", "bands_fft"]:
                m, s = ms(name, lambda r, k=k, src=src: r[src][k])
                row += [f"{m:.4f}", f"{s:.4f}"]
        lines.append(",".join(row))
    with open(os.path.join(args.out, "metricas.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ---- modos dominantes de la primera ventana
    with open(os.path.join(args.out, "modos_dominantes.csv"), "w", encoding="utf-8") as f:
        f.write("condicion,frecuencia_Hz,tasa_crecimiento_1_s,amplitud\n")
        for name in CONDITIONS:
            r = results[name][0]
            order = np.argsort(-r["amp"])
            shown = 0
            for i in order:
                if r["freq"][i] > 0 and shown < 5:
                    f.write(f"{name},{r['freq'][i]:.1f},{r['growth'][i]:.1f},{r['amp'][i]:.3f}\n")
                    shown += 1

    plot_eigs(results, args.out)
    plot_spectra(results, data, args)
    plot_bands(results, args)
    plot_errors(results, args.out)
    plot_recon(results, data, args)
    print(f"Listo. Resultados en: {args.out}/")


# ----------------------------------------------------------------- figuras
def plot_eigs(results, out):
    fig, axs = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    th = np.linspace(0, 2 * np.pi, 400)
    for ax, name in zip(axs.ravel(), CONDITIONS):
        lam = results[name][0]["lam"]
        ax.plot(np.cos(th), np.sin(th), "k--", lw=1, label="Círculo unitario")
        ax.scatter(lam.real, lam.imag, s=35, color=COLORS[name], label="Eigenvalores DMD")
        ax.set_title(name)
        ax.set_aspect("equal")
    for ax in axs[1]:
        ax.set_xlabel("Parte real de λ [-]")
    for ax in axs[:, 0]:
        ax.set_ylabel("Parte imaginaria de λ [-]")
    axs[0, 0].legend(loc="lower left", fontsize=10)
    fig.suptitle("Eigenvalores de DMD en el plano complejo (1.ª ventana)")
    fig.savefig(os.path.join(out, "fig1_eigenvalores.png"))
    plt.close(fig)


def plot_spectra(results, data, args):
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fmax = 6000
    for ax, name in zip(axs.ravel(), CONDITIONS):
        w = data[name][0]
        f, P = spectrum_fft(w)
        mag = np.sqrt(P)
        ax.plot(f, mag / mag.max(), color="lightgray", lw=1, label="FFT (línea base)")
        r = results[name][0]
        sel = (r["freq"] > 0) & (r["freq"] < fmax)
        ax.stem(r["freq"][sel], r["amp"][sel] / r["amp"][sel].max(), linefmt=COLORS[name],
                markerfmt="o", basefmt=" ", label="Modos DMD")
        for i, (k, f0) in enumerate(FAULT_FREQ.items()):
            ax.axvline(f0, ls="--", lw=1, color="k",
                       label="Frec. de falla (BPFO, BSF, BPFI)" if i == 0 else None)
        ax.set_xlim(0, fmax)
        ax.set_ylim(0, 1.15)
        ax.set_title(name)
    for ax in axs[1]:
        ax.set_xlabel("Frecuencia [Hz]")
    for ax in axs[:, 0]:
        ax.set_ylabel("Amplitud normalizada [-]")
    axs[0, 0].legend(fontsize=10, loc="upper right")
    fig.suptitle("Frecuencias de los modos DMD vs. espectro FFT (1.ª ventana)")
    fig.savefig(os.path.join(args.out, "fig2_frecuencias_modos.png"))
    plt.close(fig)


def plot_bands(results, args):
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.8), sharey=False)
    names = list(CONDITIONS)
    x = np.arange(len(names))
    titles = {k: f"{k} = {f:.1f} Hz (±{args.tol:g} Hz)" for k, f in FAULT_FREQ.items()}
    titles["HF"] = "Banda alta 1–6 kHz (resonancia)"
    for ax, k in zip(axs, get_bands(args.tol)):
        for off, src, lab, col in [(-0.2, "bands_dmd", "DMD", "tab:blue"),
                                   (0.2, "bands_fft", "FFT", "tab:gray")]:
            m = [np.mean([r[src][k] for r in results[n]]) for n in names]
            s = [np.std([r[src][k] for r in results[n]]) for n in names]
            lo = np.minimum(s, m)  # la energía no puede ser negativa
            ax.bar(x + off, m, 0.38, yerr=[lo, s], capsize=3, color=col, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20)
        ax.set_title(titles[k], fontsize=12)
        ax.set_ylabel("Energía relativa en banda [-]")
    axs[0].legend()
    fig.suptitle("Energía relativa por banda de frecuencia (media ± desv. entre ventanas)")
    fig.savefig(os.path.join(args.out, "fig3_energia_bandas.png"))
    plt.close(fig)


def plot_errors(results, out):
    names = list(CONDITIONS)
    x = np.arange(len(names))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for off, key, lab, col in [(-0.2, "pred_err", "DMD", "tab:blue"),
                               (0.2, "persist_err", "Persistencia (línea base)", "tab:gray")]:
        m = [np.mean([r[key] for r in results[n]]) for n in names]
        s = [np.std([r[key] for r in results[n]]) for n in names]
        axs[0].bar(x + off, m, 0.38, yerr=s, capsize=3, color=col, label=lab)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(names, rotation=20)
    axs[0].set_ylabel("Error relativo de predicción a 1 paso [-]")
    axs[0].set_title("Predicción a un paso")
    axs[0].legend()
    m = [np.mean([r["rec_err"] for r in results[n]]) for n in names]
    s = [np.std([r["rec_err"] for r in results[n]]) for n in names]
    axs[1].bar(x, m, 0.5, yerr=s, capsize=3, color=[COLORS[n] for n in names])
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(names, rotation=20)
    axs[1].set_ylabel("Error relativo de reconstrucción [-]")
    axs[1].set_title("Reconstrucción (primeras 500 columnas)")
    fig.suptitle("Errores de DMD (media ± desv. entre ventanas)")
    fig.savefig(os.path.join(out, "fig4_errores.png"))
    plt.close(fig)


def plot_recon(results, data, args):
    fig, axs = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    n = 300
    t = np.arange(n) * DT * 1e3
    for ax, name in zip(axs.ravel(), CONDITIONS):
        w = data[name][0]
        r = results[name][0]
        ax.plot(t, w[:n], color="gray", lw=1.5, label="Original")
        ax.plot(t, r["recon_x"][:n], color=COLORS[name], lw=1.2, ls="--", label="Reconstruida (DMD)")
        ax.set_title(f"{name} (error rec. = {r['rec_err']:.2f})")
    for ax in axs[1]:
        ax.set_xlabel("Tiempo [ms]")
    for ax in axs[:, 0]:
        ax.set_ylabel("Aceleración DE [g]")
    axs[0, 0].legend(fontsize=10)
    fig.suptitle("Señal original vs. reconstruida con DMD (1.ª ventana)")
    fig.savefig(os.path.join(args.out, "fig5_reconstruccion.png"))
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data", help="carpeta con 97.mat, 105.mat, 118.mat, 130.mat")
    p.add_argument("--out", default="results")
    p.add_argument("--win", type=int, default=4096, help="muestras por ventana")
    p.add_argument("--delay", type=int, default=300, help="filas de la matriz de Hankel")
    p.add_argument("--rank", type=int, default=30, help="rango de truncamiento r")
    p.add_argument("--tol", type=float, default=8.0, help="semiancho de banda alrededor de f_falla [Hz]")
    p.add_argument("--max_windows", type=int, default=25)
    p.add_argument("--demo", action="store_true", help="usar datos sintéticos")
    a = p.parse_args()
    if a.demo:
        a.data = "demo_data"
        make_demo(a.data)
        a.out = "results_demo"
    run(a)