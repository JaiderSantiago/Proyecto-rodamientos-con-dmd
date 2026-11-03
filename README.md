# Proyecto-rodamientos-con-dmd
Prueba de concepto para el diagnóstico de fallas en rodamientos mediante Dynamic Mode Decomposition (DMD) aplicado a señales de vibración del dataset CWRU.

# DMD estándar sobre señales de vibración del CWRU (prueba de concepto)

Compara 4 condiciones de un rodamiento (normal, pista interna, bola, pista externa) usando
Dynamic Mode Decomposition (DMD) estándar, con FFT y persistencia como líneas base.

## Datos
Descargar del [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file)
(12k Drive End Bearing Fault Data, 0 HP, falla de 0.007") y ponerlos en `data/`:

| Condición | Archivo |
|---|---|
| Normal (Normal Baseline, 0 HP) | `97.mat` |
| Pista interna 0.007" | `105.mat` |
| Bola 0.007" | `118.mat` |
| Pista externa 0.007" (6 en punto) | `130.mat` |

## Uso
```bash
pip install -r requirements.txt
python dmd_cwru.py --data data/          # datos reales
python dmd_cwru.py --demo                # datos SINTÉTICOS, solo para probar que el código corre
```
Parámetros: `--win 4096` (muestras por ventana), `--delay 300` (filas de Hankel),
`--rank 30` (rango r), `--tol 8` (semiancho de banda en Hz), `--max_windows 25`.

## Qué hace
1. Carga la señal `*_DE_time` de cada archivo.
2. Divide en ventanas de 4096 muestras (0.34 s a 12 kHz), sin traslape, y quita la media.
3. Matriz de Hankel (embedding de retardos): cada fila es la señal desplazada y cada columna un instante.
4. DMD exacta con SVD truncada a rango r: `X = U S V*`, `A~ = U* Y V S^-1`, eigenvalores y modos.
   Frecuencia = angle(λ)/(2π·dt); tasa de crecimiento = ln|λ|/dt.
5. Métricas y figuras en `results/`.

## Salidas
- `fig1_eigenvalores.png`, `fig2_frecuencias_modos.png`, `fig3_energia_bandas.png`,
  `fig4_errores.png`, `fig5_reconstruccion.png`
- `metricas.csv` (media ± desviación sobre ventanas) y `modos_dominantes.csv`

## Líneas base
- Espectro: FFT con ventana Hann.
- Predicción a un paso: persistencia (x_{k+1} = x_k).

## Nota de interpretación
Las frecuencias de falla teóricas (BPFI, BPFO, BSF) corresponden al rodamiento SKF 6205-2RS a 1797 RPM.
En la práctica la energía de la falla suele aparecer en la banda de resonancia (kHz) modulada por esas
frecuencias, no directamente en ellas, por eso se reporta también la banda de 1–6 kHz.
