# -*- coding: utf-8 -*-
"""
Created on Tue Sep 01 12:23:31 2026

@author: angel
"""

# -*- coding: utf-8 -*-
"""
TFM - Proyeccion actual: prediccion y pesos para el proximo trimestre
======================================================================
Usa el modelo entrenado con todo el historico disponible para:
  - Predecir que activos liderarán el proximo trimestre
  - Calcular los pesos recomendados con 2 enfoques:
       - Estrategia 35/35/15/15
       - Proporcional al score
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False
    print("ERROR: joblib no disponible. Instala con: pip install joblib")

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")


# CONFIGURACION
RUTA_PANEL      = "data/processed/panel_modelo_final.csv"
DIR_MODELOS     = "data/processed/modelos_guardados"
RUTA_MODELO     = None

PESO_SOBRE      = 0.35
PESO_INFRA      = 0.15
PESO_ESTATICO   = 0.25
CARTERA_ACTUAL = {
    "SPY": 0.25,
    "TLT": 0.25,
    "GLD": 0.25,
    "BIL": 0.25,
}

FLOOR_DEFENSIVO = {
    "TLT": 0.10,   # minimo 10% en bonos largos siempre
    "BIL": 0.05,   # minimo 5% en cash siempre
}


# CARGA DEL MODELO Y EL PANEL

if not JOBLIB_OK:
    raise RuntimeError("joblib no disponible. Instala con: pip install joblib")

ruta_modelo = RUTA_MODELO
if ruta_modelo is None and os.path.isdir(DIR_MODELOS):
    archivos = sorted([f for f in os.listdir(DIR_MODELOS) if f.endswith(".joblib")])
    if archivos:
        ruta_modelo = os.path.join(DIR_MODELOS, archivos[-1])

if ruta_modelo is None or not os.path.exists(ruta_modelo):
    raise FileNotFoundError(
        f"No se encontro modelo en {DIR_MODELOS}. "
        "Ejecuta Modelado.py primero.")

pkg       = joblib.load(ruta_modelo)
modelo    = pkg["modelo"]
escalador = pkg.get("escalador")
features  = pkg["features"]
cols_dummy = pkg["cols_dummy"]
TICKERS   = pkg["tickers"]

print(f"Modelo cargado: {os.path.basename(ruta_modelo)}")
print(f"  Tipo:        {pkg['modelo_nombre']}")
print(f"  Entrenado:   {pkg['fecha_entreno']}")
print(f"  Spearman test: {pkg.get('spearman_test_medio', float('nan')):+.4f}")
print(f"  AUC test:      {pkg.get('auc_test_medio', float('nan')):.4f}")

panel = pd.read_csv(RUTA_PANEL)
panel["fecha"] = pd.to_datetime(panel["fecha"]).dt.normalize()
ret_hist = (panel.pivot(index="fecha", columns="ticker", values="close")
                 .pct_change().dropna(how="all"))

fecha_max = panel["fecha"].max()
fechas_completas = (panel.groupby("fecha")["ticker"].count()
                        .loc[lambda s: s == len(TICKERS)].index)
fecha_pred = fechas_completas.max()
print(f"\nUltima fecha disponible:             {fecha_max.date()}")
print(f"Ultima fecha con los {len(TICKERS)} activos: {fecha_pred.date()}")


# CONSTRUIR FEATURES Y PREDECIR
ultima_obs = panel[panel["fecha"] == fecha_pred].copy()
faltan = [f for f in features if f not in ultima_obs.columns]
if faltan:
    raise ValueError(
        f"Features del modelo no encontradas en el panel: {faltan}\n")

dummies = pd.get_dummies(ultima_obs["ticker"], prefix="sector")
for c in cols_dummy:
    if c not in dummies.columns:
        dummies[c] = 0

X_pred = pd.concat([
    ultima_obs[features].reset_index(drop=True),
    dummies[cols_dummy].astype(int).reset_index(drop=True)
], axis=1)

X_pred_m = escalador.transform(X_pred) if escalador is not None else X_pred
proba = modelo.predict(X_pred_m)   # score de ranking continuo (no probabilidad)

resultados = (pd.DataFrame({"ticker": ultima_obs["ticker"].values, "score": proba})
                .sort_values("score", ascending=False)
                .reset_index(drop=True))
resultados["ranking"] = range(1, len(resultados) + 1)
resultados["top2_predicho"] = resultados["ranking"] <= 2



# CALCULAR PESOS PARA LAS 2 ESTRATEGIAS

# Estrategia fija 35/35/15/15
pesos_fijo = {r["ticker"]: (PESO_SOBRE if r["top2_predicho"] else PESO_INFRA)
              for _, r in resultados.iterrows()}

# Estrategia Proporcional a la probabilidad
total_p = resultados["score"].sum()
pesos_prop = {r["ticker"]: r["score"] / total_p for _, r in resultados.iterrows()}


# RESULTADOS EN CONSOLA

print(f"\n{'='*70}")
print(f"PROYECCION PROXIMO TRIMESTRE (datos hasta {fecha_pred.date()})")
print(f"{'='*70}")
print(f"\n{'Rank':>5}  {'Activo':>8}  {'Score':>10}")
print("-" * 30)
for _, r in resultados.iterrows():
    print(f"  {int(r['ranking']):3d}.    {r['ticker']:>6s}    {r['score']:>8.4f}")

print(f"\n{'='*70}")
print("PESOS POR ESTRATEGIA")
print(f"{'='*70}")
print(f"\n  {'Activo':>7}  {'Fijo 35/35':>11}  {'Proporcional':>13}")
print("  " + "-" * 38)
for t in TICKERS:
    print(f"  {t:>7}    {pesos_fijo[t]*100:7.1f}%      "
         f"{pesos_prop[t]*100:9.1f}%")



# DASHBOARD
fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    f"Dashboard de proyeccion -- proximo trimestre\n"
    f"Datos hasta {fecha_pred.date()} | Modelo: {pkg['modelo_nombre']}",
    fontsize=13, fontweight="bold")

gs = gridspec.GridSpec(2, 3, figure=fig)

# Score de ranking predicho
ax1 = fig.add_subplot(gs[0, 0])
colores_p = ["steelblue" if t else "lightgray" for t in resultados["top2_predicho"]]
bars = ax1.barh(resultados["ticker"], resultados["score"],
                color=colores_p, edgecolor="white")
_smin, _smax = resultados["score"].min(), resultados["score"].max()
_margen = max((_smax - _smin) * 0.25, 0.02)
ax1.set_xlim(_smin - _margen, _smax + _margen)
ax1.set_xlabel("Score de ranking (regresor)")
ax1.set_title("Score predicho por activo\n(azul = 2 mejores del ranking)")
for bar, pv in zip(bars, resultados["score"]):
    ax1.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
             f" {pv:.3f}", va="center", fontsize=9)

# Pesos fijo
ax2 = fig.add_subplot(gs[0, 1])
vals_f = [pesos_fijo[t] for t in TICKERS]
colores_f = ["steelblue" if pesos_fijo[t] > PESO_ESTATICO else "lightcoral"
             for t in TICKERS]
ax2.pie(vals_f, labels=TICKERS, colors=colores_f,
        autopct="%1.0f%%", startangle=90, textprops={"fontsize": 10})
ax2.set_title("Pesos fijo 35/35/15/15")

# Proporcional
ax3 = fig.add_subplot(gs[0, 2])
vals_pr = [pesos_prop[t] for t in TICKERS]
ax3.pie(vals_pr, labels=TICKERS, autopct="%1.1f%%", startangle=90,
        colors=plt.cm.Blues(np.linspace(0.3, 0.9, len(TICKERS))),
        textprops={"fontsize": 10})
ax3.set_title("Pesos proporcionales")


# Rebalanceo: barras comparativas
ax5 = fig.add_subplot(gs[1, :2])
x   = np.arange(len(TICKERS))
w   = 0.2
ax5.bar(x - w, [CARTERA_ACTUAL.get(t, PESO_ESTATICO)*100 for t in TICKERS],
        w, label="Actual", color="lightgray", edgecolor="white")
ax5.bar(x - 0.5*w, [pesos_fijo[t]*100 for t in TICKERS],
        w, label="Fijo 35/35", color="steelblue", edgecolor="white")
ax5.bar(x + w, [pesos_prop[t]*100 for t in TICKERS],
        w, label="Proporcional", color="darkorange", edgecolor="white")
ax5.set_xticks(x); ax5.set_xticklabels(TICKERS)
ax5.set_ylabel("Peso (%)")
ax5.set_title("Comparacion de pesos por enfoque")
ax5.legend(fontsize=8)

# Retorno historico reciente de cada activo
ax6 = fig.add_subplot(gs[1, 2:])
ret_acum = (ret_hist[TICKERS].tail(52).fillna(0) + 1).cumprod() * 100
for t in TICKERS:
    ax6.plot(ret_acum.index, ret_acum[t], label=t, linewidth=1.5)
ax6.axhline(100, color="gray", linestyle="--", linewidth=0.8)
ax6.set_ylabel("Retorno acumulado (base 100)")
ax6.set_title("Retorno de los 4 activos (ultimas 52 semanas)")
ax6.legend(fontsize=9)

plt.tight_layout()
plt.savefig("data/processed/fig_proyeccion_dashboard.png", dpi=100)
plt.show()

