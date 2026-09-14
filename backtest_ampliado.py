# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 23:53:00 2026

@author: angel
"""

# -*- coding: utf-8 -*-
"""
TFM - Backtesting
=======================================================
Compara 3 estrategias sobre el historico completo:

  - Cartera permanente: 25/25/25/25 siempre (baseline Harry Browne)
  - TACTICA 35/35: 35/35/15/15 segun el ranking predicho por el modelo
  - PROPORCIONAL al score: pesos proporcionales al score predicho por el regresor

Metricas: rentabilidad total, CAGR, volatilidad, Sharpe, Sortino,
          max drawdown, hit rate, turnover, rebalanceos efectivos.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False


from sklearn.base import clone
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")
sns.set_style("darkgrid")


# CONFIGURACION

RUTA_PANEL      = "data/processed/panel_modelo_final.csv"
DIR_MODELOS     = "data/processed/modelos_guardados"
RUTA_MODELO     = None       
TARGET_CONTINUO = "target_3m"
VENTANA_RODANTE = 156        
HORIZONTE_SEM   = 13
EMBARGO_SEM     = 4
TASA_RF_ANUAL   = 0.02
PESO_SOBRE      = 0.35
PESO_INFRA      = 0.15
PESO_ESTATICO   = 0.25
UMBRAL_TURNOVER = 0.05       # no rebalancear si cambio < 5% en todos los activos


# CARGA
panel = pd.read_csv(RUTA_PANEL)
panel["fecha"] = pd.to_datetime(panel["fecha"]).dt.normalize()
TICKERS = sorted(panel["ticker"].unique())

print(f"Panel: {panel.shape} | Activos: {TICKERS}")
print(f"Rango: {panel.fecha.min().date()} -> {panel.fecha.max().date()}")

# Cargar modelo guardado
modelo_pkg = None
if JOBLIB_OK:
    ruta = RUTA_MODELO
    if ruta is None and os.path.isdir(DIR_MODELOS):
        archivos = sorted([f for f in os.listdir(DIR_MODELOS)
                          if f.endswith(".joblib")])
        if archivos:
            ruta = os.path.join(DIR_MODELOS, archivos[-1])
    if ruta and os.path.exists(ruta):
        modelo_pkg = joblib.load(ruta)
        print(f"\nModelo cargado: {ruta}")
        print(f"  Tipo: {modelo_pkg['modelo_nombre']} | "
             f"Entrenado: {modelo_pkg['fecha_entreno']}")

features   = modelo_pkg["features"]   if modelo_pkg else []
COLS_DUMMY = modelo_pkg["cols_dummy"] if modelo_pkg else [f"sector_{t}"
                                                          for t in TICKERS[1:]]


def matriz_features(df, feats, cols_dummy):
    dummies = pd.get_dummies(df["ticker"], prefix="sector")
    for c in cols_dummy:
        if c not in dummies.columns:
            dummies[c] = 0
    return pd.concat([df[feats].reset_index(drop=True),
                      dummies[cols_dummy].astype(int).reset_index(drop=True)], axis=1)


# PREDICCIONES WALK-FORWARD
predicciones = pd.DataFrame()

if features:
    est     = modelo_pkg["modelo"]
    escalar = modelo_pkg.get("escalador") is not None

    datos = (panel.dropna(subset=features + [TARGET_CONTINUO])
                  .sort_values("fecha").reset_index(drop=True))
    fechas_u = np.sort(datos["fecha"].unique())
    primera  = (pd.Timestamp(fechas_u[0])
                + pd.Timedelta(weeks=VENTANA_RODANTE + HORIZONTE_SEM + EMBARGO_SEM))
    fechas_pred = fechas_u[fechas_u >= primera.to_numpy()]

    print(f"\nGenerando predicciones "
         f"({pd.Timestamp(fechas_pred[0]).date()} -> "
         f"{pd.Timestamp(fechas_pred[-1]).date()})...")

    filas_pred = []
    for fecha in fechas_pred:
        f_purga = pd.Timestamp(fecha) - pd.Timedelta(weeks=HORIZONTE_SEM)
        f_emb   = f_purga - pd.Timedelta(weeks=EMBARGO_SEM)
        f_ini   = f_emb - pd.Timedelta(weeks=VENTANA_RODANTE)
        train   = datos[(datos.fecha >= f_ini) & (datos.fecha < f_emb)]
        pred_df = datos[datos.fecha == fecha]
        if len(train) < 80 or len(pred_df) < len(TICKERS):
            continue
        X_tr = matriz_features(train, features, COLS_DUMMY)
        y_tr = train[TARGET_CONTINUO]   # regresor: objetivo continuo
        X_pr = matriz_features(pred_df, features, COLS_DUMMY)
        if escalar:
            esc    = StandardScaler().fit(X_tr)
            X_tr_m = esc.transform(X_tr)
            X_pr_m = esc.transform(X_pr)
        else:
            X_tr_m, X_pr_m = X_tr, X_pr
        modelo_f = clone(est)
        modelo_f.fit(X_tr_m, y_tr)
        proba = modelo_f.predict(X_pr_m)   # score de ranking continuo
        for i, (_, fila) in enumerate(pred_df.iterrows()):
            filas_pred.append({"fecha": fecha, "ticker": fila["ticker"],
                               "score": float(proba[i])})

    predicciones = pd.DataFrame(filas_pred)
    print(f"  {len(predicciones):,} filas | {predicciones.fecha.nunique()} fechas")



# FUNCIONES DE PESOS
def pesos_estaticos(tickers):
    return {t: PESO_ESTATICO for t in tickers}


def pesos_tacticos(pred_fecha, tickers):
    if pred_fecha.empty:
        return pesos_estaticos(tickers)
    ranking = pred_fecha.sort_values("score", ascending=False)["ticker"].tolist()
    return {t: (PESO_SOBRE if i < 2 else PESO_INFRA)
            for i, t in enumerate(ranking)}


def pesos_proporcionales(pred_fecha, tickers):
    if pred_fecha.empty:
        return pesos_estaticos(tickers)
    total = pred_fecha["score"].sum()
    if total == 0:
        return pesos_estaticos(tickers)
    d = dict(zip(pred_fecha["ticker"], pred_fecha["score"] / total))
    return {t: d.get(t, 0.0) for t in tickers}


def aplicar_umbral(pesos_nuevos, pesos_prev):
    # No rebalancear si el cambio maximo en cualquier activo < UMBRAL_TURNOVER.
    if pesos_prev is None:
        return pesos_nuevos, True
    if max(abs(pesos_nuevos.get(t, 0) - pesos_prev.get(t, 0))
           for t in pesos_nuevos) > UMBRAL_TURNOVER:
        return pesos_nuevos, True
    return pesos_prev, False


# SIMULACION SEMANAL
ret_w = (panel.pivot(index="fecha", columns="ticker", values="close")
              .pct_change().dropna(how="all"))

fecha_inicio = (predicciones.fecha.min() if not predicciones.empty
                else panel.fecha.min() + pd.Timedelta(weeks=VENTANA_RODANTE))
fechas_sim   = ret_w.index[ret_w.index >= fecha_inicio]
print(f"\nSimulacion: {pd.Timestamp(fechas_sim[0]).date()} -> "
     f"{pd.Timestamp(fechas_sim[-1]).date()}")


def simular(nombre, fn_pesos):
    valor, serie = 100.0, []
    pesos_prev, n_rebal, turnover_total = None, 0, 0.0
    for fecha in fechas_sim:
        if fecha not in ret_w.index:
            continue
        ret_sem = ret_w.loc[fecha]
        if not predicciones.empty and fecha in predicciones.fecha.values:
            pred_f = predicciones[predicciones.fecha == fecha]
            pesos_nuevos = fn_pesos(pred_f)
            pesos, rebal = aplicar_umbral(pesos_nuevos, pesos_prev)
            if rebal and pesos_prev is not None:
                n_rebal += 1
                turnover_total += sum(abs(pesos.get(t, 0) - pesos_prev.get(t, 0))
                                      for t in TICKERS) / 2
            pesos_prev = pesos
        elif pesos_prev is None:
            pesos_prev = {t: PESO_ESTATICO for t in TICKERS}
        pesos = pesos_prev
        ret_c = sum(pesos.get(t, 0) * ret_sem.get(t, 0.0) for t in TICKERS)
        valor *= (1 + ret_c)
        serie.append({"fecha": fecha, "valor": valor, "ret": ret_c,
                      "estrategia": nombre})
    df = pd.DataFrame(serie)
    return df, n_rebal, (turnover_total / n_rebal if n_rebal > 0 else 0)


# Estrategia estatica (sin predicciones)
def fn_estat(pred_f): return pesos_estaticos(TICKERS)
sim_estat, _, _ = simular("Estatica 25/25/25/25", fn_estat)
carteras = {"Estatica 25/25/25/25": (sim_estat, 0, 0)}

if not predicciones.empty:
    def fn_tact(pred_f): return pesos_tacticos(pred_f, TICKERS)
    sim_tact, nr_t, to_t = simular("Tactica 35/35/15/15", fn_tact)
    carteras["Tactica 35/35/15/15"] = (sim_tact, nr_t, to_t)

    def fn_prop(pred_f): return pesos_proporcionales(pred_f, TICKERS)
    sim_prop, nr_p, to_p = simular("Proporcional al score", fn_prop)
    carteras["Proporcional al score"] = (sim_prop, nr_p, to_p)


# METRICAS
def metricas(df_serie, n_rebal, turnover_medio):
    df   = df_serie.set_index("fecha").sort_index()
    rets = df["ret"].dropna()
    n    = len(rets)
    ret_total = df["valor"].iloc[-1] / 100 - 1
    cagr = (1 + ret_total) ** (52 / n) - 1
    vol  = rets.std() * np.sqrt(52)
    sharpe = (cagr - TASA_RF_ANUAL) / vol if vol > 0 else np.nan
    ret_neg = rets[rets < 0]
    sd = ret_neg.std() * np.sqrt(52)
    sortino = (cagr - TASA_RF_ANUAL) / sd if sd > 0 else np.nan
    vals = df["valor"].values
    dd   = (vals - np.maximum.accumulate(vals)) / np.maximum.accumulate(vals)
    return {
        "Retorno total":  f"{ret_total*100:.1f}%",
        "CAGR":           f"{cagr*100:.2f}%",
        "Volatilidad":    f"{vol*100:.2f}%",
        "Sharpe":         f"{sharpe:.3f}",
        "Sortino":        f"{sortino:.3f}",
        "Max Drawdown":   f"{dd.min()*100:.1f}%",
        "N rebalanceos":  str(n_rebal),
        "Turnover medio": f"{turnover_medio*100:.1f}%",
    }

print(f"\n{'='*80}")
print("METRICAS COMPARATIVAS")
print(f"{'='*80}")
rows = []
for nombre, (df_c, nr, to) in carteras.items():
    m = metricas(df_c, nr, to)
    m["Estrategia"] = nombre
    rows.append(m)
df_met = pd.DataFrame(rows).set_index("Estrategia")
print(df_met.to_string())


# GRAFICAS
COLORES = {
    "Estatica 25/25/25/25": "gray",
    "Tactica 35/35/15/15":  "steelblue",
    "Proporcional al score": "darkorange",
}

fig, ax = plt.subplots(figsize=(13, 6))
for nombre, (df_c, _, _) in carteras.items():
    d = df_c.set_index("fecha").sort_index()
    ax.plot(d.index, d["valor"], label=nombre,
            color=COLORES.get(nombre, "blue"), linewidth=1.8)
ax.set_ylabel("Valor (base 100)")
ax.set_title("Evolucion de las carteras (base 100)")
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("data/processed/fig_bt_valor.png", dpi=100)
plt.show()

# Drawdown
fig, ax = plt.subplots(figsize=(13, 4))
for nombre, (df_c, _, _) in carteras.items():
    d   = df_c.set_index("fecha").sort_index()
    v   = d["valor"].values
    dd  = (v - np.maximum.accumulate(v)) / np.maximum.accumulate(v) * 100
    ax.plot(d.index, dd, label=nombre, color=COLORES.get(nombre, "blue"),
            linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.7)
ax.set_ylabel("Drawdown (%)")
ax.set_title("Drawdown por estrategia")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("data/processed/fig_bt_drawdown.png", dpi=100)
plt.show()

# --- 7c. Retorno anual por estrategia ---
df_all = pd.concat([df_c for df_c, _, _ in carteras.values()], ignore_index=True)
df_all["anio"] = pd.to_datetime(df_all["fecha"]).dt.year
df_pivot = (df_all.groupby(["anio", "estrategia"])["ret"]
                  .apply(lambda x: (1 + x).prod() - 1)
                  .reset_index()
                  .pivot(index="anio", columns="estrategia", values="ret") * 100)
df_pivot = df_pivot[[c for c in COLORES if c in df_pivot.columns]]
fig, ax = plt.subplots(figsize=(14, 5))
df_pivot.plot(kind="bar", ax=ax,
              color=[COLORES[c] for c in df_pivot.columns],
              width=0.75)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Retorno (%)")
ax.set_title("Retorno anual por estrategia")
ax.legend(fontsize=8, loc="lower right")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("data/processed/fig_bt_anual.png", dpi=100)
plt.show()

# Metricas
metricas_radar = ["CAGR", "Sharpe", "Sortino", "Max Drawdown"]
fig, ax = plt.subplots(1, len(metricas_radar), figsize=(14, 4))
for i, met in enumerate(metricas_radar):
    vals_num = []
    nombres  = []
    for nombre, (df_c, nr, to) in carteras.items():
        m = metricas(df_c, nr, to)
        try:
            v = float(m[met].replace("%", ""))
        except Exception:
            v = 0
        vals_num.append(v)
        nombres.append(nombre[:12])
    colores_bar = [COLORES.get(list(carteras.keys())[j], "gray")
                   for j in range(len(nombres))]
    ax[i].bar(range(len(nombres)), vals_num, color=colores_bar)
    ax[i].set_xticks(range(len(nombres)))
    ax[i].set_xticklabels(nombres, rotation=30, fontsize=7, ha="right")
    ax[i].set_title(met, fontsize=10)
    ax[i].axhline(0, color="black", linewidth=0.6)
plt.suptitle("Comparacion de metricas por estrategia", fontsize=12)
plt.tight_layout()
plt.savefig("data/processed/fig_bt_metricas.png", dpi=100)
plt.show()
