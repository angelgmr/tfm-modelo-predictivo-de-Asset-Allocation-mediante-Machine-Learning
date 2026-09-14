# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 09:05:00 2026

@author: angel
"""

# -*- coding: utf-8 -*-
"""
TFM - Modelo
===================================================================================
Seleccion, entrenamiento y test del modelo
- Carga del panel y separacion train/validacion/test
- Walk-forward con purga y embargo (bloque train+validacion)
- Metricas por fold: AUC, hit rate top2, lift, F1, accuracy
- Importancia de features por permutacion
- Explicabilidad SHAP
- Matriz de confusion (ultimo fold de validacion)
- TEST FINAL. Bloque congelado, evaluado una sola vez
- Guardar modelo entrenado con todo el historico
------------------------------
Modelo elegido: ExtraTreesRegressor (regresion sobre target_3m continuo)
Features: 20 Features finales seleccionadas
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from scipy.stats import spearmanr


try:
    import shap
    SHAP_DISPONIBLE = True
except ImportError:
    SHAP_DISPONIBLE = False

try:
    import joblib
    JOBLIB_DISPONIBLE = True
except ImportError:
    JOBLIB_DISPONIBLE = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
sns.set_style("darkgrid")



# CONFIGURACION
RUTA_PANEL      = "data/processed/panel_modelo_final.csv"
TARGET_CONTINUO = "target_3m"
TARGET_CONTINUO    = "target_3m"

# Parametros de walk-forward
HORIZONTE_SEMANAS = 13   # semanas del retorno forward que predice el target
VENTANA_RODANTE   = 156  # semanas de train en cada fold (3 años)
EMBARGO_SEMANAS   = 4    # semanas de embargo entre train y test
N_FOLDS           = 8    # folds del bloque train+validación

# Bloque de test congelado
N_ANIOS_TEST_FIJO = 2    # años más recientes reservados como test final
N_FOLDS_TEST_FIJO = 6    # subdivision del bloque de test (6 meses cada fold)

SEED        = 42
DIR_MODELOS = "data/processed/modelos_guardados"

# Pesos de la estrategia tactica (usados en la simulación del test)
# Se busca sobre ponderar a 35/35/15/15 en vez de 40/40/10/10 de la cartera permanente
PESO_SOBRE = 0.35
PESO_INFRA = 0.15

# FEATURES FINALES SELECCIONADAS EN ANALISIS_DEPURACION
FEATURES_FINALES = [
    "beta_Momentum_Global_ExUS_score",
    "beta_ret_vnq", "beta_ret_vtv", "beta_ret_vug",
    "beta_ret_iwm", "beta_ret_hyg", "beta_ret_slv",
    "beta_breakeven_5y", "beta_percentil_yield_10y",
    "beta_Curva_Invertida_score", "beta_vix", "impacto_vix_nivel",
    "beta_condiciones_financieras", "beta_Liquidez_FED_score",
    "beta_Riesgo_Sistemico_score",
    "mom_12_1", "ret_12m", "cruce_ma90_ma200",
    "downside_vol_6m", "amihud_3m",
]

# MODELO FINAL ELEGIDO
# Tras probar previamente en otro script con validación anidada y múltiples combinaciones de variables
# y parámetros, el modelo que mejor predice y es más robusto es ExtraTreesRegressor.
# Opciones probadas que mejor se ajustaban a este proyecto: "ExtraTrees", "Ridge", "ElasticNet", "GradBoosting", "MLP"
MODELO_FINAL_FIJO = "ExtraTrees"

# Modelos
CONFIGS_MODELO = {
    "ExtraTrees": (
        ExtraTreesRegressor(random_state=SEED, n_jobs=-1),
        {"n_estimators": [200], "max_depth": [3, 4, 6], "min_samples_leaf": [10, 20]},
        False,   
    ),
    "Ridge": (
        Ridge(),
        {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        True,
    ),
    "ElasticNet": (
        ElasticNet(max_iter=5000),
        {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.1, 0.5, 0.9]},
        True,
    ),
    "GradBoosting": (
        GradientBoostingRegressor(random_state=SEED),
        {"n_estimators": [100, 200], "max_depth": [2, 3], "learning_rate": [0.05, 0.1]},
        False,
    ),
    "MLP": (
        MLPRegressor(max_iter=500, random_state=SEED, early_stopping=True),
        {"hidden_layer_sizes": [(16, 8), (32, 16)], "alpha": [0.1, 1.0]},
        True,
    ),
}
if MODELO_FINAL_FIJO not in CONFIGS_MODELO:
    raise ValueError(
        f"MODELO_FINAL_FIJO='{MODELO_FINAL_FIJO}' no reconocido. "
        f"Opciones: {list(CONFIGS_MODELO)}")

est_base, grid_modelo, escalar_modelo = CONFIGS_MODELO[MODELO_FINAL_FIJO]


# CARGA Y SEPARACION TRAIN+VALIDACION / TEST

panel = pd.read_csv(RUTA_PANEL)
panel["fecha"] = pd.to_datetime(panel["fecha"]).dt.normalize()

features = [f for f in FEATURES_FINALES if f in panel.columns]
faltan   = [f for f in FEATURES_FINALES if f not in panel.columns]
if faltan:
    print(f"\nAVISO: estas features no estan en el panel (re-ejecuta el EDA):")
    print(f"  {faltan}\n")

# check para que no haya columnas del futuro como feature
cols_futuro = [c for c in features
               if c.startswith(("ret_fwd_", "exceso_", "target_",))]
if cols_futuro:
    raise ValueError(f"Features con informacion futura: {cols_futuro}")

datos = (panel.dropna(subset=features + [TARGET_CONTINUO])
              .sort_values("fecha")
              .reset_index(drop=True))


TICKERS    = sorted(datos["ticker"].unique())
COLS_DUMMY = [f"sector_{t}" for t in TICKERS[1:]]

# Bloque de test: reservado siempre, nunca visto durante la validacion
fecha_max       = datos["fecha"].max()
fecha_corte     = fecha_max - pd.Timedelta(weeks=52 * N_ANIOS_TEST_FIJO)
datos_train_val = datos[
    datos["fecha"] < fecha_corte - pd.Timedelta(weeks=HORIZONTE_SEMANAS + EMBARGO_SEMANAS)
].copy()
datos_test_fijo = datos[
    datos["fecha"] >= fecha_corte - pd.Timedelta(weeks=VENTANA_RODANTE)
].copy()

print(f"Panel: {datos.shape}  |  Features: {len(features)}  |  Activos: {TICKERS}")
print(f"Target continuo ({TARGET_CONTINUO}): media por activo:")
for t in TICKERS:
    m = datos[datos.ticker == t][TARGET_CONTINUO].mean()
    print(f"  {t}: {m:+.4f}")
print(f"\nSeparacion:")
print(f"  Train+Validacion: hasta {datos_train_val['fecha'].max().date()} "
     f"({len(datos_train_val):,} filas)")
print(f"  Test congelado  : desde {fecha_corte.date()} "
     f"({len(datos_test_fijo):,} filas)")
print(f"\nModelo: {MODELO_FINAL_FIJO}")


# FUNCIONES AUXILIARES
def matriz_features(df):
    dummies = pd.get_dummies(df["ticker"], prefix="sector")
    for c in COLS_DUMMY:
        if c not in dummies.columns:
            dummies[c] = 0
    return pd.concat([
        df[features].reset_index(drop=True),
        dummies[COLS_DUMMY].astype(int).reset_index(drop=True)
    ], axis=1)


def _spearman_scorer(estimator, X, y):
    # Scorer de Spearman 
    pred = estimator.predict(X)
    r, _ = spearmanr(pred, y)
    return r if not np.isnan(r) else 0.0


def generar_folds(df, ventana, horizonte, embargo, n_folds, col="fecha"):
    # Walk-forward con ventana rodante fija, purga y embargo
    fechas = np.sort(df[col].unique())
    primer_test = (pd.Timestamp(fechas[0])
                   + pd.Timedelta(weeks=ventana + horizonte + embargo))
    fechas_test = fechas[fechas >= primer_test.to_numpy()]
    n = len(fechas_test)
    if n < n_folds * 4:
        raise ValueError(f"Pocas fechas testables ({n}) para {n_folds} folds.")
    tam = n // n_folds
    folds = []
    for i in range(n_folds):
        ini, fin = i * tam, (i * tam + tam if i < n_folds - 1 else n)
        f_test_ini  = fechas_test[ini]
        f_test_fin  = fechas_test[fin - 1]
        f_purga     = pd.Timestamp(f_test_ini) - pd.Timedelta(weeks=horizonte)
        f_emb       = f_purga - pd.Timedelta(weeks=embargo)
        f_train_ini = f_emb - pd.Timedelta(weeks=ventana)
        idx_tr = df.index[(df[col] >= f_train_ini) & (df[col] < f_emb)].to_numpy()
        idx_te = df.index[(df[col] >= f_test_ini) & (df[col] <= f_test_fin)].to_numpy()
        if len(idx_tr) > 60 and len(idx_te) > 8:
            folds.append({"fold": i + 1, "idx_train": idx_tr, "idx_test": idx_te,
                          "test_ini": f_test_ini, "test_fin": f_test_fin})
    return folds

def spearman_por_fecha(df_fechas, score):
    # Correlacion de Spearman media entre el score predicho y target_3m, calculada por fecha
    d = df_fechas[["fecha", TARGET_CONTINUO]].copy()
    d["_s"] = np.asarray(score)
    corrs = []
    for _, g in d.groupby("fecha"):
        if len(g) >= 2:
            r, _ = spearmanr(g["_s"], g[TARGET_CONTINUO])
            if not np.isnan(r):
                corrs.append(r)
    return np.mean(corrs) if corrs else np.nan


def hit_rate_topN(df_fechas, score, n=2):
    # Fraccion de veces que el top-N predicho coincide con el top-N real
    d = df_fechas[["fecha", TARGET_CONTINUO]].copy()
    d["_s"] = np.asarray(score)
    aciertos, azar = [], []
    for _, g in d.groupby("fecha"):
        if len(g) < n:
            continue
        verdaderos = set(g.nlargest(n, TARGET_CONTINUO).index)
        predichos  = set(g.nlargest(n, "_s").index)
        aciertos.append(len(verdaderos & predichos) / n)
        azar.append(n / len(g))
    return (np.mean(aciertos), np.mean(azar)) if aciertos else (np.nan, np.nan)


def entrenar_modelo(X_tr, y_tr):
    # Ajusta MODELO_FINAL_FIJO (regresor) con GridSearchCV.
    # Entrena sobre TARGET_CONTINUO (target_3m).
    # Devuelve (modelo, escalador)
    if escalar_modelo:
        esc = StandardScaler().fit(X_tr)
        X_m = esc.transform(X_tr)
    else:
        esc = None
        X_m = X_tr

    if grid_modelo:
        busq = GridSearchCV(clone(est_base), grid_modelo,
                           cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
        busq.fit(X_m, y_tr)
        modelo = busq.best_estimator_
    else:
        modelo = clone(est_base)
        modelo.fit(X_m, y_tr)

    return modelo, esc

    return modelo, esc


def aplicar_escalador(X, esc):
    return esc.transform(X) if esc is not None else X


def top2_bin_desde_target(df):
    # Construye el top2 binario dinamicamente desde target_3m: en cada fecha, los 2 activos con mayor retorno relativo = 1, el resto = 0.
    return (df.groupby("fecha")[TARGET_CONTINUO]
              .transform(lambda x: x >= x.nlargest(2).min())
              .astype(int))


# WALK-FORWARD (bloque train+validacion)
folds = generar_folds(datos_train_val, VENTANA_RODANTE, HORIZONTE_SEMANAS,
                      EMBARGO_SEMANAS, N_FOLDS)

print(f"\n{'='*70}")
print(f"WALK-FORWARD VALIDACION -- {MODELO_FINAL_FIJO}")
print(f"({N_FOLDS} folds | ventana {VENTANA_RODANTE} sem | "
     f"purga {HORIZONTE_SEMANAS} sem | embargo {EMBARGO_SEMANAS} sem)")
print(f"{'='*70}")

resultados_val = []
importancias_folds = []
_modelo_ult = _esc_ult = _X_te_ult = _te_df_ult = None

for f in folds:
    tr = datos_train_val.loc[f["idx_train"]]
    te = datos_train_val.loc[f["idx_test"]]
    X_tr, X_te = matriz_features(tr), matriz_features(te)
    y_tr_cont = tr[TARGET_CONTINUO]
    y_te_bin  = top2_bin_desde_target(te)   

    modelo, esc = entrenar_modelo(X_tr, y_tr_cont)
    X_tr_m = aplicar_escalador(X_tr, esc)
    X_te_m = aplicar_escalador(X_te, esc)

    score_te = modelo.predict(X_te_m)        
    hit, azar = hit_rate_topN(te, score_te)
    sp = spearman_por_fecha(te, score_te)

    resultados_val.append({
        "fold":          f["fold"],
        "test_ini":      pd.Timestamp(f["test_ini"]).date(),
        "test_fin":      pd.Timestamp(f["test_fin"]).date(),
        "spearman":      sp,
        "roc_auc":       roc_auc_score(y_te_bin, score_te),
        "hit_rate_top2": hit,
        "lift_vs_azar":  hit / azar if (azar and azar > 0) else np.nan,
    })

    # Importancia por permutacion en el MISMO fold
    _res_imp = permutation_importance(modelo, X_te_m, te[TARGET_CONTINUO],
                                      n_repeats=5, random_state=SEED,
                                      scoring=_spearman_scorer, n_jobs=-1)
    importancias_folds.append(
        pd.Series(_res_imp.importances_mean, index=features + COLS_DUMMY))

    _modelo_ult = modelo
    _esc_ult    = esc
    _X_te_ult   = X_te_m
    _te_df_ult  = te

df_val = pd.DataFrame(resultados_val)
print(df_val[["fold", "test_ini", "test_fin", "spearman",
              "roc_auc", "hit_rate_top2", "lift_vs_azar"]].to_string(index=False))
print(f"\n{'Media':12s} Spearman={df_val.spearman.mean():+.4f}  "
     f"AUC={df_val.roc_auc.mean():.4f}  "
     f"Hit={df_val.hit_rate_top2.mean():.4f}  "
     f"Lift={df_val.lift_vs_azar.mean():.4f}")
print(f"{'Mediana':12s} Spearman={df_val.spearman.median():+.4f}  "
     f"AUC={df_val.roc_auc.median():.4f}")
print(f"{'Std':12s} Spearman={df_val.spearman.std():.4f}  "
     f"AUC={df_val.roc_auc.std():.4f}")

# Graficas walk-forward
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, color, label, azar_ref in [
        (axes[0], "roc_auc",  "steelblue",  "AUC",      0.5),
        (axes[1], "spearman", "darkorange",  "Spearman", 0.0)]:
    ax.bar(df_val.fold, df_val[col], color=color)
    ax.axhline(df_val[col].mean(),  color="red",  linestyle="--",
               label=f"Media {df_val[col].mean():+.3f}")
    ax.axhline(df_val[col].median(), color="purple", linestyle=":",
               label=f"Mediana {df_val[col].median():+.3f}")
    ax.axhline(azar_ref, color="gray", linestyle=":", alpha=0.5, label="Azar")
    ax.set_xlabel("Fold"); ax.set_ylabel(label)
    ax.set_title(f"{label} por fold -- {MODELO_FINAL_FIJO}"); ax.legend(fontsize=8)
plt.suptitle(f"Walk-forward validacion -- {MODELO_FINAL_FIJO} (regresor)", fontsize=12)
plt.tight_layout()
plt.savefig("data/processed/fig_mod_walkforward.png", dpi=100)
plt.show()


# IMPORTANCIA DE FEATURES
print(f"\n{'='*70}")
print("IMPORTANCIA DE FEATURES (permutacion, promedio de todos los folds)")
print(f"{'='*70}")

df_imp    = pd.DataFrame(importancias_folds)
imp_media = df_imp.mean().sort_values(ascending=False)
imp_std   = df_imp.std().reindex(imp_media.index)

df_imp_resumen = pd.DataFrame(
    {"perm_media": imp_media, "perm_std": imp_std}).head(25)
print(df_imp_resumen.to_string())

fig, ax = plt.subplots(figsize=(10, 7))
colores = ["steelblue" if v >= 0 else "salmon"
           for v in df_imp_resumen.perm_media]
ax.barh(df_imp_resumen.index, df_imp_resumen.perm_media, color=colores)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Importancia por permutacion (delta AUC de ranking)")
ax.set_title(f"Importancia de features -- {MODELO_FINAL_FIJO}")
plt.tight_layout()
plt.savefig("data/processed/fig_mod_importancia.png", dpi=100)
plt.show()


# Grafica Spearman y hit rate por fold 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Hit rate
hits = df_val["hit_rate_top2"].tolist()
etiquetas = [f"F{int(r.fold)}\n{str(r.test_ini)[:7]}" for _, r in df_val.iterrows()]
colores_hit = ["steelblue" if h >= 0.5 else "salmon" for h in hits]
axes[0].bar(range(len(hits)), hits, color=colores_hit, alpha=0.85)
axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1.2, label="Azar (50%)")
axes[0].axhline(np.mean(hits), color="red", linestyle="-.", linewidth=1.2,
               label=f"Media {np.mean(hits):.3f}")
for i, h in enumerate(hits):
    axes[0].text(i, h + 0.005, f"x{h/0.5:.2f}", ha="center", va="bottom",
                fontsize=8, color="navy")
axes[0].set_xticks(range(len(etiquetas)))
axes[0].set_xticklabels(etiquetas, fontsize=8)
axes[0].set_ylabel("Hit rate top2"); axes[0].set_ylim(0, 1)
axes[0].set_title(f"Hit rate top2 -- {MODELO_FINAL_FIJO}")
axes[0].legend(fontsize=9)

# Spearman
sp = df_val["spearman"].tolist()
colores_sp = ["steelblue" if s >= 0 else "salmon" for s in sp]
axes[1].bar(range(len(sp)), sp, color=colores_sp, alpha=0.85)
axes[1].axhline(0, color="gray", linestyle="--", linewidth=1.2, label="Azar (0)")
axes[1].axhline(np.mean(sp), color="red", linestyle="-.", linewidth=1.2,
               label=f"Media {np.mean(sp):+.3f}")
axes[1].set_xticks(range(len(etiquetas)))
axes[1].set_xticklabels(etiquetas, fontsize=8)
axes[1].set_ylabel("Spearman por fecha (medio)")
axes[1].set_title("Spearman de ranking por fold\n(azul = mejor que azar, rojo = peor)")
axes[1].legend(fontsize=9)

plt.suptitle(f"Rendimiento por fold -- {MODELO_FINAL_FIJO} (regresor)", fontsize=12)
plt.tight_layout()
plt.savefig("data/processed/fig_mod_hitrate_temporal.png", dpi=100)
plt.show()


# Grafica dispersion Spearman vs AUC de ranking por fold
fig, ax = plt.subplots(figsize=(7, 6))

for _, r in df_val.iterrows():
    sp_f  = r["spearman"]
    auc_f = r["roc_auc"]
    fold_id = int(r["fold"])
    if sp_f >= 0 and auc_f >= 0.5:
        color, estilo = "steelblue", "o"
    elif sp_f >= 0:
        color, estilo = "darkorange", "s"
    elif auc_f >= 0.5:
        color, estilo = "mediumpurple", "^"
    else:
        color, estilo = "salmon", "X"
    ax.scatter(auc_f, sp_f, s=120, color=color, marker=estilo, zorder=3)
    ax.annotate(f"F{fold_id}", (auc_f, sp_f),
               textcoords="offset points", xytext=(6, 4), fontsize=9)

ax.axhline(0,   color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
ax.axhline(df_val.spearman.mean(), color="darkorange", linestyle=":",
           linewidth=1, label=f"Media Spearman {df_val.spearman.mean():+.3f}")
ax.axvline(df_val.roc_auc.mean(), color="steelblue", linestyle=":",
           linewidth=1, label=f"Media AUC {df_val.roc_auc.mean():.3f}")
ax.set_xlabel("AUC de ranking"); ax.set_ylabel("Spearman por fecha")
ax.set_title(f"Spearman vs AUC por fold -- {MODELO_FINAL_FIJO}")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig("data/processed/fig_mod_dispersion_spearman_auc.png", dpi=100)
plt.show()



# EXPLICABILIDAD SHAP
if not SHAP_DISPONIBLE:
    print("\nSHAP no disponible. Instala con: pip install shap")
elif not hasattr(_modelo_ult, "estimators_"):
    print(f"\nSHAP TreeExplainer requiere modelo de arboles. "
         f"{MODELO_FINAL_FIJO} no es de arboles -- seccion omitida.")
else:
    print(f"\n{'='*70}")
    print(f"EXPLICABILIDAD SHAP -- {MODELO_FINAL_FIJO} (ultimo fold)")
    print(f"{'='*70}")
    explainer = shap.TreeExplainer(_modelo_ult)
    sv = explainer.shap_values(_X_te_ult)

    if isinstance(sv, list):
        sv = sv[0]

    cols_shap = features + COLS_DUMMY
    imp_shap = (pd.Series(np.abs(sv).mean(axis=0), index=cols_shap)
                  .sort_values(ascending=False))
    print("Importancia SHAP media (top 15):")
    print(imp_shap.head(15).to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(imp_shap.head(15).index[::-1],
            imp_shap.head(15).values[::-1], color="steelblue")
    ax.set_xlabel("|SHAP value| medio")
    ax.set_title(f"Importancia SHAP -- {MODELO_FINAL_FIJO}")
    plt.tight_layout()
    plt.savefig("data/processed/fig_shap_importancia.png", dpi=100)
    plt.show()

    shap.summary_plot(sv, _X_te_ult, feature_names=cols_shap,
                     max_display=15, show=False)
    plt.tight_layout()
    plt.savefig("data/processed/fig_shap_summary.png", dpi=100)
    plt.show()



# TEST FINAL: bloque congelado, evaluado solo una vez
print(f"\n{'='*70}")
print(f"TEST FINAL")
print(f"Modelo: {MODELO_FINAL_FIJO}  |  Features: {len(features)}")
print(f"{'='*70}")

folds_test = generar_folds(datos_test_fijo, VENTANA_RODANTE, HORIZONTE_SEMANAS,
                           EMBARGO_SEMANAS, N_FOLDS_TEST_FIJO)

resultados_test = []
_filas_test_graf = []    
for f in folds_test:
    tr_f = datos_test_fijo.loc[f["idx_train"]]
    te_f = datos_test_fijo.loc[f["idx_test"]]
    X_tr_f, X_te_f = matriz_features(tr_f), matriz_features(te_f)
    y_tr_cont_f = tr_f[TARGET_CONTINUO]
    y_te_bin_f  = top2_bin_desde_target(te_f)   

    modelo_f, esc_f = entrenar_modelo(X_tr_f, y_tr_cont_f)
    X_te_f_m = aplicar_escalador(X_te_f, esc_f)

    score_f  = modelo_f.predict(X_te_f_m)
    sp_f     = spearman_por_fecha(te_f, score_f)
    auc_f    = roc_auc_score(y_te_bin_f, score_f)

    resultados_test.append({
        "fold":      f["fold"],
        "periodo":   (f"{pd.Timestamp(f['test_ini']).date()} -> "
                     f"{pd.Timestamp(f['test_fin']).date()}"),
        "spearman":  sp_f,
        "roc_auc":   auc_f,
    })
    print(f"  Fold {f['fold']} ({resultados_test[-1]['periodo']}): "
         f"Spearman={sp_f:+.4f}  AUC={auc_f:.4f}")

    # Guardar predicciones de este fold para las graficas (evita re-entrenar)
    _df_f = te_f[["fecha", "ticker", TARGET_CONTINUO]].copy()
    _df_f["score"] = score_f
    _df_f["fold"]  = f["fold"]
    _df_f["rank_real"] = _df_f.groupby("fecha")[TARGET_CONTINUO].rank(
        ascending=False, method="first").astype(int)
    _df_f["rank_pred"] = _df_f.groupby("fecha")["score"].rank(
        ascending=False, method="first").astype(int)
    _df_f["top2_real"] = (_df_f["rank_real"] <= 2).astype(int)
    _df_f["top2_pred"] = (_df_f["rank_pred"] <= 2).astype(int)
    _filas_test_graf.append(_df_f)

df_test = pd.DataFrame(resultados_test)
print(f"\nRESULTADO FINAL:")
print(f"  Spearman medio: {df_test.spearman.mean():+.4f} "
     f"(mediana {df_test.spearman.median():+.4f}, "
     f"std {df_test.spearman.std():.4f})")
print(f"  AUC medio     : {df_test.roc_auc.mean():.4f} "
     f"(mediana {df_test.roc_auc.median():.4f})")


# GRAFICAS DEL TEST FINAL
# Predicciones ya acumuladas en el bucle del test (sin re-entrenar)
_df_test_g = pd.concat(_filas_test_graf, ignore_index=True)

# Gráfica Spearman y AUC por fold
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, col, color, label, azar in [
        (axes[0], "spearman", "steelblue",  "Spearman", 0.0),
        (axes[1], "roc_auc",  "darkorange",  "AUC",     0.5)]:
    vals = df_test[col].tolist()
    etq  = [f"F{int(r.fold)}\n{str(r.periodo)[:7]}" for _, r in df_test.iterrows()]
    colores_b = [color if v > azar else "salmon" for v in vals]
    ax.bar(range(len(vals)), vals, color=colores_b, alpha=0.85)
    ax.axhline(np.mean(vals), color="red", linestyle="--",
               label=f"Media {np.mean(vals):+.3f}")
    ax.axhline(azar, color="gray", linestyle=":", alpha=0.7, label="Azar")
    ax.set_xticks(range(len(etq)))
    ax.set_xticklabels(etq, fontsize=8)
    ax.set_ylabel(label)
    ax.set_title(f"{label} por fold -- TEST ({MODELO_FINAL_FIJO})")
    ax.legend(fontsize=8)

plt.suptitle(f"Resultados del test congelado ({N_ANIOS_TEST_FIJO} años) -- {MODELO_FINAL_FIJO}",
             fontsize=12)
plt.tight_layout()
plt.savefig("data/processed/fig_test_walkforward.png", dpi=100)
plt.show()

# Gráfica comparacion validacion vs test
# Permite ver si hay degradacion entre validación y test
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

metricas_comp = [("spearman", "Spearman", 0.0), ("roc_auc", "AUC", 0.5)]
for ax, (met, label, azar) in zip(axes, metricas_comp):
    val_media  = df_val[met].mean()
    val_std    = df_val[met].std()
    test_media = df_test[met].mean()
    test_std   = df_test[met].std()

    x = [0, 1]
    medias = [val_media, test_media]
    stds   = [val_std, test_std]
    colors_c = ["steelblue", "darkorange"]
    bars = ax.bar(x, medias, color=colors_c, alpha=0.85, width=0.5)
    ax.errorbar(x, medias, yerr=stds, fmt="none", color="black",
                capsize=5, linewidth=1.5)
    ax.axhline(azar, color="gray", linestyle=":", alpha=0.7, label="Azar")
    ax.set_xticks(x)
    ax.set_xticklabels(["Validacion\n(walk-forward)", f"Test\n(congelado)"])
    ax.set_ylabel(label)
    ax.set_title(f"{label}: validacion vs test")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, medias):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f"{val:+.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.suptitle(f"Comparacion validacion vs test -- {MODELO_FINAL_FIJO}", fontsize=12)
plt.tight_layout()
plt.savefig("data/processed/fig_test_comparacion.png", dpi=100)
plt.show()


# Gráfica Retorno acumulado simulado en el test
# Compara tres carteras sobre el bloque de test:
#   - Ranking PREDICHO: regla 35/35/15/15 sobre el ranking del modelo
#   - Proporcional score:pesos proporcionales al score continuo del regresor
#   - Cartera Permanente: 25/25/25/25
if "close" in datos_test_fijo.columns:
    _ret_w = (datos_test_fijo.pivot(index="fecha", columns="ticker", values="close")
              .pct_change().dropna(how="all"))
    _fechas_sim = sorted(_df_test_g["fecha"].unique())
    _val = {"real": 100.0, "pred": 100.0, "prop": 100.0, "estat": 100.0}
    _series = {k: [] for k in _val}

    for _fecha in _fechas_sim:
        if _fecha not in _ret_w.index:
            continue
        _ret = _ret_w.loc[_fecha]
        _g = _df_test_g[_df_test_g.fecha == _fecha]
        if len(_g) < len(TICKERS):
            continue

        # Pesos por ranking (real y predicho): PESO_SOBRE a los 2 primeros
        _top2_pred = set(_g[_g["rank_pred"] <= 2]["ticker"])
        _top2_real = set(_g[_g["rank_real"] <= 2]["ticker"])
        _w_pred  = {t: (PESO_SOBRE if t in _top2_pred else PESO_INFRA) for t in TICKERS}
        _w_real  = {t: (PESO_SOBRE if t in _top2_real else PESO_INFRA) for t in TICKERS}

        # Pesos proporcionales al score continuo, reescalado a [0,1] dentro
        # de la fecha para que sean positivos y sumen 1. Es el uso directo
        # del regresor: los activos con score mas alto pesan mas, de forma
        # gradual en vez de en dos escalones.
        _sc = _g.set_index("ticker")["score"].reindex(TICKERS)
        _rango = _sc.max() - _sc.min()
        if _rango > 0:
            _norm = (_sc - _sc.min()) / _rango + 0.15   # suelo 0.15 -> nadie a cero
        else:
            _norm = pd.Series(1.0, index=TICKERS)
        _w_prop = (_norm / _norm.sum()).to_dict()

        _val["real"]  *= (1 + sum(_w_real[t]  * _ret.get(t, 0) for t in TICKERS))
        _val["pred"]  *= (1 + sum(_w_pred[t]  * _ret.get(t, 0) for t in TICKERS))
        _val["prop"]  *= (1 + sum(_w_prop[t]  * _ret.get(t, 0) for t in TICKERS))
        _val["estat"] *= (1 + sum(0.25        * _ret.get(t, 0) for t in TICKERS))
        for _k in _val:
            _series[_k].append({"fecha": _fecha, "valor": _val[_k]})

    if _series["pred"]:
        _dfs = {k: pd.DataFrame(v).set_index("fecha") for k, v in _series.items()}

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(_dfs["pred"].index,  _dfs["pred"].valor,
                color="steelblue", linewidth=2,
                label=f"Ranking PREDICHO {int(PESO_SOBRE*100)}/{int(PESO_SOBRE*100)}/"
                      f"{int(PESO_INFRA*100)}/{int(PESO_INFRA*100)}")
        ax.plot(_dfs["prop"].index,  _dfs["prop"].valor,
                color="darkorange", linewidth=2,
                label="Cartera proporcional a prediccion del modelo")
        ax.plot(_dfs["estat"].index, _dfs["estat"].valor,
                color="gray", linewidth=1.5, alpha=0.85,
                label="Cartera permanente 25/25/25/25")
        ax.axhline(100, color="black", linewidth=0.6, linestyle=":")
        ax.set_ylabel("Valor (base 100)")
        ax.set_title(f"Retorno simulado en el TEST -- {MODELO_FINAL_FIJO}\n"
                     f"(test congelado de {N_ANIOS_TEST_FIJO} años | base 100)")
        ax.legend(fontsize=8, loc="upper left")
        plt.tight_layout()
        plt.savefig("data/processed/fig_test_retorno_simulado.png", dpi=100)
        plt.show()

        # Resumen numerico
        print(f"\n  Retorno total en el test ({N_ANIOS_TEST_FIJO} años):")
        for _k, _lab in [("pred", "Ranking PREDICHO"),
                         ("prop", "Proporcional score"),
                         ("estat","Cartera Permanente original")]:
            _r = _dfs[_k].valor.iloc[-1] / 100 - 1
            print(f"    {_lab}: {_r*100:+7.2f}%")


# GUARDAR MODELO ENTRENADO CON TODO EL HISTORICO
if not JOBLIB_DISPONIBLE:
    print("\njoblib no instalado. Instala con: pip install joblib")
else:
    print(f"\n{'='*70}")
    print(f"{MODELO_FINAL_FIJO} GUARDADO")
    print(f"{'='*70}")
    # Se entrena con todo el historico (train_val + test)
    X_todo = matriz_features(datos)
    y_todo = datos[TARGET_CONTINUO]
    modelo_prod, esc_prod = entrenar_modelo(X_todo, y_todo)
    X_todo_m  = aplicar_escalador(X_todo, esc_prod)
    score_prod = modelo_prod.predict(X_todo_m)

    print(f"Entrenado con: {datos['fecha'].min().date()} -> "
         f"{datos['fecha'].max().date()} ({len(datos):,} filas)")

    os.makedirs(DIR_MODELOS, exist_ok=True)
    paquete = {
        "modelo":              modelo_prod,
        "escalador":           esc_prod,
        "features":            features,
        "cols_dummy":          COLS_DUMMY,
        "tickers":             TICKERS,
        "target":              TARGET_CONTINUO,
        "tipo":                "regresor",
        "modelo_nombre":       MODELO_FINAL_FIJO,
        "fecha_entreno":       datetime.today().strftime("%Y-%m-%d"),
        "spearman_test_medio": df_test.spearman.mean(),
        "auc_test_medio":      df_test.roc_auc.mean(),
    }
    ruta = (f"{DIR_MODELOS}/modelo_{MODELO_FINAL_FIJO}_"
            f"{datetime.today().strftime('%Y%m%d')}.joblib")
    joblib.dump(paquete, ruta)
    print(f"Guardado en: {ruta}")

