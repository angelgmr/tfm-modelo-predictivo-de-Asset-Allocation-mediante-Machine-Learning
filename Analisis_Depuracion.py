# -*- coding: utf-8 -*-
"""
Created on Sat Aug 8 20:10:11 2026

@author: angel
"""

# -*- coding: utf-8 -*-

"""
TFM - Pipeline EDA
=======================================================================
Análisis y depuración de datos.
- Construcción de betas y variables sintéticas
- Eliminación de variables originales con IC < que su beta construida
- Tratamiento de nulos y atípicos
- Correlación Spearman
- Chi cuadrado y V de Cramer
- IC + consistencia de signo
- VIF
- Seleccion de features
"""
 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant
import os
from collections import Counter
 
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 170)
sns.set_style("darkgrid")
 
RUTA_PANEL = "data/processed/panel_modelo.csv"
TARGET_CONTINUO = "target_3m"
HORIZONTE_SEMANAS = 13
VENTANA_RODANTE = 156
N_FOLDS = 5
UMBRAL_CORR = 0.75   
 

# CARGA 
panel = pd.read_csv(RUTA_PANEL)
panel["fecha"] = pd.to_datetime(panel["fecha"]).dt.normalize()

# Se excluye la columna Activos_cartera
panel = panel.drop(columns=["n_ACTIVOS_CARTERA"], errors="ignore")

print(f"Panel cargado: {panel.shape}")
 
# VARIABLES SINTETICAS DE REGIMEN
VENTANA_PERCENTIL_REGIMEN = 100  # ~2 años para el percentil rodante

def _serie_plana(col):

    return panel.drop_duplicates("fecha").set_index("fecha")[col].sort_index()

def _percentil_rodante(s):
    # solo usa los ultimos VENTANA_PERCENTIL_REGIMEN datos.
    return s.rolling(VENTANA_PERCENTIL_REGIMEN, min_periods=52).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).mean() if len(x) > 1 else np.nan,
        raw=False)

def _percentil_expanding(s):
    # historico completo hasta cada fecha

    return s.expanding(min_periods=26).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).mean() if len(x) > 1 else np.nan,
        raw=False)

def _media_tolerante(*series):
    # Media de varias series que IGNORA componentes puntualmente nulos
    
    df_temp = pd.concat(series, axis=1)
    return df_temp.mean(axis=1, skipna=True)


print(f"\n{'='*70}\nVARIABLES SINTETICAS DE REGIMEN\n{'='*70}")

# --- Liquidez_FED_score ---
if all(c in panel.columns for c in ["crec_balance_fed_6m", "d_reservas_bancarias_6m", "reverse_repo_pct2y"]):
    pct_balance_fed = _percentil_expanding(_serie_plana("crec_balance_fed_6m"))  
    pct_reservas = _percentil_expanding(_serie_plana("d_reservas_bancarias_6m"))
    pct_reverse_repo = _percentil_rodante(_serie_plana("reverse_repo_pct2y"))
    score_liquidez_fed = _media_tolerante(pct_balance_fed, pct_reservas, 1 - pct_reverse_repo)
    panel["Liquidez_FED_score"] = panel["fecha"].map(score_liquidez_fed)
    n_validos = panel["Liquidez_FED_score"].notna().sum()
    print(f"  Liquidez_FED_score: {n_validos} validos ({100*n_validos/len(panel):.1f}%)")
else:
    print("  AVISO: faltan componentes de Liquidez_FED_score, se omite")

# ---  Riesgo_Sistemico_score ---
_componentes_riesgo = ["tightening_credito", "d_desempleo_6m", "d_produccion_ind_12m",
                       "d_deuda_publica_pib_1y", "d_deuda_hogares_pib_1y"]
if all(c in panel.columns for c in _componentes_riesgo):
    pct_tightening = _percentil_expanding(_serie_plana("tightening_credito")) 
    pct_desempleo_delta = _percentil_expanding(_serie_plana("d_desempleo_6m"))  
    pct_produccion_delta = _percentil_expanding(_serie_plana("d_produccion_ind_12m")) 
    pct_deuda_publica = _percentil_expanding(_serie_plana("d_deuda_publica_pib_1y"))  
    pct_deuda_hogares = _percentil_expanding(_serie_plana("d_deuda_hogares_pib_1y"))
    score_riesgo_sistemico = _media_tolerante(pct_tightening, pct_desempleo_delta,
                                              1 - pct_produccion_delta, pct_deuda_publica,
                                              pct_deuda_hogares)
    panel["Riesgo_Sistemico_score"] = panel["fecha"].map(score_riesgo_sistemico)
    n_validos = panel["Riesgo_Sistemico_score"].notna().sum()
    print(f"  Riesgo_Sistemico_score: {n_validos} validos ({100*n_validos/len(panel):.1f}%)")
else:
    faltan = [c for c in _componentes_riesgo if c not in panel.columns]
    print(f"  AVISO: faltan componentes de Riesgo_Sistemico_score ({faltan})")

# ---  RiskOnOff_score ---
if "Liquidez_FED_score" in panel.columns and "Riesgo_Sistemico_score" in panel.columns:
    panel["RiskOnOff_score"] = panel["Riesgo_Sistemico_score"] - panel["Liquidez_FED_score"]
    n_validos = panel["RiskOnOff_score"].notna().sum()
    print(f"  RiskOnOff_score: {n_validos} validos ({100*n_validos/len(panel):.1f}%)")
else:
    print("  AVISO: faltan Liquidez_FED_score o Riesgo_Sistemico_score")

# --- Curva_Invertida_score ---
if all(c in panel.columns for c in ["spread_curva", "spread_curva_3m10y", "spread_curva_10y30y"]):
    pct_sc1 = _percentil_rodante(_serie_plana("spread_curva"))
    pct_sc2 = _percentil_rodante(_serie_plana("spread_curva_3m10y"))
    pct_sc3 = _percentil_rodante(_serie_plana("spread_curva_10y30y"))
    score_curva_invertida = 1 - _media_tolerante(pct_sc1, pct_sc2, pct_sc3)
    panel["Curva_Invertida_score"] = panel["fecha"].map(score_curva_invertida)
    n_validos = panel["Curva_Invertida_score"].notna().sum()
    print(f"  Curva_Invertida_score: {n_validos} validos ({100*n_validos/len(panel):.1f}%)")
else:
    print("  AVISO: faltan componentes de Curva_Invertida_score")

# --- Momentum_Global_ExUS_score ---
if all(c in panel.columns for c in ["ret_vgk", "ret_ewj", "ret_eem"]):
    pct_vgk = _percentil_rodante(_serie_plana("ret_vgk"))
    pct_ewj = _percentil_rodante(_serie_plana("ret_ewj"))
    pct_eem = _percentil_rodante(_serie_plana("ret_eem"))
    score_momentum_global = _media_tolerante(pct_vgk, pct_ewj, pct_eem)
    panel["Momentum_Global_ExUS_score"] = panel["fecha"].map(score_momentum_global)
    n_validos = panel["Momentum_Global_ExUS_score"].notna().sum()
    print(f"  Momentum_Global_ExUS_score: {n_validos} validos ({100*n_validos/len(panel):.1f}%)")
else:
    print("  AVISO: faltan componentes de Momentum_Global_ExUS_score")


# CONSTRUCCION DE BETAS PARA TODAS LAS FEATURES PLANAS 
VENTANA_BETA = 52  # semanas
  
FACTORES_PRIORITARIOS = {
    "vix_nivel":      ("beta_vix",       "pct_change"),  # retorno semanal del VIX
    "dolar_nivel":    ("beta_dolar",     "pct_change"),  # retorno semanal del dolar
    "tipo_real_10y":  ("beta_realrate",  "diff"),        # cambio semanal (es un tipo, no un indice)
    "fedfunds_nivel": ("beta_fedfunds",  "diff"),        # cambio semanal
    "breakeven_10y":  ("beta_breakeven", "diff"),        # cambio semanal
}
 
print("\n" + "="*70 + "\nCONSTRUYENDO BETAS (cambio semanal puro)\n" + "="*70)
for col_factor, (nombre_beta, transformacion) in FACTORES_PRIORITARIOS.items():
    if col_factor not in panel.columns:
        print(f"  AVISO: falta {col_factor} en el panel, se omite {nombre_beta}")
        continue
    factor_serie = panel.drop_duplicates("fecha").set_index("fecha")[col_factor].sort_index()
    factor_cambio = (factor_serie.pct_change() if transformacion == "pct_change"
                     else factor_serie.diff())
 
    resultado = pd.Series(index=panel.index, dtype=float)
    for ticker, grupo in panel.groupby("ticker"):
        grupo = grupo.sort_values("fecha")
        ret_1w = grupo["close"].pct_change()
        factor_alineado = factor_cambio.reindex(grupo["fecha"]).to_numpy()
        factor_pd = pd.Series(factor_alineado, index=grupo.index)
        cov = ret_1w.rolling(VENTANA_BETA).cov(factor_pd)
        var = factor_pd.rolling(VENTANA_BETA).var()
        resultado.loc[grupo.index] = (cov / var).to_numpy()
 
    panel[nombre_beta] = resultado
    n_validos = panel[nombre_beta].notna().sum()
    print(f"  {nombre_beta:16s} <- {col_factor:16s} ({transformacion:11s}): "
         f"{n_validos} validos ({100*n_validos/len(panel):.1f}%)")
 
# --- Resto de features ---
 
FACTORES_CAMBIO_DIRECTO = [
    "d_spread_curva_3m", "d_tipo_real_3m", "d_fedfunds_6m", "d_spread_hy_3m",
    "crec_balance_fed_6m", "d_reservas_bancarias_6m", "inflacion_yoy",
    "d_desempleo_6m", "d_produccion_ind_12m", "d_confianza_3m", "d_m2_12m",
    "d_petroleo_wti_3m", "d_cobre_3m", "d_gas_natural_3m",
    "vix_cambio_1m", "vix_pendiente", "d_dolar_3m",
    "ret_slv", "ret_eem", "ret_iwm", "ret_vtv", "ret_vug", "ret_hyg",
    "ret_vnq",             
    "d_prob_recesion_3m",   
    "ret_vgk", "ret_aaxj", "ret_ewj",
    "d_tipo_bce_6m",
    "crec_prestamos_totales_6m", "d_deuda_publica_pib_1y", "d_deuda_hogares_pib_1y",
]
FACTORES_NIVEL_DIFF = [
    "spread_curva", "spread_curva_3m10y", "yield_10y", "yield_2y", "yield_30y",
    "spread_curva_10y30y", "breakeven_5y", "spread_high_yield",
    "condiciones_financieras", "stress_financiero", "reverse_repo_pct2y",
    "desempleo_nivel", "spread_wti_brent",
    "prob_recesion",
    "tightening_credito",
    "oro_real",   
    "oro_real_percentil_5y",
    "percentil_yield_10y",  
    "tipo_bce",
    "deuda_publica_pib", "deuda_hogares_pib",
    "Liquidez_FED_score", "Riesgo_Sistemico_score", "RiskOnOff_score",
    "Curva_Invertida_score", "Momentum_Global_ExUS_score",
]
 
print("\n" + "="*70 + "\nCONSTRUYENDO BETAS\n" + "="*70)
 
nuevas_betas = []
for grupo, transformacion in [(FACTORES_CAMBIO_DIRECTO, "directo"),
                              (FACTORES_NIVEL_DIFF, "diff")]:
    for col_factor in grupo:
        if col_factor not in panel.columns:
            print(f"  AVISO: {col_factor} no esta en el panel")
            continue
        nombre_beta = f"beta_{col_factor}"
 
        factor_serie = panel.drop_duplicates("fecha").set_index("fecha")[col_factor].sort_index()
        factor_cambio = factor_serie if transformacion == "directo" else factor_serie.diff()
 
        resultado = pd.Series(index=panel.index, dtype=float)
        for ticker, grupo_t in panel.groupby("ticker"):
            grupo_t = grupo_t.sort_values("fecha")
            ret_1w = grupo_t["close"].pct_change()
            factor_alineado = factor_cambio.reindex(grupo_t["fecha"]).to_numpy()
            factor_pd = pd.Series(factor_alineado, index=grupo_t.index)
            cov = ret_1w.rolling(VENTANA_BETA).cov(factor_pd)
            var = factor_pd.rolling(VENTANA_BETA).var()
            resultado.loc[grupo_t.index] = (cov / var).to_numpy()
 
        panel[nombre_beta] = resultado
        nuevas_betas.append(nombre_beta)
 
n_validas = sum(panel[b].notna().sum() > 0 for b in nuevas_betas)
print(f"Betas construidas: {len(nuevas_betas)} ({n_validas} con datos validos)")
print(nuevas_betas)
 
 
# Interaciones macro por beta. impacto = nivel_actual x beta)
 
PARES_INTERACCION = {
    "vix_nivel":     "beta_vix_cambio_1m",
    "breakeven_5y":  "beta_breakeven_5y",
    "yield_10y":     "beta_yield_10y",
    "ret_eem":       "beta_ret_eem",
}
 
print("\n" + "="*70 + "\nCONSTRUYENDO INTERACCIONES MACRO x BETA (priorizadas)\n" + "="*70)
nuevas_interacciones = []
for macro_col, beta_col in PARES_INTERACCION.items():
    if macro_col in panel.columns and beta_col in panel.columns:
        nombre = f"impacto_{macro_col}"
        panel[nombre] = panel[macro_col] * panel[beta_col]
        nuevas_interacciones.append(nombre)
        print(f"  {nombre} = {macro_col} x {beta_col}")
    else:
        print(f"  AVISO: falta {macro_col} o {beta_col}")
 
 
# LIMPIA ORIGINAL vs BETA: por cada par, se elimina la mas debil
 
datos_ic_rapido = panel.dropna(subset=[TARGET_CONTINUO])
pares_factor_beta = [(f, f"beta_{f}") for f in FACTORES_CAMBIO_DIRECTO + FACTORES_NIVEL_DIFF
                     if f in panel.columns and f"beta_{f}" in panel.columns]
 
print(f"\n{'='*70}\nPODA ORIGINAL vs BETA (se elimina la mas debil de cada par)\n{'='*70}")
eliminadas_por_par = []
for original, beta in pares_factor_beta:
    ic_original = abs(datos_ic_rapido[original].corr(datos_ic_rapido[TARGET_CONTINUO], method="spearman"))
    ic_beta = abs(datos_ic_rapido[beta].corr(datos_ic_rapido[TARGET_CONTINUO], method="spearman"))
    if pd.isna(ic_original) or pd.isna(ic_beta):
        continue
    if ic_beta >= ic_original:
        panel = panel.drop(columns=[original])
        eliminadas_por_par.append((original, ic_original, beta, ic_beta))
    else:
        panel = panel.drop(columns=[beta])
        eliminadas_por_par.append((beta, ic_beta, original, ic_original))
 
for eliminada, ic_perdedora, ganadora, ic_ganadora in eliminadas_por_par:
    print(f"  Eliminada: {eliminada:28s} (|IC|={ic_perdedora:.4f})  "
         f"<- gana {ganadora:28s} (|IC|={ic_ganadora:.4f})")
print(f"\n{len(eliminadas_por_par)} features eliminadas de {len(pares_factor_beta)} pares comparados")
 
 
# DEFINICION DEL UNIVERSO DE FEATURES
 
no_features = {"fecha", "ticker", "close", "volume", "n_sectores"} | \
    {c for c in panel.columns if c.startswith(("ret_fwd_", "exceso_",
                                               "target_", "top2_"))}
features = [c for c in panel.columns if c not in no_features
            and panel[c].dtype in [np.float64, np.int64, float, int]]
 
datos = panel.dropna(subset=[TARGET_CONTINUO]).copy()
datos = datos.sort_values(["ticker", "fecha"]).reset_index(drop=True)
 
print(f"Panel: {panel.shape} | Features candidatas: {len(features)}")
print(f"Rango: {panel.fecha.min().date()} -> {panel.fecha.max().date()}")
print(f"Activos: {sorted(panel.ticker.unique())}")
print(f"Filas con target: {len(datos)}")
 
 

# NULOS Y MISSING 
print("\n" + "="*70 + "\n2. ANALISIS DE NULOS Y MISSING\n" + "="*70)
 
# Calentamiento dinamico en vez de fijo 
datos_ord = datos.sort_values(["ticker", "fecha"])
datos_ord["semana_ticker"] = datos_ord.groupby("ticker").cumcount()
 
def primera_posicion_valida(serie):
    valida = serie.notna().to_numpy()
    return valida.argmax() if valida.any() else len(serie)
 
resumen_nulos = []
for f in features:
    total = len(datos_ord)
    n_nulos = datos_ord[f].isna().sum()
    pct_total = 100 * n_nulos / total
 
    calentamiento_por_ticker = datos_ord.groupby("ticker")[f].transform(primera_posicion_valida)
    es_calentamiento = datos_ord["semana_ticker"] < calentamiento_por_ticker
    nulos_inicio = datos_ord.loc[es_calentamiento, f].isna().sum()
    nulos_fuera = n_nulos - nulos_inicio
    pct_fuera = 100 * nulos_fuera / total
 
    nulos_por_ticker = (datos_ord.loc[~es_calentamiento]
                        .groupby("ticker")[f].apply(lambda s: s.isna().sum()))
    max_concentracion = (nulos_por_ticker.max() / nulos_fuera) if nulos_fuera > 0 else 0
    ticker_concentrado = nulos_por_ticker.idxmax() if nulos_fuera > 0 else None
 
    if n_nulos == 0:
        tipo = "ninguno"
    elif pct_fuera < 2.0:
        tipo = "estructural_calentamiento"
    elif max_concentracion > 0.85:
        tipo = f"estructural_activo({ticker_concentrado})"
    else:
        tipo = "puntual"

    calentamiento_medio_semanas = calentamiento_por_ticker.groupby(
        datos_ord["ticker"]).first().mean()
    pct_calentamiento = 100 * calentamiento_medio_semanas / (
        datos_ord.groupby("ticker").size().mean())
 
    resumen_nulos.append({"feature": f, "n_nulos": n_nulos, "pct_total": pct_total,
                          "nulos_fuera_calentamiento": nulos_fuera,
                          "pct_fuera": pct_fuera, "tipo_missing": tipo,
                          "calentamiento_semanas": calentamiento_medio_semanas,
                          "pct_calentamiento": pct_calentamiento})
 
df_nulos = pd.DataFrame(resumen_nulos).sort_values("pct_total", ascending=False)
print(df_nulos.to_string(index=False))
 
# EXCLUSION: features con >50% de nulos
 
UMBRAL_EXCLUSION_NULOS = 50.0
excluidas_por_nulos = df_nulos[df_nulos.pct_total > UMBRAL_EXCLUSION_NULOS]["feature"].tolist()
if excluidas_por_nulos:
    print(f"\nEXCLUIDAS por tener > {UMBRAL_EXCLUSION_NULOS:.0f}% de nulos totales "
         f"({len(excluidas_por_nulos)}):")
    for f in excluidas_por_nulos:
        pct = df_nulos.loc[df_nulos.feature == f, "pct_total"].values[0]
        print(f"  {f:30s} {pct:.1f}% nulos")
    features = [f for f in features if f not in excluidas_por_nulos]
    panel = panel.drop(columns=excluidas_por_nulos, errors="ignore")
    df_nulos = df_nulos[~df_nulos.feature.isin(excluidas_por_nulos)].reset_index(drop=True)

else:
    print(f"\nNinguna feature supera el {UMBRAL_EXCLUSION_NULOS:.0f}% de nulos totales.")
 

# EXCLUSION: calentamiento estructural DEMASIADO LARGO 
UMBRAL_CALENTAMIENTO_SEMANAS = 260  # ~5 años

# Exclusion MANUAL: features que no cruzan UMBRAL_CALENTAMIENTO_SEMANAS
EXCLUSION_MANUAL = ["beta_ret_aaxj"]
 
excluidas_por_calentamiento = df_nulos[
    df_nulos.calentamiento_semanas > UMBRAL_CALENTAMIENTO_SEMANAS]["feature"].tolist()
if excluidas_por_calentamiento:
    print(f"\nEXCLUIDAS por calentamiento estructural > {UMBRAL_CALENTAMIENTO_SEMANAS} semanas "
         f"(~{UMBRAL_CALENTAMIENTO_SEMANAS/52:.0f} años) ({len(excluidas_por_calentamiento)}):")
    for f in excluidas_por_calentamiento:
        row = df_nulos.loc[df_nulos.feature == f].iloc[0]
        print(f"  {f:30s} {row.calentamiento_semanas:.0f} semanas "
             f"({row.pct_calentamiento:.1f}% del panel)")
    features = [f for f in features if f not in excluidas_por_calentamiento]
    df_nulos = df_nulos[~df_nulos.feature.isin(excluidas_por_calentamiento)].reset_index(drop=True)
    panel = panel.drop(columns=excluidas_por_calentamiento, errors="ignore")
else:
    print(f"\nNinguna feature supera el calentamiento maximo de "
         f"{UMBRAL_CALENTAMIENTO_SEMANAS} semanas.")

# --- Exclusion MANUAL
excluidas_manual = [f for f in EXCLUSION_MANUAL if f in features]
if excluidas_manual:
    for f in excluidas_manual:
        print(f"  {f}")
    features = [f for f in features if f not in excluidas_manual]
    df_nulos = df_nulos[~df_nulos.feature.isin(excluidas_manual)].reset_index(drop=True)
    panel = panel.drop(columns=excluidas_manual, errors="ignore")
 
# Grafica nulos
UMBRAL_CALENTAMIENTO_NOTABLE = 78  # 1.5x el estandar de 52 semanas

problematicas = df_nulos[df_nulos.pct_fuera > 0].sort_values("pct_fuera")
calentamiento_notable = df_nulos[
    (df_nulos.calentamiento_semanas > UMBRAL_CALENTAMIENTO_NOTABLE) &
    (df_nulos.pct_fuera == 0)
].sort_values("calentamiento_semanas")

fig, axes = plt.subplots(1, 2, figsize=(14, max(4, max(len(problematicas), len(calentamiento_notable)) * 0.3)))

if len(problematicas) > 0:
    def color_por_tipo(tipo):
        if tipo == "puntual":
            return "indianred"
        return "darkorange"  # estructural_activo(TICKER)
    axes[0].barh(problematicas.feature, problematicas.pct_fuera,
                color=[color_por_tipo(t) for t in problematicas.tipo_missing])
    axes[0].set_xlabel("% nulos (fuera del calentamiento)")
    axes[0].set_title("Nulos GENUINAMENTE problematicos\n(rojo=puntual, naranja=estructural de un activo)")
else:
    axes[0].text(0.5, 0.5, "Ninguna feature con nulos\nfuera del calentamiento normal",
                ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

if len(calentamiento_notable) > 0:
    axes[1].barh(calentamiento_notable.feature, calentamiento_notable.calentamiento_semanas,
                color="steelblue")
    axes[1].axvline(52, color="gray", linestyle="--", linewidth=1, label="52 sem (estandar)")
    axes[1].axvline(UMBRAL_CALENTAMIENTO_SEMANAS, color="red", linestyle="--",
                    linewidth=1, label="umbral de exclusion")
    axes[1].set_xlabel("Semanas de calentamiento")
    axes[1].set_title("Calentamiento NOTABLEMENTE largo\n(sin nulos genuinos, solo historico corto)")
    axes[1].legend(fontsize=8)
else:
    axes[1].text(0.5, 0.5, "Ninguna feature con calentamiento\nnotablemente largo (>78 semanas)",
                ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

plt.suptitle("Diagnostico de nulos -- separado por tipo real de causa", fontsize=12)
plt.tight_layout()
plt.savefig("data/processed/eda2_fig1_nulos.png", dpi=100)
plt.show()

 
# Solucion diferenciada por tipo de missing
puntuales = df_nulos[df_nulos.tipo_missing == "puntual"]["feature"].tolist()
estructurales_activo = df_nulos[df_nulos.tipo_missing.str.startswith(
    "estructural_activo", na=False)]["feature"].tolist()
 
if puntuales:
    for f in puntuales:
        datos[f] = datos.groupby("ticker")[f].ffill()
 
if estructurales_activo:
    for f in estructurales_activo:
        datos[f] = datos[f].fillna(0)
   
if not puntuales and not estructurales_activo:
    print("\nNo hay missing puntual ni estructural-de-activo: sin necesidad de imputacion.")
 
# Propagar la misma imputacion al PANEL completo
for f in puntuales:
    panel[f] = panel.groupby("ticker")[f].ffill()
for f in estructurales_activo:
    panel[f] = panel[f].fillna(0)
 

# RECALCULO DE NULOS TRAS LA IMPUTACION 
if puntuales or estructurales_activo:
    datos_ord_post = datos.sort_values(["ticker", "fecha"]).copy()
    datos_ord_post["semana_ticker"] = datos_ord_post.groupby("ticker").cumcount()
 
    for idx, row in df_nulos.iterrows():
        f = row["feature"]
        if f not in puntuales and f not in estructurales_activo:
            continue
        total = len(datos_ord_post)
        n_nulos_nuevo = datos_ord_post[f].isna().sum()
        pct_total_nuevo = 100 * n_nulos_nuevo / total
        calentamiento_post = datos_ord_post.groupby("ticker")[f].transform(primera_posicion_valida)
        es_calentamiento_post = datos_ord_post["semana_ticker"] < calentamiento_post
        nulos_inicio_nuevo = datos_ord_post.loc[es_calentamiento_post, f].isna().sum()
        pct_fuera_nuevo = 100 * (n_nulos_nuevo - nulos_inicio_nuevo) / total
 
        df_nulos.loc[idx, "pct_total"] = pct_total_nuevo
        df_nulos.loc[idx, "pct_fuera"] = pct_fuera_nuevo
        df_nulos.loc[idx, "nulos_fuera_calentamiento"] = n_nulos_nuevo - nulos_inicio_nuevo
        print(f"  {f:30s} pct_total: {row['pct_total']:.1f}% -> {pct_total_nuevo:.1f}% "
             f"(imputado)")
 
 

# Detalle datos features 
print("\n" + "="*70 + "\n3. Detalle features\n" + "="*70)
 
tabla_tipos = []
for f in features:
    s = datos[f].dropna()
    nuniq = s.nunique()
    dtype_real = str(s.dtype)
    if nuniq <= 2:
        tipo_log = "binaria"
    elif nuniq <= 15:
        tipo_log = "categorica_candidata"
    else:
        tipo_log = "continua"
    tabla_tipos.append({"feature": f, "dtype": dtype_real,
                        "n_unicos": nuniq, "tipo_logico": tipo_log,
                        "min": round(s.min(), 4), "max": round(s.max(), 4)})
 
df_tipos = pd.DataFrame(tabla_tipos).sort_values("n_unicos")
print(df_tipos.to_string(index=False))
 
candidatas_cat = df_tipos[df_tipos.tipo_logico == "categorica_candidata"]["feature"].tolist()
binarias = df_tipos[df_tipos.tipo_logico == "binaria"]["feature"].tolist()


# VALORES ATIPICOS 
print("\n" + "="*70 + "\n4. VALORES ATIPICOS\n" + "="*70)
 
def info_outliers(df, feat, factor=3):
    s = df[feat].dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    li, ls = q1 - factor*iqr, q3 + factor*iqr
    mask = (s < li) | (s > ls)
    return {"n": mask.sum(), "pct": 100*mask.mean(),
            "skew": round(s.skew(), 2), "limite_inf": round(li, 4),
            "limite_sup": round(ls, 4)}
 
tabla_out = pd.DataFrame([{"feature": f, **info_outliers(datos, f)}
                          for f in features]).sort_values("pct", ascending=False)
print(tabla_out.to_string(index=False))
 
# Boxplots dinamicos: se generan solo para features con outliers > 1%
con_outliers = tabla_out[tabla_out.pct > 1.0]["feature"].tolist()
n_cols = 4
n_rows = max(1, (len(con_outliers) + n_cols - 1) // n_cols)
if con_outliers:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    for ax, feat in zip(axes_flat, con_outliers):
        datos.boxplot(column=feat, ax=ax)
        n_out = tabla_out.loc[tabla_out.feature == feat, "n"].values[0]
        pct_out = tabla_out.loc[tabla_out.feature == feat, "pct"].values[0]
        ax.set_title(f"{feat}\n({n_out} outliers, {pct_out:.1f}%)", fontsize=8)
        ax.set_xlabel("")
    for ax in list(axes_flat)[len(con_outliers):]:
        ax.set_visible(False)
    plt.suptitle("Outliers por feature (IQR x3) — solo con >1% outliers", fontsize=11)
    plt.tight_layout()
    plt.savefig("data/processed/eda2_fig2_outliers.png", dpi=100)
    plt.show()
 
 

# CORRELACION (Spearman)
print("\n" + "="*70 + "\n5. CORRELACION (Spearman)\n" + "="*70)

corr = datos[features].corr(method="spearman")

 
mask_triu = np.triu(np.ones(corr.shape), k=1).astype(bool)
pares_altos = (corr.abs().where(mask_triu).stack()
               .sort_values(ascending=False)
               .rename("corr_abs")
               .reset_index()
               .rename(columns={"level_0": "var1", "level_1": "var2"}))
pares_altos_075 = pares_altos[pares_altos.corr_abs > UMBRAL_CORR]
 
print(f"\nPares con |corr| > {UMBRAL_CORR}:")
if len(pares_altos_075) > 0:
    print(pares_altos_075.to_string(index=False))
    todos = list(pares_altos_075.var1) + list(pares_altos_075.var2)
    conteo = Counter(todos)
    hubs = {k: v for k, v in conteo.items() if v >= 2}
    
else:
    print("  Ningun par supera el umbral.")
 
correlacionadas_con = {f: [] for f in features}
for _, row in pares_altos_075.iterrows():
    correlacionadas_con[row.var1].append(f"{row.var2}({row.corr_abs:.2f})")
    correlacionadas_con[row.var2].append(f"{row.var1}({row.corr_abs:.2f})")
df_correlacionadas = pd.DataFrame({
    "feature": list(correlacionadas_con.keys()),
    "n_correlacionadas_075": [len(v) for v in correlacionadas_con.values()],
    "correlacionada_con": [", ".join(v) if v else "" for v in correlacionadas_con.values()],
})
 
 

# CHI-CUADRADO, P-VALOR Y V DE CRAMER 
print("\n" + "="*70 + "\n6. CHI-CUADRADO Y V DE CRAMER\n" + "="*70)
 
def cramers_v(chi2, n, r, c):
    return np.sqrt(chi2 / (n * (min(r, c) - 1)))
 
res_chi2 = []
for f in features:
    try:
        tmp = datos[[f, TARGET_CONTINUO]].dropna()
        tmp["q"] = pd.qcut(tmp[f], 4, duplicates="drop")
        if tmp["q"].nunique() < 2:
            continue
        ct = pd.crosstab(tmp["q"], pd.qcut(tmp[TARGET_CONTINUO], 4, duplicates="drop"))
        chi2_val, p, _, _ = chi2_contingency(ct)
        v = cramers_v(chi2_val, ct.values.sum(), *ct.shape)
        res_chi2.append({"feature": f, "chi2": round(chi2_val, 2),
                         "p_valor": p, "cramer_v": round(v, 4)})
    except Exception:
        pass
 
df_chi2 = pd.DataFrame(res_chi2).sort_values("cramer_v", ascending=False)
print(df_chi2.to_string(index=False))
 
plt.figure(figsize=(10, max(5, len(df_chi2)*0.28)))
colores_v = ["seagreen" if v > 0.10 else ("orange" if v > 0.05 else "lightgray")
             for v in df_chi2["cramer_v"]]
plt.barh(df_chi2["feature"], df_chi2["cramer_v"], color=colores_v)
plt.axvline(0.10, color="green", linestyle="--", label="V=0.10 (moderado)")
plt.axvline(0.05, color="orange", linestyle="--", label="V=0.05 (debil)")
plt.xlabel("V de Cramer")
plt.title("V de Cramer con target_3m por cuartiles (verde>=0.10, naranja>=0.05)")
plt.gca().invert_yaxis()
plt.legend()
plt.tight_layout()
plt.savefig("data/processed/eda2_fig4_cramer.png", dpi=100)
plt.show()
 

# BALANCE DEL DATASET Y VARIABLE OBJETIVO 
print("\n" + "="*70 + "\n7. BALANCE Y VARIABLE OBJETIVO\n" + "="*70)
 
print(f"\nFilas totales del panel filtrado: {len(datos)}")
print(f"Activos: {sorted(datos.ticker.unique())}")
print(f"\nTarget_3m por activo:")
for t in sorted(datos.ticker.unique()):
    s = datos[datos.ticker == t][TARGET_CONTINUO]
    print(f"  {t}: media={s.mean():.4f}  std={s.std():.4f}  n={s.notna().sum()}")

fig_bal, axes_bal = plt.subplots(1, 2, figsize=(12, 4))
datos.groupby("ticker")[TARGET_CONTINUO].mean().plot(
    kind="bar", ax=axes_bal[0], color="steelblue")
axes_bal[0].axhline(0, color="red", linewidth=1.0)
axes_bal[0].set_title("Retorno relativo medio por activo (target_3m)")
axes_bal[0].set_ylabel("target_3m medio")

datos.groupby(datos.fecha.dt.year)[TARGET_CONTINUO].mean().plot(
    ax=axes_bal[1], marker="o")
axes_bal[1].axhline(0, color="red", linestyle="--", label="sin diferencia")
axes_bal[1].set_title("Media anual del target continuo (control de estabilidad)")
axes_bal[1].legend()
plt.tight_layout()
plt.savefig("data/processed/eda2_fig5_balance.png", dpi=100)
plt.show()
 
 
# ESTABILIDAD UNIVARIANTE POR FOLD 
print("\n" + "="*70 + "\n8. ESTABILIDAD UNIVARIANTE (IC + consistencia de signo)\n" + "="*70)
 
def generar_folds(df, ventana, horizonte, n_folds, col="fecha"):
    fechas = np.sort(df[col].unique())
    primer_test = pd.Timestamp(fechas[0]) + pd.Timedelta(weeks=ventana + horizonte)
    fechas_test = fechas[fechas >= primer_test.to_numpy()]
    n = len(fechas_test)
    if n < n_folds * 8:
        raise ValueError(f"Pocas fechas testables ({n}) para {n_folds} folds.")
    tam = n // n_folds
    folds = []
    for i in range(n_folds):
        ini, fin = i*tam, (i*tam + tam if i < n_folds-1 else n)
        f_test_ini, f_test_fin = fechas_test[ini], fechas_test[fin-1]
        idx_te = df.index[(df[col] >= f_test_ini) & (df[col] <= f_test_fin)].to_numpy()
        if len(idx_te) > 10:
            folds.append({"fold": i+1, "idx_test": idx_te,
                          "test_ini": f_test_ini, "test_fin": f_test_fin})
    return folds
 
datos_est = datos.dropna(subset=[TARGET_CONTINUO]).reset_index(drop=True)
folds = generar_folds(datos_est, VENTANA_RODANTE, HORIZONTE_SEMANAS, N_FOLDS)

for f in folds:
    print(f"  Fold {f['fold']}: test {pd.Timestamp(f['test_ini']).date()} -> "
         f"{pd.Timestamp(f['test_fin']).date()} ({len(f['idx_test'])} obs)")
 
estabilidad = []
matriz_ic = {}         
for feat in features:
    ics = []
    for f in folds:
        te = datos_est.loc[f["idx_test"]]
        ic = te[feat].corr(te[TARGET_CONTINUO], method="spearman")
        ics.append(ic)
    ics = np.array(ics)
    matriz_ic[feat] = ics
    cons = (np.sign(ics) == np.sign(np.nansum(ics))).mean()
    estabilidad.append({
        "feature": feat, "ic_medio": np.nanmean(ics), "ic_std": np.nanstd(ics),
        "consistencia_signo": cons, "score": cons * abs(np.nanmean(ics))
    })

df_est = pd.DataFrame(estabilidad).sort_values("score", ascending=False)
print("\nRanking de estabilidad univariante:")
print(df_est.to_string(index=False))

plt.figure(figsize=(10, max(6, len(df_est)*0.25)))
t_plot = df_est.sort_values("score")
plt.barh(t_plot.feature, t_plot.score,
        color=plt.cm.RdYlGn(t_plot.consistencia_signo))
plt.xlabel("Score = consistencia x |IC medio|")
plt.title("Estabilidad univariante (verde=señal consistente, rojo=cambia de signo por regimen)")
plt.tight_layout()
plt.savefig("data/processed/eda2_fig6_estabilidad.png", dpi=100)
plt.show()

# heatmap del IC fold a fold
_top_est = df_est.head(25)["feature"].tolist()
_mat_ic = pd.DataFrame({f: matriz_ic[f] for f in _top_est}).T
_mat_ic.columns = [f"F{i+1}\n{pd.Timestamp(folds[i]['test_ini']).strftime('%Y-%m')}"
                   for i in range(len(folds))]

fig, ax = plt.subplots(figsize=(max(8, len(folds) * 1.3),
                                max(7, len(_top_est) * 0.32)))
sns.heatmap(_mat_ic, cmap="RdBu_r", center=0, vmin=-0.6, vmax=0.6,
            annot=True, fmt=".2f", annot_kws={"size": 7}, linewidths=0.4,
            cbar_kws={"label": "IC (Spearman con target_3m)"}, ax=ax)
ax.set_title("IC por fold y feature (rojo = IC positivo, azul = IC negativo)\n"
             "Columnas palidas = regimen donde ninguna feature tiene señal",
             fontsize=11)
ax.set_xlabel("Fold de validacion (inicio del periodo de test)")
plt.tight_layout()
plt.savefig("data/processed/eda2_fig7_ic_por_fold.png", dpi=100)
plt.show()
 
 
# =============================================================
# Features planas y VIF
# =============================================================
 
print("\n" + "="*70 + "\n9. Features planas y VIF\n" + "="*70)

def ratio_cs(df, feat):
    vt = df[feat].var()
    if vt == 0 or pd.isna(vt):
        return np.nan
    vd = df.groupby("fecha")[feat].transform("std").pow(2).mean()
    return vd / vt
 
cs_ratios = pd.Series({f: ratio_cs(datos_est, f) for f in features},
                      name="ratio_cross_sectional").sort_values()
planas = cs_ratios[cs_ratios < 0.05].index.tolist()
print(f"\nFeatures PLANAS (mismo valor para los 4 activos esa fecha, ratio<0.05):")
print(f"  {planas}")
print(f"  Estas features solo discriminan a traves de sus betas sectoriales.")
 
# VIF 
print("\nVIF de features::")
no_planas = [f for f in features if f not in planas]
datos_vif = datos_est[no_planas].dropna()
if len(datos_vif) > 3000:
    datos_vif = datos_vif.sample(3000, random_state=42)
datos_vif_c = add_constant(datos_vif)
vif_res = []
for i, f in enumerate(no_planas):
    try:
        v = variance_inflation_factor(datos_vif_c.values, i+1)
    except Exception:
        v = np.nan
    vif_res.append({"feature": f, "VIF": round(v, 1)})
df_vif = pd.DataFrame(vif_res).sort_values("VIF", ascending=False)
print(df_vif[df_vif.VIF > 10].to_string(index=False))
if (df_vif.VIF > 10).sum() == 0:
    print("  Ningun VIF > 10 (sin multicolinealidad seria entre las no-planas)")
 

# GRUPOS tipologías de variables
GRUPOS_BASE = {
    # --- Variables sinteticas de regimen ---
    "Liquidez_FED_score": "Riesgo_Liquidez", "Riesgo_Sistemico_score": "Credito_Deuda_Sistema",
    "RiskOnOff_score": "Actividad_Regimen", "Curva_Invertida_score": "Tipos_Curva",
    "Momentum_Global_ExUS_score": "Activos_Contexto",
    # --- Momentum y tendencia ---
    "ret_1m": "Momentum_Tendencia", "ret_3m": "Momentum_Tendencia",
    "ret_6m": "Momentum_Tendencia", "ret_12m": "Momentum_Tendencia",
    "mom_12_1": "Momentum_Tendencia", "acel_momentum": "Momentum_Tendencia",
    "prox_max_52w": "Momentum_Tendencia", "consistencia_mom": "Momentum_Tendencia",
    "mom_absoluto": "Momentum_Tendencia", "ratio_ma90": "Momentum_Tendencia",
    "ratio_ma200": "Momentum_Tendencia", "cruce_ma90_ma200": "Momentum_Tendencia",
    "rank_mom_6m": "Momentum_Tendencia", "rank_mom_12m": "Momentum_Tendencia",
    "z_ret_1m": "Momentum_Tendencia", "persist_rank": "Momentum_Tendencia",
    "relmom_ret_3m": "Momentum_Tendencia", "relmom_ret_6m": "Momentum_Tendencia",
    "relmom_ret_12m": "Momentum_Tendencia",
    # --- Riesgo y liquidez ---
    "vol_3m": "Riesgo_Liquidez", "vol_12m": "Riesgo_Liquidez",
    "downside_vol_6m": "Riesgo_Liquidez", "max_dd_12m": "Riesgo_Liquidez",
    "amihud_3m": "Riesgo_Liquidez", "tend_volumen": "Riesgo_Liquidez",
    "rank_vol": "Riesgo_Liquidez",
    # --- Tipos y curva ---
    "spread_curva": "Tipos_Curva", "d_spread_curva_3m": "Tipos_Curva",
    "spread_curva_3m10y": "Tipos_Curva", "yield_10y": "Tipos_Curva",
    "yield_2y": "Tipos_Curva", "yield_30y": "Tipos_Curva",
    "spread_curva_10y30y": "Tipos_Curva", "tipo_real_10y": "Tipos_Curva",
    "d_tipo_real_3m": "Tipos_Curva", "breakeven_10y": "Tipos_Curva",
    "breakeven_5y": "Tipos_Curva", "fedfunds_nivel": "Tipos_Curva",
    "d_fedfunds_6m": "Tipos_Curva", "percentil_yield_10y": "Tipos_Curva",
    "beta_realrate": "Tipos_Curva", "beta_fedfunds": "Tipos_Curva", "beta_breakeven": "Tipos_Curva",
    # --- Credito, liquidez bancaria y deuda del sistema ---
    "spread_high_yield": "Credito_Deuda_Sistema", "d_spread_hy_3m": "Credito_Deuda_Sistema",
    "condiciones_financieras": "Credito_Deuda_Sistema", "stress_financiero": "Credito_Deuda_Sistema",
    "reverse_repo_pct2y": "Credito_Deuda_Sistema", "crec_balance_fed_6m": "Credito_Deuda_Sistema",
    "d_reservas_bancarias_6m": "Credito_Deuda_Sistema",
    "crec_prestamos_totales_6m": "Credito_Deuda_Sistema", "deuda_publica_pib": "Credito_Deuda_Sistema",
    "d_deuda_publica_pib_1y": "Credito_Deuda_Sistema", "deuda_hogares_pib": "Credito_Deuda_Sistema",
    "d_deuda_hogares_pib_1y": "Credito_Deuda_Sistema",
    # --- Actividad economica y regimen ---
    "inflacion_yoy": "Actividad_Regimen", "desempleo_nivel": "Actividad_Regimen",
    "d_desempleo_6m": "Actividad_Regimen", "d_produccion_ind_12m": "Actividad_Regimen",
    "d_confianza_3m": "Actividad_Regimen", "d_m2_12m": "Actividad_Regimen",
    "prob_recesion": "Actividad_Regimen", "d_prob_recesion_3m": "Actividad_Regimen",
    "tightening_credito": "Actividad_Regimen",
    # --- Commodities ---
    "d_petroleo_wti_3m": "Commodities", "spread_wti_brent": "Commodities",
    "d_cobre_3m": "Commodities", "d_gas_natural_3m": "Commodities",
    # --- Sentimiento de mercado ---
    "vix_nivel": "Sentimiento_Mercado", "vix_cambio_1m": "Sentimiento_Mercado",
    "vix_pendiente": "Sentimiento_Mercado", "beta_vix": "Sentimiento_Mercado",
    "dolar_nivel": "Sentimiento_Mercado", "d_dolar_3m": "Sentimiento_Mercado",
    "tipo_bce": "Sentimiento_Mercado", "d_tipo_bce_6m": "Sentimiento_Mercado",
    "beta_dolar": "Sentimiento_Mercado",
    # --- Otros activos de contexto (ETFs de geografías y estilos) ---
    "ret_slv": "Activos_Contexto", "ret_eem": "Activos_Contexto",
    "ret_iwm": "Activos_Contexto", "ret_vtv": "Activos_Contexto",
    "ret_vug": "Activos_Contexto", "ret_hyg": "Activos_Contexto",
    "ret_vnq": "Activos_Contexto", "ret_vgk": "Activos_Contexto",
    "ret_aaxj": "Activos_Contexto", "ret_ewj": "Activos_Contexto",
    # --- Valoracion ---
    "oro_real": "Valoracion", "oro_real_percentil_5y": "Valoracion",
}


def asignar_grupo(nombre_feature):
    origen = nombre_feature
    for prefijo in ("beta_", "impacto_"):
        if nombre_feature.startswith(prefijo):
            origen = nombre_feature[len(prefijo):]
            break
    return GRUPOS_BASE.get(origen, "SIN_CLASIFICAR")


# TABLA RESUMEN CONSOLIDADA 
print("\n" + "="*70 + "\n10. RESUMEN CONSOLIDADO PARA SELECCION\n" + "="*70)
 
resumen_final = df_est[["feature", "ic_medio", "ic_std", "consistencia_signo", "score"]].copy()
resumen_final = resumen_final.merge(
    df_nulos[["feature", "pct_total", "pct_fuera", "tipo_missing"]], on="feature", how="left")
resumen_final = resumen_final.merge(
    df_chi2[["feature", "cramer_v"]], on="feature", how="left")
resumen_final = resumen_final.merge(
    cs_ratios.reset_index().rename(columns={"index": "feature",
                                            "ratio_cross_sectional": "ratio_cs"}),
    on="feature", how="left")
vif_dict = df_vif.set_index("feature")["VIF"].to_dict()
resumen_final["VIF"] = resumen_final["feature"].map(vif_dict)
resumen_final["es_plana"] = resumen_final["feature"].isin(planas)
resumen_final = resumen_final.merge(df_correlacionadas, on="feature", how="left")
resumen_final["grupo"] = resumen_final["feature"].map(asignar_grupo)

sin_clasificar = resumen_final[resumen_final["grupo"] == "SIN_CLASIFICAR"]["feature"].tolist()
if sin_clasificar:
    print(f"\nAVISO: {len(sin_clasificar)} features sin grupo asignado "
         f"(revisar GRUPOS_BASE): {sin_clasificar}")

resumen_final = resumen_final.sort_values("score", ascending=False)
 
 
print(f"\nListado final ({len(resumen_final)} candidatas")
 
print("\n" + "="*70)
print("SELECCION DE FEATURES")
print("="*70)
for i, row in enumerate(resumen_final.itertuples(), 1):
    plana_tag = " [PLANA]" if row.es_plana else ""
    nulos_tag = f" [NULOS {row.pct_fuera:.1f}% fuera de calentamiento" \
               f"{' PUNTUAL' if row.tipo_missing=='puntual' else ''}]" \
                if row.pct_fuera > 2 else ""
    corr_tag = (f" [CORR>0.75 con {row.n_correlacionadas_075}: {row.correlacionada_con}]"
               if row.n_correlacionadas_075 > 0 else "")
    print(f"  {i:2d}. {row.feature:30s} score={row.score:.4f}  "
         f"cons={row.consistencia_signo:.1f}  IC={row.ic_medio:+.4f}"
         f"{plana_tag}{nulos_tag}{corr_tag}")

_top_ic = resumen_final.sort_values("ic_medio", key=abs, ascending=False)["feature"].head(30).tolist()
_top_ic = [f for f in _top_ic if f in datos.columns]
_corr_sub = datos[_top_ic].corr(method="spearman")
_tiene_alta_corr = (_corr_sub.abs().where(~np.eye(len(_top_ic), dtype=bool)).max() > 0.7)
_feats_corr = _tiene_alta_corr[_tiene_alta_corr].index.tolist()
if len(_feats_corr) < 4:
    _feats_corr = _top_ic
corr_plot = datos[_feats_corr].corr(method="spearman")
mask_triu_plot = np.triu(np.ones(corr_plot.shape, dtype=bool))
n = len(_feats_corr)
fig, ax = plt.subplots(figsize=(max(10, n * 0.55), max(8, n * 0.45)))
sns.heatmap(corr_plot, mask=mask_triu_plot, cmap="coolwarm",
            center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
            annot_kws={"size": 7}, linewidths=0.4, ax=ax,
            cbar_kws={"shrink": 0.6})
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
plt.title(f"Correlacion de Spearman -- top 30 por IC con |corr|>0.7 ({n} features mostradas)")
plt.tight_layout()
plt.savefig("data/processed/eda2_fig3_correlacion.png", dpi=100)
plt.show()



# SELECCION FINAL DE FEATURES
MODO_SELECCION = "final"
FEATURES_SELECCIONADAS = [
    "beta_Momentum_Global_ExUS_score", "beta_ret_vnq", "beta_breakeven_5y",
    "beta_ret_vtv", "beta_ret_vug", "beta_Liquidez_FED_score",
    "beta_Riesgo_Sistemico_score", "beta_Curva_Invertida_score",
    "beta_vix", "beta_ret_iwm", "impacto_vix_nivel", "beta_ret_hyg",
    "downside_vol_6m", "mom_12_1", "beta_percentil_yield_10y", "ret_12m",
    "amihud_3m", "beta_ret_slv", "cruce_ma90_ma200", "beta_condiciones_financieras",
]

# --- Resolver seleccion ---
modo = MODO_SELECCION.strip().lower()
if modo == "final":
    disponibles   = [f for f in FEATURES_SELECCIONADAS if f in resumen_final["feature"].values]
    no_disp       = [f for f in FEATURES_SELECCIONADAS if f not in resumen_final["feature"].values]
    if no_disp:
        print(f"  AVISO: estas features no estan disponibles en este panel: {no_disp}")
    seleccion_final = disponibles
    print(f"  Modo: FINAL ({len(seleccion_final)} features del proyecto)")
elif modo == "all":
    seleccion_final = resumen_final["feature"].tolist()
    print(f"  Modo: ALL ({len(seleccion_final)} features)")
elif modo.startswith("top"):
    try:
        k = int(modo[3:])
    except ValueError:
        k = 20
    seleccion_final = resumen_final["feature"].head(k).tolist()
    print(f"  Modo: TOP{k} ({len(seleccion_final)} features por score)")

print(f"\nFEATURES SELECCIONADAS ({len(seleccion_final)}):")
for f in seleccion_final:
    row = resumen_final[resumen_final.feature == f].iloc[0]
    print(f"  - {f:38s} score={row.score:.4f}  IC={row.ic_medio:+.4f}  grupo={row.grupo}")




# GENERAR DATASET FINAL
print(f"\n{'='*70}")
print("DATASET FINAL PARA MODELADO")
print(f"{'='*70}")

TARGET_CONTINUO  = "target_3m"

panel_betas = panel.copy()
panel_betas["fecha"] = pd.to_datetime(panel_betas["fecha"]).dt.normalize()

# Verificar que todas las features seleccionadas existen
faltantes = [f for f in seleccion_final if f not in panel_betas.columns]
if faltantes:
    print(f"  AVISO: features no encontradas en el panel: {faltantes}")
    seleccion_final = [f for f in seleccion_final if f in panel_betas.columns]

# Informacion del grupo de cada feature (del resumen EDA)
resumen_merge = resumen_final.set_index("feature")

print(f"\nComposicion por grupo:")
grupos_sel = {}
for f in seleccion_final:
    g = resumen_merge.loc[f, "grupo"] if f in resumen_merge.index else "SIN_CLASIFICAR"
    grupos_sel[g] = grupos_sel.get(g, 0) + 1
for g, n in sorted(grupos_sel.items(), key=lambda x: -x[1]):
    print(f"  {g:25s}: {n}")

# Aplicar imputacion puntual (ffill por ticker, fuera del calentamiento)
# Reutilizar primera_posicion_valida (definida en la seccion de analisis de nulos)

panel_ord = panel_betas.sort_values(["ticker", "fecha"])
panel_ord["_sem"] = panel_ord.groupby("ticker").cumcount()
for f in seleccion_final:
    cal = panel_ord.groupby("ticker")[f].transform(primera_posicion_valida)
    nulos_fuera = panel_ord.loc[panel_ord["_sem"] >= cal, f].isna().sum()
    if nulos_fuera > 0:
        panel_ord[f] = panel_ord.groupby("ticker")[f].ffill()
panel_betas = panel_ord.drop(columns=["_sem"]).reset_index(drop=True)

# Construir el panel final
cols_mantener = (["fecha", "ticker", "close", "volume"] + seleccion_final +
                 [TARGET_CONTINUO, "target_1m", "target_6m"])
cols_mantener = [c for c in cols_mantener if c in panel_betas.columns]

panel_final = panel_betas[cols_mantener].copy()
panel_final.to_csv("data/processed/panel_modelo_final.csv", index=False)
pd.Series(seleccion_final, name="feature").to_csv(
    "data/processed/features_finales_pp.csv", index=False)

print(f"\nPanel final guardado: data/processed/panel_modelo_final.csv")
print(f"  Filas  : {len(panel_final):,}")
print(f"  Activos: {sorted(panel_final.ticker.unique())}")
print(f"  Rango  : {panel_final.fecha.min()} -> {panel_final.fecha.max()}")