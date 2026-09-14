# -*- coding: utf-8 -*-
"""
Created on Tue Aug 4 19:04:19 2026

@author: angel
"""

# -*- coding: utf-8 -*-
"""
TFM - Pipeline de descarga
============================================================================= 
Fuentes de datos: 
    Precios ETFs: yfinance y stooq
    Variables macroeconómicas: FRED
"""
 
import os
import re
import logging
from datetime import datetime
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import yfinance as yf
import pandas_datareader.data as web
 
 
# VARIABLES 
FECHA_INICIO = "2004-01-01"   # margen para que las features de 52 semanas de SPY/TLT/GLD esten "calientes" antes de que BIL aparezca en 2007-06
FECHA_FIN = datetime.today().strftime("%Y-%m-%d")
 
# Los 4 activos de la cartera permanente
ACTIVOS_CARTERA = ["SPY", "TLT", "GLD", "BIL"]
 
# Variables de contexto de mercado
CONTEXTO = [
    "^VIX", "DX-Y.NYB", #VIX (volatilidad de mercado) 
    "SLV",    # ETF plata (complementa a GLD, ratio oro/plata como risk-on/off)
    "EEM",    # ETF emergentes (apetito de riesgo global)
    "IWM",    # ETF small caps (Russell 2000, mas sensible al ciclo domestico)
    "VTV",    # ETF RV value
    "VUG",    # ETF RV growth
    "HYG",    # ETF credito high yield
    "VNQ",    # ETF REITs (real estate)
    "VGK",    # ETF RV Europa (Vanguard FTSE Europe)
    "AAXJ",   # ETF RV Asia ex-Japon (iShares MSCI All Country Asia ex Japan)
    "EWJ",    # ETF RV Japon (iShares MSCI Japan)
]
 
# Series FRED: Features macroeconómicas
SERIES_FRED = {
    "T10Y2Y":       ("spread_curva",            1),
    "T10Y3M":       ("spread_curva_3m10y",      1),
    "DGS10":        ("yield_10y",               1),
    "DGS2":         ("yield_2y",                1),
    "DGS30":        ("yield_30y",                1),  
    "DFII10":       ("tipo_real_10y",           1),
    "T10YIE":       ("breakeven_10y",           1),
    "T5YIE":        ("breakeven_5y",            1),
    "BAMLH0A0HYM2": ("spread_high_yield",       1),
    "RRPONTSYD":    ("reverse_repo",            1),
    "WALCL":        ("balance_fed",             4),
    "WRESBAL":      ("reservas_bancarias",      4),
    "NFCI":         ("condiciones_financieras", 4),
    "STLFSI4":      ("stress_financiero",       4),
    "CPIAUCSL":     ("cpi",                    15),
    "UNRATE":       ("desempleo",               7),
    "INDPRO":       ("produccion_industrial",  16),
    "UMCSENT":      ("confianza_consumidor",   14),
    "M2SL":         ("m2",                     30),
    "DFF":          ("fedfunds",                1),
    "DCOILWTICO":   ("petroleo_wti",            1),
    "DCOILBRENTEU": ("petroleo_brent",          1),
    "PCOPPUSDM":    ("cobre",                  30),
    "PNGASUSUSDM":  ("gas_natural",             30),
    "RECPROUSM156N":("prob_recesion",           30),  
    "DRTSCILM":     ("tightening_credito",      90),  
    "ECBDFR":       ("tipo_bce",                 1),  
    "TOTLL":        ("prestamos_totales",        4),  
    "GFDEGDQ188S":  ("deuda_publica_pib",        90),  
    "HDTGPDUSQ163N":("deuda_hogares_pib",        90), 
}
 
# Horizontes de prediccion en SEMANAS
HORIZONTES = {
    "1m": 4,
    "3m": 13,   #TARGET
    "6m": 26,
}
HORIZONTE_PRINCIPAL = "3m" 
# Ventanas rolling en SEMANAS
W_1M, W_3M, W_6M, W_12M = 4, 13, 26, 52
 
DIR_RAW = "data/raw"
DIR_PROCESSED = "data/processed" 
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__) 
 
# 2. DESCARGA
 
def descargar_yfinance(tickers, inicio, fin):
    # Descarga de yahoo finance los precios de los ETF de los activos de la cartera y variables de contexto
    datos = yf.download(tickers, start=inicio, end=fin, auto_adjust=True,
                        group_by="ticker", progress=False, threads=True)
    resultado = {}
    for t in tickers:
        try:
            df = datos[t].copy() if len(tickers) > 1 else datos.copy()
            df = df.dropna(how="all")
            if not df.empty:
                resultado[t] = df
        except Exception as e:
            log.warning("  %s: no parseable en la respuesta de yfinance (%s)", t, e)
    log.info("  yfinance: %d/%d tickers", len(resultado), len(tickers))
    return resultado


def descargar_stooq(tickers, inicio, fin):
    # Si falla la descarga de precios para algún acitvo en yahoo finance se descarga de stooq
    resultado = {}
    for t in tickers:
        try:
            df = web.DataReader(t.replace("^", ""), "stooq", inicio, fin)
            df = df.sort_index()
            if not df.empty:
                resultado[t] = df
                log.info("  stooq OK: %s", t)
        except Exception as e:
            log.warning("  stooq fallo %s: %s", t, e)
    return resultado


def obtener_precios(tickers, inicio, fin):
    # Descarga precios con fallback automatico yfinance -> Stooq
    log.info("Descargando precios (%d tickers)...", len(tickers))
    datos = {}
    try:
        datos = descargar_yfinance(tickers, inicio, fin)
    except Exception as e:
        log.warning("yfinance fallo: %s", e)

    faltantes = [t for t in tickers if t not in datos]
    if faltantes:
        log.warning("Fallback Stooq: %s", ", ".join(faltantes))
        datos.update(descargar_stooq(faltantes, inicio, fin))

    log.info("Precios obtenidos: %d/%d", len(datos), len(tickers))

    faltantes_definitivos = [t for t in tickers if t not in datos]
    if faltantes_definitivos:
        log.warning("="*70)
        log.warning("TICKERS SIN DATOS TRAS YFINANCE + STOOQ: %s",
                   ", ".join(faltantes_definitivos))
        log.warning("Features sin información disponible")
        log.warning("="*70)

    return datos


def obtener_macro(series, inicio, fin):
    # Descarga FRED aplicando el lag de publicacion de cada serie.
    log.info("Descargando macro FRED (%d series)...", len(series))
    frames = []
    UMBRAL_DIAS_SIN_ACTUALIZAR = 180  # ~6 meses
    for codigo, (nombre, lag_dias) in series.items():
        try:
            s = web.DataReader(codigo, "fred", inicio, fin)
            s.columns = [nombre]
            s = s.dropna()
            if len(s) > 0:
                dias_desde_ultimo = (pd.Timestamp(fin) - s.index.max()).days
                if dias_desde_ultimo > UMBRAL_DIAS_SIN_ACTUALIZAR:
                    log.warning(
                        "  %s (%s): SIN ACTUALIZAR desde hace %d dias (ultimo dato: %s). ",
                        codigo, nombre, dias_desde_ultimo, s.index.max().date())
            # Desplazar el indice hacia adelante = simular fecha de publicacion
            s.index = pd.to_datetime(s.index) + pd.Timedelta(days=lag_dias)
            # Se recorta para que ninguna fila tenga fecha de publicacion posterior a D-1.
            fecha_limite = pd.Timestamp(fin) - pd.Timedelta(days=1)
            n_antes = len(s)
            s = s[s.index <= fecha_limite]
            if len(s) < n_antes:
                log.info("  %s: recortadas %d filas con fecha de publicacion "
                        "posterior a D-1 (%s)", codigo, n_antes - len(s), fecha_limite.date())
            frames.append(s)
            log.info("  %s -> %s (lag %dd): %d obs",
                     codigo, nombre, lag_dias, len(s))
        except Exception as e:
            log.warning("  %s: fallo (%s)", codigo, e)
    macro = pd.concat(frames, axis=1, sort=True).sort_index()
    macro.index.name = "fecha"
    return macro


def a_semanal(dict_precios):
    #Se convierte a frecuencia semanal
    log.info("Convirtiendo a frecuencia semanal (viernes)...")
    semanal = {}
    for ticker, df in dict_precios.items():
        s = pd.DataFrame()
        s["close"] = df["Close"].resample("W-FRI").last()
        if "Volume" in df.columns:
            s["volume"] = df["Volume"].resample("W-FRI").sum()
        else:
            s["volume"] = np.nan
        s = s.dropna(subset=["close"])
        semanal[ticker] = s
    return semanal
 
 
def macro_semanal(macro, idx_semanal):
    #Se convierte a frecuencia semanal y aplica ffill hacia delante
    log.info("Alineando macro a frecuencia semanal...")
    idx_diario = pd.date_range(macro.index.min(), macro.index.max(), freq="D")
    macro_d = macro.reindex(idx_diario).ffill()
    macro_w = macro_d.reindex(idx_semanal).ffill()
    macro_w.index.name = "fecha"
    return macro_w

 
def _retornos(serie, n):
    return serie / serie.shift(n) - 1
 
 
def features_por_ticker(df):    
    #Features simples por activo de la cartera
    #df : DataFrame de un activo con columnas close, volume (indice = fecha semanal)
    f = pd.DataFrame(index=df.index)
    close = df["close"]
    ret_1w = close.pct_change()
    
     # --- Momentum ---
    f["ret_1m"] = _retornos(close, W_1M)      # retorno de las ultimas 4 semanas
    f["ret_3m"] = _retornos(close, W_3M)      # retorno de las ultimas 13 semanas
    f["ret_6m"] = _retornos(close, W_6M)      # retorno de las ultimas 26 semanas
    f["ret_12m"] = _retornos(close, W_12M)    # retorno de las ultimas 52 semanas
    f["mom_12_1"] = close.shift(W_1M) / close.shift(W_12M) - 1   # 12m excluyendo ultimo mes
    f["acel_momentum"] = f["ret_3m"] - f["ret_6m"] / 2           # momentum acelerando/agotandose
    f["prox_max_52w"] = close / close.rolling(W_12M, min_periods=26).max()  # cercania al maximo 52 sem
    f["consistencia_mom"] = (ret_1w > 0).rolling(W_6M, min_periods=10).mean()  # % semanas positivas, 6m
    f["mom_absoluto"] = (f["ret_12m"] > 0).astype(float)         # 1 si retorno 12m > 0
    f.loc[f["ret_12m"].isna(), "mom_absoluto"] = np.nan
 
    # --- Riesgo ---
    f["vol_3m"] = ret_1w.rolling(W_3M).std() * np.sqrt(52)       # volatilidad anualizada, 3m
    f["vol_12m"] = ret_1w.rolling(W_12M).std() * np.sqrt(52)     # volatilidad anualizada, 12m
    f["downside_vol_6m"] = (ret_1w.where(ret_1w < 0)
                            .rolling(W_6M, min_periods=5).std() * np.sqrt(52))  # solo retornos negativos
    roll_max = close.rolling(W_12M, min_periods=10).max()
    f["max_dd_12m"] = (close / roll_max - 1).rolling(W_12M, min_periods=10).min()  # maxima caida, 12m
 
    # --- Tendencia (medias moviles, todo propio del activo) ---
    ma90 = close.rolling(18).mean()    # ~90 dias
    ma200 = close.rolling(40).mean()   # ~200 dias
    f["ratio_ma90"] = close / ma90 - 1      # posicion vs media movil ~90d
    f["ratio_ma200"] = close / ma200 - 1    # posicion vs media movil ~200d
    f["cruce_ma90_ma200"] = ma90 / ma200 - 1  # golden/death cross
 
    # --- Liquidez ---
    dollar_vol = df["volume"] * close
    f["amihud_3m"] = (ret_1w.abs() / dollar_vol.replace(0, np.nan)
                      ).rolling(W_3M).mean() * 1e9   # iliquidez de Amihud, 3m
    f["tend_volumen"] = (df["volume"].rolling(W_1M).mean() /
                         df["volume"].rolling(W_12M).mean() - 1)  # volumen reciente vs normal
 
    return f
 
 
 
def features_macro(macro_w, ctx_vix_dolar):
    #Variables macroeconómicas
    m = pd.DataFrame(index=macro_w.index)
 
    # --- Tipos y curva ---
    m["spread_curva"] = macro_w["spread_curva"]                       # nivel 10y-2y
    m["d_spread_curva_3m"] = macro_w["spread_curva"].diff(W_3M)        # cambio en 3m
    m["spread_curva_3m10y"] = macro_w["spread_curva_3m10y"]            # nivel 10y-3m (Fed NY)
    m["yield_10y"] = macro_w["yield_10y"]                              # nivel tipo 10a
    m["yield_2y"] = macro_w["yield_2y"]                                # nivel tipo 2a
    m["yield_30y"] = macro_w["yield_30y"]                              # nivel tipo 30a (tramo real de TLT)
    m["spread_curva_10y30y"] = macro_w["yield_30y"] - macro_w["yield_10y"]  # pendiente tramo largo
    m["tipo_real_10y"] = macro_w["tipo_real_10y"]                      # nivel tipo real (TIPS)
    m["d_tipo_real_3m"] = macro_w["tipo_real_10y"].diff(W_3M)          # cambio en 3m
    m["breakeven_10y"] = macro_w["breakeven_10y"]                      # inflacion esperada 10a
    m["breakeven_5y"] = macro_w["breakeven_5y"]                        # inflacion esperada 5a
    m["fedfunds_nivel"] = macro_w["fedfunds"]                          # tipo de la Fed
    m["d_fedfunds_6m"] = macro_w["fedfunds"].diff(W_6M)                # cambio del tipo Fed, 6m
 
    # --- Credito y condiciones financieras ---
    m["spread_high_yield"] = macro_w["spread_high_yield"]              # nivel spread HY
    m["d_spread_hy_3m"] = macro_w["spread_high_yield"].diff(W_3M)      # cambio en 3m
    m["condiciones_financieras"] = macro_w["condiciones_financieras"]  # indice NFCI
    m["stress_financiero"] = macro_w["stress_financiero"]              # indice STLFSI4
 
    # --- Liquidez / balance de la Fed ---
    m["reverse_repo_pct2y"] = (macro_w["reverse_repo"]
                               .rolling(104, min_periods=26).rank(pct=True)) # percentil 2a
    m["crec_balance_fed_6m"] = (macro_w["balance_fed"] /
                                macro_w["balance_fed"].shift(W_6M) - 1)  # expansion/contraccion Fed
    m["d_reservas_bancarias_6m"] = macro_w["reservas_bancarias"].pct_change(W_6M)  # cambio reservas
 
    # --- Inflacion y actividad real ---
    m["inflacion_yoy"] = macro_w["cpi"].pct_change(W_12M)              # inflacion interanual
    m["desempleo_nivel"] = macro_w["desempleo"]                        # nivel tasa de paro
    m["d_desempleo_6m"] = macro_w["desempleo"].diff(W_6M)              # cambio en puntos, 6m
    m["d_produccion_ind_12m"] = macro_w["produccion_industrial"].pct_change(W_12M)  # crec. interanual
    m["d_confianza_3m"] = macro_w["confianza_consumidor"].pct_change(W_3M)  # cambio confianza, 3m
    m["d_m2_12m"] = macro_w["m2"].pct_change(W_12M)                    # crecimiento M2 interanual
 
    # --- Materias primas ---
    m["d_petroleo_wti_3m"] = macro_w["petroleo_wti"].pct_change(W_3M)  # cambio WTI, 3m
    m["spread_wti_brent"] = macro_w["petroleo_brent"] - macro_w["petroleo_wti"]  # diferencial
    m["d_cobre_3m"] = macro_w["cobre"].pct_change(W_3M)                # cambio cobre, 3m
    m["d_gas_natural_3m"] = macro_w["gas_natural"].pct_change(W_3M)    # cambio gas natural, 3m
 
    # --- Contexto de mercado (VIX y dolar, en bruto) ---
    m["vix_nivel"] = ctx_vix_dolar["vix_nivel"]                        # nivel del VIX
    m["vix_cambio_1m"] = ctx_vix_dolar["vix_cambio_1m"]                # cambio del VIX, 1m
    m["vix_pendiente"] = ctx_vix_dolar["vix_pendiente"]                # VIX vs su media 6m
    # vix_estructura_3m_spot ELIMINADA (ver construir_contexto_vix_dolar)
    m["dolar_nivel"] = ctx_vix_dolar["dolar_nivel"]                    # nivel indice dolar
    m["d_dolar_3m"] = ctx_vix_dolar["d_dolar_3m"]                      # cambio del dolar, 3m
 
    # Retornos de activos de contexto
    for col in ["ret_slv", "ret_eem", "ret_iwm", "ret_vtv", "ret_vug",
               "ret_hyg", "ret_vnq", "ret_vgk", "ret_aaxj", "ret_ewj"]:
        m[col] = ctx_vix_dolar[col]
 
    # --- Regimen ---
    m["prob_recesion"] = macro_w["prob_recesion"]                          # nivel (0-100)
    m["d_prob_recesion_3m"] = macro_w["prob_recesion"].diff(W_3M)          # cambio en 3m
    m["tightening_credito"] = macro_w["tightening_credito"]                # nivel (% neto)
 
    # --- Globales ---
    m["tipo_bce"] = macro_w["tipo_bce"]                                    # nivel tipo BCE
    m["d_tipo_bce_6m"] = macro_w["tipo_bce"].diff(W_6M)                    # cambio en 6m
 
    # --- Deuda y credito ---
    m["crec_prestamos_totales_6m"] = macro_w["prestamos_totales"].pct_change(W_6M)
    m["deuda_publica_pib"] = macro_w["deuda_publica_pib"]                  # nivel % PIB
    m["d_deuda_publica_pib_1y"] = macro_w["deuda_publica_pib"].diff(W_12M)  # cambio en 1a
    m["deuda_hogares_pib"] = macro_w["deuda_hogares_pib"]                  # nivel % PIB
    m["d_deuda_hogares_pib_1y"] = macro_w["deuda_hogares_pib"].diff(W_12M)  # cambio en 1a
 
    # --- Valoracion ---
    m["oro_real"] = np.nan # Se completa en panel ya que se necesita el precio del oro
    m["percentil_yield_10y"] = macro_w["yield_10y"].rolling(260, min_periods=104).rank(pct=True)                                                                            
 
    return m
 
 
def construir_contexto_vix_dolar(semanal, idx_semanal):

    # Contexto de mercado: VIX y dolar en bruto y activos de contexto.
    ctx = pd.DataFrame(index=idx_semanal)
 
    if "^VIX" in semanal:
        vix = semanal["^VIX"]["close"].reindex(ctx.index)
        ctx["vix_nivel"] = vix
        ctx["vix_cambio_1m"] = vix / vix.shift(W_1M) - 1
        ctx["vix_pendiente"] = vix / vix.rolling(W_6M).mean() - 1
    else:
        ctx["vix_nivel"] = np.nan
        ctx["vix_cambio_1m"] = np.nan
        ctx["vix_pendiente"] = np.nan 

    if "DX-Y.NYB" in semanal:
        dolar = semanal["DX-Y.NYB"]["close"].reindex(ctx.index)
        ctx["dolar_nivel"] = dolar
        ctx["d_dolar_3m"] = dolar.pct_change(W_3M)
    else:
        ctx["dolar_nivel"] = np.nan
        ctx["d_dolar_3m"] = np.nan
 
    # Retornos semanales de los activos de contexto:
    ACTIVOS_CONTEXTO_RET = ["SLV", "EEM", "IWM", "VTV", "VUG", "HYG", "VNQ",
                           "VGK", "AAXJ", "EWJ"]
    for ticker_ctx in ACTIVOS_CONTEXTO_RET:
        nombre_col = f"ret_{ticker_ctx.lower()}"
        if ticker_ctx in semanal:
            ctx[nombre_col] = semanal[ticker_ctx]["close"].reindex(ctx.index).pct_change()
        else:
            ctx[nombre_col] = np.nan
 
    return ctx
 
 
def features_cross_sectional(panel):

    g = panel.groupby("fecha")
 
    panel["rank_mom_6m"] = g["ret_6m"].rank(pct=True)
    panel["rank_mom_12m"] = g["ret_12m"].rank(pct=True)
    panel["rank_vol"] = g["vol_3m"].rank(pct=True)
 
    media = g["ret_1m"].transform("mean")
    desv = g["ret_1m"].transform("std")
    panel["z_ret_1m"] = (panel["ret_1m"] - media) / desv.replace(0, np.nan)
 
    panel["n_ACTIVOS_CARTERA"] = g["ticker"].transform("count")    
 
    # --- Momentum relativo ---
    for horizonte_mom in ["ret_3m", "ret_6m", "ret_12m"]:
        suma_grupo = g[horizonte_mom].transform("sum")
        n_grupo = g[horizonte_mom].transform("count")
        media_otros = (suma_grupo - panel[horizonte_mom]) / (n_grupo - 1).replace(0, np.nan)
        panel[f"relmom_{horizonte_mom}"] = panel[horizonte_mom] - media_otros
 
    panel = panel.sort_values(["ticker", "fecha"])
    panel["persist_rank"] = (
        panel.groupby("ticker")["rank_mom_6m"]
             .transform(lambda x: x.rolling(W_3M, min_periods=4).mean())
    )
    panel = panel.sort_values(["fecha", "ticker"]).reset_index(drop=True)
 
    return panel
 
 
def construir_targets(panel, horizontes):
    # Se construyen targets: ret_fwd, ret_relativo y ranking relativo
    log.info("Construyendo targets...")
    panel = panel.sort_values(["ticker", "fecha"]).reset_index(drop=True)

    for nombre, h in horizontes.items():
        panel[f"ret_fwd_{nombre}"] = (
            panel.groupby("ticker")["close"]
                 .transform(lambda x: x.shift(-h) / x - 1)
        )
        media_fecha = panel.groupby("fecha")[f"ret_fwd_{nombre}"].transform("mean")
        panel[f"exceso_{nombre}"] = panel[f"ret_fwd_{nombre}"] - media_fecha
        panel[f"target_{nombre}"] = (
            panel.groupby("fecha")[f"exceso_{nombre}"].rank(pct=True)
        )

    return panel
 

def construir_panel(semanal, macro_w, activos):
    # Se construye el panel con todas las features 
    ctx_vix_dolar = construir_contexto_vix_dolar(semanal, macro_w.index)
    f_macro = features_macro(macro_w, ctx_vix_dolar)
 
    filas = []
    for ticker in activos:
        if ticker not in semanal:
            log.warning("  %s no disponible", ticker)
            continue
        df = semanal[ticker].copy()
        f = features_por_ticker(df)
        f["ticker"] = ticker
        f["close"] = df["close"]
        f["volume"] = df["volume"]
        filas.append(f)
 
    panel = pd.concat(filas)
    panel.index.name = "fecha"
    panel = panel.reset_index()
 
    f_macro = f_macro.reset_index()
    panel = panel.merge(f_macro, on="fecha", how="left")

    if "GLD" in semanal:
        precio_gld = semanal["GLD"]["close"].reindex(macro_w.index).ffill()
        cpi_serie = macro_w["cpi"]
        oro_real_por_fecha = (precio_gld / cpi_serie).reset_index()
        oro_real_por_fecha.columns = ["fecha", "oro_real"]
        panel = panel.drop(columns=["oro_real"], errors="ignore").merge(
            oro_real_por_fecha, on="fecha", how="left")
        # Se normaliza a percentil de los ultimos 5 años
        panel = panel.sort_values("fecha")
        panel["oro_real_percentil_5y"] = (
            panel.drop_duplicates("fecha").set_index("fecha")["oro_real"]
                 .rolling(260, min_periods=104).rank(pct=True)
                 .reindex(panel["fecha"]).to_numpy()
        )
 
    panel = features_cross_sectional(panel)

    # Se excluye fechas donde no estan todos los activos objetivo disponibles
    panel = panel[panel["n_ACTIVOS_CARTERA"] == len(activos)].reset_index(drop=True)

    return panel
 
 
def guardar(panel, semanal, macro):
    # Se guarda la informacion descargada en panel
    os.makedirs(DIR_RAW, exist_ok=True)
    os.makedirs(DIR_PROCESSED, exist_ok=True)
    hoy = datetime.today().strftime("%Y-%m-%d")
 
    cierres = pd.DataFrame({t: d["close"] for t, d in semanal.items()})
    cierres.to_csv(f"{DIR_RAW}/precios_{hoy}.csv")
    macro.to_csv(f"{DIR_RAW}/macro_{hoy}.csv")
    panel.to_csv(f"{DIR_PROCESSED}/panel_modelo.csv", index=False)
 
    log.info("Guardado: %s/precios_%s.csv", DIR_RAW, hoy)
    log.info("Guardado: %s/macro_%s.csv", DIR_RAW, hoy)
    log.info("Guardado: %s/panel_modelo.csv", DIR_PROCESSED)


def resumen(panel):
    h = HORIZONTE_PRINCIPAL
    col_target = f"target_{h}"
 
    print("\n" + "=" * 68)
    print("RESUMEN DEL PANEL - CARTERA PERMANENTE TACTICA")
    print("=" * 68)
    print(f"Filas              : {len(panel):,}")
    print(f"Fechas unicas      : {panel['fecha'].nunique():,}")
    print(f"Rango              : {panel['fecha'].min().date()} -> "
          f"{panel['fecha'].max().date()}")
    print(f"Activos objetivo   : {sorted(panel['ticker'].unique())}")
    print(f"Columnas           : {len(panel.columns)}")
 
    n_feat = len([c for c in panel.columns if not c.startswith(
        ("target_", "ret_fwd_", "exceso_"))
        and c not in ["fecha", "ticker", "close", "volume"]])
    print(f"Features           : {n_feat}")
 
    validas = panel[col_target].notna().sum()
    print(f"Filas con target   : {validas:,} ({100*validas/len(panel):.1f}%)")
 
    print("\nCobertura por activo:")
    cob = panel.groupby("ticker").agg(
        inicio=("fecha", "min"), fin=("fecha", "max"), n=("fecha", "count")
    ).sort_values("inicio")
    for t, r in cob.iterrows():
        print(f"  {t:6s} {r.inicio.date()} -> {r.fin.date()}  ({r.n:,} sem)")
 
    print("\nActivos disponibles por fecha:")
    n_por_fecha = panel.groupby("fecha")["ticker"].count()
    for anio in [2005, 2006, 2007, 2008, 2015, 2024]:
        sub = n_por_fecha[n_por_fecha.index.year == anio]
        if len(sub) > 0:
            print(f"  {anio}: {int(sub.median())} activos")

    todas_features = [c for c in panel.columns if c not in
                      ["fecha", "ticker", "close", "volume"] and not
                      c.startswith(("target_", "ret_fwd_", "exceso_"))]
    pct_nulos_global = 100 * panel[todas_features].isna().mean().mean()
    print(f"\n% nulos medio (todas las features, incluye calentamiento normal "
         f"de ~52 semanas): {pct_nulos_global:.1f}%")

    print(f"\nDistribucion del target principal ({h}):")
    print(f"  min={panel[col_target].min():.3f}  "
          f"media={panel[col_target].mean():.3f}  "
          f"max={panel[col_target].max():.3f}")
    # Distribución del ranking continuo por activo
    print("\nRanking relativo medio por activo:")
    for t in sorted(panel["ticker"].unique()):
        sub = panel[panel["ticker"] == t][col_target]
        print(f"  {t}: media={sub.mean():.3f}  std={sub.std():.3f}")
 
 

def main():
    # Ejecucion
    log.info("INICIO | %s -> %s", FECHA_INICIO, FECHA_FIN)
 
    todos = ACTIVOS_CARTERA + CONTEXTO
    dict_precios = obtener_precios(todos, FECHA_INICIO, FECHA_FIN)
    if not dict_precios:
        raise RuntimeError("No se pudo descargar ningun precio")
 
    macro = obtener_macro(SERIES_FRED, FECHA_INICIO, FECHA_FIN) 
    semanal = a_semanal(dict_precios)
 
    ref = "SPY" if "SPY" in semanal else list(semanal.keys())[0]
    idx_sem = semanal[ref].index
    macro_w = macro_semanal(macro, idx_sem)
 
    panel = construir_panel(semanal, macro_w, ACTIVOS_CARTERA)
    panel = construir_targets(panel, HORIZONTES)
 
    panel = panel.sort_values(["fecha", "ticker"]).reset_index(drop=True)
 
    guardar(panel, semanal, macro)
    resumen(panel)
 
    log.info("PROCESO COMPLETADO")
    return panel
 
 
if __name__ == "__main__":
    panel = main()