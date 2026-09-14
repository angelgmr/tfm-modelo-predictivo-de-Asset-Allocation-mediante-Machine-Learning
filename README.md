# Rotación Táctica de Activos basada en la Cartera Permanente
### Modelo predictivo de Asset Allocation mediante Machine Learning
**TFM — Máster Big Data, Data Science & IA | Ángel García-Mochales Ruiz | 2026**

---

## Descripción
Pipeline completo de predicción táctica sobre la Cartera Permanente de Harry Browne (SPY, TLT, GLD, BIL). El modelo predice el ranking relativo de los 4 activos cada trimestre y genera recomendaciones de pesos de cartera accionables. Resultado en test congelado: Spearman +0.433, AUC 0.718, +10pp sobre la cartera estática en 2 años.

---

## Requisitos
```bash
pip install -r requirements.txt
```

## Ejecución del pipeline completo
Los scripts se ejecutan en orden desde la carpeta raíz del repositorio:

```bash
# 1. Descarga de precios (yfinance/Stooq) y variables macro (FRED)
python scripts/descarga_datos.py

# 2. Análisis, depuración de datos y construcción del dataset final
python scripts/Analisis_Depuracion.py

# 3. Entrenamiento, validación walk-forward y evaluación en test congelado
python scripts/Modelado.py

# 4. Backtesting de las 3 estrategias de cartera
python scripts/backtest.py

# 5. Proyección de pesos para el próximo trimestre
python scripts/proyeccion_actual.py
```

> **Nota**: los pasos 1-3 pueden tardar 2-3 horas en total. Si el modelo ya está entrenado, ejecutar solo el paso 5 tarda segundos.

---

## Ejecución rápida (modelo ya entrenado)
Si `data/processed/modelos_guardados/` contiene un modelo `.joblib` previo:
```bash
python scripts/proyeccion_actual.py
```
Genera en segundos el ranking predicho y los pesos recomendados para el trimestre actual.

---

## Estructura del repositorio
```
📁 scripts/              → pipeline de 5 scripts principales + 2 de soporte
📁 data/
   📁 raw/               → datos descargados (generado por descarga_datos.py)
   📁 processed/         → dataset final, modelos guardados y gráficas
📄 README.md
📄 requirements.txt
```

---

## Scripts de soporte
| Script | Descripción |
|---|---|
| `Validacion_anidada.py` | Exploración de configuraciones alternativas de features |
| `InformacionFeatures.py` | Análisis detallado y gráficas históricas de las features |

---

## Resultados principales
| Estrategia | CAGR | Sharpe | Sortino | Rebalanceos |
|---|---|---|---|---|
| Cartera Permanente (25/25/25/25) | 6.00% | 0.562 | 0.813 | 0 |
| Táctica 35/35/15/15 | 7.52% | 0.643 | 0.879 | 45 |
| Proporcional al score | 6.89% | 0.651 | 0.932 | 17 |

---

## Fuentes de datos
- **Precios ETFs**: [Yahoo Finance](https://finance.yahoo.com) vía `yfinance`
- **Variables macro**: [FRED - Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org)

---

## Contacto
Ángel García-Mochales Ruiz · [LinkedIn](https://www.linkedin.com/in/tu_perfil)
