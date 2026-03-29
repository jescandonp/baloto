# 🎯 Sistema Baloto ML Avanzado v8.0

Sistema de predicción de lotería colombiana con Machine Learning, ensemble de modelos,
backtesting comparativo y modo FUN. Soporta **Baloto**, **Revancha** y **MiLoto**.

---

## 🚀 Características

### 🤖 Machine Learning
- **Ensemble de 4 modelos**: Random Forest, XGBoost, LightGBM, HistGradientBoosting
- **80+ features** por sorteo: frecuencia, retardos, patrones, distribución par/impar, gaps
- **Votación ponderada** entre modelos (XGBoost con peso 1.2x)
- **Score ML real**: las probabilidades del ensemble alimentan directamente el ranking

### 💾 Persistencia de modelos
- Los modelos se guardan automáticamente en `saved_models/` tras cada entrenamiento
- Al iniciar, el sistema carga el modelo más reciente — **sin re-entrenar**
- Formato: `{juego}_{YYYYMMDD}_{HHMMSS}.joblib`

### 📈 Backtesting walk-forward
**Modo estándar** — evalúa la estrategia ML sorteo a sorteo y muestra aciertos reales.

**Modo comparativo** — compara 5 estrategias en el mismo período:
| Estrategia | Descripción |
|---|---|
| ML top-N | Modelo entrenado, top probabilidades |
| Frecuencia total | 5 números más frecuentes en toda la historia |
| Frecuencia últ. 30 | 5 más frecuentes en los últimos 30 sorteos |
| Frecuencia últ. 10 | 5 más frecuentes en los últimos 10 sorteos |
| Aleatorio ×20 | Promedio de 20 corridas aleatorias por sorteo |

Genera una **tabla ranking** con aciertos promedio, tasa y `vs aleatorio ±`.

### 🎲 Modo FUN
6 sabores de suerte para generar 1-20 combinaciones:
| # | Sabor | Lógica |
|---|---|---|
| 1 | 🎰 Puro azar | 100% aleatorio |
| 2 | 🔥 Caliente | Hot numbers recientes + azar |
| 3 | ❄️ Frío | Números que no salen hace tiempo + azar |
| 4 | ☯️ Equilibrio | Mitad calientes + mitad fríos |
| 5 | 🤞 Tu número | Fijas un número, el resto al azar |
| 6 | 🎂 Fecha especial | Extrae números de una fecha como ancla |

### 🎯 Predicción múltiple
Genera hasta 20 secuencias con 4 estrategias (`top_n`, `diverse`, `stochastic`, `mixed`)
y las rankea por score compuesto:

| Componente | Peso | Descripción |
|---|---|---|
| ML Probability | 30% | Prob. real del ensemble para esos números |
| Pattern Match | 25% | Similitud con patrones históricos ganadores |
| Distribution | 20% | Balance par/impar, rangos, gaps |
| Diversity | 15% | Variedad entre los números |
| Hot/Cold | 10% | Equilibrio calientes/fríos |

---

## 📋 Instalación

```bash
pip install pandas numpy scikit-learn xgboost lightgbm joblib
```

---

## 🎮 Uso rápido

```bash
python main.py
```

### Flujo típico (primera vez)
```
1. Opción 1 → cargar CSVs (ruta de carpeta o archivo)
2. Opción 2 → entrenar modelos (guarda en saved_models/ automáticamente)
3. Opción 4 → predicción múltiple
```

### Flujo típico (sesiones siguientes)
```
1. Opción 1 → cargar CSVs  ← activa los modelos guardados automáticamente
2. Opción 4 → predicción múltiple  ← sin re-entrenar
```

### Menú principal
```
1. 📂 Cargar datos CSV
2. 🤖 Entrenar modelos (ML Avanzado)
3. 🔮 Predicción simple (1 secuencia)
4. 🎯 Predicción múltiple (Top 10 secuencias)
5. 📊 Ver estadísticas
6. ⚙️  Configurar entrenamiento
7. 📈 Backtesting (Validación histórica)
8. 🎲 Modo FUN (combinaciones de suerte)
0. 🚪 Salir
```

---

## 📁 Estructura del proyecto

```
baloto_system/
├── main.py                  # UI principal y orquestación
├── data_manager.py          # Carga y limpieza de CSVs
├── feature_engineering.py   # 80+ features temporales/estadísticas
├── models.py                # Wrapper simple/ensemble con save/load
├── ensemble_models.py       # Ensemble de 4 modelos ML
├── advanced_predictor.py    # Generación múltiple + scoring real
├── backtester.py            # Walk-forward validation + comparativo
├── statistics_analyzer.py   # Análisis estadístico (hot/cold, delays)
├── verify_system.py         # Verificación del sistema
├── saved_models/            # Modelos entrenados (.joblib)
└── requirements.txt
```

---

## 📊 Formato de los CSVs

**Baloto/Revancha** (`baloto_revancha_resultados_completo.csv`):
```
Fecha | Números Baloto | Superbalota Baloto | Números Revancha | Superbalota Revancha | Juego
```

**MiLoto** (`miloto_resultados_completo.csv`):
```
Fecha | Números MiLoto
```

Números en formato `01-05-12-23-43`. El sistema acepta separadores `-`, `,`, `;` o `/`.

---

## ⚠️ Nota importante

> Este sistema es para **análisis estadístico y entretenimiento**.
> La lotería es un juego de azar — ningún modelo puede garantizar aciertos.
> El backtesting comparativo incluye una línea base aleatoria para
> mantener expectativas honestas sobre el rendimiento real del ML.
> **Juega con responsabilidad.** 🍀

---

**Desarrollado con Python + scikit-learn + XGBoost + LightGBM** 🐍📊🤖
