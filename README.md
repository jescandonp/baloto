# Sistema Baloto ML Avanzado v7.0

Sistema de predicción de lotería mejorado con Machine Learning avanzado, ensemble de modelos y generación de múltiples secuencias candidatas.

## 🚀 Nuevas Características (v7.0)

### 1. **Feature Engineering Avanzado** (80+ features)
- **Análisis Multi-Ventana**: Ventanas de 5, 10 y 20 sorteos
- **Tendencias Temporales**: Números calientes/fríos, ciclos, retardos
- **Análisis de Patrones**: Pares consecutivos, gaps, repeticiones
- **Distribuciones**: Par/impar, rangos, suma total, asimetría

### 2. **Ensemble de Modelos ML**
- **Random Forest**: Robusto y rápido (300 estimators)
- **XGBoost**: Alta precisión con boosting avanzado (500 iter)
- **LightGBM**: Eficiente con grandes datasets (500 iter)
- **HistGradientBoosting**: Rápido y maneja NaNs (500 iter)
- **Votación Ponderada**: Combinación inteligente de predicciones

### 3. **Predicción Múltiple**
- **Top-N Strategy**: Números con mayor probabilidad ML
- **Diverse Strategy**: Maximiza diversidad entre números
- **Stochastic Strategy**: Sampling probabilístico
- **Mixed Strategy**: Combinación de todas las anteriores

### 4. **Scoring Avanzado**
- **ML Probability**: Basado en modelos entrenados
- **Pattern Matching**: Similitud con patrones históricos
- **Distribution Score**: Balance de distribución
- **Diversity Score**: Variedad de números
- **Hot/Cold Balance**: Equilibrio caliente/frío

### 5. **Configuración Flexible**
- Modo Simple (rápido, ~30-60 seg)
- Modo Ensemble (preciso, ~2-5 min)
- Cambio dinámico entre modos

## 📋 Requisitos

### Básicos (incluidos)
```bash
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
```

### Opcionales (para máxima potencia)
```bash
xgboost>=1.5.0
lightgbm>=3.3.0
```

### Instalación Completa
```bash
pip install pandas numpy scikit-learn xgboost lightgbm
```

## 🎮 Modo de Uso

### 1. Iniciar el Sistema
```bash
cd c:\Users\jmep2\Downloads\Personal\AI\baloto_system
python main.py
```

### 2. Flujo Típico

#### Opción 1: Cargar Datos CSV
- Proporcionar ruta a archivos CSV de Baloto/MiLoto
- El sistema carga y procesa automáticamente

#### Opción 2: Entrenar Modelos
- **Modo Simple**: ~30-60 segundos
  - 1 modelo HistGradientBoosting optimizado
  - 500 iteraciones con early stopping
  
- **Modo Ensemble**: ~2-5 minutos
  - 4 modelos distintos (RF, XGB, LGB, Hist)
  - Validación cruzada
  - Votación ponderada

#### Opción 3: Predicción Simple
- Genera 1 secuencia óptima
- Muestra probabilidades individuales
- Score de confianza del ensemble
- Top 10 números con barras de probabilidad

#### Opción 4: Predicción Múltiple
- Genera hasta 20 secuencias candidatas
- 4 estrategias disponibles:
  - `top_n`: Enfocado en probabilidad ML
  - `diverse`: Maximiza diversidad
  - `stochastic`: Sampling probabilístico
  - `mixed`: Combina todas (recomendado)
  
- **Output detallado**:
  ```
  #1 [Score: 0.85] [Confianza: 85%]
     Números: 05 - 12 - 23 - 34 - 41
     SuperBalota: 07
     💡 Método: Top 5 más probables | Alta similitud con patrones 
        históricos | Distribución óptima (par/impar, rangos, gaps) | 
        Ratio par/impar: 2/3 | Suma total: 115
     📊 Scores: Pattern=0.82 | Dist=0.91 | Div=0.78 | Hot/Cold=0.88
  ```

#### Opción 5: Ver Estadísticas
- Números más calientes (histórico)
- Números más demorados
- Análisis por juego

#### Opción 6: Configurar Entrenamiento
- Cambiar entre modo Simple y Ensemble
- Ver modo actual

## 📊 Comparación: Modelo Antiguo vs Nuevo

| Característica | v6.0 (Antiguo) | v7.0 (Nuevo) |
|---------------|----------------|--------------|
| **Features** | ~60 | ~80-100 |
| **Algoritmos ML** | 1 (HistGradient) | 4 (RF+XGB+LGB+Hist) |
| **Iteraciones** | 100 | 500 |
| **Validación** | No | Sí (Cross-validation) |
| **Predicciones** | 1 secuencia | Hasta 20 secuencias |
| **Scoring** | Single probability | 5 scores combinados |
| **Confianza** | No disponible | Sí (model agreement) |
| **Tiempo entrenamiento** | ~30 seg | 30 seg - 5 min |
| **Precisión estimada** | Buena | Excelente |

## 🔧 Arquitectura del Sistema

```
baloto_system/
├── main.py                      # UI principal y orquestación
├── data_manager.py              # Carga y limpieza de datos
├── feature_engineering.py       # 🆕 80+ features avanzadas
├── models.py                    # Wrapper de modelos (simple/ensemble)
├── ensemble_models.py           # 🆕 Ensemble de 4 modelos ML
├── advanced_predictor.py        # 🆕 Generación múltiple + scoring
├── statistics_analyzer.py       # Análisis estadístico
└── verify_system.py             # Testing y verificación
```

## 🎯 Estrategias de Predicción Recomendadas

### Para Baloto/Revancha
- **Estrategia**: `mixed` (combina todas)
- **Secuencias**: 10-15
- **Modo**: Ensemble (para máxima precisión)

### Para MiLoto
- **Estrategia**: `diverse` (mayor variedad)
- **Secuencias**: 5-10
- **Modo**: Ensemble o Simple

### Análisis de Resultados
1. **Revisar Top 3 secuencias** - Tienen los mejores scores combinados
2. **Verificar distribución** - Balance par/impar ideal: 2/3 o 3/2
3. **Comparar con históricos** - Pattern match score > 0.7 es bueno
4. **Considerar Hot/Cold** - Balance óptimo aumenta chances

## 📈 Métricas de Rendimiento

El sistema utiliza múltiples métricas para evaluar cada predicción:

- **ML Probability (30%)**: Probabilidad del modelo ML
- **Pattern Match (25%)**: Similitud con patrones ganadores históricos
- **Distribution (20%)**: Balance de distribución estadística
- **Diversity (15%)**: Variedad entre números seleccionados
- **Hot/Cold Balance (10%)**: Equilibrio números calientes/fríos

**Score Total**: Promedio ponderado de todas las métricas (0-1)

## 🐛 Troubleshooting

### XGBoost/LightGBM no disponibles
```
⚠️ XGBoost no disponible. Instala con: pip install xgboost
```
**Solución**: Instalar librerías opcionales o usar modo Simple

### Error al cargar datos
```
❌ Archivos no encontrados
```
**Solución**: Verificar rutas de archivos CSV, pueden ser relativas o absolutas

### Memoria insuficiente
```
MemoryError durante entrenamiento
```
**Solución**: Usar modo Simple en lugar de Ensemble

## 📝 Notas Importantes

> **⚠️ IMPORTANTE**: Este es un sistema de análisis estadístico y ML para fines educativos. 
> La lotería es un juego de azar y no existe garantía de aciertos.

> **💡 TIP**: Para mejores resultados, usa el modo Ensemble con la estrategia `mixed` 
> y analiza las Top 3-5 secuencias generadas.

> **🔥 RENDIMIENTO**: El ensemble requiere más recursos pero ofrece predicciones 
> significativamente más robustas gracias a la votación entre 4 modelos distintos.

## 🎓 Ejemplo de Sesión Completa

```
🎯 SISTEMA AVANZADO DE BALOTO v7.0 ML Enhanced
============================================================

--- MENÚ PRINCIPAL ---
Juegos cargados: Ninguno
Modo entrenamiento: Ensemble

👉 Opción: 1
Ruta CSV Baloto/Revancha: datos/
Ruta CSV MiLoto: datos/
✅ Datos cargados exitosamente:
   - Baloto: 1250 sorteos
   - Revancha: 1250 sorteos
   - Miloto: 890 sorteos

👉 Opción: 2
⚙️ Preparando datos para BALOTO...
✅ Ensemble inicializado con 4 modelos: ['RandomForest', 'HistGradient', 'XGBoost', 'LightGBM']

🎯 Entrenando Ensemble para BALOTO
   📊 Muestras: 1230 | Features: 187

   🔄 Entrenando RandomForest...
      ✅ RandomForest completado (Score: 0.673)
   🔄 Entrenando HistGradient...
      ✅ HistGradient completado (Score: 0.698)
   🔄 Entrenando XGBoost...
      ✅ XGBoost completado (Score: 0.712)
   🔄 Entrenando LightGBM...
      ✅ LightGBM completado (Score: 0.705)

✅ Ensemble entrenado: 4 modelos activos
   ✅ Modelo baloto entrenado exitosamente.

👉 Opción: 4
Juego: baloto
¿Cuántas secuencias generar? (1-20) [10]: 10
Estrategia (top_n/diverse/stochastic/mixed) [mixed]: mixed

🔄 Generando 10 secuencias con estrategia 'mixed'...

======================================================================
🎯 TOP 10 SECUENCIAS RECOMENDADAS
======================================================================

#1 [Score: 0.87] [Confianza: 87%]
   Números: 05 - 12 - 23 - 34 - 41
   SuperBalota: 07
   💡 Método: Top 5 más probables | Alta similitud con patrones históricos | 
      Distribución óptima | Ratio par/impar: 2/3 | Suma total: 115
   📊 Scores: Pattern=0.82 | Dist=0.91 | Div=0.78 | Hot/Cold=0.92
   -----------------------------------------------------------------

#2 [Score: 0.84] [Confianza: 84%]
   Números: 03 - 15 - 22 - 35 - 42
   SuperBalota: 11
   💡 Método: Máxima diversidad entre números | Patrones moderadamente similares | 
      Ratio par/impar: 3/2 | Suma total: 117
   📊 Scores: Pattern=0.78 | Dist=0.88 | Div=0.91 | Hot/Cold=0.85
   -----------------------------------------------------------------

...
```

## 🌟 Mejoras Futuras Potenciales

- [ ] Red neuronal LSTM para secuencias temporales
- [ ] Optimización automática de hiperparámetros (Optuna)
- [ ] Interfaz gráfica (GUI)
- [ ] API REST para integración
- [ ] Análisis de resultados posteriores (backtesting)
- [ ] Exportación de predicciones a CSV/PDF

---

**Desarrollado para análisis estadístico avanzado de loterías** 🎲📊🤖
