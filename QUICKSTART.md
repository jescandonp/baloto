# 🚀 Guía de Inicio Rápido - Baloto ML v7.0

## Paso 1: Instalar Dependencias (Primera Vez)

### Opción A: Instalación Básica (Funcional sin extras)
```bash
pip install pandas numpy scikit-learn
```

### Opción B: Instalación Completa (Recomendada para máximo rendimiento)
```bash
pip install pandas numpy scikit-learn xgboost lightgbm
```

> **Nota**: Si ves advertencias sobre XGBoost/LightGBM, el sistema funcionará de todos modos con Random Forest y HistGradientBoosting.

---

## Paso 2: Ejecutar el Sistema

```bash
cd c:\Users\jmep2\Downloads\Personal\AI\baloto_system
python main.py
```

---

## Paso 3: Flujo Básico de Uso

### 1️⃣ Cargar Datos (Opción 1)
```
👉 Opción: 1
Ruta CSV Baloto/Revancha: [tu_carpeta_con_csvs]
Ruta CSV MiLoto: [tu_carpeta_con_csvs]
```

**Tip**: Puedes proporcionar la carpeta completa, el sistema buscará automáticamente los archivos.

### 2️⃣ Entrenar Modelos (Opción 2)
```
👉 Opción: 2
```

**Tiempos esperados**:
- Modo Simple: ~30-60 segundos
- Modo Ensemble (con XGB/LGB): ~2-5 minutos

### 3️⃣ Generar Predicción

#### Opción 3: Predicción Simple (1 secuencia)
```
👉 Opción: 3
Juego: baloto
```

**Output**:
- 1 secuencia óptima
- Score de confianza
- Probabilidades individuales

#### Opción 4: Predicción Múltiple (Top 10 secuencias) ⭐
```
👉 Opción: 4
Juego: baloto
¿Cuántas secuencias generar? [10]: 10
Estrategia [mixed]: mixed
```

**Output**:
- 10 secuencias rankeadas por score
- Explicación de cada secuencia
- Scores detallados

---

## 🎯 Estrategias Recomendadas

### Para Baloto/Revancha
```
Secuencias: 10-15
Estrategia: mixed
```

### Para MiLoto
```
Secuencias: 5-10
Estrategia: diverse
```

---

## ⚙️ Configuración Avanzada (Opción 6)

### Cambiar entre Modo Simple y Ensemble

```
👉 Opción: 6
👉 Opción: 2

Seleccionar modo:
1. Simple - Rápido (~30 seg)
2. Ensemble - Preciso (~2-5 min)
```

**Después de cambiar, re-entrena los modelos (Opción 2)**

---

## 🔍 Interpretando los Resultados

### Predicción Múltiple - Ejemplo de Output

```
#1 [Score: 0.87] [Confianza: 87%]
   Números: 05 - 12 - 23 - 34 - 41
   SuperBalota: 07
   💡 Alta similitud con patrones históricos | Distribución óptima
```

**Qué significa**:
- **Score 0.87**: Muy bueno (>0.8 es excelente)
- **Confianza 87%**: Alto acuerdo entre los 4 modelos
- **Explicación**: Razones por las que fue seleccionada

### Scores Detallados (Top 3)

```
📊 Scores: Pattern=0.82 | Dist=0.91 | Div=0.78 | Hot/Cold=0.92
```

- **Pattern**: Similitud con históricos (>0.7 es bueno)
- **Dist**: Balance de distribución (>0.8 es ideal)
- **Div**: Diversidad de números (>0.7 es bueno)
- **Hot/Cold**: Balance caliente/frío (>0.8 es ideal)

---

## ❓ Solución de Problemas Comunes

### "XGBoost/LightGBM no disponible"
✅ **Normal**: El sistema funciona sin ellos
💡 **Solución**: `pip install xgboost lightgbm` para mejor rendimiento

### "Error cargando datos"
✅ **Verifica**: Rutas de archivos CSV correctas
💡 **Tip**: Usa rutas absolutas o relativas correctamente

### "Predictor avanzado no disponible"
✅ **Causa**: No has entrenado los modelos
💡 **Solución**: Ejecuta Opción 2 primero

### El entrenamiento es muy lento
✅ **Normal**: Ensemble con XGB/LGB toma ~2-5 min
💡 **Alternativa**: Cambia a modo Simple (Opción 6)

---

## 📚 Documentación Completa

Ver [README.md](file:///c:/Users/jmep2/Downloads/Personal/AI/baloto_system/README.md) para documentación completa.

Ver [walkthrough.md](file:///C:/Users/jmep2/.gemini/antigravity/brain/30da9972-562e-4113-b594-a5d742aeb3cd/walkthrough.md) para detalles técnicos de las mejoras.

---

## 🎉 ¡Listo para Usar!

Tu sistema Baloto ML v7.0 está completamente configurado y listo para generar predicciones avanzadas con:
- ✅ 80+ features de análisis
- ✅ 4 algoritmos ML en ensemble
- ✅ Hasta 20 secuencias por predicción
- ✅ Scoring multinivel
- ✅ Explicaciones automáticas

**¡Buena suerte! 🍀🎯**
