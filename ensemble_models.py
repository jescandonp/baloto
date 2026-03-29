"""
Módulo de Ensemble de Modelos ML para predicción avanzada de lotería.
Incluye múltiples algoritmos y estrategias de combinación.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Importaciones opcionales (instalar si no están disponibles)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost no disponible. Instala con: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️ LightGBM no disponible. Instala con: pip install lightgbm")


class EnsembleModels:
    """
    Sistema de ensemble que combina múltiples algoritmos ML para 
    maximizar la precisión de las predicciones.
    """
    
    def __init__(self, game_type, use_all_models=True):
        """
        Args:
            game_type: Tipo de juego (baloto, revancha, miloto)
            use_all_models: Si True, usa todos los modelos disponibles
        """
        self.game_type = game_type
        self.use_all_models = use_all_models
        self.models = {}
        self.sb_models = {}
        self.is_trained = False
        self.sb_is_trained = False
        self.feature_count = None
        
        # Inicializar modelos disponibles
        self._initialize_models()
    
    def _initialize_models(self):
        """Inicializa todos los modelos base del ensemble."""
        
        # 1. Random Forest - Robusto y rápido
        rf_base = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.models['RandomForest'] = MultiOutputClassifier(rf_base, n_jobs=-1)
        self.sb_models['RandomForest'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # 2. HistGradientBoosting - Rápido y maneja NaNs
        hist_base = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_depth=7,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        self.models['HistGradient'] = MultiOutputClassifier(hist_base, n_jobs=-1)
        self.sb_models['HistGradient'] = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_depth=7,
            random_state=42
        )
        
        # 3. XGBoost - Alta precisión (si está disponible)
        if HAS_XGBOOST and self.use_all_models:
            xgb_base = xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            self.models['XGBoost'] = MultiOutputClassifier(xgb_base, n_jobs=-1)
            self.sb_models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=7,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
        
        # 4. LightGBM - Rápido y eficiente (si está disponible)
        if HAS_LIGHTGBM and self.use_all_models:
            lgb_base = lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=7,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            self.models['LightGBM'] = MultiOutputClassifier(lgb_base, n_jobs=-1)
            self.sb_models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=7,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        print(f"✅ Ensemble inicializado con {len(self.models)} modelos: {list(self.models.keys())}")
    
    def train(self, X, y, y_sb=None, validate=True):
        """
        Entrena todos los modelos del ensemble.
        
        Args:
            X: Features array
            y: Target array
            y_sb: Superbalota targets (opcional)
            validate: Si True, realiza validación cruzada
        
        Returns:
            bool: True si el entrenamiento fue exitoso
        """
        if len(X) < 50:
            print(f"⚠️ Datos insuficientes para entrenar {self.game_type} ({len(X)} muestras)")
            return False
        
        self.feature_count = X.shape[1]
        print(f"\n🎯 Entrenando Ensemble para {self.game_type.upper()}")
        print(f"   📊 Muestras: {len(X)} | Features: {self.feature_count}")
        
        # Entrenar cada modelo (iterar sobre copia para poder eliminar fallidos)
        failed_models = []
        for name, model in list(self.models.items()):
            try:
                print(f"\n   🔄 Entrenando {name}...")
                model.fit(X, y)

                # Validación cruzada opcional (solo para el primer output)
                if validate and len(X) > 100:
                    try:
                        score = self._quick_validation(model, X, y)
                        print(f"      ✅ {name} completado (Score: {score:.3f})")
                    except:
                        print(f"      ✅ {name} completado")
                else:
                    print(f"      ✅ {name} completado")

            except Exception as e:
                print(f"      ❌ Error en {name}: {e}")
                failed_models.append(name)

        for name in failed_models:
            del self.models[name]
        
        if not self.models:
            print("❌ Ningún modelo se entrenó exitosamente")
            return False
        
        self.is_trained = True
        
        # Entrenar modelos de Superbalota si aplica
        if y_sb is not None:
            print(f"\n   🔄 Entrenando modelos Superbalota...")
            # Normalizar y_sb a 0-indexado para compatibilidad con XGBoost
            y_sb_min = int(np.min(y_sb))
            y_sb_norm = y_sb - y_sb_min  # ej: 1-16 → 0-15

            failed_sb = []
            for name, model in list(self.sb_models.items()):
                if name in self.models:  # Solo si el modelo principal funcionó
                    try:
                        model.fit(X, y_sb_norm)
                    except Exception as e:
                        print(f"      ⚠️ Error en SB-{name}: {e}")
                        failed_sb.append(name)

            for name in failed_sb:
                del self.sb_models[name]

            if self.sb_models:
                self.sb_is_trained = True
                self._sb_label_offset = y_sb_min  # Guardar offset para revertir en predicción
                print(f"      ✅ {len(self.sb_models)} modelos SB entrenados")
        
        print(f"\n✅ Ensemble entrenado: {len(self.models)} modelos activos")
        return True
    
    def _quick_validation(self, model, X, y):
        """Validación rápida usando el primer output."""
        # Tomar solo el primer estimador y primer target
        if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
            first_estimator = model.estimators_[0]
            y_first = y[:, 0]
            # Cross-validation con 3 folds
            scores = cross_val_score(first_estimator, X, y_first, cv=3, scoring='accuracy')
            return scores.mean()
        return 0.0
    
    def predict_proba(self, X):
        """
        Predice probabilidades usando ensemble voting.
        
        Returns:
            Lista de arrays con probabilidades para cada número
        """
        if not self.is_trained or not self.models:
            return None
        
        # Obtener predicciones de cada modelo
        all_predictions = []
        weights = []
        
        for name, model in self.models.items():
            try:
                preds = model.predict_proba(X)
                all_predictions.append(preds)
                # Peso basado en tipo de modelo (personalizables)
                weight = 1.2 if name == 'XGBoost' else 1.0
                weights.append(weight)
            except Exception as e:
                print(f"⚠️ Error en predicción de {name}: {e}")
        
        if not all_predictions:
            return None
        
        # Votación ponderada: Promediar probabilidades
        weights = np.array(weights) / sum(weights)  # Normalizar pesos
        
        # all_predictions[i] es una lista de arrays (uno por output)
        # Necesitamos promediar cada output
        n_outputs = len(all_predictions[0])
        ensemble_proba = []
        
        for output_idx in range(n_outputs):
            # Obtener predicciones de este output de todos los modelos
            output_preds = [pred[output_idx] for pred in all_predictions]
            # Promediar con pesos
            weighted_pred = np.average(output_preds, axis=0, weights=weights)
            ensemble_proba.append(weighted_pred)
        
        return ensemble_proba
    
    def predict_sb_proba(self, X):
        """Predice probabilidades de Superbalota usando ensemble."""
        if not self.sb_is_trained or not self.sb_models:
            return None
        
        all_predictions = []
        weights = []
        
        for name, model in self.sb_models.items():
            try:
                preds = model.predict_proba(X)
                all_predictions.append(preds)
                weight = 1.2 if name == 'XGBoost' else 1.0
                weights.append(weight)
            except Exception as e:
                print(f"⚠️ Error en predicción SB de {name}: {e}")
        
        if not all_predictions:
            return None
        
        # Promediar probabilidades
        weights = np.array(weights) / sum(weights)
        ensemble_proba = np.average(all_predictions, axis=0, weights=weights)
        
        return ensemble_proba
    
    def get_top_numbers(self, X, n_top=10):
        """
        Obtiene los n números más probables según el ensemble.
        
        Returns:
            Lista de tuplas (numero, probabilidad)
        """
        probas_list = self.predict_proba(X)
        if not probas_list:
            return []
        
        number_probs = []
        for i, probas in enumerate(probas_list):
            # probas[0] contiene [prob_clase_0, prob_clase_1, ...]
            # Queremos la probabilidad de clase 1 (que el número salga)
            if len(probas[0]) > 1:
                p_success = probas[0][1]
            else:
                p_success = 0
            number_probs.append((i + 1, p_success))
        
        # Ordenar por probabilidad descendente
        number_probs.sort(key=lambda x: x[1], reverse=True)
        return number_probs[:n_top]
    
    def get_top_sb(self, X, n_top=5):
        """
        Obtiene las superbalotas más probables según el ensemble.
        
        Returns:
            Lista de tuplas (numero_sb, probabilidad)
        """
        if not self.sb_is_trained:
            return []
        
        probas = self.predict_sb_proba(X)
        if probas is None:
            return []
        
        # Obtener clases del primer modelo SB disponible y revertir offset de normalización
        first_model = list(self.sb_models.values())[0]
        classes = first_model.classes_
        offset = getattr(self, '_sb_label_offset', 0)

        sb_probs = []
        for cls, prob in zip(classes, probas[0]):
            sb_probs.append((int(cls) + offset, prob))  # Revertir 0-15 → 1-16
        
        sb_probs.sort(key=lambda x: x[1], reverse=True)
        return sb_probs[:n_top]
    
    def get_model_agreement(self, X):
        """
        Calcula el nivel de acuerdo entre modelos (confidence).
        
        Returns:
            float: Score de confianza [0-1]
        """
        if not self.is_trained or len(self.models) < 2:
            return 1.0
        
        # Obtener top 10 de cada modelo
        top_numbers_per_model = []
        
        for name, model in self.models.items():
            try:
                preds = model.predict_proba(X)
                model_tops = []
                for i, probas in enumerate(preds):
                    p_success = probas[0][1] if len(probas[0]) > 1 else 0
                    model_tops.append((i + 1, p_success))
                model_tops.sort(key=lambda x: x[1], reverse=True)
                top_10 = set([n for n, p in model_tops[:10]])
                top_numbers_per_model.append(top_10)
            except:
                continue
        
        if len(top_numbers_per_model) < 2:
            return 1.0
        
        # Calcular intersección promedio
        agreements = []
        for i in range(len(top_numbers_per_model)):
            for j in range(i + 1, len(top_numbers_per_model)):
                intersection = len(top_numbers_per_model[i] & top_numbers_per_model[j])
                agreements.append(intersection / 10.0)
        
        return np.mean(agreements) if agreements else 1.0
