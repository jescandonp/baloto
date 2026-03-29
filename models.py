from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.base import BaseEstimator
import numpy as np
import joblib
from datetime import datetime

# Importar ensemble si está disponible
try:
    from baloto_system.ensemble_models import EnsembleModels
    HAS_ENSEMBLE = True
except ImportError:
    HAS_ENSEMBLE = False

class Models:
    """Clase para manejar el entrenamiento y predicción con Machine Learning."""
    
    def __init__(self, game_type, use_ensemble=True):
        """
        Args:
            game_type: Tipo de juego (baloto, revancha, miloto)
            use_ensemble: Si True, usa ensemble de modelos (más lento pero más preciso)
        """
        self.game_type = game_type
        self.use_ensemble = use_ensemble and HAS_ENSEMBLE
        
        if self.use_ensemble:
            print(f"   🎯 Usando Ensemble de Modelos Avanzado")
            self.model = EnsembleModels(game_type, use_all_models=True)
            self.is_trained = False
            self.sb_is_trained = False
        else:
            print(f"   🎯 Usando Modelo Simple (HistGradientBoosting)")
            # Modelo simple original (backward compatibility)
            try:
                base_model = HistGradientBoostingClassifier(
                    max_iter=500,  # Incrementado de 100 a 500
                    learning_rate=0.05,  # Reducido para más estabilidad
                    max_depth=7,  # Incrementado de 5 a 7
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1
                )
            except:
                # Fallback para sklearn viejos
                base_model = GradientBoostingClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42
                )
                
            self.model = MultiOutputClassifier(base_model, n_jobs=-1)
            self.sb_model = HistGradientBoostingClassifier(
                max_iter=500,
                learning_rate=0.05,
                max_depth=7,
                random_state=42
            )
            self.is_trained = False
            self.sb_is_trained = False
        
    def train(self, X, y, y_sb=None):
        """Entrena el modelo (ensemble o simple según configuración)."""
        if len(X) < 50:
            print(f"⚠️ Datos insuficientes para entrenar {self.game_type} ({len(X)} muestras)")
            return False
        
        if self.use_ensemble:
            # Entrenar ensemble
            return self.model.train(X, y, y_sb, validate=True)
        else:
            # Entrenar modelo simple
            print(f"🔄 Entrenando modelo números para {self.game_type}...")
            self.model.fit(X, y)
            self.is_trained = True
            
            if y_sb is not None:
                print(f"🔄 Entrenando modelo Superbalota para {self.game_type}...")
                self.sb_model.fit(X, y_sb)
                self.sb_is_trained = True
                
            print("✅ Entrenamiento completado.")
            return True
    
    def predict_proba(self, X):
        """Retorna probabilidades para cada número."""
        if self.use_ensemble:
            return self.model.predict_proba(X)
        else:
            if not self.is_trained:
                return None
            return self.model.predict_proba(X)
        
    def predict_sb_proba(self, X):
        """Retorna probabilidades para la Superbalota."""
        if self.use_ensemble:
            return self.model.predict_sb_proba(X)
        else:
            if not self.sb_is_trained:
                return None
            return self.sb_model.predict_proba(X)

    def get_top_numbers(self, X, n_top=5):
        """Obtiene los n números más probables para una entrada X."""
        if self.use_ensemble:
            return self.model.get_top_numbers(X, n_top)
        else:
            # Implementación original
            probas_list = self.predict_proba(X)
            if not probas_list:
                return []
                
            number_probs = []
            for i, probas in enumerate(probas_list):
                p_success = probas[0][1] if len(probas[0]) > 1 else 0
                number_probs.append((i + 1, p_success))
                
            number_probs.sort(key=lambda x: x[1], reverse=True)
            return number_probs[:n_top]

    def get_top_sb(self, X, n_top=3):
        """Obtiene las superbalotas más probables."""
        if self.use_ensemble:
            return self.model.get_top_sb(X, n_top)
        else:
            # Implementación original
            if not self.sb_is_trained:
                return []
                
            probas = self.sb_model.predict_proba(X)[0]
            classes = self.sb_model.classes_
            
            sb_probs = []
            for cls, prob in zip(classes, probas):
                sb_probs.append((cls, prob))
                
            sb_probs.sort(key=lambda x: x[1], reverse=True)
            return sb_probs[:n_top]
    
    def get_confidence_score(self, X):
        """
        Obtiene score de confianza de la predicción.
        Solo disponible en ensemble.
        """
        if self.use_ensemble and hasattr(self.model, 'get_model_agreement'):
            return self.model.get_model_agreement(X)
        return 1.0  # Confianza 100% por defecto en modelo simple

    def save_model(self, filepath):
        """Guarda el modelo entrenado con todos sus metadatos."""
        bundle = {
            'game_type': self.game_type,
            'use_ensemble': self.use_ensemble,
            'model': self.model,
            'sb_model': getattr(self, 'sb_model', None),
            'is_trained': self.is_trained,
            'sb_is_trained': self.sb_is_trained,
            'saved_at': datetime.now().isoformat()
        }
        joblib.dump(bundle, filepath)

    def load_model(self, filepath):
        """Carga un modelo guardado. Retorna la fecha de guardado o None si falla."""
        try:
            bundle = joblib.load(filepath)
            self.model = bundle['model']
            self.is_trained = bundle.get('is_trained', True)
            self.sb_is_trained = bundle.get('sb_is_trained', False)
            if not self.use_ensemble and bundle.get('sb_model') is not None:
                self.sb_model = bundle['sb_model']
            return bundle.get('saved_at', 'fecha desconocida')
        except Exception as e:
            print(f"⚠️ No se pudo cargar modelo desde {filepath}: {e}")
            return None
