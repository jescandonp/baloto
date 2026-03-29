import os
import sys

# Permitir ejecutar el script desde la carpeta baloto_system o desde una superior
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from datetime import datetime
from baloto_system.data_manager import DataManager
from baloto_system.feature_engineering import FeatureEngineer
from baloto_system.models import Models
from baloto_system.statistics_analyzer import StatisticsAnalyzer
import pandas as pd

class BalotoSystem:
    def __init__(self):
        self.dm = DataManager()
        self.fe = FeatureEngineer()
        self.stats = StatisticsAnalyzer()
        self.models = {} 
        self.loaded_games = []
        self.use_ensemble = True  # Usar ensemble por defecto
        self.advanced_predictors = {}  # Predictores avanzados

    def start(self):
        print("🎯 SISTEMA AVANZADO DE BALOTO v7.0 ML Enhanced")
        print("="*60)
        
        # Intentar cargar datos por defecto si existen
        self.load_default_data()
        
        while True:
            self.print_menu()
            choice = input("\n👉 Opción: ").strip()
            
            if choice == '1':
                self.load_custom_data()
            elif choice == '2':
                self.train_models()
            elif choice == '3':
                self.generate_prediction_simple()
            elif choice == '4':
                self.generate_prediction_multiple()
            elif choice == '5':
                self.show_statistics()
            elif choice == '6':
                self.configure_training()
            elif choice == '0':
                print("👋 Hasta luego!")
                break
            else:
                print("❌ Opción inválida.")

    def print_menu(self):
        print("\n--- MENÚ PRINCIPAL ---")
        print(f"Juegos cargados: {', '.join(self.loaded_games) if self.loaded_games else 'Ninguno'}")
        mode_str = "Ensemble" if self.use_ensemble else "Simple"
        print(f"Modo entrenamiento: {mode_str}")
        print("1. 📂 Cargar datos CSV")
        print("2. 🤖 Entrenar modelos (ML Avanzado)")
        print("3. 🔮 Predicción simple (1 secuencia)")
        print("4. 🎯 Predicción múltiple (Top 10 secuencias)")
        print("5. 📊 Ver estadísticas")
        print("6. ⚙️  Configurar entrenamiento")
        print("0. 🚪 Salir")

    def load_default_data(self):
        # Archivos esperados por defecto (rutas relativas al script)
        files = {
            'baloto': os.path.join(current_dir, 'baloto_revancha_resultados_completo.csv'),
            'miloto': os.path.join(current_dir, 'miloto_resultados_completo.csv'),
            'colorloto': os.path.join(current_dir, 'colorloto_resultados_completo.csv'),
        }
        # Verificar si existen
        if os.path.exists(files['baloto']) and os.path.exists(files['miloto']):
            print("\n🔄 Cargando archivos por defecto...")
            self.dm.load_data(files['baloto'], files['miloto'], files.get('colorloto'))
            self.loaded_games = [k for k, v in self.dm.data.items() if not v.empty]
        else:
            print("\nℹ️ No se encontraron archivos de datos por defecto.")

    def load_custom_data(self):
        print("\nℹ️ Puedes ingresar la ruta del archivo específico O la carpeta donde están.")
        b_input = input("Ruta CSV Baloto/Revancha: ").strip().strip('"').strip("'")
        
        # Lógica inteligente para completar rutas
        b_file = b_input
        if os.path.isdir(b_input):
            b_file = os.path.join(b_input, 'baloto_revancha_resultados_completo.csv')
            print(f"   📂 Carpeta detectada. Buscando: {os.path.basename(b_file)}")

        m_input = input("Ruta CSV MiLoto: ").strip().strip('"').strip("'")
        m_file = m_input
        if os.path.isdir(m_input):
            m_file = os.path.join(m_input, 'miloto_resultados_completo.csv')
            print(f"   📂 Carpeta detectada. Buscando: {os.path.basename(m_file)}")

        if os.path.exists(b_file) and os.path.exists(m_file):
            # Buscar opcional ColorLoto en la misma carpeta si es directorio
            c_file = None
            if os.path.isdir(b_input):
                possible_c = os.path.join(b_input, 'colorloto_resultados_completo.csv')
                if os.path.exists(possible_c):
                    c_file = possible_c

            self.dm.load_data(b_file, m_file, c_file)
            self.loaded_games = [k for k, v in self.dm.data.items() if not v.empty]
        else:
            print(f"❌ Archivos no encontrados:\n   - {b_file}\n   - {m_file}")

    def train_models(self):
        if not self.loaded_games:
            print("⚠️ Carga datos primero.")
            return

        for game in self.loaded_games:
            print(f"\n⚙️ Preparando datos para {game.upper()}...")
            df = self.dm.data[game]
            config = self.dm.game_configs[game]
            
            # Preparar datos
            X, y, y_sb = self.fe.prepare_training_data(df, config)
            if X is None:
                print(f"   ⚠️ Datos insuficientes para {game}")
                continue
                
            # Inicializar y entrenar modelo
            if game not in self.models:
                self.models[game] = Models(game, use_ensemble=self.use_ensemble)
                
            success = self.models[game].train(X, y, y_sb)
            if success:
                print(f"   ✅ Modelo {game} entrenado exitosamente.")
                
                # Inicializar predictor avanzado
                try:
                    from baloto_system.advanced_predictor import AdvancedPredictor
                    self.advanced_predictors[game] = AdvancedPredictor(
                        self.models[game], df, config
                    )
                except ImportError:
                    pass

    def generate_prediction_simple(self):
        """Predicción simple (1 secuencia) - modo clásico."""
        if not self.models:
            print("⚠️ Entrena los modelos primero (Opción 2).")
            return
            
        game = input("Juego (baloto/revancha/miloto): ").strip().lower()
        if game not in self.models:
            print("❌ Modelo no entrenado o juego inválido.")
            return
            
        print(f"\n🔮 Generando predicción para {game.upper()}...")
        df = self.dm.data[game]
        config = self.dm.game_configs[game]
        model = self.models[game]
        
        # Preparar input (última ventana)
        X_pred = self.fe.prepare_prediction_input(df, config)
        if X_pred is None:
            print("❌ Error preparando entrada.")
            return
            
        # Predecir números principales
        top_nums = model.get_top_numbers(X_pred, n_top=config['numbers_count'] + 10)
        
        # Obtener score de confianza
        confidence = model.get_confidence_score(X_pred)
        
        print(f"\n🎯 PREDICCIÓN (Confianza: {confidence*100:.0f}%)")
        print("="*50)
        
        recommended = [n for n, p in top_nums[:config['numbers_count']]]
        print(f"\n   Números: {' - '.join([f'{n:02d}' for n in sorted(recommended)])}")
        
        # Predecir SuperBalota (si aplica)
        if config.get('has_superbalota'):
            top_sb = model.get_top_sb(X_pred, n_top=3)
            if top_sb:
                sb_recommended, sb_prob = top_sb[0]
                print(f"   SuperBalota: {sb_recommended:02d} ({sb_prob:.1%})")
                print(f"\n   Alternativas SB: {', '.join([f'{sb:02d} ({p:.1%})' for sb, p in top_sb[:3]])}")

        print("\n   📊 Probabilidades Top 10:")
        for n, p in top_nums[:10]:
            bar = '█' * int(p * 50)
            print(f"   #{n:02d}: {bar} {p:.1%}")
    
    def generate_prediction_multiple(self):
        """Predicción múltiple (Top 10 secuencias) - modo avanzado."""
        if not self.models:
            print("⚠️ Entrena los modelos primero (Opción 2).")
            return
            
        game = input("Juego (baloto/revancha/miloto): ").strip().lower()
        if game not in self.models:
            print("❌ Modelo no entrenado o juego inválido.")
            return
        
        if game not in self.advanced_predictors:
            print("⚠️ Predictor avanzado no disponible. Entrenando modelos...")
            return
            
        print(f"\n🎯 Generando múltiples predicciones para {game.upper()}...")
        df = self.dm.data[game]
        config = self.dm.game_configs[game]
        predictor = self.advanced_predictors[game]
        
        # Preparar input
        X_pred = self.fe.prepare_prediction_input(df, config)
        if X_pred is None:
            print("❌ Error preparando entrada.")
            return
        
        # Generar múltiples secuencias
        try:
            n_sequences = int(input("¿Cuántas secuencias generar? (1-20) [10]: ").strip() or "10")
        except ValueError:
            print("⚠️ Valor inválido, usando 10 por defecto.")
            n_sequences = 10
        n_sequences = max(1, min(20, n_sequences))
        
        strategy = input("Estrategia (top_n/diverse/stochastic/mixed) [mixed]: ").strip() or "mixed"
        
        print(f"\n🔄 Generando {n_sequences} secuencias con estrategia '{strategy}'...")
        sequences = predictor.generate_multiple_sequences(X_pred, n_sequences, strategy)
        
        # Mostrar resultados
        print(f"\n{'='*70}")
        print(f"🎯 TOP {len(sequences)} SECUENCIAS RECOMENDADAS")
        print(f"{'='*70}")
        
        for seq in sequences:
            rank = seq['rank']
            nums = seq['numbers']
            sb = seq['superbalota']
            score = seq['total_score']
            
            print(f"\n#{rank} [Score: {score:.2f}] [Confianza: {score*100:.0f}%]")
            print(f"   Números: {' - '.join([f'{n:02d}' for n in nums])}")
            
            if sb is not None:
                print(f"   SuperBalota: {sb:02d}")
            
            # Explicación
            explanation = predictor.explain_sequence(seq)
            print(f"   💡 {explanation}")
            
            # Scores detallados (solo para top 3)
            if rank <= 3:
                scores = seq['scores']
                print(f"   📊 Scores: Pattern={scores['pattern_match']:.2f} | "
                      f"Dist={scores['distribution']:.2f} | "
                      f"Div={scores['diversity']:.2f} | "
                      f"Hot/Cold={scores['hot_cold_balance']:.2f}")
            
            print(f"   {'-'*65}")

    def show_statistics(self):
        if not self.loaded_games:
            print("⚠️ Carga datos primero.")
            return
            
        game = input("Juego (baloto/revancha/miloto): ").strip().lower()
        if game not in self.dm.data:
            print("❌ Juego no cargado.")
            return

        df = self.dm.data[game]
        config = self.dm.game_configs[game]
        
        print(f"\n📊 ESTADÍSTICAS {game.upper()}")
        
        # Frecuencia
        freq_stats = self.stats.analyze_frequency(df, config)
        print("\n🔥 Números más calientes (Histórico):")
        for n, c in freq_stats['hot_numbers']:
            print(f"   {n:02d}: {c} veces")
            
        # Retardos
        delays = self.stats.analyze_delays(df, config)
        print("\n⏰ Números más demorados:")
        for n, d in delays[:5]:
            print(f"   {n:02d}: hace {d} sorteos")
    
    def configure_training(self):
        """Configurar opciones de entrenamiento."""
        print("\n⚙️  CONFIGURACIÓN DE ENTRENAMIENTO")
        print("="*50)
        print(f"\n1. Modo actual: {'Ensemble (Avanzado)' if self.use_ensemble else 'Simple (Rápido)'}")
        print("2. Cambiar modo de entrenamiento")
        print("0. Volver")
        
        choice = input("\n👉 Opción: ").strip()
        
        if choice == '2':
            print("\nModos disponibles:")
            print("1. Simple (HistGradientBoosting) - Rápido, ~30-60 seg")
            print("2. Ensemble (RF+XGB+LGB+Hist) - Preciso, ~2-5 min")
            
            mode_choice = input("\nSeleccionar modo (1/2): ").strip()
            
            if mode_choice == '1':
                self.use_ensemble = False
                print("\n✅ Cambiado a modo Simple. Re-entrena los modelos para aplicar.")
            elif mode_choice == '2':
                self.use_ensemble = True
                print("\n✅ Cambiado a modo Ensemble. Re-entrena los modelos para aplicar.")
            else:
                print("❌ Opción inválida.")

if __name__ == "__main__":
    system = BalotoSystem()
    system.start()
