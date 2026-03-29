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
import glob as glob_module

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
            elif choice == '7':
                self.run_backtest()
            elif choice == '8':
                self.run_fun_mode()
            elif choice == '0':
                print("👋 Hasta luego!")
                break
            else:
                print("❌ Opción inválida.")

    def print_menu(self):
        print("\n--- MENÚ PRINCIPAL ---")
        print(f"Juegos cargados: {', '.join(self.loaded_games) if self.loaded_games else 'Ninguno'}")
        models_str = f"Modelos listos: {', '.join(self.models.keys()) if self.models else 'Ninguno'}"
        print(models_str)
        mode_str = "Ensemble" if self.use_ensemble else "Simple"
        print(f"Modo entrenamiento: {mode_str}")
        print("1. 📂 Cargar datos CSV")
        print("2. 🤖 Entrenar modelos (ML Avanzado)")
        print("3. 🔮 Predicción simple (1 secuencia)")
        print("4. 🎯 Predicción múltiple (Top 10 secuencias)")
        print("5. 📊 Ver estadísticas")
        print("6. ⚙️  Configurar entrenamiento")
        print("7. 📈 Backtesting (Validación histórica)")
        print("8. 🎲 Modo FUN (combinaciones de suerte)")
        print("0. 🚪 Salir")

    def _get_model_dir(self):
        """Retorna la carpeta de modelos guardados, creándola si no existe."""
        model_dir = os.path.join(current_dir, 'saved_models')
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    def _save_models(self):
        """Guarda todos los modelos entrenados en disco."""
        model_dir = self._get_model_dir()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for game, model in self.models.items():
            if model.is_trained:
                filepath = os.path.join(model_dir, f'{game}_{timestamp}.joblib')
                model.save_model(filepath)
                print(f"   💾 Modelo {game} guardado: {os.path.basename(filepath)}")

    def _load_saved_models(self):
        """Carga los modelos más recientes desde disco si existen."""
        model_dir = self._get_model_dir()
        loaded_any = False

        for game in self.loaded_games:
            pattern = os.path.join(model_dir, f'{game}_*.joblib')
            files = sorted(glob_module.glob(pattern))
            if not files:
                continue

            newest = files[-1]  # Los nombres son YYYYMMDD_HHMMSS, orden léxico = cronológico
            model = Models(game, use_ensemble=self.use_ensemble)
            saved_at = model.load_model(newest)
            if saved_at:
                self.models[game] = model
                # Inicializar predictor avanzado
                try:
                    from baloto_system.advanced_predictor import AdvancedPredictor
                    df = self.dm.data[game]
                    config = self.dm.game_configs[game]
                    self.advanced_predictors[game] = AdvancedPredictor(model, df, config)
                except ImportError:
                    pass
                print(f"   ✅ Modelo {game} cargado (guardado el {saved_at[:19]})")
                loaded_any = True

        if loaded_any:
            print("   💡 Modelos listos. No es necesario re-entrenar.")

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
            self._load_saved_models()
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
            self._load_saved_models()
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

        # Guardar todos los modelos entrenados
        print("\n💾 Guardando modelos en disco...")
        self._save_models()

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

    def run_fun_mode(self):
        """Modo FUN: combinaciones de suerte con distintos sabores."""
        import random as _random
        from collections import Counter

        print("\n🎲 MODO FUN — ¡Al final es suerte!")
        print("=" * 55)
        print(f"Juegos disponibles: {', '.join(self.loaded_games) if self.loaded_games else 'todos'}")
        game = input("Juego (baloto/revancha/miloto) [baloto]: ").strip().lower() or "baloto"

        # Configuración del juego
        config = self.dm.game_configs.get(game)
        if config is None:
            print("❌ Juego no reconocido.")
            return

        lo, hi   = config['number_range']
        k        = config['numbers_count']
        has_sb   = config.get('has_superbalota', False)
        sb_lo, sb_hi = config.get('superbalota_range', (1, 1))
        all_nums = list(range(lo, hi + 1))

        # Datos históricos para hot numbers (si están disponibles)
        df = self.dm.data.get(game)
        hot_numbers = []
        cold_numbers = []
        if df is not None and not df.empty:
            lookback = min(30, len(df))
            cnt = Counter()
            for nums in df.iloc[-lookback:]['Numeros']:
                cnt.update(nums)
            sorted_by_freq = [n for n, _ in cnt.most_common()]
            hot_numbers  = sorted_by_freq[:10]
            cold_numbers = sorted_by_freq[-10:]

        print("\n¿Cuántas combinaciones quieres generar? (1-20) [5]: ", end="")
        try:
            n_combos = int(input().strip() or "5")
            n_combos = max(1, min(20, n_combos))
        except ValueError:
            n_combos = 5

        print("\nElige el sabor de suerte:")
        print("  1. 🎰 Puro azar          — 100% aleatorio")
        print("  2. 🔥 Sabor caliente      — mezcla números calientes + azar")
        print("  3. ❄️  Sabor frío          — apuesta por los que no han salido")
        print("  4. ☯️  Equilibrio          — mitad calientes, mitad fríos + azar")
        print("  5. 🤞 Tu número de suerte — tú fijas un número, el resto al azar")
        print("  6. 🎂 Fecha especial       — usa dígitos de una fecha como ancla")
        flavor = input("\nSabor (1-6) [1]: ").strip() or "1"

        # Número fijo del usuario (sabor 5)
        fixed_num = None
        if flavor == "5":
            try:
                fixed_num = int(input(f"Ingresa tu número de suerte ({lo}-{hi}): ").strip())
                if not (lo <= fixed_num <= hi):
                    print(f"⚠️ Fuera de rango. Usando aleatorio.")
                    fixed_num = None
            except ValueError:
                fixed_num = None

        # Fecha especial (sabor 6)
        date_anchors = []
        if flavor == "6":
            raw = input("Ingresa una fecha (DD/MM/AAAA): ").strip()
            try:
                parts = raw.replace("-", "/").split("/")
                digits = []
                for p in parts:
                    digits += [int(p), int(p) % (hi + 1) or lo]
                # Filtrar válidos y únicos
                date_anchors = list({n for n in digits if lo <= n <= hi})[:3]
                print(f"   Anclas extraídas: {date_anchors}")
            except Exception:
                print("⚠️ No se pudo parsear la fecha. Usando azar puro.")
                date_anchors = []

        # ---- Generación de combinaciones ----
        print(f"\n{'='*55}")
        print(f"🎲 {n_combos} COMBINACIONES — {game.upper()}")
        print(f"{'='*55}")

        used = set()
        generated = 0
        attempts  = 0

        while generated < n_combos and attempts < n_combos * 50:
            attempts += 1
            pool = all_nums.copy()

            if flavor == "1":   # Puro azar
                picked = _random.sample(pool, k)

            elif flavor == "2": # Calientes
                if len(hot_numbers) >= k:
                    n_hot  = _random.randint(max(2, k // 2), min(k - 1, len(hot_numbers)))
                    hot    = _random.sample(hot_numbers, n_hot)
                    remain = [n for n in pool if n not in hot]
                    rest   = _random.sample(remain, k - n_hot)
                    picked = hot + rest
                else:
                    picked = _random.sample(pool, k)

            elif flavor == "3": # Fríos
                if len(cold_numbers) >= k:
                    n_cold = _random.randint(max(2, k // 2), min(k - 1, len(cold_numbers)))
                    cold   = _random.sample(cold_numbers, n_cold)
                    remain = [n for n in pool if n not in cold]
                    rest   = _random.sample(remain, k - n_cold)
                    picked = cold + rest
                else:
                    picked = _random.sample(pool, k)

            elif flavor == "4": # Equilibrio
                n_hot  = k // 2
                n_cold = k // 2
                avail_hot  = hot_numbers  if len(hot_numbers)  >= n_hot  else pool
                avail_cold = cold_numbers if len(cold_numbers) >= n_cold else pool
                h = _random.sample(avail_hot,  min(n_hot,  len(avail_hot)))
                c = _random.sample([x for x in avail_cold if x not in h],
                                   min(n_cold, len([x for x in avail_cold if x not in h])))
                remain = [n for n in pool if n not in h and n not in c]
                fill   = _random.sample(remain, k - len(h) - len(c))
                picked = h + c + fill

            elif flavor == "5": # Número fijo
                anchor = fixed_num if fixed_num else _random.choice(pool)
                remain = [n for n in pool if n != anchor]
                rest   = _random.sample(remain, k - 1)
                picked = [anchor] + rest

            elif flavor == "6": # Fecha
                anchors = [n for n in date_anchors if n in pool][:min(3, k - 1)]
                remain  = [n for n in pool if n not in anchors]
                rest    = _random.sample(remain, k - len(anchors))
                picked  = anchors + rest

            else:
                picked = _random.sample(pool, k)

            combo = tuple(sorted(picked))
            if combo in used or len(combo) != k:
                continue

            used.add(combo)
            generated += 1

            sb_str = ""
            if has_sb:
                sb = _random.randint(sb_lo, sb_hi)
                sb_str = f"  +SB: {sb:02d}"

            nums_str = " - ".join(f"{n:02d}" for n in combo)
            print(f"  #{generated:2d}  [ {nums_str} ]{sb_str}")

        print(f"\n{'='*55}")
        print("  🍀 ¡Buena suerte! Recuerda: juega con responsabilidad.")
        print(f"{'='*55}\n")

    def run_backtest(self):
        """Backtesting: valida el modelo contra sorteos históricos."""
        if not self.loaded_games:
            print("⚠️ Carga datos primero.")
            return

        print("\n📈 BACKTESTING - Validación histórica")
        print("="*50)
        print(f"Juegos disponibles: {', '.join(self.loaded_games)}")
        game = input("Juego a evaluar (baloto/revancha/miloto): ").strip().lower()
        if game not in self.loaded_games:
            print("❌ Juego no disponible.")
            return

        try:
            n_test = int(input("¿Cuántos sorteos evaluar? (5-50) [20]: ").strip() or "20")
            n_test = max(5, min(50, n_test))
        except ValueError:
            n_test = 20

        print("\nModos disponibles:")
        print("  1. Estándar    — ML top-N detallado por sorteo")
        print("  2. Comparativo — ML vs Frecuencia vs Aleatorio en tabla")
        mode = input("Modo (1/2) [1]: ").strip() or "1"

        from baloto_system.backtester import Backtester
        bt = Backtester(self.fe, self.dm.game_configs)

        if mode == "2":
            bt.run_comparison(game, self.dm.data[game], n_test=n_test)
        else:
            bt.run(game, self.dm.data[game], n_test=n_test)


if __name__ == "__main__":
    system = BalotoSystem()
    system.start()
