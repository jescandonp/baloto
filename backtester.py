"""
Backtesting de walk-forward: entrena en datos históricos, predice el siguiente
sorteo y mide cuántos números acierta vs. el resultado real.
"""

import io
import contextlib
from collections import Counter


class Backtester:
    """Validación histórica walk-forward para cualquier juego de lotería."""

    def __init__(self, feature_engineer, game_configs):
        self.fe = feature_engineer
        self.game_configs = game_configs

    def run(self, game, full_df, n_test=20):
        """
        Ejecuta backtesting walk-forward.

        Para cada sorteo i en los últimos n_test:
          1. Entrena con full_df[:i]
          2. Predice los números del sorteo i
          3. Compara con full_df[i]['Numeros']
          4. Registra aciertos

        Args:
            game:     Nombre del juego (baloto, revancha, miloto)
            full_df:  DataFrame completo con todos los sorteos
            n_test:   Cantidad de sorteos a evaluar (cola del dataset)

        Returns:
            Lista de dicts con resultados por sorteo, o None si falló.
        """
        from baloto_system.models import Models

        config = self.game_configs[game]
        n = len(full_df)
        min_train = 50

        if n < n_test + min_train:
            print(f"❌ Datos insuficientes. Se necesitan al menos {n_test + min_train} "
                  f"sorteos (tienes {n}).")
            return None

        print(f"\n{'='*65}")
        print(f"📈 BACKTESTING WALK-FORWARD — {game.upper()}")
        print(f"{'='*65}")
        print(f"   Sorteos a evaluar : {n_test}")
        print(f"   Sorteos de entren.: {n - n_test} → {n - 1} (crece en cada paso)")
        print(f"   Modo              : Simple (rápido)")
        est_secs = n_test * 25
        print(f"   Tiempo estimado   : ~{est_secs // 60}m {est_secs % 60}s\n")

        results = []
        total_hits = 0

        for step, test_idx in enumerate(range(n - n_test, n)):
            train_df = full_df.iloc[:test_idx].copy()

            # Preparar datos de entrenamiento
            X_tr, y_tr, y_sb_tr = self.fe.prepare_training_data(train_df, config)
            if X_tr is None:
                continue

            # Entrenar modelo simple (suprimir output)
            model = Models(game, use_ensemble=False)
            with contextlib.redirect_stdout(io.StringIO()):
                trained = model.train(X_tr, y_tr, y_sb_tr)

            if not trained or not model.is_trained:
                continue

            # Preparar input de predicción
            X_pred = self.fe.prepare_prediction_input(train_df, config)
            if X_pred is None:
                continue

            # Predecir top N números
            top_nums = model.get_top_numbers(X_pred, n_top=config['numbers_count'])
            predicted = set(num for num, prob in top_nums)
            actual = set(full_df.iloc[test_idx]['Numeros'])
            hits = len(predicted & actual)
            total_hits += hits

            draw_date = str(full_df.iloc[test_idx]['Fecha'])[:10]
            pred_str = ' '.join(f'{num:02d}' for num in sorted(predicted))
            real_str = ' '.join(f'{num:02d}' for num in sorted(actual))
            hit_marker = '⭐' * hits if hits > 0 else '·'

            print(f"   [{step+1:2d}/{n_test}] {draw_date} | "
                  f"Pred: {pred_str} | Real: {real_str} | "
                  f"{hits}/5 {hit_marker}")

            results.append({
                'draw_idx': test_idx,
                'date': draw_date,
                'predicted': sorted(predicted),
                'actual': sorted(actual),
                'hits': hits
            })

        if not results:
            print("❌ No se pudo completar el backtesting.")
            return None

        self._print_summary(game, results, total_hits)
        return results

    def _print_summary(self, game, results, total_hits):
        """Imprime el resumen estadístico del backtesting."""
        n = len(results)
        avg_hits = total_hits / n
        hit_dist = Counter(r['hits'] for r in results)
        best = max(results, key=lambda x: x['hits'])

        print(f"\n{'='*65}")
        print(f"📊 RESUMEN — {game.upper()}")
        print(f"{'='*65}")
        print(f"   Sorteos evaluados : {n}")
        print(f"   Aciertos promedio : {avg_hits:.2f} / 5")
        print(f"   Tasa de acierto   : {avg_hits / 5 * 100:.1f}%")
        print(f"\n   Distribución de aciertos:")
        for k in range(6):
            count = hit_dist.get(k, 0)
            pct = count / n * 100
            bar = '█' * count
            print(f"   {k} aciertos: {bar:<10} {count:2d}x  ({pct:.0f}%)")
        print(f"\n   Mejor predicción  : {best['date']} — {best['hits']} aciertos")
        print(f"      Pred: {' '.join(f'{x:02d}' for x in best['predicted'])}")
        print(f"      Real: {' '.join(f'{x:02d}' for x in best['actual'])}")
        print(f"{'='*65}\n")
