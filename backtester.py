"""
Backtesting de walk-forward: entrena en datos históricos, predice el siguiente
sorteo y mide cuántos números acierta vs. el resultado real.

Soporta dos modos:
  - run()             : evalúa la estrategia ML estándar (top-N)
  - run_comparison()  : compara ML, frecuencia histórica, frecuencia reciente
                        y una línea base aleatoria en el mismo período
"""

import io
import random
import contextlib
from collections import Counter


class Backtester:
    """Validación histórica walk-forward para cualquier juego de lotería."""

    def __init__(self, feature_engineer, game_configs):
        self.fe = feature_engineer
        self.game_configs = game_configs

    # ------------------------------------------------------------------
    # Modo 1: backtesting simple (estrategia ML top-N)
    # ------------------------------------------------------------------

    def run(self, game, full_df, n_test=20):
        """Ejecuta backtesting walk-forward con la estrategia ML top-N."""
        config = self.game_configs[game]
        n = len(full_df)

        if not self._check_data(n, n_test):
            return None

        print(f"\n{'='*65}")
        print(f"📈 BACKTESTING WALK-FORWARD — {game.upper()}")
        print(f"{'='*65}")
        print(f"   Sorteos a evaluar : {n_test}")
        print(f"   Tiempo estimado   : ~{n_test * 25 // 60}m {n_test * 25 % 60}s\n")

        results = []
        total_hits = 0

        for step, test_idx in enumerate(range(n - n_test, n)):
            train_df = full_df.iloc[:test_idx].copy()
            model, X_pred = self._train_and_prepare(game, train_df, config)
            if model is None:
                continue

            predicted = self._strat_ml_topn(model, X_pred, train_df, config)
            actual = set(full_df.iloc[test_idx]['Numeros'])
            hits = len(predicted & actual)
            total_hits += hits

            draw_date = str(full_df.iloc[test_idx]['Fecha'])[:10]
            pred_str = ' '.join(f'{x:02d}' for x in sorted(predicted))
            real_str = ' '.join(f'{x:02d}' for x in sorted(actual))
            print(f"   [{step+1:2d}/{n_test}] {draw_date} | "
                  f"Pred: {pred_str} | Real: {real_str} | "
                  f"{hits}/5 {'⭐' * hits if hits else '·'}")

            results.append({'date': draw_date, 'predicted': sorted(predicted),
                             'actual': sorted(actual), 'hits': hits})

        if not results:
            print("❌ No se pudo completar el backtesting.")
            return None

        self._print_summary(game, results, total_hits)
        return results

    # ------------------------------------------------------------------
    # Modo 2: backtesting comparativo (múltiples estrategias)
    # ------------------------------------------------------------------

    def run_comparison(self, game, full_df, n_test=20):
        """
        Compara 5 estrategias sobre el mismo período walk-forward:
          1. ML top-N          → modelo entrenado, top probabilidades
          2. Frecuencia total  → 5 números más frecuentes en toda la historia
          3. Frecuencia 30     → 5 más frecuentes en los últimos 30 sorteos
          4. Frecuencia 10     → 5 más frecuentes en los últimos 10 sorteos
          5. Aleatorio         → selección aleatoria (promedio de 20 corridas)
        """
        config = self.game_configs[game]
        n = len(full_df)

        if not self._check_data(n, n_test):
            return None

        n_rand_runs = 20  # corridas aleatorias por sorteo para estabilizar el promedio
        strategy_names = [
            'ML top-N',
            'Frecuencia total',
            'Frecuencia últ.30',
            'Frecuencia últ.10',
            f'Aleatorio (x{n_rand_runs})',
        ]
        hits_per_strategy = {name: [] for name in strategy_names}

        print(f"\n{'='*70}")
        print(f"📊 BACKTESTING COMPARATIVO — {game.upper()}")
        print(f"{'='*70}")
        print(f"   Sorteos : {n_test}  |  Estrategias : {len(strategy_names)}")
        est = n_test * 28
        print(f"   Tiempo estimado : ~{est // 60}m {est % 60}s\n")
        header = f"   {'Fecha':<12} {'ML':>5} {'F.Total':>8} {'F.30':>6} {'F.10':>6} {'Azar':>6}"
        print(header)
        print(f"   {'-'*52}")

        for step, test_idx in enumerate(range(n - n_test, n)):
            train_df = full_df.iloc[:test_idx].copy()
            actual   = set(full_df.iloc[test_idx]['Numeros'])
            date_str = str(full_df.iloc[test_idx]['Fecha'])[:10]

            # Estrategia 1: ML
            model, X_pred = self._train_and_prepare(game, train_df, config)
            ml_hits = 0
            if model is not None:
                pred_ml = self._strat_ml_topn(model, X_pred, train_df, config)
                ml_hits = len(pred_ml & actual)

            # Estrategia 2: Frecuencia total
            pred_freq_all    = self._strat_freq(train_df, config, lookback=None)
            freq_all_hits    = len(pred_freq_all & actual)

            # Estrategia 3: Frecuencia últimas 30
            pred_freq_30     = self._strat_freq(train_df, config, lookback=30)
            freq_30_hits     = len(pred_freq_30 & actual)

            # Estrategia 4: Frecuencia últimas 10
            pred_freq_10     = self._strat_freq(train_df, config, lookback=10)
            freq_10_hits     = len(pred_freq_10 & actual)

            # Estrategia 5: Aleatorio (promedio de n_rand_runs corridas)
            rand_hits_total  = sum(
                len(self._strat_random(config) & actual)
                for _ in range(n_rand_runs)
            )
            rand_hits_avg    = rand_hits_total / n_rand_runs

            hits_per_strategy['ML top-N'].append(ml_hits)
            hits_per_strategy['Frecuencia total'].append(freq_all_hits)
            hits_per_strategy['Frecuencia últ.30'].append(freq_30_hits)
            hits_per_strategy['Frecuencia últ.10'].append(freq_10_hits)
            hits_per_strategy[f'Aleatorio (x{n_rand_runs})'].append(rand_hits_avg)

            print(f"   {date_str:<12} "
                  f"{ml_hits:>5} "
                  f"{freq_all_hits:>8} "
                  f"{freq_30_hits:>6} "
                  f"{freq_10_hits:>6} "
                  f"{rand_hits_avg:>6.2f}")

        self._print_comparison_summary(game, strategy_names, hits_per_strategy,
                                       n_test, config)
        return hits_per_strategy

    # ------------------------------------------------------------------
    # Estrategias de predicción
    # ------------------------------------------------------------------

    def _strat_ml_topn(self, model, X_pred, train_df, config):
        """Estrategia ML: top-N números por probabilidad del modelo."""
        top = model.get_top_numbers(X_pred, n_top=config['numbers_count'])
        return set(num for num, _ in top)

    def _strat_freq(self, train_df, config, lookback=None):
        """Estrategia de frecuencia: los N más frecuentes en el histórico."""
        k = config['numbers_count']
        if lookback is not None:
            lookback = min(lookback, len(train_df))
            subset = train_df.iloc[-lookback:]
        else:
            subset = train_df

        cnt = Counter()
        for nums in subset['Numeros']:
            cnt.update(nums)

        # Si hay empates, completar con números del rango no presentes
        top = set(num for num, _ in cnt.most_common(k))
        if len(top) < k:
            lo, hi = config['number_range']
            for num in range(lo, hi + 1):
                if len(top) >= k:
                    break
                top.add(num)
        return top

    def _strat_random(self, config):
        """Estrategia aleatoria: selección uniforme sin reemplazo."""
        lo, hi = config['number_range']
        return set(random.sample(range(lo, hi + 1), config['numbers_count']))

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _check_data(self, n, n_test, min_train=50):
        if n < n_test + min_train:
            print(f"❌ Datos insuficientes. Necesitas ≥ {n_test + min_train} sorteos (tienes {n}).")
            return False
        return True

    def _train_and_prepare(self, game, train_df, config):
        """Entrena un modelo simple silenciosamente y prepara el input."""
        from baloto_system.models import Models

        X_tr, y_tr, y_sb_tr = self.fe.prepare_training_data(train_df, config)
        if X_tr is None:
            return None, None

        model = Models(game, use_ensemble=False)
        with contextlib.redirect_stdout(io.StringIO()):
            trained = model.train(X_tr, y_tr, y_sb_tr)

        if not trained or not model.is_trained:
            return None, None

        X_pred = self.fe.prepare_prediction_input(train_df, config)
        return (model, X_pred) if X_pred is not None else (None, None)

    # ------------------------------------------------------------------
    # Impresión de resultados
    # ------------------------------------------------------------------

    def _print_summary(self, game, results, total_hits):
        """Resumen del backtesting simple."""
        n = len(results)
        avg_hits = total_hits / n
        hit_dist = Counter(r['hits'] for r in results)
        best = max(results, key=lambda x: x['hits'])

        print(f"\n{'='*65}")
        print(f"📊 RESUMEN — {game.upper()}")
        print(f"{'='*65}")
        print(f"   Sorteos evaluados : {n}")
        print(f"   Aciertos promedio : {avg_hits:.2f} / 5  →  {avg_hits/5*100:.1f}%")
        print(f"\n   Distribución:")
        for k in range(6):
            count = hit_dist.get(k, 0)
            bar = '█' * count
            print(f"   {k} aciertos: {bar:<12} {count:2d}x  ({count/n*100:.0f}%)")
        print(f"\n   Mejor : {best['date']} — {best['hits']} aciertos")
        print(f"      Pred: {' '.join(f'{x:02d}' for x in best['predicted'])}")
        print(f"      Real: {' '.join(f'{x:02d}' for x in best['actual'])}")
        print(f"{'='*65}\n")

    def _print_comparison_summary(self, game, strategy_names, hits_per_strategy,
                                   n_test, config):
        """Tabla comparativa de todas las estrategias."""
        n_nums = config['numbers_count']
        lo, hi  = config['number_range']
        expected_random = n_nums * n_nums / (hi - lo + 1)

        print(f"\n{'='*70}")
        print(f"🏆 TABLA COMPARATIVA — {game.upper()}")
        print(f"{'='*70}")
        print(f"   (Referencia teórica aleatoria: {expected_random:.2f} aciertos / {expected_random/n_nums*100:.1f}%)\n")

        # Calcular stats por estrategia
        stats = []
        for name in strategy_names:
            h = hits_per_strategy[name]
            avg  = sum(h) / len(h)
            rate = avg / n_nums * 100
            best = max(h)
            stats.append((name, avg, rate, best))

        # Ordenar de mejor a peor
        stats.sort(key=lambda x: x[1], reverse=True)

        winner = stats[0][0]
        max_avg = stats[0][1]

        print(f"   {'Estrategia':<22} {'Prom':>6} {'Tasa':>7} {'Mejor':>6}  {'Barra'}")
        print(f"   {'-'*60}")
        for name, avg, rate, best in stats:
            bar_len = int(rate / 2)          # escala: 0-50% → 0-25 chars
            bar = '█' * bar_len
            crown = ' 👑' if name == winner else ''
            vs_random = avg - expected_random
            sign = '+' if vs_random >= 0 else ''
            print(f"   {name:<22} {avg:>6.2f} {rate:>6.1f}%  {best:>5}   {bar}{crown}")
            print(f"   {'':22} {'vs aleatorio: ' + sign + f'{vs_random:.2f}':>28}")

        print(f"\n   🥇 Mejor estrategia : {winner}  ({stats[0][1]:.2f} hits promedio)")
        print(f"   📌 Sorteos evaluados: {n_test}")
        print(f"{'='*70}\n")
