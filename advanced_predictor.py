"""
Sistema de Predicción Avanzada con generación de múltiples secuencias candidatas.
Incluye scoring, ranking y análisis de diversidad.
"""

import numpy as np
from collections import Counter
from itertools import combinations
import random


class AdvancedPredictor:
    """
    Generador de múltiples secuencias de predicción con scoring sofisticado.
    """
    
    def __init__(self, model, data_df, config):
        """
        Args:
            model: Modelo entrenado (con métodos get_top_numbers y get_top_sb)
            data_df: DataFrame con datos históricos
            config: Configuración del juego
        """
        self.model = model
        self.data_df = data_df
        self.config = config
        self.historical_patterns = self._analyze_historical_patterns()
        self._current_number_probs = {}  # Cache de probabilidades para scoring
    
    def _analyze_historical_patterns(self):
        """Analiza patrones históricos para scoring."""
        patterns = {
            'pair_frequencies': Counter(),
            'sum_distribution': [],
            'gap_distribution': [],
            'even_odd_ratio': []
        }
        
        for _, row in self.data_df.iterrows():
            nums = sorted(row['Numeros'])
            
            # Pares de números
            for pair in combinations(nums, 2):
                patterns['pair_frequencies'][pair] += 1
            
            # Suma total
            patterns['sum_distribution'].append(sum(nums))
            
            # Gaps entre números
            if len(nums) > 1:
                gaps = [nums[i+1] - nums[i] for i in range(len(nums) - 1)]
                patterns['gap_distribution'].extend(gaps)
            
            # Ratio par/impar
            even_count = sum(1 for n in nums if n % 2 == 0)
            patterns['even_odd_ratio'].append(even_count / len(nums))
        
        return patterns
    
    def generate_multiple_sequences(self, X_pred, n_sequences=10, strategy='mixed'):
        """
        Genera múltiples secuencias candidatas con diferentes estrategias.
        
        Args:
            X_pred: Input features para predicción
            n_sequences: Número de secuencias a generar
            strategy: 'top_n' | 'diverse' | 'mixed' | 'stochastic'
        
        Returns:
            Lista de diccionarios con secuencias y metadata
        """
        sequences = []
        
        # Obtener probabilidades de números y cachearlas para scoring
        top_numbers = self.model.get_top_numbers(X_pred, n_top=30)
        self._current_number_probs = {n: p for n, p in top_numbers}
        
        # Obtener superbalotas (si aplica)
        top_sb = []
        if self.config.get('has_superbalota'):
            top_sb = self.model.get_top_sb(X_pred, n_top=10)
        
        if strategy == 'top_n':
            sequences = self._generate_top_n(top_numbers, top_sb, n_sequences)
        elif strategy == 'diverse':
            sequences = self._generate_diverse(top_numbers, top_sb, n_sequences)
        elif strategy == 'stochastic':
            sequences = self._generate_stochastic(top_numbers, top_sb, n_sequences)
        else:  # mixed (default)
            # Mezclar estrategias
            n_top = n_sequences // 3
            n_diverse = n_sequences // 3
            n_stochastic = n_sequences - n_top - n_diverse
            
            sequences.extend(self._generate_top_n(top_numbers, top_sb, n_top))
            sequences.extend(self._generate_diverse(top_numbers, top_sb, n_diverse))
            sequences.extend(self._generate_stochastic(top_numbers, top_sb, n_stochastic))
        
        # Calcular scores para cada secuencia
        scored_sequences = []
        for seq in sequences:
            score_info = self._calculate_sequence_score(seq['numbers'])
            seq['scores'] = score_info
            seq['total_score'] = score_info['total']
            scored_sequences.append(seq)
        
        # Ordenar por score total
        scored_sequences.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Agregar ranking
        for i, seq in enumerate(scored_sequences):
            seq['rank'] = i + 1
        
        return scored_sequences[:n_sequences]
    
    def _generate_top_n(self, top_numbers, top_sb, n_sequences):
        """Genera secuencias usando los números con mayor probabilidad."""
        sequences = []
        nums_count = self.config['numbers_count']
        
        for i in range(n_sequences):
            # Tomar los top N números con un offset
            start_idx = i
            end_idx = start_idx + nums_count
            selected_nums = [n for n, p in top_numbers[start_idx:end_idx]]
            
            # Asegurar que tengamos suficientes números
            if len(selected_nums) < nums_count:
                # Completar con números adicionales
                remaining = [n for n, p in top_numbers if n not in selected_nums]
                selected_nums.extend(remaining[:nums_count - len(selected_nums)])
            
            selected_nums = sorted(selected_nums[:nums_count])
            
            # Seleccionar superbalota
            sb = None
            if top_sb and i < len(top_sb):
                sb, sb_prob = top_sb[i]
            
            sequences.append({
                'numbers': selected_nums,
                'superbalota': sb,
                'strategy': 'top_n',
                'generation_method': f'Top {nums_count} más probables (offset {i})'
            })
        
        return sequences
    
    def _generate_diverse(self, top_numbers, top_sb, n_sequences):
        """Genera secuencias maximizando diversidad."""
        sequences = []
        nums_count = self.config['numbers_count']
        used_combinations = set()
        
        # Pool de números candidatos (top 20)
        candidates = [n for n, p in top_numbers[:20]]
        
        attempts = 0
        max_attempts = n_sequences * 10
        
        while len(sequences) < n_sequences and attempts < max_attempts:
            attempts += 1
            
            # Seleccionar números con diversidad
            selected = []
            available = candidates.copy()
            
            for _ in range(nums_count):
                if not available:
                    break
                
                # Seleccionar número que maximice distancia mínima
                best_num = None
                best_min_dist = -1
                
                for num in available:
                    if not selected:
                        best_num = num
                        break
                    
                    min_dist = min(abs(num - s) for s in selected)
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_num = num
                
                if best_num:
                    selected.append(best_num)
                    available.remove(best_num)
            
            selected = sorted(selected)
            combo_key = tuple(selected)
            
            # Verificar que no se repita
            if combo_key not in used_combinations and len(selected) == nums_count:
                used_combinations.add(combo_key)
                
                sb = None
                if top_sb and len(sequences) < len(top_sb):
                    sb, _ = top_sb[len(sequences)]
                
                sequences.append({
                    'numbers': selected,
                    'superbalota': sb,
                    'strategy': 'diverse',
                    'generation_method': 'Máxima diversidad entre números'
                })
        
        return sequences
    
    def _generate_stochastic(self, top_numbers, top_sb, n_sequences):
        """Genera secuencias usando sampling estocástico con probabilidades."""
        sequences = []
        nums_count = self.config['numbers_count']
        
        # Preparar probabilidades normalizadas
        numbers = [n for n, p in top_numbers[:25]]
        probs = np.array([p for n, p in top_numbers[:25]])
        probs = probs / probs.sum()  # Normalizar
        
        used_combinations = set()
        attempts = 0
        max_attempts = n_sequences * 10
        
        while len(sequences) < n_sequences and attempts < max_attempts:
            attempts += 1
            
            # Sampling sin reemplazo
            selected_idx = np.random.choice(
                len(numbers), 
                size=nums_count, 
                replace=False, 
                p=probs
            )
            selected = sorted([numbers[i] for i in selected_idx])
            combo_key = tuple(selected)
            
            if combo_key not in used_combinations:
                used_combinations.add(combo_key)
                
                sb = None
                if top_sb and len(sequences) < len(top_sb):
                    sb_probs = np.array([p for _, p in top_sb], dtype=float)
                    sb_probs /= sb_probs.sum()
                    sb_idx = np.random.choice(len(top_sb), p=sb_probs)
                    sb, _ = top_sb[sb_idx]
                
                sequences.append({
                    'numbers': selected,
                    'superbalota': sb,
                    'strategy': 'stochastic',
                    'generation_method': 'Sampling probabilístico'
                })
        
        return sequences
    
    def _calculate_sequence_score(self, numbers):
        """
        Calcula múltiples scores para una secuencia.
        
        Returns:
            Dict con diferentes componentes de score
        """
        scores = {}
        
        # 1. ML Probability Score - promedio de probabilidades del ensemble para estos números
        if self._current_number_probs:
            seq_probs = [self._current_number_probs.get(n, 0.0) for n in numbers]
            raw_avg = float(np.mean(seq_probs)) if seq_probs else 0.0
            all_probs = list(self._current_number_probs.values())
            global_mean = float(np.mean(all_probs)) if all_probs else 0.01
            # Una secuencia promedio → ~0.5, una con los mejores números → ~1.0
            scores['ml_prob'] = min(raw_avg / (2 * global_mean), 1.0)
        else:
            scores['ml_prob'] = 0.5
        
        # 2. Historical Pattern Score
        scores['pattern_match'] = self._pattern_similarity_score(numbers)
        
        # 3. Distribution Score
        scores['distribution'] = self._distribution_score(numbers)
        
        # 4. Diversity Score
        scores['diversity'] = self._diversity_score(numbers)
        
        # 5. Hot/Cold Balance Score
        scores['hot_cold_balance'] = self._hot_cold_score(numbers)
        
        # Score total (ponderado)
        weights = {
            'ml_prob': 0.3,
            'pattern_match': 0.25,
            'distribution': 0.2,
            'diversity': 0.15,
            'hot_cold_balance': 0.1
        }
        
        scores['total'] = sum(scores[key] * weights[key] for key in weights.keys())
        
        return scores
    
    def _pattern_similarity_score(self, numbers):
        """Score basado en similitud con patrones históricos."""
        score = 0.0
        
        # 1. Pares comunes
        pairs = list(combinations(numbers, 2))
        common_pairs = sum(1 for pair in pairs if self.historical_patterns['pair_frequencies'][pair] > 0)
        score += (common_pairs / len(pairs)) * 0.5 if pairs else 0
        
        # 2. Suma similar a histórica
        num_sum = sum(numbers)
        hist_sums = self.historical_patterns['sum_distribution']
        if hist_sums:
            mean_sum = np.mean(hist_sums)
            std_sum = np.std(hist_sums)
            if std_sum > 0:
                z_score = abs((num_sum - mean_sum) / std_sum)
                # Score alto si está dentro de 1 desviación estándar
                score += max(0, 1 - z_score / 2) * 0.5
        
        return min(score, 1.0)
    
    def _distribution_score(self, numbers):
        """Score basado en distribución balanceada."""
        score = 0.0
        
        # 1. Balance par/impar (ideal: 2-3 o 3-2)
        even_count = sum(1 for n in numbers if n % 2 == 0)
        if even_count in [2, 3]:
            score += 0.3
        
        # 2. Distribución por rangos
        range_max = self.config['number_range'][1]
        range_width = range_max // 4
        range_counts = [0] * 4
        for n in numbers:
            range_idx = min((n - 1) // range_width, 3)
            range_counts[range_idx] += 1
        
        # Penalizar si todos están en 1-2 rangos
        non_zero_ranges = sum(1 for c in range_counts if c > 0)
        score += (non_zero_ranges / 4) * 0.4
        
        # 3. Gaps razonables
        if len(numbers) > 1:
            gaps = [numbers[i+1] - numbers[i] for i in range(len(numbers) - 1)]
            avg_gap = np.mean(gaps)
            # Ideal: gaps entre 5-10
            if 5 <= avg_gap <= 10:
                score += 0.3
        
        return min(score, 1.0)
    
    def _diversity_score(self, numbers):
        """Score basado en diversidad de números."""
        if len(numbers) < 2:
            return 0.0
        
        # Calcular distancias mínimas entre números
        min_dists = []
        for i, n in enumerate(numbers):
            other_nums = numbers[:i] + numbers[i+1:]
            if other_nums:
                min_dist = min(abs(n - o) for o in other_nums)
                min_dists.append(min_dist)
        
        avg_min_dist = np.mean(min_dists) if min_dists else 0
        # Normalizar (ideal: distancia mín promedio > 5)
        score = min(avg_min_dist / 5.0, 1.0)
        
        return score
    
    def _hot_cold_score(self, numbers):
        """Score basado en balance de números calientes y fríos."""
        # Analizar últimos 30 sorteos
        lookback = min(30, len(self.data_df))
        recent_nums = []
        for i in range(len(self.data_df) - lookback, len(self.data_df)):
            recent_nums.extend(self.data_df.iloc[i]['Numeros'])
        
        freq_counter = Counter(recent_nums)

        if not freq_counter:
            return 0.4

        # Clasificar números de la secuencia
        freq_values = list(freq_counter.values())
        hot_threshold = np.percentile(freq_values, 70)
        cold_threshold = np.percentile(freq_values, 30)
        hot_count = sum(1 for n in numbers if freq_counter.get(n, 0) >= hot_threshold)
        cold_count = sum(1 for n in numbers if freq_counter.get(n, 0) <= cold_threshold)
        
        # Balance ideal: 2-3 calientes, 1-2 fríos
        if 2 <= hot_count <= 3 and 1 <= cold_count <= 2:
            return 1.0
        elif 1 <= hot_count <= 4:
            return 0.7
        else:
            return 0.4
    
    def explain_sequence(self, sequence):
        """
        Genera explicación textual de por qué se seleccionó una secuencia.
        
        Returns:
            str: Explicación legible
        """
        nums = sequence['numbers']
        scores = sequence['scores']
        
        explanations = []
        
        # Estrategia usada
        explanations.append(f"Método: {sequence['generation_method']}")
        
        # Score de patrones
        if scores['pattern_match'] > 0.7:
            explanations.append("Alta similitud con patrones históricos ganadores")
        elif scores['pattern_match'] > 0.5:
            explanations.append("Patrones moderadamente similares a históricos")
        
        # Distribución
        if scores['distribution'] > 0.7:
            explanations.append("Distribución óptima (par/impar, rangos, gaps)")
        
        # Balance caliente/frío
        if scores['hot_cold_balance'] > 0.8:
            explanations.append("Excelente balance entre números calientes y fríos")
        
        # Información adicional
        even_count = sum(1 for n in nums if n % 2 == 0)
        explanations.append(f"Ratio par/impar: {even_count}/{len(nums) - even_count}")
        
        num_sum = sum(nums)
        explanations.append(f"Suma total: {num_sum}")
        
        return " | ".join(explanations)
