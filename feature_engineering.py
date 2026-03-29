import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations

class FeatureEngineer:
    """Genera características (features) avanzadas para el entrenamiento de modelos ML."""
    
    def prepare_training_data(self, data_df, config, window_sizes=[5, 10, 20]):
        """
        Prepara X (features) y y (targets) para entrenamiento con features avanzadas.
        
        Args:
            data_df: DataFrame con datos históricos
            config: Configuración del juego
            window_sizes: Lista de tamaños de ventana para análisis temporal
        
        Returns:
            X: Features array
            y: Target array (binary vector para cada número)
            y_sb: Superbalota targets (si aplica)
        """
        max_window = max(window_sizes)
        if len(data_df) < max_window + 1:
            return None, None, None
            
        X = []
        y = []
        y_sb = []
        
        # Pre-calcular rango total
        total_nums = config['number_range'][1] - config['number_range'][0] + 1
        min_num = config['number_range'][0]
        max_num = config['number_range'][1]
        
        for i in range(max_window, len(data_df)):
            features = []
            
            # ===== FEATURES MULTI-VENTANA =====
            for window_size in window_sizes:
                window_features = self._extract_window_features(
                    data_df, i, window_size, config, total_nums, min_num, max_num
                )
                features.extend(window_features)
            
            # ===== FEATURES TEMPORALES AVANZADAS =====
            temporal_features = self._extract_temporal_features(
                data_df, i, config, total_nums, min_num
            )
            features.extend(temporal_features)
            
            # ===== FEATURES DE PATRONES =====
            pattern_features = self._extract_pattern_features(
                data_df, i, max_window, config, min_num
            )
            features.extend(pattern_features)
            
            # ===== FEATURES DE DISTRIBUCIÓN =====
            distribution_features = self._extract_distribution_features(
                data_df, i, max_window, config, min_num, max_num
            )
            features.extend(distribution_features)
            
            X.append(features)
            
            # --- TARGET GENERATION ---
            target_nums = data_df.iloc[i]['Numeros']
            target_vector = [0] * total_nums
            for n in target_nums:
                idx = n - min_num
                if 0 <= idx < total_nums:
                    target_vector[idx] = 1
            y.append(target_vector)
            
            # Superbalota target
            if config.get('has_superbalota') and 'Superbalota' in data_df.columns:
                sb = data_df.iloc[i]['Superbalota']
                y_sb.append(int(sb) if pd.notna(sb) else 0)
            
        y_sb_res = np.array(y_sb) if y_sb else None
        return np.array(X), np.array(y), y_sb_res

    def _extract_window_features(self, data_df, idx, window_size, config, total_nums, min_num, max_num):
        """Extrae features basadas en una ventana temporal específica."""
        features = []
        
        # 1. Historia cruda (números normalizados)
        for j in range(idx - window_size, idx):
            draw_nums = data_df.iloc[j]['Numeros']
            features.extend([(n - min_num) / total_nums for n in draw_nums])
        
        # 2. Frecuencias en la ventana
        recent_flat = [n for k in range(idx-window_size, idx) for n in data_df.iloc[k]['Numeros']]
        freqs = Counter(recent_flat)
        
        # Top 5 más frecuentes
        top_common = [n for n, _ in freqs.most_common(5)]
        while len(top_common) < 5: 
            top_common.append(0)
        features.extend([(n - min_num)/total_nums if n > 0 else 0 for n in top_common])
        
        # 3. Estadísticas agregadas
        if recent_flat:
            features.append(np.mean(recent_flat) / total_nums)
            features.append(np.std(recent_flat) / total_nums)
            features.append(np.median(recent_flat) / total_nums)
            features.append(np.min(recent_flat) / total_nums)
            features.append(np.max(recent_flat) / total_nums)
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # 4. Heat map (frecuencia normalizada por número)
        heat_vector = [0] * total_nums
        for n in recent_flat:
            idx_heat = n - min_num
            if 0 <= idx_heat < total_nums:
                heat_vector[idx_heat] += 1
        # Normalizar por el tamaño de ventana
        features.extend([h / window_size for h in heat_vector])
        
        # 5. Retardos (sorteos desde última aparición)
        delays = self._calculate_delays(data_df, idx, window_size, config, total_nums, min_num)
        features.extend(delays)
        
        return features
    
    def _calculate_delays(self, data_df, idx, window_size, config, total_nums, min_num):
        """Calcula el retardo (sorteos desde última aparición) para cada número."""
        delays = [window_size] * total_nums  # Default: no apareció en ventana
        
        for j in range(idx - 1, max(0, idx - window_size), -1):
            draw_nums = data_df.iloc[j]['Numeros']
            delay_val = idx - j
            for n in draw_nums:
                idx_delay = n - min_num
                if 0 <= idx_delay < total_nums and delays[idx_delay] == window_size:
                    delays[idx_delay] = delay_val
        
        # Normalizar delays
        return [d / window_size for d in delays]
    
    def _extract_temporal_features(self, data_df, idx, config, total_nums, min_num):
        """Extrae features de tendencias temporales."""
        features = []
        
        # 1. Tendencia de frecuencias (últimos 5, 10, 20 sorteos)
        for lookback in [5, 10, 20]:
            if idx >= lookback:
                recent_nums = [n for k in range(idx-lookback, idx) for n in data_df.iloc[k]['Numeros']]
                freq_trend = len(set(recent_nums)) / total_nums  # Diversidad
                features.append(freq_trend)
            else:
                features.append(0)
        
        # 2. Números "calientes" vs "fríos" (últimos 30 sorteos)
        lookback_hot = min(30, idx)
        if lookback_hot > 0:
            all_recent = [n for k in range(idx-lookback_hot, idx) for n in data_df.iloc[k]['Numeros']]
            freq_count = Counter(all_recent)
            
            # Top 5 calientes
            hot_nums = [n for n, _ in freq_count.most_common(5)]
            while len(hot_nums) < 5:
                hot_nums.append(0)
            features.extend([(n - min_num) / total_nums if n > 0 else 0 for n in hot_nums])
            
            # Bottom 5 fríos (números que menos aparecieron)
            all_possible = set(range(config['number_range'][0], config['number_range'][1] + 1))
            appeared = set(all_recent)
            cold_candidates = all_possible - appeared
            cold_nums = list(cold_candidates)[:5]
            while len(cold_nums) < 5:
                cold_nums.append(0)
            features.extend([(n - min_num) / total_nums if n > 0 else 0 for n in cold_nums])
        else:
            features.extend([0] * 10)
        
        # 3. Día de la semana
        # Si idx está fuera de rango (predicción), usar el último día + estimación
        if idx >= len(data_df):
            last_date = data_df.iloc[-1]['Fecha']
            # Estimar siguiente día (asumiendo sorteo cada 3-4 días)
            features.append(last_date.dayofweek / 6.0)
        else:
            current_date = data_df.iloc[idx]['Fecha']
            features.append(current_date.dayofweek / 6.0)
        
        # 4. Mes del año (ciclicidad)
        if idx >= len(data_df):
            last_date = data_df.iloc[-1]['Fecha']
            features.append(last_date.month / 12.0)
        else:
            current_date = data_df.iloc[idx]['Fecha']
            features.append(current_date.month / 12.0)
        
        return features
    
    def _extract_pattern_features(self, data_df, idx, window_size, config, min_num):
        """Extrae features de patrones y secuencias."""
        features = []
        
        # 1. Análisis de pares consecutivos
        pairs_count = 0
        for j in range(max(0, idx - window_size), idx):
            nums = sorted(data_df.iloc[j]['Numeros'])
            for i_pair in range(len(nums) - 1):
                if nums[i_pair + 1] - nums[i_pair] == 1:
                    pairs_count += 1
        features.append(pairs_count / window_size if window_size > 0 else 0)
        
        # 2. Análisis de gaps (distancia promedio entre números)
        avg_gap = 0
        gap_std = 0
        if idx > 0:
            last_nums = sorted(data_df.iloc[idx - 1]['Numeros'])
            if len(last_nums) > 1:
                gaps = [last_nums[i+1] - last_nums[i] for i in range(len(last_nums) - 1)]
                avg_gap = np.mean(gaps) / config['number_range'][1]
                gap_std = np.std(gaps) / config['number_range'][1]
        features.extend([avg_gap, gap_std])
        
        # 3. Frecuencia de repeticiones inmediatas
        repeat_count = 0
        if idx >= 2:
            prev_nums = set(data_df.iloc[idx - 1]['Numeros'])
            prev_prev_nums = set(data_df.iloc[idx - 2]['Numeros'])
            repeat_count = len(prev_nums & prev_prev_nums) / config['numbers_count']
        features.append(repeat_count)
        
        # 4. Análisis de posiciones (números en posiciones específicas)
        position_features = [0] * config['numbers_count']
        if idx >= window_size:
            for pos in range(config['numbers_count']):
                pos_nums = []
                for j in range(idx - window_size, idx):
                    nums_sorted = sorted(data_df.iloc[j]['Numeros'])
                    if pos < len(nums_sorted):
                        pos_nums.append(nums_sorted[pos])
                if pos_nums:
                    position_features[pos] = np.mean(pos_nums) / config['number_range'][1]
        features.extend(position_features)
        
        return features
    
    def _extract_distribution_features(self, data_df, idx, window_size, config, min_num, max_num):
        """Extrae features de distribución estadística."""
        features = []
        
        if idx == 0:
            return [0] * 10  # Features por defecto
        
        last_nums = data_df.iloc[idx - 1]['Numeros']
        
        # 1. Ratio Par/Impar
        even_count = sum(1 for n in last_nums if n % 2 == 0)
        odd_count = len(last_nums) - even_count
        features.append(even_count / len(last_nums))
        features.append(odd_count / len(last_nums))
        
        # 2. Distribución por rangos
        range_width = (max_num - min_num + 1) // 4
        range_counts = [0] * 4
        for n in last_nums:
            range_idx = min((n - min_num) // range_width, 3)
            range_counts[range_idx] += 1
        features.extend([c / len(last_nums) for c in range_counts])
        
        # 3. Suma total de números
        total_sum = sum(last_nums)
        avg_sum = (min_num + max_num) * config['numbers_count'] / 2
        features.append(total_sum / avg_sum if avg_sum > 0 else 0)
        
        # 4. Rango (max - min)
        num_range = max(last_nums) - min(last_nums)
        max_possible_range = max_num - min_num
        features.append(num_range / max_possible_range if max_possible_range > 0 else 0)
        
        # 5. Coeficiente de variación
        if len(last_nums) > 1:
            cv = np.std(last_nums) / np.mean(last_nums) if np.mean(last_nums) > 0 else 0
            features.append(min(cv, 1.0))  # Limitar a 1
        else:
            features.append(0)
        
        # 6. Asimetría (skewness aproximada)
        if len(last_nums) > 2:
            mean_val = np.mean(last_nums)
            std_val = np.std(last_nums)
            if std_val > 0:
                skew = np.mean([((x - mean_val) / std_val) ** 3 for x in last_nums])
                features.append(np.tanh(skew))  # Normalizar a [-1, 1]
            else:
                features.append(0)
        else:
            features.append(0)
        
        return features

    def prepare_prediction_input(self, data_df, config, window_sizes=[5, 10, 20]):
        """Prepara el vector de entrada para predecir el SIGUIENTE sorteo."""
        max_window = max(window_sizes)
        if len(data_df) < max_window:
            return None
            
        idx = len(data_df)  # Índice imaginario del siguiente sorteo
        
        total_nums = config['number_range'][1] - config['number_range'][0] + 1
        min_num = config['number_range'][0]
        max_num = config['number_range'][1]
        
        features = []
        
        # Usar misma lógica que prepare_training_data
        for window_size in window_sizes:
            window_features = self._extract_window_features(
                data_df, idx, window_size, config, total_nums, min_num, max_num
            )
            features.extend(window_features)
        
        temporal_features = self._extract_temporal_features(
            data_df, idx, config, total_nums, min_num
        )
        features.extend(temporal_features)
        
        pattern_features = self._extract_pattern_features(
            data_df, idx, max_window, config, min_num
        )
        features.extend(pattern_features)
        
        distribution_features = self._extract_distribution_features(
            data_df, idx, max_window, config, min_num, max_num
        )
        features.extend(distribution_features)
        
        return np.array([features])
