from collections import Counter, defaultdict
import numpy as np
from datetime import datetime

class StatisticsAnalyzer:
    """Realiza análisis estadísticos avanzados sobre los datos."""
    
    def analyze_frequency(self, df_processed, config, limit=None):
        """Devuelve números más frecuentes (calientes) y menos frecuentes (fríos)."""
        if df_processed.empty:
            return {'frequency': {}, 'hot_numbers': [], 'cold_numbers': [], 'total_draws': 0}
        
        data = df_processed[-limit:] if limit else df_processed
        all_nums = [n for idx, row in data.iterrows() for n in row['Numeros']]
        
        freq = Counter(all_nums)
        total_draws = len(data)
        
        stats = {
            'frequency': freq,
            'hot_numbers': freq.most_common(10),
            'cold_numbers': freq.most_common()[:-11:-1], # Los últimos 10 invertidos
            'total_draws': total_draws
        }
        return stats

    def analyze_delays(self, df_processed, config):
        """Calcula hace cuántos sorteos no sale cada número."""
        if df_processed.empty: return []
        
        last_appearance = {}
        total_draws = len(df_processed)
        
        # Inicializar todos con total_draws (nunca salieron)
        start = config['number_range'][0]
        end = config['number_range'][1]
        
        for n in range(start, end + 1):
            last_appearance[n] = -1
            
        # Recorrer de atrás hacia adelante es más rápido para encontrar la última vez
        # pero iterar normal y actualizar es O(N) también
        for idx, row in df_processed.iterrows():
            for n in row['Numeros']:
                last_appearance[n] = idx
                
        # Calcular retardo (sorteos desde la última vez)
        delays = []
        current_idx = total_draws - 1
        for n in range(start, end + 1):
            last = last_appearance[n]
            delay = (current_idx - last) if last != -1 else total_draws
            delays.append((n, delay))
            
        return sorted(delays, key=lambda x: x[1], reverse=True)
        
    def analyze_superbalota(self, df_processed, config):
        """Analiza la superbalota específicamente."""
        if not config['has_superbalota'] or df_processed.empty:
            return None
            
        sbs = [row['Superbalota'] for _, row in df_processed.iterrows() if row['Superbalota']]
        if not sbs: return None
        
        freq = Counter(sbs)
        return {
            'common': freq.most_common(5),
            'rare': freq.most_common()[:-6:-1]
        }
