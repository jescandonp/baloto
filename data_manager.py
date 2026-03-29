import pandas as pd
import numpy as np
import warnings
from datetime import datetime

class DataManager:
    """Maneja la carga, limpieza y validación de datos de lotería."""
    
    def __init__(self):
        self.game_configs = {
            'baloto': {
                'number_range': (1, 43),
                'numbers_count': 5,
                'superbalota_range': (1, 16),
                'has_superbalota': True,
                'csv_cols': ['Fecha', 'Números Baloto', 'Superbalota Baloto']
            },
            'revancha': {
                'number_range': (1, 43),
                'numbers_count': 5,
                'superbalota_range': (1, 16),
                'has_superbalota': True,
                'csv_cols': ['Fecha', 'Números Revancha', 'Superbalota Revancha']
            },
            'miloto': {
                'number_range': (1, 39),
                'numbers_count': 5,
                'has_superbalota': False,
                'csv_cols': ['Fecha', 'Números MiLoto']
            }
        }
        self.data = {}

    def load_data(self, baloto_file, miloto_file, colorloto_file=None):
        """Carga los datos desde los archivos CSV."""
        try:
            # Cargar Baloto/Revancha (mismo archivo generalmente)
            df_baloto = self._safe_load_csv(baloto_file)
            self.data['baloto'] = self._process_game_data(df_baloto, 'baloto')
            self.data['revancha'] = self._process_game_data(df_baloto, 'revancha')
            
            # Cargar MiLoto
            df_miloto = self._safe_load_csv(miloto_file)
            self.data['miloto'] = self._process_game_data(df_miloto, 'miloto')
            
            print("✅ Datos cargados exitosamente:")
            for game, df in self.data.items():
                print(f"   - {game.capitalize()}: {len(df)} sorteos")
                
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            # No raise here to allow empty init in tests if needed, but in prod we might want to raise
    
    def _safe_load_csv(self, filepath):
        """Intenta cargar CSV con diferentes encodings."""
        try:
            return pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            return pd.read_csv(filepath, encoding='latin-1')

    def _process_game_data(self, df, game_type):
        """Procesa y limpia los datos específicos de un juego."""
        config = self.game_configs[game_type]
        processed_rows = []
        
        # Identificar columnas
        cols = config['csv_cols']
        # Buscar columnas en el df que coincidan parcialmente si no son exactas
        actual_cols = {}
        for target in cols:
            # 1. Exact match (case-insensitive)
            match_exact = next((c for c in df.columns if target.lower() == c.lower()), None)
            if match_exact:
                actual_cols[target] = match_exact
                continue
                
            # 2. Partial match (fallback)
            match = next((c for c in df.columns if target.lower() in c.lower()), None)
            if match:
                actual_cols[target] = match
        
        if len(actual_cols) < len(cols):
             # Si faltan columnas, retornar vacío o intentar lo que hay
             # Para simplificar, si falta la de números, devolvemos vacío
             if config['csv_cols'][1] not in actual_cols:
                 return pd.DataFrame()

        for _, row in df.iterrows():
            try:
                date_str = str(row[actual_cols[cols[0]]])
                nums_str = str(row[actual_cols[cols[1]]])
                
                # Parsear fecha
                date = pd.to_datetime(date_str, errors='coerce')
                if pd.isna(date):
                    continue
                    
                # Parsear números
                numbers = self._parse_numbers(nums_str, config)
                if not numbers:
                    continue
                
                entry = {
                    'Fecha': date,
                    'Numeros': numbers,
                    'Dia_Semana': date.day_name()
                }
                
                # Superbalota
                if config['has_superbalota'] and cols[2] in actual_cols:
                    sb_val = row[actual_cols[cols[2]]]
                    entry['Superbalota'] = self._parse_superbalota(sb_val, config)
                else:
                    entry['Superbalota'] = None
                    
                processed_rows.append(entry)
                
            except (ValueError, KeyError, TypeError):
                continue
                
        # Crear DataFrame y ordenar
        if not processed_rows:
            return pd.DataFrame()
            
        df_proc = pd.DataFrame(processed_rows)
        df_proc = df_proc.sort_values('Fecha').reset_index(drop=True)
        return df_proc

    def _parse_numbers(self, numbers_str, config):
        """Parsea string de números a lista de enteros."""
        if pd.isna(numbers_str): return None
        
        clean_str = str(numbers_str).replace(' ', '')
        separators = ['-', ',', ';', '/']
        parts = []
        
        for sep in separators:
            if sep in clean_str:
                parts = clean_str.split(sep)
                break
        else:
            # Asumir pares de 2 w/o separator si falla todo (ej: "010203")
            parts = [clean_str[i:i+2] for i in range(0, len(clean_str), 2)]
            
        try:
            nums = [int(p.strip()) for p in parts if p.strip().isdigit()]
            nums = [n for n in nums if config['number_range'][0] <= n <= config['number_range'][1]]

            if len(nums) == config['numbers_count']:
                return sorted(nums)
        except (ValueError, TypeError):
            pass
        return None

    def _parse_superbalota(self, val, config):
        """Parsea valor de superbalota."""
        try:
            n = int(float(str(val).strip()))
            if config['superbalota_range'][0] <= n <= config['superbalota_range'][1]:
                return n
        except:
            pass
        return None
