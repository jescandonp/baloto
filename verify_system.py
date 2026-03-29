import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import shutil
from baloto_system.data_manager import DataManager
from baloto_system.feature_engineering import FeatureEngineer
from baloto_system.models import Models

def generate_dummy_data():
    """Genera archivos CSV de prueba."""
    print("Generating dummy data...")
    
    # Baloto (Fecha, Números Baloto, Superbalota Baloto, Números Revancha, Superbalota Revancha)
    dates = [datetime(2023, 1, 1) + timedelta(days=i*3) for i in range(100)]
    
    baloto_data = []
    for d in dates:
        nums_b = np.random.choice(range(1, 44), 5, replace=False)
        sb_b = np.random.randint(1, 17)
        nums_r = np.random.choice(range(1, 44), 5, replace=False)
        sb_r = np.random.randint(1, 17)
        
        baloto_data.append({
            'Fecha': d.strftime('%Y-%m-%d'),
            'Números Baloto': '-'.join(map(str, sorted(nums_b))),
            'Superbalota Baloto': sb_b,
            'Números Revancha': '-'.join(map(str, sorted(nums_r))),
            'Superbalota Revancha': sb_r
        })
    pd.DataFrame(baloto_data).to_csv('dummy_baloto.csv', index=False)
    
    # MiLoto
    miloto_data = []
    for d in dates:
        nums_m = np.random.choice(range(1, 40), 5, replace=False)
        miloto_data.append({
            'Fecha': d.strftime('%Y-%m-%d'),
            'Números MiLoto': '-'.join(map(str, sorted(nums_m)))
        })
    pd.DataFrame(miloto_data).to_csv('dummy_miloto.csv', index=False)
    
    print("Dummy CSVs created.")
    return 'dummy_baloto.csv', 'dummy_miloto.csv'

def cleanup_dummy_data():
    if os.path.exists('dummy_baloto.csv'): os.remove('dummy_baloto.csv')
    if os.path.exists('dummy_miloto.csv'): os.remove('dummy_miloto.csv')

def test_pipeline():
    try:
        b_file, m_file = generate_dummy_data()
        
        print("\n--- Testing DataManager ---")
        dm = DataManager()
        dm.load_data(b_file, m_file)
        
        assert 'baloto' in dm.data
        assert 'revancha' in dm.data
        assert len(dm.data['baloto']) == 100
        print("✅ DataManager OK")
        
        print("\n--- Testing FeatureEngineer ---")
        fe = FeatureEngineer()
        config = dm.game_configs['baloto']
        X, y = fe.prepare_training_data(dm.data['baloto'], config)
        
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        assert X.shape[0] == 90 # 100 - window(10)
        assert y.shape[1] == 43 # 43 possible numbers (multi-output)
        print("✅ FeatureEngineer OK")
        
        print("\n--- Testing Models ---")
        model = Models('baloto')
        success = model.train(X, y)
        assert success
        
        # Test Prediction
        X_pred = fe.prepare_prediction_input(dm.data['baloto'], config)
        top = model.get_top_numbers(X_pred)
        print(f"Top prediction: {top}")
        assert len(top) == 5
        print("✅ Models OK")
        
        print("\n🎉 ALL TESTS PASSED")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_dummy_data()

if __name__ == "__main__":
    test_pipeline()
