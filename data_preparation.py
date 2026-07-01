import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class CMAPSSDataPreparation:
    """Data preparation for CMAPSS dataset"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.train_data = None
        self.test_data = None
        self.scaler = MinMaxScaler()
        
    def log(self, msg):
        if self.verbose:
            print(f"[INFO] {msg}")
    
    def load_data(self, train_file, test_file, rul_file):
        """Load CMAPSS dataset"""
        self.log("Loading CMAPSS data...")
        
        # Define column names
        cols = ['unit_id', 'cycle'] + [f'setting_{i}' for i in range(1, 4)] + \
               [f'sensor_{i}' for i in range(1, 22)]
        
        # Load files
        self.train_data = pd.read_csv(train_file, sep=r'\s+', header=None, names=cols)
        self.test_data = pd.read_csv(test_file, sep=r'\s+', header=None, names=cols)
        self.test_rul = pd.read_csv(rul_file, header=None, names=['rul']).values.flatten()
        
        self.log(f"Train shape: {self.train_data.shape}, Test shape: {self.test_data.shape}")
        return self.train_data, self.test_data, self.test_rul
    
    def compute_rul(self, data):
        """Compute remaining useful life for each sample"""
        self.log("Computing RUL...")
        max_cycles = data.groupby('unit_id')['cycle'].max()
        data['max_cycle'] = data['unit_id'].map(max_cycles)
        data['rul'] = data['max_cycle'] - data['cycle']
        data['rul'] = data['rul'].clip(upper=125)
        self.log(f"RUL range: {data['rul'].min()} to {data['rul'].max()}")
        return data

    def compute_test_rul(self, test_data, test_rul):
        """Compute RUL for test data using the provided RUL file"""
        self.log("Computing RUL for test data...")
        max_cycles = test_data.groupby('unit_id')['cycle'].max()
        test_data['max_cycle'] = test_data['unit_id'].map(max_cycles)
        
        # map the final RUL from test_rul array (unit_id starts from 1)
        test_data['final_rul'] = test_data['unit_id'].apply(lambda x: test_rul[x - 1])
        
        # RUL at any cycle = final_RUL + (max_cycle - current_cycle)
        test_data['rul'] = test_data['final_rul'] + test_data['max_cycle'] - test_data['cycle']
        test_data['rul'] = test_data['rul'].clip(upper=125)
        return test_data
    
    def normalize_data(self, data, fit_scaler=True):
        """Normalize sensor data to [0,1]"""
        self.log("Normalizing sensors...")
        sensor_cols = [col for col in data.columns if 'sensor' in col]
        
        if fit_scaler:
            data[sensor_cols] = self.scaler.fit_transform(data[sensor_cols])
        else:
            data[sensor_cols] = self.scaler.transform(data[sensor_cols])
        
        return data
    
    def create_sequences(self, data, seq_length=30):
        """Create rolling window sequences"""
        self.log(f"Creating sequences (length={seq_length})...")
        
        feature_cols = [col for col in data.columns if 'sensor' in col or 'setting' in col]
        X, y, unit_ids, cycles = [], [], [], []
        
        for unit_id in data['unit_id'].unique():
            unit_data = data[data['unit_id'] == unit_id].reset_index(drop=True)
            features = unit_data[feature_cols].values
            ruls = unit_data['rul'].values
            cycle_vals = unit_data['cycle'].values
            
            for i in range(len(unit_data) - seq_length):
                X.append(features[i:i+seq_length])
                y.append(ruls[i+seq_length])
                unit_ids.append(unit_id)
                cycles.append(cycle_vals[i+seq_length])
        
        return np.array(X), np.array(y), np.array(unit_ids), np.array(cycles)
    
    def prepare(self, train_file, test_file, rul_file, seq_length=30):
        """Complete pipeline"""
        # Load
        self.load_data(train_file, test_file, rul_file)
        
        # Compute RUL
        self.train_data = self.compute_rul(self.train_data)
        self.test_data = self.compute_test_rul(self.test_data, self.test_rul)
        
        # Normalize (fit on train only!)
        self.train_data = self.normalize_data(self.train_data, fit_scaler=True)
        self.test_data = self.normalize_data(self.test_data, fit_scaler=False)
        
        # Create sequences
        X_train, y_train, train_ids, train_cycles = self.create_sequences(self.train_data, seq_length)
        X_test, y_test, test_ids, test_cycles = self.create_sequences(self.test_data, seq_length)
        
        self.log(f"[DONE] Complete! X_train: {X_train.shape}, y_train: {y_train.shape}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'train_ids': train_ids, 'test_ids': test_ids,
            'test_cycles': test_cycles,
            'scaler': self.scaler
        }
