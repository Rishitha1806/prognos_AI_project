import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

class LSTMModelBuilder:
    """Build and train LSTM models for RUL prediction"""
    
    def __init__(self):
        self.model = None
        self.history = None
    
    def build_simple_lstm(self, seq_length=30, num_features=24, units=50):
        """Simple LSTM: Input → LSTM → Dense"""
        model = Sequential([
            LSTM(units, activation='relu', input_shape=(seq_length, num_features)),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.0003, clipnorm=1.0), loss='mse', metrics=['mae'])
        return model
    
    def build_stacked_lstm(self, seq_length=30, num_features=24, units=[50, 50]):
        """Stacked LSTM: Input → LSTM → LSTM → Dense"""
        model = Sequential([
            LSTM(units[0], activation='relu', return_sequences=True,
                 input_shape=(seq_length, num_features)),
            Dropout(0.2),
            LSTM(units[1], activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.0003, clipnorm=1.0), loss='mse', metrics=['mae'])
        return model
    
    def train(self, model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train model with early stopping"""
        print(f"Training on {X_train.shape[0]} samples...")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        self.history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.model = model
        print(f"[DONE] Training complete!")
        return model, self.history
    
    def plot_history(self):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid()
        
        axes[1].plot(self.history.history['mae'], label='Train MAE')
        axes[1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[1].set_ylabel('MAE (cycles)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_title('Model MAE')
        axes[1].legend()
        axes[1].grid()
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save trained model"""
        self.model.save(filepath)
        print(f"[DONE] Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath, compile=False)
        print(f"[DONE] Model loaded from {filepath}")
        return self.model
