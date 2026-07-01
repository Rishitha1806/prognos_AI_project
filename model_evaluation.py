import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class RULEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self):
        self.metrics = {}
    
    def compute_rmse(self, y_true, y_pred):
        """Root Mean Squared Error"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        self.metrics['rmse'] = rmse
        return rmse
    
    def compute_mae(self, y_true, y_pred):
        """Mean Absolute Error"""
        mae = mean_absolute_error(y_true, y_pred)
        self.metrics['mae'] = mae
        return mae
    
    def compute_r2(self, y_true, y_pred):
        """R² Score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        self.metrics['r2'] = r2
        return r2
    
    def analyze(self, y_true, y_pred):
        """Comprehensive analysis"""
        print("\n" + "="*70)
        print("MODEL EVALUATION RESULTS")
        print("="*70)
        
        rmse = self.compute_rmse(y_true, y_pred)
        mae = self.compute_mae(y_true, y_pred)
        r2 = self.compute_r2(y_true, y_pred)
        
        print(f"\nMetrics:")
        print(f"  RMSE: {rmse:.2f} cycles")
        print(f"  MAE:  {mae:.2f} cycles")
        print(f"  R²:   {r2:.4f}")
        
        # Error analysis
        errors = y_pred - y_true
        print(f"\nError Statistics:")
        print(f"  Mean Error: {np.mean(errors):.2f}")
        print(f"  Std Error:  {np.std(errors):.2f}")
        print(f"  Min Error:  {np.min(errors):.2f}")
        print(f"  Max Error:  {np.max(errors):.2f}")
        
        return {
            'rmse': rmse, 'mae': mae, 'r2': r2,
            'errors': errors
        }
    
    def plot_results(self, y_true, y_pred):
        """Visualize predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Predicted vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[0, 0].set_xlabel('Actual RUL')
        axes[0, 0].set_ylabel('Predicted RUL')
        axes[0, 0].set_title('Predicted vs Actual')
        axes[0, 0].grid()
        
        # Error distribution
        errors = y_pred - y_true
        axes[0, 1].hist(errors, bins=50, edgecolor='black')
        axes[0, 1].axvline(0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Error (cycles)')
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].grid()
        
        # Error by range
        ranges = [(0, 50), (50, 150), (150, 300)]
        range_errors = []
        for low, high in ranges:
            mask = (y_true >= low) & (y_true < high)
            if np.sum(mask) > 0:
                range_errors.append(np.mean(np.abs(errors[mask])))
            else:
                range_errors.append(0)
        
        axes[1, 0].bar([f'{l}-{h}' for l, h in ranges], range_errors)
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('Error by RUL Range')
        axes[1, 0].grid()
        
        # Residuals
        axes[1, 1].scatter(y_pred, errors, alpha=0.5)
        axes[1, 1].axhline(0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Predicted RUL')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residual Plot')
        axes[1, 1].grid()
        
        plt.tight_layout()
        plt.show()
