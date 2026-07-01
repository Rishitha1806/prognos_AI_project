import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class Dashboard:
    """Create maintenance dashboards"""
    
    def __init__(self, unit_ids, rul_predictions, cycles):
        self.unit_ids = unit_ids
        self.cycles = cycles
        # If rul_predictions is 2D (like from LSTM output), flatten it
        if isinstance(rul_predictions, np.ndarray) and rul_predictions.ndim > 1:
            self.ruls = rul_predictions.flatten()
        else:
            self.ruls = rul_predictions
            
        self.df = pd.DataFrame({
            'unit_id': self.unit_ids,
            'cycle': self.cycles,
            'rul': self.ruls
        })
        self.latest_df = self.df.groupby('unit_id').last().reset_index()
    
    def create_fleet_dashboard(self):
        """Fleet health overview"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # 1. Health gauge
        ax1 = fig.add_subplot(gs[0, :2])
        healthy = len(self.latest_df[self.latest_df['rul'] > 100])
        health_score = 100 * healthy / len(self.latest_df)
        color = 'green' if health_score > 70 else 'yellow' if health_score > 40 else 'red'
        ax1.barh([0], [health_score], color=color, height=0.5, edgecolor='black', linewidth=2)
        ax1.set_yticks([0])
        ax1.set_yticklabels(['Fleet Health'])
        ax1.set_xlim(0, 100)
        ax1.text(health_score/2, 0, f'{health_score:.1f}%', ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')
        ax1.set_title('Fleet Health Status')
        ax1.grid()
        
        # 2. Status distribution
        ax2 = fig.add_subplot(gs[0, 2])
        healthy_c = len(self.latest_df[self.latest_df['rul'] > 100])
        warning = len(self.latest_df[(self.latest_df['rul'] > 50) & (self.latest_df['rul'] <= 100)])
        critical = len(self.latest_df[(self.latest_df['rul'] > 20) & (self.latest_df['rul'] <= 50)])
        failed = len(self.latest_df[self.latest_df['rul'] <= 20])
        
        sizes = [healthy_c, warning, critical, failed]
        colors = ['green', 'yellow', 'orange', 'red']
        
        active_sizes = [s for s in sizes if s > 0]
        active_labels = [l for s, l in zip(sizes, ['Healthy', 'Warning', 'Critical', 'Failed']) if s > 0]
        active_colors = [c for s, c in zip(sizes, colors) if s > 0]
        
        ax2.pie(active_sizes, labels=active_labels, colors=active_colors, 
               autopct='%1.0f%%', textprops={'fontsize': 10})
        ax2.set_title('Unit Status Distribution')
        
        # 3. RUL histogram
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.hist(self.latest_df['rul'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax3.axvline(100, color='green', linestyle='--', linewidth=2, label='Healthy')
        ax3.axvline(50, color='orange', linestyle='--', linewidth=2, label='Warning')
        ax3.axvline(20, color='red', linestyle='--', linewidth=2, label='Critical')
        ax3.set_xlabel('Predicted RUL (cycles)')
        ax3.set_ylabel('Number of Units')
        ax3.set_title('RUL Distribution')
        ax3.legend()
        ax3.grid()
        
        # 4. Key metrics
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis('off')
        metrics_text = f"""
        KEY METRICS
        
        Total Units: {len(self.latest_df)}
        Avg RUL: {self.latest_df['rul'].mean():.1f}
        Min RUL: {self.latest_df['rul'].min():.1f}
        Max RUL: {self.latest_df['rul'].max():.1f}
        """
        ax4.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 5. Alert zones
        ax5 = fig.add_subplot(gs[2, :])
        sorted_ruls = np.sort(self.latest_df['rul'].values)
        colors_plot = ['green' if r > 100 else 'yellow' if r > 50 else 'orange' if r > 20 else 'red' 
                       for r in sorted_ruls]
        ax5.scatter(range(len(sorted_ruls)), sorted_ruls, c=colors_plot, s=50, alpha=0.7, edgecolor='black')
        ax5.axhline(100, color='green', linestyle='--', linewidth=2)
        ax5.axhline(50, color='orange', linestyle='--', linewidth=2)
        ax5.axhline(20, color='red', linestyle='--', linewidth=2)
        ax5.set_xlabel('Equipment Unit')
        ax5.set_ylabel('Predicted RUL (cycles)')
        ax5.set_title('Fleet Alert Status Map')
        ax5.grid()
        
        plt.suptitle('PrognosAI: Fleet Health Executive Dashboard', fontsize=14, fontweight='bold')
        plt.show()
    
    def create_unit_dashboard(self, unit_id):
        """Single unit detail dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        
        unit_df = self.df[self.df['unit_id'] == unit_id].sort_values('cycle')
        unit_rul = unit_df['rul'].values
        unit_cycles = unit_df['cycle'].values
        
        if len(unit_rul) == 0:
            print(f"Unit {unit_id} not found")
            return
        
        current_rul = unit_rul[-1] if len(unit_rul) > 0 else unit_rul[0]
        
        # Status indicator
        if current_rul > 100:
            status, color = 'HEALTHY', 'green'
        elif current_rul > 50:
            status, color = 'WARNING', 'yellow'
        elif current_rul > 20:
            status, color = 'CRITICAL', 'orange'
        else:
            status, color = 'FAILURE', 'red'
        
        # 1. RUL trend
        axes[0, 0].plot(unit_cycles, unit_rul, linewidth=2, marker='o', color='steelblue')
        axes[0, 0].fill_between(unit_cycles, 0, unit_rul, alpha=0.2)
        axes[0, 0].axhline(50, color='orange', linestyle='--', alpha=0.7)
        axes[0, 0].axhline(20, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_ylabel('RUL (cycles)')
        axes[0, 0].set_title(f'Unit {unit_id}: RUL Trend')
        axes[0, 0].grid()
        
        # 2. Degradation rate
        if len(unit_rul) > 1:
            diffs = np.diff(unit_rul)
            axes[0, 1].plot(unit_cycles[1:], diffs, linewidth=2, marker='o', color='coral')
            axes[0, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
            axes[0, 1].set_ylabel('RUL Change (cycles)')
            axes[0, 1].set_title('Degradation Rate')
            axes[0, 1].grid()
        
        # 3. RUL bar
        axes[1, 0].barh(['Current RUL'], [current_rul], color=color, edgecolor='black', linewidth=2)
        axes[1, 0].set_xlim(0, 350)
        axes[1, 0].text(current_rul/2, 0, f'{current_rul:.0f}', ha='center', va='center',
                       fontsize=12, fontweight='bold')
        axes[1, 0].grid(axis='x')
        
        # 4. Status box
        axes[1, 1].axis('off')
        status_text = f"""
        UNIT {unit_id} STATUS
        
        Current RUL: {current_rul:.1f} cycles
        Status: {status}
        
        Recommendation:
        {['Continue monitoring', 'Schedule maintenance', 'Immediate action'][
            0 if current_rul > 50 else 1 if current_rul > 20 else 2]}
        """
        axes[1, 1].text(0.5, 0.5, status_text, fontsize=10, ha='center', va='center',
                       family='monospace', bbox=dict(boxstyle='round', facecolor=color, 
                       alpha=0.3, linewidth=3, edgecolor=color))
        
        plt.tight_layout()
        plt.show()
