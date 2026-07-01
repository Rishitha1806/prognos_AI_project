import numpy as np
import pandas as pd
from enum import Enum

class AlertLevel(Enum):
    """Alert severity levels"""
    HEALTHY = 0
    WARNING = 1
    CRITICAL = 2
    FAILURE = 3

class AlertSystem:
    """Maintenance alert system"""
    
    def __init__(self, healthy=250, warning=100, critical=30, failure=0):
        """Define thresholds"""
        self.healthy_th = healthy
        self.warning_th = warning
        self.critical_th = critical
        self.failure_th = failure
        
        print("Alert Thresholds:")
        print(f"  HEALTHY:   RUL > {healthy}")
        print(f"  WARNING:   {warning} < RUL <= {healthy}")
        print(f"  CRITICAL:  {critical} < RUL <= {warning}")
        print(f"  FAILURE:   RUL <= {critical}")
    
    def classify(self, rul):
        """Classify RUL into alert level"""
        if rul > self.healthy_th:
            return AlertLevel.HEALTHY
        elif rul > self.warning_th:
            return AlertLevel.WARNING
        elif rul > self.critical_th:
            return AlertLevel.CRITICAL
        else:
            return AlertLevel.FAILURE
    
    def get_action(self, alert_level, rul):
        """Get recommended action"""
        actions = {
            AlertLevel.HEALTHY: "Continue normal operation",
            AlertLevel.WARNING: "Monitor closely. Schedule maintenance in 2 weeks",
            AlertLevel.CRITICAL: "Schedule maintenance immediately",
            AlertLevel.FAILURE: "STOP OPERATION. Emergency maintenance required"
        }
        return actions[alert_level]
    
    def generate_alerts(self, unit_ids, rul_predictions):
        """Generate alerts for all units"""
        alerts = []
        
        for unit_id, rul in zip(unit_ids, rul_predictions):
            alert_level = self.classify(rul)
            action = self.get_action(alert_level, rul)
            
            alerts.append({
                'unit_id': unit_id,
                'rul_predicted': rul,
                'alert_level': alert_level.name,
                'recommendation': action
            })
        
        return pd.DataFrame(alerts)
    
    def print_report(self, alerts_df):
        """Print alert summary"""
        print("\n" + "="*70)
        print("MAINTENANCE ALERT REPORT")
        print("="*70)
        
        for level in ['HEALTHY', 'WARNING', 'CRITICAL', 'FAILURE']:
            count = len(alerts_df[alerts_df['alert_level'] == level])
            print(f"{level:10s}: {count:3d} units")
        
        print("\n" + "-"*70)
        critical = alerts_df[alerts_df['alert_level'].isin(['CRITICAL', 'FAILURE'])]
        if len(critical) > 0:
            print("CRITICAL UNITS REQUIRING ACTION:")
            for _, row in critical.iterrows():
                print(f"  Unit {row['unit_id']:3d}: RUL={row['rul_predicted']:6.1f} - {row['recommendation']}")
