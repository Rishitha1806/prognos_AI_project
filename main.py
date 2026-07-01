import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_preparation import CMAPSSDataPreparation
from model_development import LSTMModelBuilder
from model_evaluation import RULEvaluator
from alert_system import AlertSystem
from dashboard import Dashboard

def main():
    # ===== STEP 1: DATA PREPARATION =====
    print("="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70)
    prep = CMAPSSDataPreparation()
    
    try:
        datasets = prep.prepare('train_FD001.txt', 'test_FD001.txt', 'RUL_FD001.txt')
    except FileNotFoundError:
        print("Error: Dataset files not found. Please ensure 'train_FD001.txt', 'test_FD001.txt', and 'RUL_FD001.txt' are in the same directory.")
        return

    X_train, y_train = datasets['X_train'], datasets['y_train']
    X_test, y_test = datasets['X_test'], datasets['y_test']
    test_ids = datasets['test_ids']
    test_cycles = datasets['test_cycles']

    # Split train into train/val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )



    print(f"[DONE] Data prepared: X_train {X_tr.shape}, y_train {y_tr.shape}")

    # ===== STEP 2: MODEL TRAINING =====
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("="*70)
    builder = LSTMModelBuilder()
    
    model_path = 'rul_model.h5'
    if os.path.exists(model_path):
        print(f"Loading previously trained model from {model_path} (Skipping epochs!)...")
        trained_model = builder.load_model(model_path)
    else:
        model = builder.build_stacked_lstm(
            seq_length=30,
            num_features=24,
            units=[50, 50]
        )

        trained_model, history = builder.train(
            model, X_tr, y_tr, X_val, y_val,
            epochs=100, 
            batch_size=32
        )

        # Optional: Plot history
        # builder.plot_history()
        
        # Save the model
        builder.save_model(model_path)
        print("[DONE] Model trained and saved!")

    # ===== STEP 3: EVALUATION =====
    print("\n" + "="*70)
    print("STEP 3: EVALUATION")
    print("="*70)
    y_pred = trained_model.predict(X_test)

    print(f"\nSanity Check - y_test std: {y_test.std():.2f}")
    print(f"Sanity Check - y_pred std: {y_pred.std():.2f}")
    if y_pred.std() < 5.0:
        print("WARNING: Model may have collapsed to the mean!")

    evaluator = RULEvaluator()
    results = evaluator.analyze(y_test, y_pred.flatten())
    # Optional: Plot results
    # evaluator.plot_results(y_test, y_pred.flatten())
    print("[DONE] Model evaluated!")

    # ===== STEP 4: ALERTS =====
    print("\n" + "="*70)
    print("STEP 4: ALERTS")
    print("="*70)
    alert_system = AlertSystem(
        healthy=100,
        warning=50,
        critical=20
    )

    df_alerts = pd.DataFrame({'unit_id': test_ids, 'rul_pred': y_pred.flatten()})
    latest_per_engine = df_alerts.groupby('unit_id').last().reset_index()
    
    alerts = alert_system.generate_alerts(
        latest_per_engine['unit_id'].values, 
        latest_per_engine['rul_pred'].values
    )
    alert_system.print_report(alerts)
    print("[DONE] Alerts generated!")

    # ===== STEP 5: DASHBOARD =====
    print("\n" + "="*70)
    print("STEP 5: DASHBOARD")
    print("="*70)
    dashboard = Dashboard(test_ids, y_pred, test_cycles)
    
    # Optional: Create dashboard visualizations
    dashboard.create_fleet_dashboard()
    # Using Unit 3 for visualization as Unit 1 only has 1 sequence window
    dashboard.create_unit_dashboard(unit_id=3)
    print("[DONE] Dashboard setup complete! Uncomment dashboard methods in main.py to see plots.")

    print("\n" + "="*70)
    print("[DONE] ALL MILESTONES COMPLETE! [DONE]")
    print("="*70)

if __name__ == "__main__":
    main()
