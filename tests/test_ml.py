import pytest
import pandas as pd
import numpy as np
import os
from quantum_routing.ml.predictor import TrafficPredictor

@pytest.fixture
def dummy_traffic_data():
    records = []
    # Create 30 time steps for 2 pairs
    for step in range(30):
        for src, dst in [(0, 1), (2, 3)]:
            records.append({
                'time_step': step,
                'source': src,
                'destination': dst,
                'demand': float(np.random.uniform(10, 50)),
                'is_congestion_event': bool(np.random.random() < 0.2)
            })
    return pd.DataFrame(records)

def test_predictor_init():
    predictor = TrafficPredictor()
    assert not predictor.is_trained
    assert predictor.rf_regressor is not None
    assert predictor.rf_classifier is not None

def test_create_features(dummy_traffic_data):
    predictor = TrafficPredictor()
    df_features = predictor.create_features(dummy_traffic_data)
    
    # 30 steps total, but shift(2) drops the first 2 steps for each pair
    assert len(df_features) == (30 - 2) * 2
    
    # Check features exist
    expected_cols = [
        'time_sin', 'time_cos', 'demand_lag_1', 'demand_lag_2',
        'demand_roll_mean_3', 'demand_roll_std_3',
        'demand_roll_mean_5', 'demand_roll_std_5'
    ]
    for col in expected_cols:
        assert col in df_features.columns
        
    # Ensure no NaNs remain
    assert not df_features.isnull().values.any()

def test_temporal_split(dummy_traffic_data):
    predictor = TrafficPredictor()
    df_features = predictor.create_features(dummy_traffic_data)
    
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = predictor.temporal_train_test_split(df_features, test_ratio=0.2)
    
    # Test ratio 0.2 of 26 remaining time steps ~ 5 steps
    # Check sizes
    assert len(X_train) + len(X_test) == len(df_features)
    assert len(X_train) > len(X_test)
    assert len(y_reg_train) == len(X_train)
    assert len(y_clf_train) == len(X_train)
    
    # Ensure temporal ordering: all train time steps must be strictly less than all test time steps
    # Since we can't extract time_step easily from X_train, we check the original df
    train_indices = X_train.index
    test_indices = X_test.index
    train_times = df_features.loc[train_indices, 'time_step']
    test_times = df_features.loc[test_indices, 'time_step']
    
    assert train_times.max() < test_times.min()

def test_train_and_evaluate(dummy_traffic_data, tmp_path):
    predictor = TrafficPredictor()
    
    # Use tmp_path to not pollute the real results directory
    output_dir = tmp_path / "results"
    
    predictor.train_and_evaluate(dummy_traffic_data, output_dir=str(output_dir))
    
    assert predictor.is_trained
    assert 'regression' in predictor.metrics
    assert 'classification' in predictor.metrics
    
    # Check artifacts
    assert (output_dir / "ml_metrics.json").exists()
    assert (output_dir / "models" / "rf_regressor.joblib").exists()
    assert (output_dir / "models" / "rf_classifier.joblib").exists()
