import pandas as pd
import numpy as np
import os
import json
import joblib
import argparse
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

class TrafficPredictor:
    """
    Machine Learning model for predicting future network congestion.
    """

    def __init__(self):
        """
        Initialize the predictor models.
        """
        self.rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.xgb_regressor = xgb.XGBRegressor(n_estimators=100, random_state=42) if HAS_XGBOOST else None
        self.xgb_classifier = xgb.XGBClassifier(n_estimators=100, random_state=42) if HAS_XGBOOST else None
        
        self.is_trained = False
        self.metrics = {}

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates lag features, rolling statistics, and time-of-day features.
        
        Args:
            df (pd.DataFrame): Raw historical traffic data.
            
        Returns:
            pd.DataFrame: DataFrame with engineered features.
        """
        df = df.copy()
        df = df.sort_values(by=['source', 'destination', 'time_step'])
        
        # Time-of-day features (assuming cyclic pattern over e.g. 24 steps/hours)
        max_time = df['time_step'].max()
        if max_time > 0:
            df['time_sin'] = np.sin(2 * np.pi * df['time_step'] / max_time)
            df['time_cos'] = np.cos(2 * np.pi * df['time_step'] / max_time)
        else:
            df['time_sin'] = 0
            df['time_cos'] = 0
            
        # Group by edge to compute lags and rolling stats
        grouped = df.groupby(['source', 'destination'])
        
        # Lag features
        df['demand_lag_1'] = grouped['demand'].shift(1)
        df['demand_lag_2'] = grouped['demand'].shift(2)
        
        # Rolling statistics (using last 3 and 5 steps)
        df['demand_roll_mean_3'] = grouped['demand'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df['demand_roll_std_3'] = grouped['demand'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).std())
        df['demand_roll_mean_5'] = grouped['demand'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        df['demand_roll_std_5'] = grouped['demand'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).std())
        
        # Target variables
        # We predict the current 'demand' and 'is_congestion_event' using past (lagged/rolled) features.
        # Drop rows with NaN values created by lagging/rolling (mainly the first few time steps per pair)
        df.dropna(inplace=True)
        
        return df

    def temporal_train_test_split(self, df: pd.DataFrame, test_ratio: float = 0.2):
        """
        Splits data chronologically to respect temporal ordering.
        
        Args:
            df (pd.DataFrame): Feature dataframe.
            test_ratio (float): Proportion of data to use for testing.
            
        Returns:
            tuple: X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test
        """
        unique_times = sorted(df['time_step'].unique())
        split_idx = int(len(unique_times) * (1 - test_ratio))
        split_time = unique_times[split_idx]
        
        train_df = df[df['time_step'] < split_time]
        test_df = df[df['time_step'] >= split_time]
        
        feature_cols = [
            'time_sin', 'time_cos', 'demand_lag_1', 'demand_lag_2',
            'demand_roll_mean_3', 'demand_roll_std_3',
            'demand_roll_mean_5', 'demand_roll_std_5'
        ]
        
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        
        y_reg_train = train_df['demand']
        y_reg_test = test_df['demand']
        
        y_clf_train = train_df['is_congestion_event'].astype(int)
        y_clf_test = test_df['is_congestion_event'].astype(int)
        
        return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test

    def train_and_evaluate(self, traffic_data: pd.DataFrame, output_dir: str = "results"):
        """
        Trains models and evaluates their performance.
        
        Args:
            traffic_data (pd.DataFrame): Historical traffic data.
            output_dir (str): Directory to save metrics and models.
        """
        print("Creating features...")
        df_features = self.create_features(traffic_data)
        
        print("Splitting data chronologically...")
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = self.temporal_train_test_split(df_features)
        
        self.metrics = {
            'regression': {},
            'classification': {}
        }
        
        # 1. Random Forest (Regression & Classification)
        print("Training Random Forest...")
        self.rf_regressor.fit(X_train, y_reg_train)
        self.rf_classifier.fit(X_train, y_clf_train)
        
        rf_reg_preds = self.rf_regressor.predict(X_test)
        rf_clf_preds = self.rf_classifier.predict(X_test)
        
        self.metrics['regression']['RandomForest'] = {
            'MAE': mean_absolute_error(y_reg_test, rf_reg_preds),
            'RMSE': np.sqrt(mean_squared_error(y_reg_test, rf_reg_preds))
        }
        
        self.metrics['classification']['RandomForest'] = {
            'Accuracy': accuracy_score(y_clf_test, rf_clf_preds),
            'Precision': precision_score(y_clf_test, rf_clf_preds, zero_division=0),
            'Recall': recall_score(y_clf_test, rf_clf_preds, zero_division=0),
            'F1': f1_score(y_clf_test, rf_clf_preds, zero_division=0)
        }
        
        # 2. XGBoost (Regression & Classification)
        if HAS_XGBOOST:
            print("Training XGBoost...")
            self.xgb_regressor.fit(X_train, y_reg_train)
            self.xgb_classifier.fit(X_train, y_clf_train)
            
            xgb_reg_preds = self.xgb_regressor.predict(X_test)
            xgb_clf_preds = self.xgb_classifier.predict(X_test)
            
            self.metrics['regression']['XGBoost'] = {
                'MAE': mean_absolute_error(y_reg_test, xgb_reg_preds),
                'RMSE': np.sqrt(mean_squared_error(y_reg_test, xgb_reg_preds))
            }
            
            self.metrics['classification']['XGBoost'] = {
                'Accuracy': accuracy_score(y_clf_test, xgb_clf_preds),
                'Precision': precision_score(y_clf_test, xgb_clf_preds, zero_division=0),
                'Recall': recall_score(y_clf_test, xgb_clf_preds, zero_division=0),
                'F1': f1_score(y_clf_test, xgb_clf_preds, zero_division=0)
            }
        
        self.is_trained = True
        
        # Save artifacts
        self.save_artifacts(output_dir)

    def save_artifacts(self, output_dir: str):
        """
        Saves trained models and metrics to disk.
        """
        os.makedirs(output_dir, exist_ok=True)
        models_dir = os.path.join(output_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Save metrics
        metrics_file = os.path.join(output_dir, 'ml_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
        # Save models
        joblib.dump(self.rf_regressor, os.path.join(models_dir, 'rf_regressor.joblib'))
        joblib.dump(self.rf_classifier, os.path.join(models_dir, 'rf_classifier.joblib'))
        
        if HAS_XGBOOST:
            joblib.dump(self.xgb_regressor, os.path.join(models_dir, 'xgb_regressor.joblib'))
            joblib.dump(self.xgb_classifier, os.path.join(models_dir, 'xgb_classifier.joblib'))
            
        print(f"Artifacts saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train ML Traffic Predictor")
    parser.add_argument("--data", type=str, default="data/traffic.csv", help="Path to traffic CSV data")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save models and metrics")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found. Generate it first.")
        return

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    
    predictor = TrafficPredictor()
    predictor.train_and_evaluate(df, output_dir=args.output_dir)
    print("Training complete.")
    print(json.dumps(predictor.metrics, indent=2))

if __name__ == '__main__':
    main()
