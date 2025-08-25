from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
import requests
import time
from typing import Dict, List, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Adding project path to sys.path
sys.path.append('/opt/airflow/pipeline_code')

# Project Configuration
class Config:
    # Paths
    PROJECT_ROOT = Path.cwd()
    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    DATA_TRANSFORMED = PROJECT_ROOT / "data" / "transformed"
    MODELS = PROJECT_ROOT / "models"
    LOGS = PROJECT_ROOT / "logs"
    REPORTS = PROJECT_ROOT / "reports"
    
    
    # Database
    DB_PATH = PROJECT_ROOT / "data" / "churn_db.sqlite"
    DB_CUSTOMER = PROJECT_ROOT / "data" / "customers.db"

    # Feature Store
    FEATURE_STORE_PATH = PROJECT_ROOT / "feature_store"
    
    def __init__(self):
        # Create directories
        for path in [self.DATA_RAW, self.DATA_PROCESSED, self.DATA_TRANSFORMED, 
                    self.MODELS, self.LOGS, self.REPORTS, self.FEATURE_STORE_PATH]:
            path.mkdir(parents=True, exist_ok=True)

config = Config()

# ============================================================================
# 3. LOGGING UTILITY
# ============================================================================

class Logger:
    def __init__(self, name: str, log_file: str = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # File handler
            if log_file:
                file_handler = logging.FileHandler(config.LOGS / log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def warning(self, message: str):
        self.logger.warning(message)

# ============================================================================
# 4. DATA FETCH
# ============================================================================

class DataFetch:
    def __init__(self):
        self.logger = Logger("DataFetch", "data_fetch.log")
        np.random.seed(42)
    
    def customer_data(self) -> pd.DataFrame:
        """customer demographics data"""
        self.logger.info(f"Fetching customer records")

        # GitHub URLs
        data = pd.read_csv("https://raw.githubusercontent.com/jyotirvasu/customer_churn/refs/heads/main/customers_20250823_011314.csv", header=0)
        
        df = pd.DataFrame(data)
        self.logger.info(f"Fetched {len(df)} Customer records")
        return df
    
    def transaction_data(self) -> pd.DataFrame:
        """transaction data"""
        self.logger.info(f"Fetching Transaction data for customers")
        
        # GitHub URLs
        data = pd.read_csv("https://raw.githubusercontent.com/jyotirvasu/customer_churn/refs/heads/main/transactions_20250823_011319.csv", header=0)
        
        df = pd.DataFrame(data)
        self.logger.info(f"Fetched {len(df)} transaction records")
        return df
    
    def support_data(self) -> pd.DataFrame:
        """Generate synthetic support ticket data"""
        self.logger.info(f"Fetching support data for customers")

         # GitHub URLs
        data = pd.read_csv("https://raw.githubusercontent.com/jyotirvasu/customer_churn/refs/heads/main/support_20250823_011320.csv", header=0)
          
        df = pd.DataFrame(data)
        self.logger.info(f"Fetched {len(df)} support tickets")
        return df

# ============================================================================
# 5. DATA INGESTION
# ============================================================================

class DataIngestion:
    def __init__(self):
        self.logger = Logger("DataIngestion", "data_ingestion.log")
        self.data_gen = DataFetch()
    
    def ingest_customer_data(self) -> bool:
        """Ingest customer demographic data"""
        try:
            self.logger.info("Starting customer data ingestion")
            
            # this is read from CSV/database
            df = self.data_gen.customer_data()
            
            # Save raw data with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = config.DATA_RAW / "customers" / f"customers_{timestamp}.csv"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(filepath, index=False)
            self.logger.info(f"Customer data ingested successfully: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Customer data ingestion failed: {str(e)}")
            return False
    
    def ingest_transaction_data(self) -> bool:
        """Ingest transaction data from API"""
        try:
            self.logger.info("Starting transaction data ingestion")            
            df = self.data_gen.transaction_data()
            
            # Save raw data with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = config.DATA_RAW / "transactions" / f"transactions_{timestamp}.csv"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(filepath, index=False)
            self.logger.info(f"Transaction data ingested successfully: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Transaction data ingestion failed: {str(e)}")
            return False
    
    def ingest_support_data(self) -> bool:
        """Ingest support ticket data"""
        try:
            self.logger.info("Starting support data ingestion")            
            df = self.data_gen.support_data()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = config.DATA_RAW / "support" / f"support_{timestamp}.csv"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(filepath, index=False)
            self.logger.info(f"Support data ingested successfully: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Support data ingestion failed: {str(e)}")
            return False

# ============================================================================
# 6. DATA VALIDATION
# ============================================================================

class DataValidator:
    def __init__(self):
        self.logger = Logger("DataValidator", "data_validation.log")
    
    def validate_customer_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate customer data quality"""
        self.logger.info("Validating customer data")
        
        if isinstance(df, dict):
            df = pd.DataFrame(df)
            
        print(df.columns)
        print(df.head())
        
        validation_results = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_customers': df['customer_id'].duplicated().sum(),
            'data_types_correct': True,
            'age_range_valid': True,
            'income_positive': True,
            'issues': []
        }
        
        # Check data types
        expected_types = {
            'customer_id': 'object',
            'age': 'int64',
            'income': 'int64',
            'churn': 'int64'
        }
        
        for col, expected_type in expected_types.items():
            if col in df.columns and str(df[col].dtype) != expected_type:
                validation_results['data_types_correct'] = False
                validation_results['issues'].append(f"Column {col} has incorrect data type")
        
        # #Check age range
        # if df['age'].min() < 18 or df['age'].max() > 100:
        #     validation_results['age_range_valid'] = False
        #     validation_results['issues'].append("Age values outside valid range (18-100)")
        
        # Check income
        if df['income'].min() < 0:
            validation_results['income_positive'] = False
            validation_results['issues'].append("Negative income values found")
        
        self.logger.info(f"Validation completed. Issues found: {len(validation_results['issues'])}")
        return validation_results
    
    def generate_quality_report(self, validation_results: Dict[str, Any]) -> None:
        """Generate data quality report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = config.REPORTS / f"data_quality_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("DATA QUALITY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            f.write(f"Total Records: {validation_results['total_records']}\n")
            f.write(f"Duplicate Customers: {validation_results['duplicate_customers']}\n\n")
            
            f.write("Missing Values by Column:\n")
            for col, missing in validation_results['missing_values'].items():
                f.write(f"  {col}: {missing}\n")
            
            f.write(f"\nData Types Correct: {validation_results['data_types_correct']}\n")
            f.write(f"Age Range Valid: {validation_results['age_range_valid']}\n")
            f.write(f"Income Values Positive: {validation_results['income_positive']}\n\n")
            
            if validation_results['issues']:
                f.write("Issues Identified:\n")
                for issue in validation_results['issues']:
                    f.write(f"  - {issue}\n")
            else:
                f.write("No critical issues identified.\n")
        
        self.logger.info(f"Quality report generated: {report_path}")

# ============================================================================
# 7. DATA PREPARATION AND CLEANING
# ============================================================================

class DataPreparation:
    def __init__(self):
        self.logger = Logger("DataPreparation", "data_preparation.log")
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def clean_customer_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess customer data"""
        self.logger.info("Cleaning customer data")
        
        df_clean = df.copy()
        
        # Handle missing values
        df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
        df_clean['income'] = df_clean['income'].fillna(df_clean['income'].median())
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates(subset=['customer_id'])
        
        # Cap age at reasonable limits
        df_clean['age'] = np.clip(df_clean['age'], 18, 80)
        
        # Encode categorical variables
        categorical_columns = ['account_type', 'location']
        for col in categorical_columns:
            le = LabelEncoder()
            df_clean[f'{col}_encoded'] = le.fit_transform(df_clean[col])
            self.label_encoders[col] = le
        
        self.logger.info(f"Data cleaned. Shape: {df_clean.shape}")
        return df_clean
    
    def aggregate_transaction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate transaction data by customer"""
        self.logger.info("Aggregating transaction data")
        
        # Convert transaction_date to datetime
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        # Calculate days since last transaction
        df['days_since_transaction'] = (datetime.now() - df['transaction_date']).dt.days
        
        # Aggregate by customer
        agg_features = df.groupby('customer_id').agg({
            'amount': ['sum', 'mean', 'count', 'std'],
            'days_since_transaction': 'min',
            'transaction_type': lambda x: x.mode().iloc[0] if not x.empty else 'Purchase',
            'merchant_category': 'nunique'
        }).reset_index()
        
        # Flatten column names
        agg_features.columns = [
            'customer_id', 'total_amount', 'avg_amount', 'transaction_count', 
            'amount_std', 'days_since_last_transaction', 'most_frequent_type', 
            'unique_merchants'
        ]
        
        # Fill NaN values
        agg_features['amount_std'] = agg_features['amount_std'].fillna(0)
        
        self.logger.info(f"Transaction aggregation completed. Shape: {agg_features.shape}")
        return agg_features
    
    def aggregate_support_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate support data by customer"""
        self.logger.info("Aggregating support data")
        
        if df.empty:
            return pd.DataFrame({'customer_id': [], 'total_tickets': [], 'avg_satisfaction': []})
        
        df['ticket_date'] = pd.to_datetime(df['ticket_date'])
        
        agg_features = df.groupby('customer_id').agg({
            'ticket_date': 'count',
            'resolution_time_hours': 'mean',
            'satisfaction_score': 'mean'
        }).reset_index()
        
        agg_features.columns = [
            'customer_id', 'total_tickets', 'avg_resolution_time', 'avg_satisfaction'
        ]
        
        self.logger.info(f"Support aggregation completed. Shape: {agg_features.shape}")
        return agg_features

# ============================================================================
# 8. FEATURE ENGINEERING AND TRANSFORMATION
# ============================================================================

class FeatureEngineer:
    def __init__(self):
        self.logger = Logger("FeatureEngineer", "feature_engineering.log")
    
    def create_features(self, customer_df: pd.DataFrame, transaction_df: pd.DataFrame, 
                       support_df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features for ML model"""
        self.logger.info("Creating engineered features")
        
        # Start with customer data
        features_df = customer_df.copy()
        
        # Merge transaction features
        features_df = features_df.merge(transaction_df, on='customer_id', how='left')
        
        # Merge support features
        features_df = features_df.merge(support_df, on='customer_id', how='left')
        
        # Fill NaN values from merges
        transaction_cols = ['total_amount', 'avg_amount', 'transaction_count', 'amount_std', 
                           'days_since_last_transaction', 'unique_merchants']
        for col in transaction_cols:
            features_df[col] = features_df[col].fillna(0)
        
        support_cols = ['total_tickets', 'avg_resolution_time', 'avg_satisfaction']
        for col in support_cols:
            features_df[col] = features_df[col].fillna(0)
        
        # Engineer new features
        features_df['income_per_product'] = features_df['income'] / (features_df['num_products'] + 1)
        features_df['amount_per_transaction'] = features_df['total_amount'] / (features_df['transaction_count'] + 1)
        features_df['tickets_per_tenure'] = features_df['total_tickets'] / (features_df['tenure_months'] + 1)
        features_df['is_high_value'] = (features_df['total_amount'] > features_df['total_amount'].quantile(0.8)).astype(int)
        features_df['is_recent_customer'] = (features_df['tenure_months'] < 12).astype(int)
        features_df['has_complaints'] = (features_df['total_tickets'] > 0).astype(int)
        
        self.logger.info(f"Feature engineering completed. Shape: {features_df.shape}")
        return features_df
    
    def save_to_database(self, df: pd.DataFrame) -> None:
        """Save features to SQLite database"""
        self.logger.info("Saving features to database")
        
        conn = sqlite3.connect(config.DB_PATH)
        df.to_sql('customer_features', conn, if_exists='replace', index=False)
        conn.close()
        
        self.logger.info("Features saved to database successfully")

# ============================================================================
# 9. FEATURE STORE
# ============================================================================

class FeatureStore:
    def __init__(self):
        self.logger = Logger("FeatureStore", "feature_store.log")
        self.metadata_file = config.FEATURE_STORE_PATH / "feature_metadata.json"
        self._initialize_metadata()
    
    def _initialize_metadata(self):
        """Initialize feature metadata"""
        metadata = {
            "features": {
                "customer_id": {"description": "Unique customer identifier", "type": "categorical"},
                "age": {"description": "Customer age", "type": "numerical"},
                "income": {"description": "Annual income", "type": "numerical"},
                "tenure_months": {"description": "Months as customer", "type": "numerical"},
                "total_amount": {"description": "Total transaction amount", "type": "numerical"},
                "transaction_count": {"description": "Number of transactions", "type": "numerical"},
                "avg_satisfaction": {"description": "Average satisfaction score", "type": "numerical"},
                "is_high_value": {"description": "High value customer flag", "type": "binary"},
                "has_complaints": {"description": "Has support tickets", "type": "binary"}
            },
            "version": "1.0",
            "created_date": datetime.now().isoformat()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_features(self, customer_ids: List[str] = None) -> pd.DataFrame:
        """Retrieve features from database"""
        self.logger.info("Retrieving features from store")
        
        conn = sqlite3.connect(config.DB_PATH)
        
        if customer_ids:
            placeholders = ','.join(['?'] * len(customer_ids))
            query = f"SELECT * FROM customer_features WHERE customer_id IN ({placeholders})"
            df = pd.read_sql_query(query, conn, params=customer_ids)
        else:
            df = pd.read_sql_query("SELECT * FROM customer_features", conn)
        
        conn.close()
        
        self.logger.info(f"Retrieved {len(df)} feature records")
        return df

# ============================================================================
# 10. MODEL BUILDING
# ============================================================================

class ChurnModel:
    def __init__(self):
        self.logger = Logger("ChurnModel", "model_building.log")
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def prepare_model_data(self, df: pd.DataFrame) -> tuple:
        """Prepare data for model training"""
        self.logger.info("Preparing model data")
        
        # Select features for modeling
        feature_cols = [
            'age', 'income', 'tenure_months', 'account_type_encoded', 'location_encoded',
            'has_credit_card', 'num_products', 'is_active_member',
            'total_amount', 'avg_amount', 'transaction_count', 'amount_std',
            'days_since_last_transaction', 'unique_merchants',
            'total_tickets', 'avg_resolution_time', 'avg_satisfaction',
            'income_per_product', 'amount_per_transaction', 'tickets_per_tenure',
            'is_high_value', 'is_recent_customer', 'has_complaints'
        ]
        
        # Filter existing columns
        existing_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = existing_cols
        
        X = df[existing_cols].copy()
        y = df['churn'].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=existing_cols)
        
        self.logger.info(f"Data prepared. Features: {len(existing_cols)}, Samples: {len(X_scaled_df)}")
        return X_scaled_df, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train multiple models and compare performance"""
        self.logger.info("Training models")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models
        model_configs = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in model_configs.items():
            self.logger.info(f"Training {name}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': report,
                'auc_score': auc_score,
                'accuracy': report['accuracy'],
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'f1_score': report['1']['f1-score']
            }
            
            # Save model
            model_path = config.MODELS / f"{name}_model.joblib"
            joblib.dump(model, model_path)
            
            self.logger.info(f"{name} - Accuracy: {report['accuracy']:.3f}, AUC: {auc_score:.3f}")
        
        # Save scaler
        scaler_path = config.MODELS / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        
        self.models = results
        return results
    
    def generate_model_report(self, results: Dict[str, Any]) -> None:
        """Generate model performance report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = config.REPORTS / f"model_performance_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("MODEL PERFORMANCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            for model_name, result in results.items():
                f.write(f"{model_name.upper()}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"Precision: {result['precision']:.4f}\n")
                f.write(f"Recall: {result['recall']:.4f}\n")
                f.write(f"F1-Score: {result['f1_score']:.4f}\n")
                f.write(f"AUC Score: {result['auc_score']:.4f}\n\n")
        
        self.logger.info(f"Model report generated: {report_path}")

# ============================================================================
# 13. ADDITIONAL UTILITIES
# ============================================================================

class DataVersioning:
    """Utility class for data versioning with DVC simulation"""
    
    def __init__(self):
        self.logger = Logger("DataVersioning", "data_versioning.log")
        self.version_file = config.PROJECT_ROOT / "data_versions.json"
        self._initialize_versions()
    
    def _initialize_versions(self):
        """Initialize version tracking"""
        if not self.version_file.exists():
            versions = {
                "versions": [],
                "current_version": None
            }
            with open(self.version_file, 'w') as f:
                json.dump(versions, f, indent=2)
    
    def create_version(self, description: str) -> str:
        """Create a new data version"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"v{timestamp}"
        
        with open(self.version_file, 'r') as f:
            versions = json.load(f)
        
        version_info = {
            "version_id": version_id,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "files": []
        }
        
        # Track current files
        for data_type in ["customers", "transactions", "support"]:
            pattern = f"{data_type}/*.csv"
            files = list(config.DATA_RAW.glob(pattern))
            if files:
                latest_file = max(files, key=os.path.getctime)
                version_info["files"].append({
                    "type": data_type,
                    "path": str(latest_file.relative_to(config.PROJECT_ROOT)),
                    "size": latest_file.stat().st_size
                })
        
        versions["versions"].append(version_info)
        versions["current_version"] = version_id
        
        with open(self.version_file, 'w') as f:
            json.dump(versions, f, indent=2)
        
        self.logger.info(f"Created data version: {version_id}")
        return version_id

class PipelineMonitor:
    """Pipeline monitoring and alerting"""
    
    def __init__(self):
        self.logger = Logger("PipelineMonitor", "pipeline_monitoring.log")
        self.metrics_file = config.REPORTS / "pipeline_metrics.json"
    
    def log_pipeline_metrics(self, stage: str, duration: float, status: str, **kwargs):
        """Log pipeline stage metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "duration_seconds": duration,
            "status": status,
            "additional_info": kwargs
        }
        
        # Load existing metrics
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = {"pipeline_runs": []}
        except Exception as e:
            # A generic catch-all for any other unexpected errors
            logging.critical(f"An unexpected error occurred while loading metrics: {e}")
            #Depending on the application, you might want to re-raise or handle this differently
            all_metrics = {"pipeline_runs": []}
        
        all_metrics["pipeline_runs"].append(metrics)
        
        # Keep only last 100 runs
        all_metrics["pipeline_runs"] = all_metrics["pipeline_runs"][-100:]
        
        with open(self.metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        self.logger.info(f"Logged metrics for {stage}: {status} in {duration:.2f}s")
    
    def get_pipeline_health(self) -> Dict[str, Any]:
        """Get pipeline health summary"""
        if not self.metrics_file.exists():
            return {"status": "no_data", "message": "No pipeline metrics available"}
        
        with open(self.metrics_file, 'r') as f:
            metrics = json.load(f)
        
        runs = metrics["pipeline_runs"]
        if not runs:
            return {"status": "no_data", "message": "No pipeline runs recorded"}
        
        recent_runs = runs[-10:]  # Last 10 runs
        success_rate = sum(1 for run in recent_runs if run["status"] == "success") / len(recent_runs)
        avg_duration = sum(run["duration_seconds"] for run in recent_runs) / len(recent_runs)
        
        return {
            "status": "healthy" if success_rate >= 0.8 else "warning",
            "success_rate": success_rate,
            "avg_duration": avg_duration,
            "last_run": recent_runs[-1]["timestamp"],
            "total_runs": len(runs)
        }


# ============================================================================
# 11. PIPELINE ORCHESTRATOR
# ============================================================================

class PipelineOrchestrator:
    def __init__(self):
        self.logger = Logger("PipelineOrchestrator", "pipeline_orchestration.log")
        self.ingestion = DataIngestion()
        self.validator = DataValidator()
        self.preparation = DataPreparation()
        self.feature_engineer = FeatureEngineer()
        self.feature_store = FeatureStore()
        self.model = ChurnModel()
        
# Default arguments
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['admin@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=15)
}

# Create DAG
dag = DAG(
    'customer_churn_pipeline',
    default_args=default_args,
    description='End-to-end customer churn prediction pipeline',
    schedule_interval=timedelta(days=1),  # Daily execution
    max_active_runs=1,
    catchup=False,
    tags=['ml', 'churn', 'data-pipeline']
)

# Task functions with monitoring and versioning
import time
from datetime import datetime

def ingest_data(**context):
    start_time = time.time()
    monitor = PipelineMonitor()
    versioning = DataVersioning()
    
    try:
        orchestrator = PipelineOrchestrator()
        success = (orchestrator.ingestion.ingest_customer_data() and
                   orchestrator.ingestion.ingest_transaction_data() and
                   orchestrator.ingestion.ingest_support_data())
        
        if not success:
            duration = time.time() - start_time
            monitor.log_pipeline_metrics("data_ingestion", duration, "failed")
            raise Exception("Data ingestion failed")
        
        # Create data version after successful ingestion
        version_id = versioning.create_version("Daily data ingestion completed")
        
        duration = time.time() - start_time
        monitor.log_pipeline_metrics(
            "data_ingestion", 
            duration, 
            "success",
            version_id=version_id,
            files_ingested=3
        )
        
        # Store version_id for downstream tasks
        context['task_instance'].xcom_push(key='data_version_id', value=version_id)
        
        return f"Data ingested successfully. Version: {version_id}"
        
    except Exception as e:
        duration = time.time() - start_time
        monitor.log_pipeline_metrics("data_ingestion", duration, "failed", error=str(e))
        raise

def validate_data(**context):
    start_time = time.time()
    monitor = PipelineMonitor()
    
    try:
        orchestrator = PipelineOrchestrator()
        customer_df = pd.read_csv(max(config.DATA_RAW.glob("customers/*.csv")))
        validation_results = orchestrator.validator.validate_customer_data(customer_df)
        
        # Generate quality report
        orchestrator.validator.generate_quality_report(validation_results)
        
        if validation_results['issues']:
            duration = time.time() - start_time
            # monitor.log_pipeline_metrics(
            #     "data_validation", 
            #     duration, 
            #     "failed",
            #     issues_found=len(validation_results['issues']),
            #     issues=validation_results['issues']
            # )
            raise Exception(f"Data validation failed: {validation_results['issues']}")
        
        duration = time.time() - start_time
        # monitor.log_pipeline_metrics(
        #     "data_validation", 
        #     duration, 
        #     "success",
        #     total_records=validation_results['total_records'],
        #     duplicate_customers=validation_results['duplicate_customers'],
        #     missing_values=sum(validation_results['missing_values'].values())
        # )
        
        return "Data validation passed"
        
    except Exception as e:
        duration = time.time() - start_time
        # monitor.log_pipeline_metrics("data_validation", duration, "failed", error=str(e))
        raise

def prepare_and_train(**context):
    start_time = time.time()
    monitor = PipelineMonitor()
    
    try:
        orchestrator = PipelineOrchestrator()
        
        # Data preparation
        customer_df = pd.read_csv(max(config.DATA_RAW.glob("customers/*.csv")))
        clean_customer_df = orchestrator.preparation.clean_customer_data(customer_df)

        transaction_df = pd.read_csv(max(config.DATA_RAW.glob("transactions/*.csv")))
        agg_transaction_df = orchestrator.preparation.aggregate_transaction_data(transaction_df)
                
        support_df = pd.read_csv(max(config.DATA_RAW.glob("support/*.csv")))
        agg_support_df = orchestrator.preparation.aggregate_support_data(support_df)

        features_df = orchestrator.feature_engineer.create_features(
            clean_customer_df, agg_transaction_df, agg_support_df
        )

        processed_path = config.DATA_PROCESSED / f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        features_df.to_csv(processed_path, index=False)
        orchestrator.feature_engineer.save_to_database(features_df)

        # Model training
        X, y = orchestrator.model.prepare_model_data(features_df)
        results = orchestrator.model.train_models(X, y)
        orchestrator.model.generate_model_report(results)
        
        # Get best model performance
        best_model = max(results.items(), key=lambda x: x[1]['auc_score'])
        best_model_name, best_model_results = best_model
        
        duration = time.time() - start_time
        monitor.log_pipeline_metrics(
            "data_preparation_training", 
            duration, 
            "success",
            features_count=len(features_df.columns),
            training_samples=len(X),
            best_model=best_model_name,
            best_auc_score=best_model_results['auc_score'],
            best_accuracy=best_model_results['accuracy']
        )
        
        # Store model performance for downstream tasks
        context['task_instance'].xcom_push(key='best_model_performance', value={
            'model_name': best_model_name,
            'auc_score': best_model_results['auc_score'],
            'accuracy': best_model_results['accuracy']
        })
        
        return f"Data preparation and training completed. Best model: {best_model_name} (AUC: {best_model_results['auc_score']:.3f})"
        
    except Exception as e:
        duration = time.time() - start_time
        monitor.log_pipeline_metrics("data_preparation_training", duration, "failed", error=str(e))
        raise

def run_inference(**context):
    start_time = time.time()
    monitor = PipelineMonitor()
    
    try:
        orchestrator = PipelineOrchestrator()
        customer_ids = ['CUST_000001', 'CUST_000002', 'CUST_000003', 'CUST_000004', 'CUST_000005']
        
        # This would need to be implemented in the orchestrator
        # inference_results = orchestrator.run_inference_pipeline(customer_ids)
        
        # For now, creating a dummy inference result
        inference_results = pd.DataFrame({
            'customer_id': customer_ids,
            'churn_probability': [0.23, 0.78, 0.45, 0.12, 0.67],
            'prediction': [0, 1, 0, 0, 1]
        })
        
        if not inference_results.empty:
            # Save inference results
            inference_path = config.REPORTS / f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            inference_results.to_csv(inference_path, index=False)
            
            # Calculate inference metrics
            high_risk_customers = (inference_results['churn_probability'] > 0.5).sum()
            avg_churn_prob = inference_results['churn_probability'].mean()
            
            duration = time.time() - start_time
            monitor.log_pipeline_metrics(
                "inference", 
                duration, 
                "success",
                customers_scored=len(inference_results),
                high_risk_customers=int(high_risk_customers),
                avg_churn_probability=float(avg_churn_prob),
                inference_file=str(inference_path)
            )
            
            return f"Inference completed for {len(inference_results)} customers. High risk: {high_risk_customers}"
        else:
            duration = time.time() - start_time
            monitor.log_pipeline_metrics("inference", duration, "failed", error="No inference results generated")
            raise Exception("Inference failed - no results generated")
            
    except Exception as e:
        duration = time.time() - start_time
        monitor.log_pipeline_metrics("inference", duration, "failed", error=str(e))
        raise

def generate_reports(**context):
    start_time = time.time()
    monitor = PipelineMonitor()
    
    try:
        # Get data from previous tasks
        data_version_id = context['task_instance'].xcom_pull(key='data_version_id', task_ids='ingest_data')
        model_performance = context['task_instance'].xcom_pull(key='best_model_performance', task_ids='prepare_and_train_data')
        
        # Generate comprehensive pipeline report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = config.REPORTS / f"pipeline_summary_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("CUSTOMER CHURN PIPELINE SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Data Version: {data_version_id}\n\n")
            
            f.write("PIPELINE CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Data stored in: {config.DATA_RAW}\n")
            f.write(f"Database path: {config.DB_PATH}\n")
            f.write(f"Models saved in: {config.MODELS}\n")
            f.write(f"Reports saved in: {config.REPORTS}\n")
            f.write(f"Logs saved in: {config.LOGS}\n\n")
            
            if model_performance:
                f.write("MODEL PERFORMANCE:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Best Model: {model_performance['model_name']}\n")
                f.write(f"AUC Score: {model_performance['auc_score']:.4f}\n")
                f.write(f"Accuracy: {model_performance['accuracy']:.4f}\n\n")
            
            # Feature store info
            orchestrator = PipelineOrchestrator()
            with open(orchestrator.feature_store.metadata_file, 'r') as metadata_file:
                metadata = json.load(metadata_file)
            
            f.write("FEATURE STORE INFO:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Version: {metadata['version']}\n")
            f.write(f"Features: {len(metadata['features'])}\n")
            f.write(f"Created: {metadata['created_date']}\n\n")
            
            # Pipeline health
            pipeline_health = monitor.get_pipeline_health()
            f.write("PIPELINE HEALTH:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Status: {pipeline_health.get('status', 'unknown')}\n")
            f.write(f"Success Rate: {pipeline_health.get('success_rate', 0):.2%}\n")
            f.write(f"Avg Duration: {pipeline_health.get('avg_duration', 0):.2f}s\n")
            f.write(f"Total Runs: {pipeline_health.get('total_runs', 0)}\n")
        
        duration = time.time() - start_time
        monitor.log_pipeline_metrics(
            "report_generation", 
            duration, 
            "success",
            report_path=str(report_path),
            pipeline_health_status=pipeline_health.get('status', 'unknown')
        )
        
        return f"Reports generated successfully: {report_path}"
        
    except Exception as e:
        duration = time.time() - start_time
        monitor.log_pipeline_metrics("report_generation", duration, "failed", error=str(e))
        raise

def pipeline_monitoring_summary(**context):
    """Generate pipeline monitoring summary and health check"""
    start_time = time.time()
    monitor = PipelineMonitor()
    
    try:
        # Get pipeline health
        health = monitor.get_pipeline_health()
        
        # Generate monitoring report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        monitoring_report_path = config.REPORTS / f"pipeline_monitoring_{timestamp}.txt"
        
        with open(monitoring_report_path, 'w') as f:
            f.write("PIPELINE MONITORING SUMMARY\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            f.write("OVERALL HEALTH:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Status: {health.get('status', 'unknown')}\n")
            f.write(f"Success Rate: {health.get('success_rate', 0):.2%}\n")
            f.write(f"Average Duration: {health.get('avg_duration', 0):.2f} seconds\n")
            f.write(f"Last Run: {health.get('last_run', 'N/A')}\n")
            f.write(f"Total Runs: {health.get('total_runs', 0)}\n\n")
            
            # Load recent metrics for detailed analysis
            if monitor.metrics_file.exists():
                with open(monitor.metrics_file, 'r') as metrics_file:
                    metrics_data = json.load(metrics_file)
                
                recent_runs = metrics_data["pipeline_runs"][-10:]  # Last 10 runs
                
                f.write("RECENT PIPELINE STAGES PERFORMANCE:\n")
                f.write("-" * 35 + "\n")
                
                # Group by stage
                stage_performance = {}
                for run in recent_runs:
                    stage = run['stage']
                    if stage not in stage_performance:
                        stage_performance[stage] = {'durations': [], 'successes': 0, 'total': 0}
                    
                    stage_performance[stage]['durations'].append(run['duration_seconds'])
                    stage_performance[stage]['total'] += 1
                    if run['status'] == 'success':
                        stage_performance[stage]['successes'] += 1
                
                for stage, perf in stage_performance.items():
                    avg_duration = sum(perf['durations']) / len(perf['durations'])
                    success_rate = perf['successes'] / perf['total']
                    f.write(f"{stage}:\n")
                    f.write(f"  Success Rate: {success_rate:.2%}\n")
                    f.write(f"  Avg Duration: {avg_duration:.2f}s\n")
                    f.write(f"  Total Runs: {perf['total']}\n\n")
        
        # Alert if pipeline health is poor
        if health.get('success_rate', 1.0) < 0.8:
            alert_message = f"ALERT: Pipeline health is poor. Success rate: {health.get('success_rate', 0):.2%}"
            monitor.logger.warning(alert_message)
            
            # In a real implementation, you might send alerts via email, Slack, etc.
            # send_alert(alert_message)
        
        duration = time.time() - start_time
        monitor.log_pipeline_metrics(
            "monitoring_summary", 
            duration, 
            "success",
            health_status=health.get('status'),
            success_rate=health.get('success_rate', 0),
            monitoring_report=str(monitoring_report_path)
        )
        
        return f"Pipeline monitoring completed. Status: {health.get('status')}. Report: {monitoring_report_path}"
        
    except Exception as e:
        duration = time.time() - start_time
        monitor.log_pipeline_metrics("monitoring_summary", duration, "failed", error=str(e))
        raise

def data_versioning_summary(**context):
    """Create data versioning summary and cleanup old versions if needed"""
    start_time = time.time()
    monitor = PipelineMonitor()
    versioning = DataVersioning()
    
    try:
        # Load version information
        with open(versioning.version_file, 'r') as f:
            versions_data = json.load(f)
        
        # Generate versioning report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioning_report_path = config.REPORTS / f"data_versioning_{timestamp}.txt"
        
        with open(versioning_report_path, 'w') as f:
            f.write("DATA VERSIONING SUMMARY\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            f.write(f"Current Version: {versions_data.get('current_version', 'N/A')}\n")
            f.write(f"Total Versions: {len(versions_data.get('versions', []))}\n\n")
            
            f.write("RECENT VERSIONS:\n")
            f.write("-" * 20 + "\n")
            
            recent_versions = versions_data.get('versions', [])[-5:]  # Last 5 versions
            for version in recent_versions:
                f.write(f"Version: {version['version_id']}\n")
                f.write(f"  Created: {version['timestamp']}\n")
                f.write(f"  Description: {version['description']}\n")
                f.write(f"  Files: {len(version['files'])}\n")
                
                total_size = sum(file_info['size'] for file_info in version['files'])
                f.write(f"  Total Size: {total_size / (1024*1024):.2f} MB\n\n")
        
        # Cleanup old versions (keep last 10)
        if len(versions_data.get('versions', [])) > 10:
            versions_to_keep = versions_data['versions'][-10:]
            versions_data['versions'] = versions_to_keep
            
            with open(versioning.version_file, 'w') as f:
                json.dump(versions_data, f, indent=2)
            
            versioning.logger.info(f"Cleaned up old versions. Kept {len(versions_to_keep)} versions.")
        
        duration = time.time() - start_time
        monitor.log_pipeline_metrics(
            "data_versioning_summary", 
            duration, 
            "success",
            current_version=versions_data.get('current_version'),
            total_versions=len(versions_data.get('versions', [])),
            versioning_report=str(versioning_report_path)
        )
        
        return f"Data versioning summary completed. Current version: {versions_data.get('current_version')}"
        
    except Exception as e:
        duration = time.time() - start_time
        monitor.log_pipeline_metrics("data_versioning_summary", duration, "failed", error=str(e))
        raise

# Updated task definitions with monitoring and versioning
ingest_task = PythonOperator(
    task_id='ingest_data',
    python_callable=ingest_data,
    dag=dag,
    doc_md="""
    ### Data Ingestion Task
    
    This task:
    - Ingests customer, transaction, and support data
    - Creates a new data version
    - Logs performance metrics
    - Handles failures gracefully
    """
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
    doc_md="""
    ### Data Validation Task
    
    This task:
    - Validates data quality
    - Generates quality reports
    - Logs validation metrics
    - Fails pipeline if critical issues found
    """
)

prepare_and_train_task = PythonOperator(
    task_id='prepare_and_train_data',
    python_callable=prepare_and_train,
    dag=dag,
    doc_md="""
    ### Data Preparation and Model Training Task
    
    This task:
    - Prepares and engineers features
    - Trains multiple ML models
    - Logs training performance
    - Selects best performing model
    """
)

run_inference_task = PythonOperator(
    task_id='run_inference',
    python_callable=run_inference,
    dag=dag,
    doc_md="""
    ### Inference Task
    
    This task:
    - Runs inference on sample customers
    - Generates churn predictions
    - Logs inference metrics
    - Saves prediction results
    """
)

report_task = PythonOperator(
    task_id='generate_reports',
    python_callable=generate_reports,
    dag=dag,
    doc_md="""
    ### Report Generation Task
    
    This task:
    - Generates comprehensive pipeline summary
    - Includes model performance metrics
    - Shows pipeline health status
    - Compiles feature store information
    """
)

# New monitoring and versioning tasks
monitoring_task = PythonOperator(
    task_id='pipeline_monitoring_summary',
    python_callable=pipeline_monitoring_summary,
    dag=dag,
    doc_md="""
    ### Pipeline Monitoring Task
    
    This task:
    - Analyzes pipeline health and performance
    - Generates monitoring reports
    - Triggers alerts if health is poor
    - Tracks success rates and durations
    """,
    trigger_rule='all_done'  # Run even if some upstream tasks fail
)

versioning_task = PythonOperator(
    task_id='data_versioning_summary',
    python_callable=data_versioning_summary,
    dag=dag,
    doc_md="""
    ### Data Versioning Task
    
    This task:
    - Creates versioning summary reports
    - Cleans up old data versions
    - Tracks data lineage
    - Manages version metadata
    """,
    trigger_rule='all_done'  # Run even if some upstream tasks fail
)

# Updated task dependencies with monitoring and versioning
ingest_task >> validate_task >> prepare_and_train_task >> run_inference_task >> report_task

# Monitoring and versioning tasks run after all main tasks complete
[report_task, run_inference_task, prepare_and_train_task] >> monitoring_task
ingest_task >> versioning_task

# Both monitoring and versioning can run in parallel at the end
[monitoring_task, versioning_task]

