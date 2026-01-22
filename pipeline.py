import os
import logging
import argparse
from typing import Tuple, List, Optional, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, roc_auc_score, 
                             matthews_corrcoef, confusion_matrix)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Configure Logging to track the pipeline's progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    """
    Configuration class to ensure reproducibility and easy management of parameters.
    This replaces hard-coded paths in the script.
    """
    def __init__(self, 
                 train_path='Tim_21/Podaci/train.csv', 
                 test_path='Tim_21/Podaci/test.csv', 
                 random_state=42, 
                 cv_folds=5, 
                 target_col='Label'):
        
        self.TRAIN_PATH = train_path
        self.TEST_PATH = test_path
        self.RANDOM_STATE = random_state
        self.CV_FOLDS = cv_folds
        self.TARGET_COL = target_col
        # Define categorical features based on your previous analysis
        self.CAT_FEATURES = ['Gender', 'Education', 'Current_Work_Status', 'AdmissionDx']


class ClinicalModelPipeline:
    """
    A robust pipeline for clinical data analysis.
    Encapsulates preprocessing, training, and evaluation logic.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.preprocessor = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads train and test datasets from the paths defined in Config."""
        if not os.path.exists(self.config.TRAIN_PATH):
            raise FileNotFoundError(f"Train file not found at {self.config.TRAIN_PATH}")
        
        logger.info("Loading datasets...")
        train_df = pd.read_csv(self.config.TRAIN_PATH, index_col=0)
        test_df = pd.read_csv(self.config.TEST_PATH, index_col=0)
        
        logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df

    def visualize_distributions(self, df: pd.DataFrame, output_dir: str = 'plots') -> None:
        """
        Generates interpretable visualizations for distribution analysis.
        Saves them to disk rather than just showing them, which is better for automated runs.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Target Class Balance
        plt.figure(figsize=(8, 6))
        sns.countplot(x=self.config.TARGET_COL, data=df)
        plt.title('Class Distribution')
        plt.savefig(f"{output_dir}/class_distribution.png")
        plt.close()
        
        # 2. Categorical features
        for col in self.config.CAT_FEATURES:
            if col in df.columns:
                plt.figure(figsize=(10, 6))
                sns.countplot(y=col, data=df, orient='h')
                plt.title(f'Distribution of {col}')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/dist_{col}.png")
                plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}/")

    def build_pipeline(self, classifier) -> ImbPipeline:
        """
        Constructs a reproducible sklearn pipeline.
        
        This solves the 'Data Leakage' issue in your original code by ensuring
        imputation strategies (mode) are learned ONLY on the training folds.
        """
        
        # Step 1: Define how to handle Categorical Data
        # 'ignore' handles unknown categories in the test set gracefully
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Step 2: Bundle preprocessing
        # We use ColumnTransformer to apply specific logic to specific columns
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, self.config.CAT_FEATURES)
            ],
            remainder='passthrough'  # Keep other (numerical) columns as they are
        )

        # Step 3: Create the Full Pipeline
        # ImbPipeline is used so SMOTE is applied correctly (only during training)
        pipeline = ImbPipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('smote', SMOTE(random_state=self.config.RANDOM_STATE)),
            ('classifier', classifier)
        ])
        
        return pipeline

    def run_cv_evaluation(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'rf') -> None:
        """
        Runs Stratified Cross-Validation and reports metrics.
        """
        
        # Select Model
        if model_type == 'xgb':
            clf = XGBClassifier(
                n_estimators=300, 
                max_depth=10, 
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=self.config.RANDOM_STATE
            )
        else:
            clf = RandomForestClassifier(
                n_estimators=500, 
                max_leaf_nodes=16, 
                random_state=self.config.RANDOM_STATE
            )

        pipeline = self.build_pipeline(clf)
        
        logger.info(f"Starting {self.config.CV_FOLDS}-Fold Cross-Validation with {model_type.upper()}...")
        
        cv = StratifiedKFold(n_splits=self.config.CV_FOLDS, shuffle=True, random_state=self.config.RANDOM_STATE)
        
        # Calculate metrics using cross_validate
        scoring = {'roc_auc': 'roc_auc', 'mcc': 'matthews_corrcoef', 'f1': 'f1'}
        from sklearn.model_selection import cross_validate
        
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        logger.info(f"Mean AUC: {np.mean(scores['test_roc_auc']):.4f}")
        logger.info(f"Mean MCC: {np.mean(scores['test_mcc']):.4f}")
        logger.info(f"Mean F1: {np.mean(scores['test_f1']):.4f}")
        
        # Final Step: Train the model on the full training data so it's ready for testing
        self.model = pipeline
        self.model.fit(X, y)
        logger.info("Final model fitted on full training set.")

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predictions and probabilities on new data.
        Handles all preprocessing internally via the pipeline.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call run_cv_evaluation first.")
        
        preds = self.model.predict(test_df)
        probs = self.model.predict_proba(test_df)
        
        results = pd.DataFrame({
            'Predicted_Label': preds,
            'Probability_0': probs[:, 0],
            'Probability_1': probs[:, 1]
        }, index=test_df.index)
        
        return results

def main():
    # Argument parsing allows this script to be run from the command line
    # Example: python clinical_pipeline.py --model xgb
    parser = argparse.ArgumentParser(description="Clinical Data Analysis Pipeline")
    parser.add_argument('--model', type=str, default='rf', choices=['rf', 'xgb'], help='Model type to use (rf or xgb)')
    args = parser.parse_args()

    # Initialize Config
    # If you want to change paths without touching code, you could add args for paths too
    config = Config()
    
    # Initialize Pipeline
    pipeline = ClinicalModelPipeline(config)

    try:
        # 1. Load Data
        train_df, test_df = pipeline.load_data()

        # 2. Split Features/Target
        X = train_df.drop(columns=[config.TARGET_COL])
        y = train_df[config.TARGET_COL]

        # 3. EDA (Saved to 'plots' folder)
        pipeline.visualize_distributions(train_df)

        # 4. Train & Evaluate
        pipeline.run_cv_evaluation(X, y, model_type=args.model)

        # 5. Predict on Test
        # Note: We pass the raw test dataframe. The pipeline handles missing columns/NAs automatically.
        predictions = pipeline.predict(test_df)
        
        # 6. Save Results
        output_file = 'predictions.csv'
        predictions.to_csv(output_file)
        logger.info(f"Predictions saved to {output_file}")

        # Feature Importance (Extracting from pipeline step)
        if args.model == 'rf':
            # We must access the steps by name to get to the feature importances
            model_step = pipeline.model.named_steps['classifier']
            preprocessor_step = pipeline.model.named_steps['preprocessor']
            
            # Get feature names from the OneHotEncoder
            feature_names = preprocessor_step.get_feature_names_out()
            importances = model_step.feature_importances_
            
            imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            imp_df = imp_df.sort_values(by='Importance', ascending=False).head(20)
            print("\nTop 20 Feature Importances:")
            print(imp_df)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        # Re-raise the exception so it shows the traceback
        raise

if __name__ == "__main__":
    main()