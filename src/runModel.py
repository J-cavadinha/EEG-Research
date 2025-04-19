import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import time
import joblib

def train_and_evaluate_models(input_file):
    try:
        data = pd.read_csv(input_file)
        print(f"Loaded data with {len(data)} trials")
        
        feature_cols = [col for col in data.columns if col not in ['trial_number', 'label']]
        X = data[feature_cols]
        y = data['label']
        
        print(f"Using {len(feature_cols)} features for training")
        
        models = {
            'SVM': (SVC(probability=True), {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }),
            'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42), {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }),
            'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            })
        }
        
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        results = {}
        for model_name, (model, param_grid) in models.items():
            print(f"\n{'-'*50}")
            print(f"Training {model_name}...")
            
            grid_search = GridSearchCV(
                model, 
                param_grid, 
                cv=kfold,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            results[model_name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'model': grid_search.best_estimator_
            }
            
            print(f"\n{model_name} results:")
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
            
            model_filename = f"{model_name.lower().replace(' ', '_')}_model_{time.strftime('%Y%m%d-%H%M%S')}.joblib"
            joblib.dump(grid_search.best_estimator_, model_filename)
            print(f"Model saved as: {model_filename}")
        
        print("\nModel Comparison:")
        print("-" * 50)
        for model_name, result in results.items():
            print(f"{model_name:20} Accuracy: {result['best_score']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"Error during model training: {e}")
        return None

if __name__ == "__main__":
    input_file = '/Users/joaomachado/Desktop/IC_V3/post_rfe_normal/normalized_features_20250417-012350.csv'
    results = train_and_evaluate_models(input_file)
    
    if results is not None:
        print("\nAll models trained successfully!")