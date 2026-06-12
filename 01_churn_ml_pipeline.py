
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# Import preprocessing & modeling
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Import evaluation metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)

# Import explainability
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Install with: pip install shap")

import pickle
import json
from datetime import datetime
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / 'European_Bank.csv'
OUTPUT_DIR = BASE_DIR

# ============================================================================
# STEP 1: DATA LOADING & INITIAL EXPLORATION
# ============================================================================

def load_and_explore_data(filepath):
    """Load data and perform initial exploration"""
    print("=" * 80)
    print("STEP 1: DATA LOADING & EXPLORATION")
    print("=" * 80)
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found: {filepath}\n"
            f"Place 'European_Bank.csv' in the project directory: {BASE_DIR}"
        )
    df = pd.read_csv(filepath)
    
    print(f"\n✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn names and types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nTarget variable distribution:\n{df['Exited'].value_counts()}")
    print(f"Churn rate: {df['Exited'].mean()*100:.2f}%")
    
    return df

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================

def preprocess_data(df):
    """Handle missing values, encode categoricals, remove non-informative features"""
    print("\n" + "=" * 80)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 80)
    
    df_processed = df.copy()
    
    # Remove non-informative features
    print("\n✓ Removing non-informative features: CustomerId, Surname, Year")
    df_processed = df_processed.drop(['CustomerId', 'Surname', 'Year'], axis=1)
    
    # Handle missing values (if any)
    if df_processed.isnull().sum().sum() > 0:
        print(f"\n⚠️ Missing values detected. Handling...")
        df_processed = df_processed.fillna(df_processed.median(numeric_only=True))
    else:
        print("\n✓ No missing values found")
    
    # Separate target variable
    X = df_processed.drop('Exited', axis=1)
    y = df_processed['Exited']
    
    # One-hot encode categorical variables
    print("\n✓ Encoding categorical variables...")
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"  Categorical columns: {categorical_cols}")
    
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
    
    print(f"\n✓ Data shape after encoding: {X_encoded.shape}")
    print(f"  Features: {list(X_encoded.columns)}")
    
    return X_encoded, y, X_encoded.columns.tolist()

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================

def engineer_features(X, y):
    """Create derived features for improved model performance"""
    print("\n" + "=" * 80)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 80)
    
    X_engineered = X.copy()
    
    # Feature 1: Balance-to-Salary Ratio
    X_engineered['Balance_Salary_Ratio'] = X_engineered['Balance'] / (X_engineered['EstimatedSalary'] + 1)
    
    # Feature 2: Product Density (products per year of tenure)
    X_engineered['Product_Density'] = X_engineered['NumOfProducts'] / (X_engineered['Tenure'] + 1)
    
    # Feature 3: Engagement-Product Interaction
    X_engineered['Engagement_Product_Score'] = (
        X_engineered['IsActiveMember'] * X_engineered['NumOfProducts']
    )
    
    # Feature 4: Age-Tenure Interaction
    X_engineered['Age_Tenure_Interaction'] = X_engineered['Age'] * X_engineered['Tenure']
    
    # Feature 5: Age Group (derived feature)
    X_engineered['Age_Group_Young'] = (X_engineered['Age'] < 30).astype(int)
    X_engineered['Age_Group_Senior'] = (X_engineered['Age'] >= 55).astype(int)
    
    # Feature 6: Credit Score Normalization
    X_engineered['CreditScore_Normalized'] = (X_engineered['CreditScore'] - X_engineered['CreditScore'].min()) / \
                                               (X_engineered['CreditScore'].max() - X_engineered['CreditScore'].min())
    
    print(f"\n✓ Created 6 engineered features:")
    print("  1. Balance_Salary_Ratio")
    print("  2. Product_Density")
    print("  3. Engagement_Product_Score")
    print("  4. Age_Tenure_Interaction")
    print("  5. Age_Group_Young")
    print("  6. Age_Group_Senior")
    print(f"\n✓ Total features after engineering: {X_engineered.shape[1]}")
    
    return X_engineered

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT & SCALING
# ============================================================================

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """Stratified train-test split and feature scaling"""
    print("\n" + "=" * 80)
    print("STEP 4: TRAIN-TEST SPLIT & SCALING")
    print("=" * 80)
    
    # Stratified split (preserve class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y
    )
    
    print(f"\n✓ Stratified train-test split (80-20):")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"  Train churn rate: {y_train.mean()*100:.2f}%")
    print(f"  Test churn rate: {y_test.mean()*100:.2f}%")
    
    # Scale numerical features
    scaler = StandardScaler()
    
    # Identify numerical columns (excluding engineered features that are already normalized)
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print(f"\n✓ Feature scaling applied to {len(numerical_cols)} numerical features")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ============================================================================
# STEP 5: MODEL DEVELOPMENT & TRAINING
# ============================================================================

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and evaluate performance"""
    print("\n" + "=" * 80)
    print("STEP 5: MODEL DEVELOPMENT & TRAINING")
    print("=" * 80)
    
    models = {}
    results = {}
    
    # Model 1: Logistic Regression (Baseline)
    print("\n[1/5] Training Logistic Regression (Baseline)...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lr_model
    
    # Model 2: Decision Tree
    print("[2/5] Training Decision Tree...")
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=42, min_samples_split=20)
    dt_model.fit(X_train, y_train)
    models['Decision Tree'] = dt_model
    
    # Model 3: Random Forest
    print("[3/5] Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, 
                                      random_state=42, n_jobs=-1, min_samples_split=20)
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    
    # Model 4: Gradient Boosting
    print("[4/5] Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, 
                                          learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train)
    models['Gradient Boosting'] = gb_model
    
    # Model 5: XGBoost (if available)
    try:
        from xgboost import XGBClassifier
        print("[5/5] Training XGBoost...")
        xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                 random_state=42, eval_metric='logloss', use_label_encoder=False)
        xgb_model.fit(X_train, y_train)
        models['XGBoost'] = xgb_model
    except:
        print("[5/5] XGBoost not available. Skipping...")
    
    print(f"\n✓ {len(models)} models trained successfully")
    
    return models

# ============================================================================
# STEP 6: MODEL EVALUATION
# ============================================================================

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Comprehensive model evaluation with multiple metrics"""
    print("\n" + "=" * 80)
    print("STEP 6: MODEL EVALUATION")
    print("=" * 80)
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_pred_proba)
        
        # Store results
        results[model_name] = {
            'model': model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred': y_test_pred,
            'y_pred_proba': y_test_pred_proba,
            'cm': confusion_matrix(y_test, y_test_pred)
        }
        
        print(f"\nTraining Accuracy:  {train_accuracy:.4f}")
        print(f"Testing Accuracy:   {test_accuracy:.4f}")
        print(f"Precision:          {precision:.4f}")
        print(f"Recall:             {recall:.4f}")
        print(f"F1-Score:           {f1:.4f}")
        print(f"ROC-AUC:            {roc_auc:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(results[model_name]['cm'])
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_test_pred))
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Train Acc': [results[m]['train_accuracy'] for m in results],
        'Test Acc': [results[m]['test_accuracy'] for m in results],
        'Precision': [results[m]['precision'] for m in results],
        'Recall': [results[m]['recall'] for m in results],
        'F1-Score': [results[m]['f1_score'] for m in results],
        'ROC-AUC': [results[m]['roc_auc'] for m in results]
    }).round(4)
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Identify best model
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    print(f"\n⭐ Best Model (by F1-Score): {best_model_name}")
    
    return results, best_model_name

# ============================================================================
# STEP 7: FEATURE IMPORTANCE & EXPLAINABILITY
# ============================================================================

def explain_models(results, best_model_name, feature_names):
    """Extract feature importance and SHAP values"""
    print("\n" + "=" * 80)
    print("STEP 7: MODEL EXPLAINABILITY")
    print("=" * 80)
    
    best_model = results[best_model_name]['model']
    
    # Feature importance (tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n✓ Feature Importance (Top 15 features):")
        print(feature_importance_df.head(15).to_string(index=False))
        
        return feature_importance_df
    
    elif best_model_name == 'Logistic Regression':
        importances = np.abs(best_model.coef_[0])
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n✓ Feature Coefficients (Top 15 features):")
        print(feature_importance_df.head(15).to_string(index=False))
        
        return feature_importance_df
    
    return None

# ============================================================================
# STEP 8: SAVE ARTIFACTS FOR STREAMLIT
# ============================================================================

def save_artifacts(models, results, best_model_name, scaler, feature_names, feature_importance_df):
    """Save trained models and metadata for Streamlit app"""
    print("\n" + "=" * 80)
    print("STEP 8: SAVING ARTIFACTS")
    print("=" * 80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save best model
    best_model = results[best_model_name]['model']
    with open(OUTPUT_DIR / 'best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("✓ Best model saved: best_model.pkl")
    
    # Save scaler
    with open(OUTPUT_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Scaler saved: scaler.pkl")
    
    # Save feature names
    with open(OUTPUT_DIR / 'feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    print("✓ Feature names saved: feature_names.pkl")
    
    # Save model results metadata
    metadata = {
        'best_model': best_model_name,
        'timestamp': datetime.now().isoformat(),
        'model_metrics': {
            model_name: {
                'test_accuracy': results[model_name]['test_accuracy'],
                'precision': results[model_name]['precision'],
                'recall': results[model_name]['recall'],
                'f1_score': results[model_name]['f1_score'],
                'roc_auc': results[model_name]['roc_auc']
            }
            for model_name in results
        }
    }
    
    with open(OUTPUT_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print("✓ Metadata saved: model_metadata.json")
    
    # Save feature importance
    feature_importance_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
    print("✓ Feature importance saved: feature_importance.csv")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "BANK CUSTOMER CHURN PREDICTION - ML PIPELINE" + " "*18 + "║")
    print("╚" + "="*78 + "╝")
    
    # Load and explore data
    df = load_and_explore_data(DATA_FILE)
    
    # Preprocess data
    X_encoded, y, feature_names_initial = preprocess_data(df)
    
    # Feature engineering
    X_engineered = engineer_features(X_encoded, y)
    feature_names = X_engineered.columns.tolist()
    
    # Train-test split and scaling
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X_engineered, y)
    
    # Train models
    models = train_models(X_train, X_test, y_train, y_test)
    
    # Evaluate models
    results, best_model_name = evaluate_models(models, X_train, X_test, y_train, y_test)
    
    # Feature importance
    feature_importance_df = explain_models(results, best_model_name, feature_names)
    
    # Save artifacts
    save_artifacts(models, results, best_model_name, scaler, feature_names, feature_importance_df)
    
    print("\n" + "="*80)
    print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nGenerated files:")
    print("  • best_model.pkl")
    print("  • scaler.pkl")
    print("  • feature_names.pkl")
    print("  • model_metadata.json")
    print("  • feature_importance.csv")
    print("\nNext: Run the Streamlit app with:")
    print("  streamlit run 02_streamlit_dashboard.py")
    print("="*80 + "\n")