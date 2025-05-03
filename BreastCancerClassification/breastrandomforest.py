import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv('breast-cancer-data-cleaned.csv')

# Check column names
print("Original column names:", df.columns.tolist())

# Process target variable
y = df['class']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\nClasses: {label_encoder.classes_}")

# Process features
X = df.drop('class', axis=1)

# Define preprocessing function
def process_range_column(x):
    """Convert range strings to numeric values (midpoint of range)"""
    try:
        if isinstance(x, str) and '-' in x:
            parts = x.split('-')
            if len(parts) == 2:
                return np.mean([int(i) for i in parts])
        return float(x)
    except:
        # Return a default value or NaN for unparseable values
        return np.nan

# Define column types
range_columns = ['age', 'tumor-size', 'inv-nodes']
categorical_cols = ['menopause', 'node-caps', 'breast', 'breast-quad', 'irradiate']
numeric_cols = ['deg-malig']

# Split data first - before any transformations to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# First manually convert range columns to numeric values
def preprocess_X(X_df):
    result = X_df.copy()
    
    # Process range columns
    for col in range_columns:
        if col in result.columns:
            result[col] = result[col].apply(process_range_column)
    
    return result

# Apply manual preprocessing to both train and test sets
X_train_processed = preprocess_X(X_train)
X_test_processed = preprocess_X(X_test)

# Print column names to verify
print("\nProcessed column names:", X_train_processed.columns.tolist())

# Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), range_columns + numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='drop'  # This handles any columns not explicitly transformed
)

# Create a preprocessing and model pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define hyperparameter grid for tuning
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

# Perform grid search with cross-validation
print("\nPerforming grid search...")
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit the grid search to the data
grid_search.fit(X_train_processed, y_train)

# Print the best parameters
print("\nBest parameters:", grid_search.best_params_)

# Use the best model to make predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_processed)

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Feature importance
if hasattr(best_model['classifier'], 'feature_importances_'):
    # Get feature names after one-hot encoding
    feature_names = []
    for name, transformer, columns in preprocessor.transformers:  # Changed from transformers_ to transformers
        if name == 'num':
            feature_names.extend(columns)
        elif name == 'cat':
            for col in columns:
                # For each categorical feature, get the categories
                if hasattr(transformer, 'categories_'):
                    for cat in transformer.categories_:
                        feature_names.extend([f"{col}_{c}" for c in cat])
                else:
                    feature_names.append(col)
    
    # Get feature importances
    importances = best_model['classifier'].feature_importances_
    
    # If the lengths don't match, use generic feature names
    if len(importances) != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    # Sort feature importances
    indices = np.argsort(importances)[::-1]
    
    # Print top features
    print("\nTop features by importance:")
    for i, idx in enumerate(indices[:10]):  # Print top 10 features
        if i < len(feature_names):
            print(f"{feature_names[idx]}: {importances[idx]:.4f}")
        else:
            print(f"Feature {idx}: {importances[idx]:.4f}")

# In your current script, add these lines to save everything needed for explanations
import joblib
import pickle

# Save the trained model and related objects
# 1. Save the full pipeline (includes preprocessor and model)
joblib.dump(best_model, 'breast_cancer_pipeline.joblib')

# 2. Save the preprocessor separately (useful for LIME/SHAP)
preprocessor_fitted = best_model.named_steps['preprocessor']
joblib.dump(preprocessor_fitted, 'breast_cancer_preprocessor.joblib')

# 3. Save the classifier separately
classifier = best_model.named_steps['classifier']
joblib.dump(classifier, 'breast_cancer_classifier.joblib')

# 4. Save the label encoder for interpreting class names
with open('breast_cancer_label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# 5. Save sample data for reference
with open('breast_cancer_sample_data.pkl', 'wb') as f:
    pickle.dump({
        'X_train': X_train_processed[:5],  # Just save a few samples
        'X_test': X_test_processed[:5],
        'column_names': X_train.columns.tolist(),
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols + range_columns
    }, f)

print("All model components saved for XAI analysis")