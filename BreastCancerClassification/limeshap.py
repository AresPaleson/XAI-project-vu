# xai_explanations.py
import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import shap
from sklearn.inspection import permutation_importance

def load_model_components():
    """Load all saved model components"""
    # Load the full pipeline
    pipeline = joblib.load('breast_cancer_pipeline.joblib')
    
    # Load individual components
    preprocessor = joblib.load('breast_cancer_preprocessor.joblib')
    classifier = joblib.load('breast_cancer_classifier.joblib')
    
    # Load label encoder
    with open('breast_cancer_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load sample data
    with open('breast_cancer_sample_data.pkl', 'rb') as f:
        sample_data = pickle.load(f)
        
    return {
        'pipeline': pipeline,
        'preprocessor': preprocessor,
        'classifier': classifier,
        'label_encoder': label_encoder,
        'sample_data': sample_data
    }

def setup_lime_explainer(components):
    """Set up LIME explainer with proper categorical handling"""
    X_sample = components['sample_data']['X_train']
    feature_names = components['sample_data']['column_names']
    categorical_cols = components['sample_data']['categorical_cols']
    
    # Create a preprocessed training dataset
    X_sample_processed = components['preprocessor'].transform(X_sample)
    
    # Convert to numpy array if it's not already
    if not isinstance(X_sample_processed, np.ndarray):
        X_sample_processed = X_sample_processed.toarray()
    
    # Get processed feature names
    processed_feature_names = get_processed_feature_names(components['preprocessor'])
    
    # Get categorical feature indices after preprocessing
    categorical_feature_indices = []
    categorical_names = {}
    
    # Create LIME explainer with preprocessed data
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_sample_processed,
        feature_names=processed_feature_names,
        class_names=components['label_encoder'].classes_,
        categorical_features=categorical_feature_indices,
        categorical_names=categorical_names,
        mode='classification',
        discretize_continuous=True,
        kernel_width=3,
        verbose=True
    )
    
    return explainer

def generate_lime_explanation(explainer, components, instance_idx=0):
    """Generate and display LIME explanation for a specific instance"""
    # Get the instance and preprocess it
    instance = components['sample_data']['X_train'].iloc[instance_idx:instance_idx+1]
    instance_processed = components['preprocessor'].transform(instance)
    
    if not isinstance(instance_processed, np.ndarray):
        instance_processed = instance_processed.toarray()
    
    # Prediction function for LIME
    def predict_fn(X):
        return components['classifier'].predict_proba(X)
    
    # Generate explanation
    explanation = explainer.explain_instance(
        instance_processed[0], 
        predict_fn,
        num_features=10,
        top_labels=1
    )
    
    # Save explanation as HTML
    explanation.save_to_file('lime_explanation.html')
    
    print("LIME explanation saved to lime_explanation.html")
    return explanation

def setup_shap_explainer(components):
    """Set up SHAP explainer using preprocessed data"""
    # Preprocess the background data
    background = components['preprocessor'].transform(
        components['sample_data']['X_train'].iloc[:50]
    )
    
    # Initialize SHAP explainer
    explainer = shap.Explainer(
        model=components['classifier'].predict_proba,
        masker=background,
        feature_names=get_processed_feature_names(components['preprocessor']),
        output_names=components['label_encoder'].classes_
    )
    
    return explainer

def get_processed_feature_names(preprocessor):
    """Get feature names after preprocessing"""
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        elif name == 'cat':
            if hasattr(transformer, 'get_feature_names_out'):
                feature_names.extend(transformer.get_feature_names_out(cols))
            else:
                feature_names.extend([f"{col}_{i}" for i, col in enumerate(cols)])
    return feature_names

def generate_shap_explanations(explainer, components, num_examples=5):
    """Generate and display SHAP explanations"""
    # Set random seed for reproducibility
    np.random.seed(42)  # Set global seed instead
    
    # Get and preprocess examples
    examples = components['preprocessor'].transform(
        components['sample_data']['X_test'].iloc[:num_examples]
    )
    
    # Calculate SHAP values
    shap_values = explainer(examples)
    
    # Plot summary
    plt.figure()
    shap.summary_plot(
        shap_values[:,:,1].values,  # For class 1 (positive class)
        examples,
        feature_names=explainer.feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()
    
    # Plot individual force plots
    for i in range(num_examples):
        plt.figure()
        shap.plots.force(
            shap_values[i][:,1],
            matplotlib=True,
            show=False,
            feature_names=explainer.feature_names
        )
        plt.tight_layout()
        plt.savefig(f'shap_force_plot_{i}.png')
        plt.close()
    
    print(f"SHAP visualizations saved (summary and {num_examples} force plots)")
    return shap_values
def main():
    print("Loading model components...")
    components = load_model_components()
    
    print("\nGenerating LIME explanation...")
    lime_explainer = setup_lime_explainer(components)
    lime_explanation = generate_lime_explanation(lime_explainer, components)
    
    print("\nGenerating SHAP explanations...")
    shap_explainer = setup_shap_explainer(components)
    shap_values = generate_shap_explanations(shap_explainer, components)
    
    print("\nXAI explanations generated successfully!")

if __name__ == "__main__":
    main()