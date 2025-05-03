import shap
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import Image
import io
import pandas as pd

# Create directory for saving explanation images
os.makedirs('explanations', exist_ok=True)

# Constants
img_size = 224
batch_size = 32
num_classes = 4

# Function to preprocess images
def preprocess_image(image):
    if isinstance(image, bytes):  # If it's bytes from parquet
        image = Image.open(io.BytesIO(image))
    if hasattr(image, 'convert'):  # If it's a PIL image
        image = image.convert('RGB').resize((img_size, img_size))
        image = np.array(image)
    return image / 255.0

# Load the trained model
try:
    model = load_model('best_model.h5')  # Try loading the best model first
    print("Best model loaded successfully!")
except:
    model = load_model('alzheimer_classifier.h5')  # Fallback to the final model
    print("Fallback model loaded successfully!")

# Class names for reference
class_names = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']

# Load test data from local parquet file
def load_local_test_data(parquet_path, num_samples=20):
    df = pd.read_parquet(parquet_path)
    
    if num_samples:
        df = df.sample(n=num_samples, random_state=40)  # << Randomly sample
    
    test_images_raw = []
    test_labels = []
    
    for _, row in df.iterrows():
        # Handle different possible column names
        img_bytes = row.get('image', row.get('bytes', None))
        if isinstance(img_bytes, dict):  # If stored as dictionary with 'bytes' key
            img_bytes = img_bytes['bytes']
        
        test_images_raw.append(img_bytes)
        test_labels.append(row['label'])
    
    test_images = np.array([preprocess_image(img) for img in test_images_raw])
    test_labels = np.array(test_labels)
    
    return test_images, test_labels

# Load local test data (adjust path as needed)
test_images, test_labels = load_local_test_data('Dataset/Data/test.parquet', num_samples=20)

# Get predictions for these samples
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

print(f"Loaded {len(test_images)} test images for explanation")

# SHAP Explanation - Modified version that doesn't require OpenCV
def explain_with_shap(model, images, class_names, num_explanations=5):
    """
    Generate SHAP explanations for the model predictions
    
    Args:
        model: Trained Keras model
        images: Numpy array of preprocessed images
        class_names: List of class names
        num_explanations: Number of explanations to generate and display
    """
    # Select a subset of images to explain
    if len(images) > num_explanations:
        images = images[:num_explanations]
    
    # Create a simpler explainer that doesn't require OpenCV
    explainer = shap.GradientExplainer(model, images)
    
    print("Generating SHAP explanations... (This may take a while)")
    
    # Compute SHAP values for all classes
    shap_values = explainer.shap_values(images)
    
    # Plot explanations for each image
    for i in range(len(images)):
        plt.figure(figsize=(15, 5))
        
        # Get the true and predicted class
        true_class = class_names[test_labels[i]]
        pred_class = class_names[np.argmax(predictions[i])]
        confidence = np.max(predictions[i])
        
        # Plot the original image
        plt.subplot(1, 3, 1)
        plt.imshow(images[i])
        plt.title(f"Original\nTrue: {true_class}\nPred: {pred_class} ({confidence:.2f})")
        plt.axis('off')
        
        # Plot SHAP values for predicted class
        plt.subplot(1, 3, 2)
        shap_image = shap_values[np.argmax(predictions[i])][i]
        plt.imshow(images[i], alpha=0.15)
        plt.imshow(shap_image, cmap='jet', alpha=0.85)
        plt.colorbar(label='SHAP value', orientation='horizontal')
        plt.title(f"SHAP - {pred_class}")
        plt.axis('off')
        
        # Plot SHAP values for true class or second prediction
        plt.subplot(1, 3, 3)
        if test_labels[i] != np.argmax(predictions[i]):
            shap_image = shap_values[test_labels[i]][i]
            title = f"SHAP - {true_class}"
        else:
            second_class = np.argsort(predictions[i])[-2]
            shap_image = shap_values[second_class][i]
            title = f"SHAP - {class_names[second_class]}"
        
        plt.imshow(images[i], alpha=0.15)
        plt.imshow(shap_image, cmap='jet', alpha=0.85)
        plt.colorbar(label='SHAP value', orientation='horizontal')
        plt.title(title)
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save the explanation
        explanation_path = f"explanations/explanation_{i}.png"
        plt.savefig(explanation_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"Explanation saved to {explanation_path}")

# Generate explanations for the first 5 images
explain_with_shap(model, test_images, class_names, num_explanations=5)

print("\nSHAP explanations completed successfully!")
print(f"Explanation images saved to 'explanations' directory")