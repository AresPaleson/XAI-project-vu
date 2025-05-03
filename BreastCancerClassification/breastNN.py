import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv('breast-cancer-data-cleaned.csv')

# Check column names to ensure correct handling
print("Original column names:", df.columns.tolist())

# Process target variable
y = df['class']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Process features
X = df.drop('class', axis=1)

# Define preprocessing function
def process_range_column(x):
    """Convert range strings to numeric values (midpoint of range)"""
    try:
        if pd.isna(x):
            return np.nan
        if isinstance(x, str) and '-' in x:
            parts = x.split('-')
            if len(parts) == 2:
                return np.mean([int(i) for i in parts])
        return float(x)
    except:
        return np.nan

# Define column types
range_columns = ['age', 'tumor-size', 'inv-nodes']
categorical_cols = ['menopause', 'node-caps', 'breast', 'breast-quad', 'irradiate']
numeric_cols = ['deg-malig']

# Split data first - before any transformations to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# Custom transformer for range columns
class RangeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.imputers = {}
        
    def fit(self, X, y=None):
        for col in self.columns:
            if col in X.columns:
                # Calculate median for imputation
                temp_col = X[col].apply(process_range_column)
                self.imputers[col] = SimpleImputer(strategy='median').fit(temp_col.to_frame())
        return self
        
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].apply(process_range_column)
                X[col] = self.imputers[col].transform(X[[col]]).ravel()
        return X

# Build the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), range_columns + numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='drop'  # Explicitly drop unhandled columns
)

# Full pipeline including range processing
full_pipeline = Pipeline([
    ('range_processor', RangeTransformer(range_columns)),
    ('preprocessor', preprocessor),
])

# Apply preprocessing
X_train_final = full_pipeline.fit_transform(X_train)
X_test_final = full_pipeline.transform(X_test)

# Check transformed shapes and missing values
print(f"\nTransformed training data shape: {X_train_final.shape}")
print(f"Transformed test data shape: {X_test_final.shape}")
print("NaN values in training:", np.isnan(X_train_final).sum())
print("NaN values in test:", np.isnan(X_test_final).sum())

# Build the model
input_shape = X_train_final.shape[1]
model = Sequential([
    Dense(64, activation='relu', input_shape=(input_shape,)),
    Dropout(0.2),  # Reduced dropout
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(2, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
]

# Add class weights to handle potential imbalance
class_counts = np.bincount(np.argmax(y_train, axis=1))
total = np.sum(class_counts)
class_weights = {i: total/count for i, count in enumerate(class_counts)}
print(f"\nClass weights: {class_weights}")

# Train the model
history = model.fit(
    X_train_final, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=16,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test_final, y_test, verbose=1)
print(f"\nTest accuracy: {test_acc:.4f}")

# Predictions
predictions = model.predict(X_test_final)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Calculate additional metrics
print("\nConfusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))

print("\nClassification Report:")
print(classification_report(
    true_classes, 
    predicted_classes,
    target_names=label_encoder.classes_
))

# Print training summary
print("\nTraining Summary:")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

# Save the final model
model.save('final_model.h5')
print("\nModel saved as 'final_model.h5'")