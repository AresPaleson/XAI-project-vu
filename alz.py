import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import pandas as pd
from PIL import Image
import io

# Constants
img_size = 224
batch_size = 32
num_classes = 4
epochs = 20

# Function to load and process data from parquet files
def load_data_from_parquet(file_path):
    df = pd.read_parquet(file_path)
    
    # Process images
    images = []
    for img_bytes in df['image']:
        # Convert bytes to PIL Image
        img = Image.open(io.BytesIO(img_bytes['bytes']))
        img = img.convert('RGB').resize((img_size, img_size))
        images.append(np.array(img))
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(df['label'])
    
    return images, labels

# Load dataset from local files
print("Loading dataset...")
train_images, train_labels = load_data_from_parquet('Dataset/Data/train.parquet')
test_images, test_labels = load_data_from_parquet('Dataset/Data/test.parquet')

# Normalize pixel values to [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert labels to one-hot encoding
train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)

# Split training data to create a validation set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    train_images, train_labels_one_hot, test_size=0.2, random_state=42)

print(f"\nDataset loaded successfully!")
print(f"Training set: {len(X_train)} images")
print(f"Validation set: {len(X_val)} images")
print(f"Test set: {len(test_images)} images")

# Data augmentation for training set
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size=batch_size
)

# No augmentation for validation set
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
val_generator = val_datagen.flow(
    X_val,
    y_val,
    batch_size=batch_size,
    shuffle=False
)

# Create test data generator
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_generator = test_datagen.flow(
    test_images,
    test_labels_one_hot,
    batch_size=batch_size,
    shuffle=False
)

# Model architecture
model = Sequential([
    Input(shape=(img_size, img_size, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
]

print("\nTraining the model...")
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // batch_size,
    validation_data=val_generator,
    validation_steps=len(X_val) // batch_size,
    epochs=epochs,
    callbacks=callbacks
)

# Evaluation on validation set
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"\nValidation Accuracy: {val_accuracy*100:.2f}%")

# Evaluation on test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")

# Save model
model.save('alzheimer_classifier.h5')
print("Model saved successfully!")