class streamlit:
    ...

streamlit
tensorflow
datasets
ultralytics
scikit-learn
matplotlib
pillow



import tensorflow as tf
import datasets
import ultralytics
import sklearn
import matplotlib

print("TensorFlow:", tf.__version__)
print("Ultralytics:", ultralytics.__version__)
print("All libraries imported successfully!")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PIL import Image
import json
import os

print("All imports completed successfully!")
print("TensorFlow version:", tf.__version__)

# Check available hardware
print("Available devices:", tf.config.list_physical_devices())



# Download the GTSRB road sign dataset

dataset = load_dataset("tanganke/gtsrb")

print("Dataset downloaded successfully!")
print(dataset)

# Check dataset structure
print("\nTraining columns:", dataset["train"].column_names)
print("Testing columns:", dataset["test"].column_names)

print("\nNumber of training images:", len(dataset["train"]))
print("Number of testing images:", len(dataset["test"]))

print("\nNumber of road sign classes:",
      len(dataset["train"].features["label"].names))



# Display 9 random road sign images

import random

class_names = dataset["train"].features["label"].names

plt.figure(figsize=(10, 8))

for i in range(9):
    index = random.randint(0, len(dataset["train"]) - 1)

    image = dataset["train"][index]["image"]
    label = dataset["train"][index]["label"]

    plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(class_names[label], fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.show()



# Convert images into numerical arrays

IMG_SIZE = 32

def prepare_images(split):
    images = []
    labels = []

    for sample in split:
        img = sample["image"].convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        images.append(np.asarray(img, dtype=np.uint8))
        labels.append(sample["label"])

    return np.stack(images), np.array(labels)

# Prepare training and testing data
X_all, y_all = prepare_images(dataset["train"])
X_test, y_test = prepare_images(dataset["test"])

print("Training images:", X_all.shape)
print("Testing images:", X_test.shape)

# Divide training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_all,
    y_all,
    test_size=0.15,
    random_state=42,
    stratify=y_all
)

# Remove unnecessary arrays to save memory
del X_all, y_all

print("\nTraining set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Testing set:", X_test.shape)



# Create the Convolutional Neural Network

from tensorflow.keras import layers, models

model = models.Sequential([

    # Input layer: normalize image pixels
    layers.Input(shape=(32, 32, 3)),
    layers.Rescaling(1./255),

    # First convolution block
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Second convolution block
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Third convolution block
    layers.Conv2D(128, (3, 3), activation="relu"),

    # Classification layers
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),

    # 43 road sign categories
    layers.Dense(43, activation="softmax")
])

# Configure the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display architecture
model.summary()



# Stop training if validation performance stops improving

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# Train the CNN
history = model.fit(
    X_train,
    y_train,
    epochs=12,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

print("AI road sign model training completed!")



# Plot training and validation accuracy

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("AI Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("AI Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()



# Evaluate model on unseen test data

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("--------------------------------")
print("ROAD SIGN MODEL EVALUATION")
print("--------------------------------")

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")



# Generate predictions on the test dataset

y_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

# Print classification report
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
plt.imshow(cm, cmap="Blues")
plt.title("Road Sign Recognition - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.colorbar()
plt.tight_layout()
plt.show()



# Create project folder
os.makedirs("models", exist_ok=True)

# Save trained AI model
model.save("models/road_sign_model.keras")

# Save class labels
with open("models/road_sign_labels.json", "w") as f:
    json.dump(class_names, f)

print("Road sign AI model saved successfully!")



from google.colab import files
from IPython.display import display

# Upload a road sign photograph
uploaded = files.upload()

# Get uploaded image
filename = next(iter(uploaded))

# Open and resize image
img = Image.open(filename).convert("RGB")
img_resized = img.resize((32, 32))

# Convert into model input
img_array = np.array(img_resized, dtype=np.float32)
img_array = np.expand_dims(img_array, axis=0)

# Predict road sign
prediction = model.predict(img_array, verbose=0)[0]

predicted_class = int(np.argmax(prediction))
confidence = float(prediction[predicted_class]) * 100

# Display results
print("AI ROAD SIGN RECOGNITION")
print("-----------------------")

display(img)

if confidence >= 60:
    print("Detected sign:", class_names[predicted_class])
    print(f"Confidence: {confidence:.2f}%")
else:
    print("Uncertain prediction. Please upload a clearer image.")
    print(f"Highest model confidence: {confidence:.2f}%")


test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=1
)

print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Test loss: {test_loss:.4f}")

import numpy as np
import matplotlib.pyplot as plt

predictions = model.predict(X_test[:12], verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(12, 8))

for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(X_test[i])

    actual = class_names[int(y_test[i])]
    predicted = class_names[int(predicted_labels[i])]

    plt.title(
        f"Actual: {actual}\nPredicted: {predicted}",
        fontsize=8
    )
    plt.axis("off")

plt.tight_layout()
plt.show()

from ultralytics import YOLO
from google.colab import files
from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt

# Load a small pre-trained YOLO model
pedestrian_model = YOLO("yolov8n.pt")

print("Pedestrian detection model loaded!")

uploaded = files.upload()

filename = next(iter(uploaded))

print("Uploaded:", filename)

image = Image.open(filename).convert("RGB")
display(image)

results = pedestrian_model.predict(
    source=filename,
    conf=0.35,
    verbose=False
)

result = results[0]

pedestrian_count = 0
pedestrian_details = []

for box in result.boxes:
    class_id = int(box.cls[0])

    # COCO class 0 represents a person
    if class_id == 0:
        pedestrian_count += 1
        confidence = float(box.conf[0]) * 100

        pedestrian_details.append(confidence)

print("AI PEDESTRIAN DETECTION REPORT")
print("--------------------------------")
print("Pedestrians detected:", pedestrian_count)

for i, confidence in enumerate(pedestrian_details, start=1):
    print(f"Person {i}: {confidence:.2f}% confidence")

annotated_image = result.plot()

plt.figure(figsize=(12, 8))
plt.imshow(annotated_image[:, :, ::-1])
plt.title("AI Pedestrian Detection")
plt.axis("off")
plt.show()

import os

os.makedirs("outputs", exist_ok=True)

output_path = "outputs/pedestrian_detection.jpg"

result.save(filename=output_path)

print("Detection image saved:", output_path)

from google.colab import files

files.download("models/road_sign_model.keras")
files.download("models/road_sign_labels.json")

from google.colab import files

files.download("yolov8n.pt")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("ROAD SIGN MODEL RESULTS")
print("-----------------------")
print(f"Accuracy: {test_accuracy * 100:.2f}%")
print(f"Loss: {test_loss:.4f}")

import numpy as np
import matplotlib.pyplot as plt

predictions = model.predict(X_test[:12], verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(12, 8))

for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(X_test[i])

    actual = class_names[int(y_test[i])]
    predicted = class_names[int(predicted_labels[i])]

    plt.title(f"Actual: {actual}\nPredicted: {predicted}", fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.show()

from ultralytics import YOLO
from google.colab import files

detector = YOLO("yolov8n.pt")

uploaded = files.upload()
filename = next(iter(uploaded))

results = detector.predict(
    source=filename,
    classes=[0],
    conf=0.35,
    verbose=False
)

print(results[0].verbose())
results[0].show()

