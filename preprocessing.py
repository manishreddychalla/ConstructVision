# import os
# import numpy as np
# import cv2

# data = []
# labels = []
# classes = ['10', '30', '50', '70', '90', '100']

# for label in classes:
#     folder_path = f"dataset/{label}/"
#     for file in os.listdir(folder_path):
#         if file.endswith('.jpg') or file.endswith('.png'):
#             img_path = os.path.join(folder_path, file)
#             img = cv2.imread(img_path)
#             img = cv2.resize(img, (128, 128))  # Resize
#             img = img / 255.0                  # Normalize
#             data.append(img)
#             labels.append(classes.index(label))  # Convert class label to number
# X = np.array(data)
# y = np.array(labels)

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
#     MaxPooling2D(2,2),
#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D(2,2),
#     Flatten(),
#     Dense(64, activation='relu'),
#     Dropout(0.2),
#     Dense(6, activation='softmax')  # 6 classes → 10%, 30%, ... 100%
# ])

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1)

# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {accuracy*100:.2f}%")

# def predict_progress(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (128, 128))
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)

#     pred = model.predict(img)
#     class_index = np.argmax(pred)
#     progress_classes = ['10%', '30%', '50%', '70%', '90%', '100%']
#     return progress_classes[class_index]

# # Example:
# # print(predict_progress("new_construction_image.jpg"))

# construction_progress_ml.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# ---------------------- Configuration ----------------------
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20
CLASSES = ['0%', '10%', '30%', '50%', '70%', '90%', '100%']  # Progress stages
NUM_CLASSES = len(CLASSES)
DATASET_PATH = 'dataset'  # Folder with class subfolders

# ---------------------- Load and Preprocess Dataset ----------------------
def load_dataset():
    data = []
    labels = []
    for label in CLASSES:
        path = os.path.join(DATASET_PATH, label)
        for file in os.listdir(path):
            if file.endswith(('.jpeg', '.jpg')):
                img_path = os.path.join(path, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, IMAGE_SIZE)
                img = img / 255.0
                data.append(img)
                labels.append(CLASSES.index(label))
    return np.array(data), np.array(labels)

# ---------------------- Build Small CNN Model ----------------------
def build_small_cnn():
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------------- Build Transfer Learning Model ----------------------
def build_transfer_model():
    base_model = MobileNetV2(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0002), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------------- Main Pipeline ----------------------
def main():
    print("Loading dataset...")
    X, y = load_dataset()
    import collections
    print(collections.Counter(y))


    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("Applying data augmentation...")
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    print("Building model...")
    # model = build_small_cnn()
    model = build_transfer_model()

    print("Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test)
    )

    print("Evaluating model...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    print("Saving model...")
    model.save("construction_progress_model.h5")
    print("Model saved as 'construction_progress_model.h5'")

# ---------------------- Predict Function ----------------------
# def predict_progress(image_path, model_path="construction_progress_model.h5"):
#     from tensorflow.keras.models import load_model
#     model = load_model(model_path)
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, IMAGE_SIZE)
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)
#     pred = model.predict(img)
#     predicted_class = np.argmax(pred)
#     if predicted_class ==0:
#         print("Upload valid Image")
#     return CLASSES[predicted_class]
def predict_progress(image_path, model_path="construction_progress_model.h5", threshold=0.2):
    from tensorflow.keras.models import load_model
    import os

    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] Image file '{image_path}' not found.")

    # Load the model
    model = load_model(model_path)

    # Read and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"[ERROR] Image at path '{image_path}' could not be read. Please check the file format.")

    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)
    predicted_class = np.argmax(pred)
    confidence = np.max(pred)

    # If confidence is too low
    if confidence < threshold:
        return "⚠️ Prediction confidence too low — please upload a valid construction progress image."

    # If predicted class is '0%'
    if predicted_class == 0:
        print(" Upload a picture related to construction — image predicted as 0% progress.")
        return f"Predicted Progress: {CLASSES[predicted_class]} (Confidence: {confidence:.2f})"

    return f"✅ Predicted Progress: {CLASSES[predicted_class]} (Confidence: {confidence:.2f})"


    # Predict
    pred = model.predict(img)
    predicted_class = np.argmax(pred)
    confidence = np.max(pred)

    if confidence < threshold:
        return "⚠️ Prediction confidence too low — please upload a valid construction progress image."
    else:
        return f"✅ Predicted Progress: {CLASSES[predicted_class]}% (Confidence: {confidence:.2f})"


# ---------------------- Run ----------------------
if __name__ == "__main__":
    main()
    print(predict_progress("test12.jpeg"))
# import os
# import cv2
# import numpy as np
# from collections import Counter
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight
# from tensorflow.keras.models import Sequential, Model, load_model
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam

# # ---------------------- Configuration ----------------------
# IMAGE_SIZE = (224, 224)  # Recommended for MobileNetV2
# BATCH_SIZE = 16
# EPOCHS_INITIAL = 15
# EPOCHS_FINE_TUNE = 10
# CLASSES = ['10', '30', '50', '70', '90', '100']
# NUM_CLASSES = len(CLASSES)
# DATASET_PATH = 'dataset'

# # ---------------------- Load and Preprocess Dataset ----------------------
# def load_dataset():
#     data, labels = [], []
#     for label in CLASSES:
#         path = os.path.join(DATASET_PATH, label)
#         if not os.path.exists(path):
#             continue
#         for file in os.listdir(path):
#             if file.lower().endswith(('.jpg', '.jpeg', '.png')):
#                 img_path = os.path.join(path, file)
#                 img = cv2.imread(img_path)
#                 img = cv2.resize(img, IMAGE_SIZE)
#                 img = img / 255.0
#                 data.append(img)
#                 labels.append(CLASSES.index(label))
#     return np.array(data), np.array(labels)

# # ---------------------- Build Transfer Learning Model ----------------------
# # def build_transfer_model(trainable=False):
# #     base_model = MobileNetV2(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), include_top=False, weights='imagenet')
# #     base_model.trainable = trainable
# #     x = base_model.output
# #     x = GlobalAveragePooling2D()(x)
# #     x = Dense(128, activation='relu')(x)
# #     x = Dropout(0.3)(x)
# #     predictions = Dense(NUM_CLASSES, activation='softmax')(x)
# #     model = Model(inputs=base_model.input, outputs=predictions)
# #     model.compile(optimizer=Adam(learning_rate=1e-4 if not trainable else 1e-5),
# #                   loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# #     return model
# def build_transfer_model(trainable=False):
#     base_model = MobileNetV2(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), include_top=False, weights='imagenet')
#     base_model.trainable = trainable  # Initially False

#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.3)(x)
#     predictions = Dense(NUM_CLASSES, activation='softmax')(x)

#     model = Model(inputs=base_model.input, outputs=predictions)
#     model.compile(optimizer=Adam(learning_rate=1e-4 if not trainable else 1e-5),
#                   loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     return model, base_model


# # ---------------------- Plot Training Graphs ----------------------
# def plot_training(history):
#     plt.figure(figsize=(12, 5))

#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'], label='Train Acc')
#     plt.plot(history.history['val_accuracy'], label='Val Acc')
#     plt.title('Accuracy')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Val Loss')
#     plt.title('Loss')
#     plt.legend()

#     plt.tight_layout()
#     plt.savefig("training_plot.png")
#     plt.show()

# # ---------------------- Main Pipeline ----------------------
# def main():
#     print("📥 Loading dataset...")
#     X, y = load_dataset()
#     class_counts = Counter(y)
#     print("📊 Class distribution:", class_counts)

#     # Check for any class with < 2 samples
#     for label, count in class_counts.items():
#         if count < 2:
#             print(f"⚠ Class '{CLASSES[label]}' has only {count} sample(s). Add more to enable stratified splitting.")
#             return

#     print("📂 Splitting dataset...")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

#     print("🧪 Applying data augmentation...")
#     datagen = ImageDataGenerator(
#         rotation_range=30,
#         zoom_range=0.3,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         brightness_range=[0.7, 1.3],
#         horizontal_flip=True,
#         fill_mode='nearest'
#     )
#     datagen.fit(X_train)

#     print("🛠 Building transfer model (initial training)...")
#     # model = build_transfer_model(trainable=False)
#     model, base_model = build_transfer_model(trainable=False)


#     class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
#     class_weights_dict = dict(enumerate(class_weights))

#     print("🚀 Training transfer model...")
#     history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
#                         epochs=EPOCHS_INITIAL,
#                         validation_data=(X_test, y_test),
#                         class_weight=class_weights_dict)

#     plot_training(history)

#     # print("🔓 Fine-tuning MobileNetV2...")
#     # for layer in model.layers[0].layers[:100]:
#     #     layer.trainable = False
#     # for layer in model.layers[0].layers[100:]:
#     #     layer.trainable = True

#     # model.compile(optimizer=Adam(learning_rate=1e-5),
#     #               loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     print("🔓 Fine-tuning MobileNetV2...")
#     for layer in base_model.layers[:100]:
#         layer.trainable = False
#     for layer in base_model.layers[100:]:
#         layer.trainable = True

#     model.compile(optimizer=Adam(learning_rate=1e-5),
#               loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     print("⚙ Fine-tuning model...")
#     fine_history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
#                              epochs=EPOCHS_FINE_TUNE,
#                              validation_data=(X_test, y_test),
#                              class_weight=class_weights_dict)

#     plot_training(fine_history)

#     print("🧪 Final Evaluation...")
#     loss, acc = model.evaluate(X_test, y_test)
#     print(f"✅ Final Test Accuracy: {acc * 100:.2f}%")

#     print("💾 Saving final model...")
#     model.save("construction_progress_model.h5")
#     print("✅ Model saved as 'construction_progress_model.h5'")

# # ---------------------- Predict Function ----------------------
# def predict_progress(image_path, model_path="construction_progress_model.h5"):
#     model = load_model(model_path)
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, IMAGE_SIZE)
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)
#     pred = model.predict(img)
#     predicted_class = np.argmax(pred)
#     confidence = np.max(pred) * 100
#     return f"{CLASSES[predicted_class]}% Complete ({confidence:.2f}% confidence)"

# # ---------------------- Run ----------------------
# if __name__ == "__main__":
#     main()
#     # Example test:
#     # print(predict_progress("test_image.jpg"))
