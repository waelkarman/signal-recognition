import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os

def load_data(data_dir):
    images = []
    labels = []
    class_names = os.listdir(data_dir)
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm')):  
                continue
            try:
                img = tf.keras.utils.load_img(img_path, target_size=(32, 32))  
                img_array = tf.keras.utils.img_to_array(img) / 255.0  
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Errore nel caricamento dell'immagine {img_path}: {e}")
    return np.array(images), np.array(labels), class_names

data_dir = "/home/wael.karman/Documents/vision/GTSRB/Training/"
images, labels, class_names = load_data(data_dir)


train = False
if train :
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    data_augmentation = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(class_names), activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    history = model.fit(
        data_augmentation.flow(X_train, y_train, batch_size=32),
        validation_data=(X_test, y_test),
        epochs=15
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    model.save("traffic_sign_classifier.h5")



predict_from_image = False
if predict_from_image :
    
    model = load_model("traffic_sign_classifier.h5")

    def predict_image(model, image_path):
        img = tf.keras.utils.load_img(image_path, target_size=(32, 32))
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        return class_names[predicted_class]

    new_image_path = "GTSRB/Training/00027/00000_00020.ppm"
    predicted_class = predict_image(model, new_image_path)
    print(f"Predicted traffic sign: {predicted_class}")



model = tf.keras.models.load_model("traffic_sign_classifier.h5")

def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, (32, 32))  
    img = image.img_to_array(img)  
    img = np.expand_dims(img, axis=0)  
    img = img / 255.0  
    return img

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # preprocessed_frame = preprocess_frame(frame)
    # predictions = model.predict(preprocessed_frame)

    frame_resized = cv2.resize(frame, (32, 32))
    frame_array = tf.keras.utils.img_to_array(frame_resized) / 255.0
    frame_array = np.expand_dims(frame_array, axis=0) 
    predictions = model.predict(frame_array)

    predicted_class = np.argmax(predictions)  
    cv2.putText(frame, f'Predizione: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()