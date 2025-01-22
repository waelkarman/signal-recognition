import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os


# 1. Load the dataset
def load_data(data_dir):
    # Assuming dataset is structured with subfolders for each class
    images = []
    labels = []
    class_names = os.listdir(data_dir)
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm')):  # Aggiungi .ppm
                continue
            try:
                img = tf.keras.utils.load_img(img_path, target_size=(32, 32))  # Resize to 32x32
                img_array = tf.keras.utils.img_to_array(img) / 255.0  # Normalize images
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Errore nel caricamento dell'immagine {img_path}: {e}")
    return np.array(images), np.array(labels), class_names

# Load data from your dataset directory
data_dir = "/home/wael.karman/Documents/vision/GTSRB/Training/"
images, labels, class_names = load_data(data_dir)

train = False
if train :
    # 2. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # 3. Data augmentation
    data_augmentation = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False
    )

    # 4. Build the model
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

    # 5. Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # 6. Train the model
    history = model.fit(
        data_augmentation.flow(X_train, y_train, batch_size=32),
        validation_data=(X_test, y_test),
        epochs=15
    )

    # 7. Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    # 8. Save the model
    model.save("traffic_sign_classifier.h5")



predict_from_image = False
if predict_from_image :
    # 9. Load and predict on new images
    model = load_model("traffic_sign_classifier.h5")

    def predict_image(model, image_path):
        img = tf.keras.utils.load_img(image_path, target_size=(32, 32))
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        return class_names[predicted_class]

    # Example usage
    new_image_path = "GTSRB/Training/00027/00000_00020.ppm"
    predicted_class = predict_image(model, new_image_path)
    print(f"Predicted traffic sign: {predicted_class}")




# Carica il modello Keras (modifica con il percorso del tuo modello .h5)
model = tf.keras.models.load_model("traffic_sign_classifier.h5")

# Definisci la funzione di preprocessing per il modello (modifica se necessario)
def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converte il frame in RGB
    img = cv2.resize(img, (32, 32))  # Ridimensiona l'immagine (modifica se il tuo modello richiede dimensioni diverse)
    img = image.img_to_array(img)  # Converte l'immagine in un array numpy
    img = np.expand_dims(img, axis=0)  # Aggiungi una dimensione per il batch
    img = img / 255.0  # Normalizza l'immagine se il tuo modello lo richiede (ad esempio, se addestrato su immagini tra 0 e 1)
    return img

# Avvia il flusso video dalla fotocamera (0 è il dispositivo predefinito, usa l'indice giusto se hai più dispositivi)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessa il frame per il modello
    preprocessed_frame = preprocess_frame(frame)

    # Classificazione del frame
    predictions = model.predict(preprocessed_frame)
    predicted_class = np.argmax(predictions)  # Ottieni la classe con probabilità massima

    # Visualizza la classe predetta sul video
    cv2.putText(frame, f'Predizione: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostra il frame con la previsione
    cv2.imshow('Video Classification', frame)

    # Interrompi se premi 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera il flusso video e chiudi la finestra
cap.release()
cv2.destroyAllWindows()