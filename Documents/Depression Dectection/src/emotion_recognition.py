import cv2
import numpy as np

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    keras = None
    layers = None
    TF_AVAILABLE = False

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class EmotionModel:
    def __init__(self, input_shape=(48, 48, 1), num_classes=7) -> None:
        self.model = self.build_model(input_shape, num_classes) if TF_AVAILABLE else None

    def build_model(self, input_shape, num_classes):
        if not TF_AVAILABLE:
            raise ImportError('TensorFlow is required to build the emotion model.')

        inputs = keras.Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs, name='emotion_cnn')
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        return model

    def train(self, x_train, y_train, x_val, y_val, epochs=30, batch_size=64):
        if not TF_AVAILABLE:
            raise RuntimeError('TensorFlow is not installed; cannot train the model.')
        return self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
        )

    def predict_emotion(self, face_image: np.ndarray) -> str:
        if self.model is None:
            return 'Neutral'

        face = cv2.resize(face_image, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)
        predictions = self.model.predict(face)
        label_index = np.argmax(predictions, axis=1)[0]
        return EMOTION_LABELS[label_index]
