import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Загрузка модели
model = load_model('trashnet_classifier.h5')

# Классы мусора (по порядку, в котором обучалась модель)
class_labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Функция обработки видеопотока с веб-камеры
def detect_trash_from_webcam(conf_threshold=0.5):
    cap = cv2.VideoCapture(0)  # 0 - основная веб-камера

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Ошибка чтения кадра!")
            break

        # Изменение размера и предобработка изображения
        input_frame = cv2.resize(frame, (224, 224))
        input_frame = np.expand_dims(input_frame, axis=0)

        # Классификация изображения
        predictions = model.predict(input_frame)
        class_id = np.argmax(predictions)  # Класс с наибольшей вероятностью
        confidence = np.max(predictions)  # Уверенность модели

        # Проверяем порог уверенности
        if confidence > conf_threshold:
            label = f"{class_labels[class_id]} ({confidence:.2f})"
        else:
            label = "Unknown"

        # Вывод информации на экран
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Trash Detection", frame)

        # Выход по клавише 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Запуск видеопотока с веб-камеры
detect_trash_from_webcam()
