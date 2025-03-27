import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Параметры
FOLDER_PATH = "C://Users/Koraku/trashnet/dataset-re"
MODEL_PATH = "trashnet_classifier.h5"
IMG_SIZE = 224  # Размер входного изображения для модели

# Метки классов (должны соответствовать порядку в модели)
class_labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Загружаем модель
model = load_model(MODEL_PATH)

# Функция для загрузки и предобработки изображения
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Загружаем изображение
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Конвертируем в RGB
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Изменяем размер
    return img

# Получаем список файлов изображений в папке
image_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith((".jpg", ".png", ".jpeg"))]

# Обрабатываем изображения
for image_file in image_files:
    image_path = os.path.join(FOLDER_PATH, image_file)
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Добавляем размерность батча

    # Делаем предсказание
    predictions = model.predict(img)[0]  # Получаем массив вероятностей (6 классов)

    # Определяем класс с максимальной вероятностью
    max_index = np.argmax(predictions)  # Индекс наибольшей вероятности
    max_label = class_labels[max_index]  # Название класса
    max_prob = predictions[max_index]  # Вероятность

    # Выводим информацию
    print(f"{image_file}: {max_label} ({max_prob:.2%})")

    # Отображаем изображение с предсказанным классом
    img_show = cv2.imread(image_path)
    img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)  # Для корректного отображения
    plt.imshow(img_show)
    plt.title(f"{max_label}: {max_prob:.2%}")
    plt.axis("off")
    plt.show()
