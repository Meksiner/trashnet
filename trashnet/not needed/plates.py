import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Загрузка вашей модели
model = load_model('trashnet_classifier.h5')  # Замените на путь к вашей модели

# Функция для обработки фотографии с использованием метода скользящего окна
def detect_trash_in_image(image_path, window_size=(224, 224), step_size=224, conf_threshold=0.5):
    # Загрузка изображения
    image = cv2.imread(image_path)

    if image is None:
        print("Не удалось загрузить изображение.")
        return

    # Получение размеров изображения
    h, w, _ = image.shape

    # Перебор всех окон изображения
    for y in range(0, h - window_size[1], step_size):
        for x in range(0, w - window_size[0], step_size):
            # Выделение области интереса (ROI) с помощью скользящего окна
            roi = image[y:y + window_size[1], x:x + window_size[0]]

            # Предобработка изображения (размер, нормализация и т.д.)
            input_frame = cv2.resize(roi, (224, 224))  # Предположим, что ваша модель принимает 224x224 изображения
            input_frame = np.expand_dims(input_frame, axis=0)  # Добавляем размерность для батча

            # Классификация мусора
            predictions = model.predict(input_frame)
            conf = np.max(predictions)  # Максимальная вероятность для мусора

            # Если вероятность превышает порог, считаем, что в области есть мусор
            if conf > conf_threshold:
                label = 'trash'  # Если уверенность достаточна, то это мусор
                # Рисуем bounding box вокруг фрагмента
                cv2.rectangle(image, (x, y), (x + window_size[0], y + window_size[1]), (255, 0, 0), 2)
                cv2.putText(image, f'{label} {conf:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Сохранение результата
    output_path = 'detected_trash_image.jpg'
    cv2.imwrite(output_path, image)

    # Отображение результата
    cv2.imshow("Detected Trash", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Обработанное изображение сохранено как {output_path}")

# Запуск обработки изображения
detect_trash_in_image('23.jpg')  # Замените на путь к вашему изображению

