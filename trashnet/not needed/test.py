import numpy as np
import cv2
from tensorflow.keras.models import load_model

# 🔹 Классы мусора
class_labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# 🔹 Загрузка модели
model = load_model("trashnet_classifier.h5")
print("✅ Модель загружена")

# 🔹 Путь к изображению (замените на своё)
image_path = "glass8.jpg"
image = cv2.imread(image_path)

if image is None:
    print("❌ Ошибка загрузки изображения")
    exit()

# 🔹 Параметры разбиения изображения
grid_size = 4  # Количество разбиений (4x4)
h, w, _ = image.shape  # Размеры изображения
cell_h, cell_w = h // grid_size, w // grid_size  # Размер одного квадрата

# 🔹 Обработка каждого квадрата
for i in range(grid_size):
    for j in range(grid_size):
        # 🔹 Вырезаем часть изображения
        x1, y1 = j * cell_w, i * cell_h
        x2, y2 = (j + 1) * cell_w, (i + 1) * cell_h
        patch = image[y1:y2, x1:x2]

        # 🔹 Подготовка фрагмента для предсказания
        img = cv2.resize(patch, (224, 224))
        img = np.expand_dims(img, axis=0)  # Добавляем размерность батча

        # 🔹 Выполнение предсказания
        pred = model.predict(img)
        label_index = np.argmax(pred)  # Индекс класса
        confidence = np.max(pred)  # Уверенность модели
        label_text = class_labels[label_index]  # Название класса

        # 🔹 Рисуем квадрат и надпись
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label_text} ({confidence:.2f})", (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# 🔹 Показ изображения
cv2.imshow("Grid Classification", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
