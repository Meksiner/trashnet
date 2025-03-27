import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Функция для загрузки и предобработки изображения
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Функция для выполнения предсказания класса мусора
def predict_trash_class(model, img_path, class_names):
    processed_img = load_and_preprocess_image(img_path)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_names[predicted_class], confidence


# Функция для выбора файла и предсказания
def upload_and_predict():
    file_path = filedialog.askopenfilename()
    if file_path:
        predicted_class, confidence = predict_trash_class(model, file_path, class_names)
        print(f"Предсказанный класс: {predicted_class} (уверенность: {confidence:.2f})")

# Загрузка сохраненной модели
model = tf.keras.models.load_model("trashnet_classifier.h5")

# Классы (укажите реальные классы)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Интерфейс tkinter
root = tk.Tk()
root.title("Загрузка изображения")
btn_upload = tk.Button(root, text="Загрузить изображение", command=upload_and_predict)
btn_upload.pack(pady=20)
root.mainloop()
